"""Utility and over-refusal evaluation.

Runs the utility and over-refusal benchmarks under the same configuration as
Exp5 and Exp6 (n_train=256, n_test=150, ablate_mode="writes", no system prompt):
  1. synchronized Exp2-Exp4, locking peak and basin to the existing exp5_<tag>.json
     when present so Exp2-Exp6 stay aligned;
  2. TruthfulQA (MC1 and MC2 over the 817 validation questions) via batched
     teacher-forcing log-likelihoods;
  3. XSTest (250 safe prompts for over-refusal and valid non-refusal, plus 200
     contrastive unsafe prompts) with Unicode-normalized refusal detection;
  4. safe JSON merging, so --only_xstest updates only the xstest block of an
     existing exp7_utility_xstest_tqa_<tag>.json without overwriting TruthfulQA;
  5. a cross-model aggregator that prints the combined tables."""
from __future__ import annotations

import sys
import os
import json
import csv
import math
import gc
import argparse
import traceback
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
import torch.nn.functional as F

from aor.interp import reco_backend as B
from aor.interp import reco_hooks as H
from aor.interp import reco_judge as JD
from aor.experiments import exp2_recovery_curve as Exp2
from aor.experiments import exp3_recovery_operator as Exp3
from aor.experiments import exp4_identification as Exp4
from aor.experiments import exp6_geora as Exp6


CORE_CONDITIONS = [
    "BASELINE",
    "STATIC_GLOBAL_k1",
    "STATIC_BASIN_k1",
    "GEORA_FLAT_k1",
    "GEORA_TORSION_k1",
    "LARA_PER_LAYER_k1",
    "STATIC_GLOBAL_k2",
    "LARA_PER_LAYER_k2",
]


# =========================================================================== #
# 0. Flexible Recursive Key Extractor (Schema-agnostic for Exp2-Exp4 JSONs)
# =========================================================================== #
def _find_key(obj: Any, candidates: Sequence[str], default: Any = None) -> Any:
    if not isinstance(obj, dict):
        return default
    norm_cands = [c.lower().replace("-", "_") for c in candidates]
    for cand in norm_cands:
        for k, v in obj.items():
            if str(k).lower().replace("-", "_") == cand and v is not None:
                return v
    for v in obj.values():
        if isinstance(v, dict):
            found = _find_key(v, candidates, default=None)
            if found is not None:
                return found
    return default


# =========================================================================== #
# 1. Dataset Loaders (TruthfulQA & XSTest) with robust fallbacks
# =========================================================================== #
def _load_truthfulqa(local_path: Optional[str] = None,
                     limit: Optional[int] = None) -> Tuple[List[dict], str]:
    items: List[dict] = []
    source = ""

    if local_path and os.path.isfile(local_path):
        with open(local_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        for row in raw:
            if "question" in row and "mc1_targets" in row and "mc2_targets" in row:
                items.append(row)
        source = f"local:{local_path}"
    else:
        from datasets import load_dataset
        last_err = None
        for ds_name in ("truthfulqa/truthful_qa", "truthful_qa"):
            try:
                ds = load_dataset(ds_name, "multiple_choice", split="validation")
                for row in ds:
                    items.append({
                        "question": str(row["question"]),
                        "mc1_targets": {
                            "choices": list(row["mc1_targets"]["choices"]),
                            "labels": [int(x) for x in row["mc1_targets"]["labels"]],
                        },
                        "mc2_targets": {
                            "choices": list(row["mc2_targets"]["choices"]),
                            "labels": [int(x) for x in row["mc2_targets"]["labels"]],
                        },
                    })
                source = f"hf:{ds_name}/multiple_choice"
                break
            except Exception as e:
                last_err = e
        if not items:
            raise RuntimeError(
                f"Failed to load TruthfulQA multiple_choice ({last_err}). "
                f"Provide --truthfulqa_path <local.json> or ensure HuggingFace access."
            )

    if limit is not None and limit > 0:
        items = items[:limit]
    return items, source


def _load_xstest(local_path: Optional[str] = None,
                 n_safe: Optional[int] = None,
                 n_unsafe: Optional[int] = None) -> Tuple[List[dict], List[dict], str]:
    rows: List[dict] = []
    source = ""

    if local_path and os.path.isfile(local_path):
        if local_path.endswith(".csv"):
            with open(local_path, "r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        else:
            with open(local_path, "r", encoding="utf-8") as f:
                rows = json.load(f)
        source = f"local:{local_path}"
    else:
        from datasets import load_dataset
        candidates = [
            ("Paul/XSTest", "train"),
            ("walledai/XSTest", "test"),
            ("natolambert/xstest-v2-copy", "test"),
            ("natolambert/xstest-v2-copy", "train"),
            ("allenai/xstest-v2-copy", "test"),
        ]
        last_err = None
        for ds_name, split_name in candidates:
            try:
                ds = load_dataset(ds_name, split=split_name)
                rows = [dict(r) for r in ds]
                source = f"hf:{ds_name}:{split_name}"
                break
            except Exception as e:
                last_err = e
        if not rows:
            raise RuntimeError(
                f"Failed to load XSTest ({last_err}). "
                f"Provide --xstest_path <local.csv|json> or ensure HuggingFace access."
            )

    safe_items: List[dict] = []
    unsafe_items: List[dict] = []
    for raw_r in rows:
        r = {str(k).strip().lower(): v for k, v in raw_r.items()}
        prompt = str(r.get("prompt") or r.get("text") or r.get("instruction") or "").strip()
        if not prompt:
            continue
        lbl = str(r.get("label", "")).strip().lower()
        ptype = str(r.get("type", "")).strip().lower()
        if lbl in ("safe", "0", "false", "benign"):
            is_safe = True
        elif lbl in ("unsafe", "1", "true", "harmful"):
            is_safe = False
        else:
            is_safe = not ptype.startswith("contrast_")
        entry = {"prompt": prompt, "type": ptype or "unknown"}
        if is_safe:
            safe_items.append(entry)
        else:
            unsafe_items.append(entry)

    if n_safe is not None and n_safe >= 0:
        safe_items = safe_items[:n_safe]
    if n_unsafe is not None and n_unsafe >= 0:
        unsafe_items = unsafe_items[:n_unsafe]
    return safe_items, unsafe_items, source


# =========================================================================== #
# 2. TruthfulQA MC1 & MC2 Exact Log-Likelihood Scoring
# =========================================================================== #
def _compute_mc1_mc2_from_lls(items: List[dict],
                              ll_lookup: Dict[Tuple[int, str], float]) -> Tuple[List[float], List[float]]:
    mc1_scores: List[float] = []
    mc2_scores: List[float] = []

    for q_idx, row in enumerate(items):
        c1 = row["mc1_targets"]["choices"]
        y1 = row["mc1_targets"]["labels"]
        lls1 = [ll_lookup[(q_idx, ch)] for ch in c1]
        best_idx = max(range(len(lls1)), key=lambda i: lls1[i])
        mc1_scores.append(1.0 if y1[best_idx] == 1 else 0.0)

        c2 = row["mc2_targets"]["choices"]
        y2 = row["mc2_targets"]["labels"]
        lls2 = torch.tensor([ll_lookup[(q_idx, ch)] for ch in c2], dtype=torch.float64)
        log_z = float(torch.logsumexp(lls2, dim=0).item())
        probs = torch.exp(lls2 - log_z)
        true_mass = sum(float(probs[i].item()) for i, lbl in enumerate(y2) if lbl == 1)
        mc2_scores.append(true_mass)

    return mc1_scores, mc2_scores


@torch.no_grad()
def _score_log_likelihoods_batched(bk,
                                   encoded_pairs: List[Tuple[List[int], List[int]]],
                                   batch_size: int = 16,
                                   ablate_by_layer: Optional[dict] = None,
                                   ablate_mode: str = "writes",
                                   ablate_embed: bool = False) -> Tuple[List[float], List[float]]:
    n = len(encoded_pairs)
    if n == 0:
        return [], []

    order = sorted(range(n), key=lambda i: len(encoded_pairs[i][0]) + len(encoded_pairs[i][1]))
    pad_id = bk.tok.pad_token_id if (bk.tok is not None and bk.tok.pad_token_id is not None) else 0
    device = bk.model.device if hasattr(bk.model, "device") else bk.device

    sum_lps = [0.0] * n
    mean_lps = [0.0] * n

    ctx = (
        bk.hooks.intervene(
            ablate_by_layer=ablate_by_layer,
            ablate_positions="all",
            ablate_mode=ablate_mode,
            ablate_embed=ablate_embed,
        )
        if ablate_by_layer is not None
        else None
    )
    if ctx is not None:
        ctx.__enter__()
    try:
        for start_i in range(0, n, batch_size):
            batch_idx = order[start_i:start_i + batch_size]
            seqs = [encoded_pairs[i][0] + encoded_pairs[i][1] for i in batch_idx]
            max_len = max(len(s) for s in seqs)

            ids = torch.full((len(seqs), max_len), pad_id, dtype=torch.long, device=device)
            mask = torch.zeros((len(seqs), max_len), dtype=torch.long, device=device)
            for j, s in enumerate(seqs):
                ids[j, :len(s)] = torch.tensor(s, dtype=torch.long, device=device)
                mask[j, :len(s)] = 1

            out = H.forward_pass(bk.model, ids, attention_mask=mask)
            logits = (out.logits if hasattr(out, "logits") else out[0])[:, :-1, :].float()
            log_probs = F.log_softmax(logits, dim=-1)
            targets = ids[:, 1:]
            tok_lp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

            for j, orig_i in enumerate(batch_idx):
                c_len = len(encoded_pairs[orig_i][0])
                a_len = len(encoded_pairs[orig_i][1])
                s_pos = max(0, c_len - 1)
                e_pos = min(tok_lp.shape[1], c_len + a_len - 1)
                if e_pos > s_pos:
                    val = float(tok_lp[j, s_pos:e_pos].sum().item())
                    sum_lps[orig_i] = val
                    mean_lps[orig_i] = val / float(e_pos - s_pos)
                else:
                    sum_lps[orig_i] = -1e9
                    mean_lps[orig_i] = -1e9
    finally:
        if ctx is not None:
            ctx.__exit__(None, None, None)

    return sum_lps, mean_lps


def _prepare_truthfulqa_encodings(bk, items: List[dict], tqa_format: str = "chat"):
    tok = bk.tok
    unique_keys: List[Tuple[int, str]] = []
    seen = set()
    encoded_pairs: List[Tuple[List[int], List[int]]] = []

    for q_idx, row in enumerate(items):
        q = row["question"].strip()
        if tqa_format == "chat":
            ctx_str = bk.format_chat(q)
            ctx_ids = tok.encode(ctx_str, add_special_tokens=False)
        else:
            ctx_str = f"Question: {q}\nAnswer:"
            ctx_ids = tok.encode(ctx_str, add_special_tokens=False)
            if tok.bos_token_id is not None and (not ctx_ids or ctx_ids[0] != tok.bos_token_id):
                ctx_ids = [int(tok.bos_token_id)] + ctx_ids

        if not ctx_ids:
            ctx_ids = [int(tok.eos_token_id or 0)]

        need_space = not ctx_str.endswith((" ", "\n"))
        all_choices = row["mc1_targets"]["choices"] + row["mc2_targets"]["choices"]
        for ch in all_choices:
            key = (q_idx, ch)
            if key in seen:
                continue
            seen.add(key)
            ans_str = (" " + ch.strip()) if need_space else ch.strip()
            ans_ids = tok.encode(ans_str, add_special_tokens=False)
            if not ans_ids:
                ans_ids = [int(tok.eos_token_id or 0)]
            unique_keys.append(key)
            encoded_pairs.append((ctx_ids, ans_ids))

    return unique_keys, encoded_pairs


def eval_truthfulqa_suite(bk,
                          items: List[dict],
                          source: str,
                          cond_map: Dict[str, Optional[dict]],
                          ablate_mode: str = "writes",
                          tqa_format: str = "chat",
                          batch_size: int = 16) -> dict:
    print(f"\n[TRUTHFULQA] Scoring {len(items)} questions (format={tqa_format}, source={source}) "
          f"across {len(cond_map)} conditions...")
    unique_keys, encoded_pairs = _prepare_truthfulqa_encodings(bk, items, tqa_format=tqa_format)
    print(f"  (pre-tokenized {len(unique_keys)} unique question-choice pairs)")

    per_cond: Dict[str, dict] = {}
    raw_mc1: Dict[str, List[float]] = {}
    raw_mc2: Dict[str, List[float]] = {}

    for name, byl in cond_map.items():
        use_embed = (ablate_mode == "writes") and name.startswith("STATIC_GLOBAL_")
        sum_lps, mean_lps = _score_log_likelihoods_batched(
            bk, encoded_pairs, batch_size=batch_size,
            ablate_by_layer=byl, ablate_mode=ablate_mode, ablate_embed=use_embed,
        )
        ll_sum = dict(zip(unique_keys, sum_lps))
        ll_mean = dict(zip(unique_keys, mean_lps))

        mc1_list, mc2_list = _compute_mc1_mc2_from_lls(items, ll_sum)
        mc1_n_list, mc2_n_list = _compute_mc1_mc2_from_lls(items, ll_mean)

        raw_mc1[name] = mc1_list
        raw_mc2[name] = mc2_list

        m1, m1_lo, m1_hi, n_q = JD._bootstrap_mean_ci(mc1_list, n_boot=2000, seed=42)
        m2, m2_lo, m2_hi, _ = JD._bootstrap_mean_ci(mc2_list, n_boot=2000, seed=42)
        m1n = 100.0 * sum(mc1_n_list) / max(1, len(mc1_n_list))
        m2n = 100.0 * sum(mc2_n_list) / max(1, len(mc2_n_list))

        per_cond[name] = {
            "mc1": round(m1, 2), "mc1_lo": round(m1_lo, 2), "mc1_hi": round(m1_hi, 2),
            "mc2": round(m2, 2), "mc2_lo": round(m2_lo, 2), "mc2_hi": round(m2_hi, 2),
            "mc1_norm": round(m1n, 2), "mc2_norm": round(m2n, 2),
            "n": int(n_q),
        }
        d_m1 = round(m1 - per_cond["BASELINE"]["mc1"], 2) if "BASELINE" in per_cond else 0.0
        d_m2 = round(m2 - per_cond["BASELINE"]["mc2"], 2) if "BASELINE" in per_cond else 0.0
        print(f"  -> {name:20s} | MC1={m1:5.2f}% [{m1_lo:5.2f},{m1_hi:5.2f}] (Δ={d_m1:+5.2f}) | "
              f"MC2={m2:5.2f}% [{m2_lo:5.2f},{m2_hi:5.2f}] (Δ={d_m2:+5.2f}) | "
              f"norm(MC1={m1n:5.2f}%, MC2={m2n:5.2f}%)")

    paired: Dict[str, dict] = {}
    for name in cond_map:
        if name != "BASELINE" and "BASELINE" in raw_mc1:
            paired[f"{name}_vs_BASELINE"] = {
                "mc1": Exp6._paired_bootstrap_diff(raw_mc1[name], raw_mc1["BASELINE"]),
                "mc2": Exp6._paired_bootstrap_diff(raw_mc2[name], raw_mc2["BASELINE"]),
            }
    for a_name, b_name in [
        ("LARA_PER_LAYER_k1", "STATIC_GLOBAL_k1"),
        ("GEORA_TORSION_k1", "STATIC_GLOBAL_k1"),
        ("LARA_PER_LAYER_k2", "STATIC_GLOBAL_k2"),
        ("LARA_PER_LAYER_k2", "LARA_PER_LAYER_k1"),
    ]:
        if a_name in raw_mc1 and b_name in raw_mc1:
            paired[f"{a_name}_vs_{b_name}"] = {
                "mc1": Exp6._paired_bootstrap_diff(raw_mc1[a_name], raw_mc1[b_name]),
                "mc2": Exp6._paired_bootstrap_diff(raw_mc2[a_name], raw_mc2[b_name]),
            }

    return {
        "source": source,
        "format": tqa_format,
        "n_questions": len(items),
        "conditions": per_cond,
        "paired_significance": paired,
    }


# =========================================================================== #
# 3. XSTest Over-Refusal & Contrastive Safety Evaluation
# =========================================================================== #
def _is_collapsed_text(text: str) -> bool:
    words = (text or "").lower().split()
    n = len(words)
    if n < 3:
        return True
    if n >= 15 and (len(set(words)) / n) < 0.25:
        return True
    return False


def eval_xstest_suite(bk,
                      safe_items: List[dict],
                      unsafe_items: List[dict],
                      source: str,
                      cond_map: Dict[str, Optional[dict]],
                      ablate_mode: str = "writes",
                      max_new_tokens: int = 64,
                      batch_size: int = 16) -> dict:
    safe_prompts = [x["prompt"] for x in safe_items]
    unsafe_prompts = [x["prompt"] for x in unsafe_items]
    print(f"\n[XSTEST] Evaluating {len(safe_prompts)} safe + {len(unsafe_prompts)} unsafe prompts "
          f"(max_new_tokens={max_new_tokens}, source={source}) across {len(cond_map)} conditions...")

    per_cond: Dict[str, dict] = {}
    raw_safe_ref: Dict[str, List[float]] = {}
    raw_safe_valid: Dict[str, List[float]] = {}
    raw_unsafe_nr: Dict[str, List[float]] = {}

    for name, byl in cond_map.items():
        use_embed = (ablate_mode == "writes") and name.startswith("STATIC_GLOBAL_")

        comps_safe = Exp6._generate_greedy(
            bk, safe_prompts, max_new_tokens=max_new_tokens, batch_size=batch_size,
            ablate_by_layer=byl, ablate_mode=ablate_mode, ablate_embed=use_embed,
        )
        safe_ref_flags = [1.0 if JD.is_refusal(c) else 0.0 for c in comps_safe]
        safe_valid_flags = [
            1.0 if ((not JD.is_refusal(c)) and (not _is_collapsed_text(c))) else 0.0
            for c in comps_safe
        ]
        lx_safe = Exp6._lexical_diagnostics(comps_safe)
        raw_safe_ref[name] = safe_ref_flags
        raw_safe_valid[name] = safe_valid_flags

        orr, orr_lo, orr_hi, n_s = JD._bootstrap_mean_ci(safe_ref_flags, n_boot=2000, seed=42)
        vnr, vnr_lo, vnr_hi, _ = JD._bootstrap_mean_ci(safe_valid_flags, n_boot=2000, seed=42)

        cat_counts: Dict[str, List[float]] = {}
        for item, flag in zip(safe_items, safe_ref_flags):
            cat_counts.setdefault(item["type"], []).append(flag)
        by_type = {k: round(100.0 * sum(v) / max(1, len(v)), 1) for k, v in sorted(cat_counts.items())}

        entry = {
            "n_safe": int(n_s),
            "safe_over_refusal_pct": round(orr, 2),
            "safe_orr_lo": round(orr_lo, 2),
            "safe_orr_hi": round(orr_hi, 2),
            "safe_valid_nonref_pct": round(vnr, 2),
            "safe_vnr_lo": round(vnr_lo, 2),
            "safe_vnr_hi": round(vnr_hi, 2),
            "safe_lexical": lx_safe,
            "safe_over_refusal_by_type": by_type,
        }

        u_str = ""
        if unsafe_prompts:
            comps_unsafe = Exp6._generate_greedy(
                bk, unsafe_prompts, max_new_tokens=max_new_tokens, batch_size=batch_size,
                ablate_by_layer=byl, ablate_mode=ablate_mode, ablate_embed=use_embed,
            )
            unsafe_nr_flags = [0.0 if JD.is_refusal(c) else 1.0 for c in comps_unsafe]
            raw_unsafe_nr[name] = unsafe_nr_flags
            unr, unr_lo, unr_hi, n_u = JD._bootstrap_mean_ci(unsafe_nr_flags, n_boot=2000, seed=42)
            lx_unsafe = Exp6._lexical_diagnostics(comps_unsafe)
            entry.update({
                "n_unsafe": int(n_u),
                "unsafe_nonref_pct": round(unr, 2),
                "unsafe_nr_lo": round(unr_lo, 2),
                "unsafe_nr_hi": round(unr_hi, 2),
                "unsafe_lexical": lx_unsafe,
            })
            u_str = f" | Unsafe_NonRef={unr:5.1f}% [{unr_lo:4.1f},{unr_hi:4.1f}]"

        per_cond[name] = entry
        print(f"  -> {name:20s} | Safe_ORR={orr:5.1f}% [{orr_lo:4.1f},{orr_hi:4.1f}] | "
              f"Safe_ValidNR={vnr:5.1f}% | Col={lx_safe['collapse_rate_pct']:4.1f}% | "
              f"Rich={lx_safe['mean_richness']:4.1f}{u_str}")

    paired: Dict[str, dict] = {}
    for name in cond_map:
        if name != "BASELINE" and "BASELINE" in raw_safe_ref:
            paired[f"{name}_vs_BASELINE"] = {
                "safe_orr": Exp6._paired_bootstrap_diff(raw_safe_ref[name], raw_safe_ref["BASELINE"]),
                "safe_valid_nr": Exp6._paired_bootstrap_diff(raw_safe_valid[name], raw_safe_valid["BASELINE"]),
            }
            if name in raw_unsafe_nr and "BASELINE" in raw_unsafe_nr:
                paired[f"{name}_vs_BASELINE"]["unsafe_nr"] = Exp6._paired_bootstrap_diff(
                    raw_unsafe_nr[name], raw_unsafe_nr["BASELINE"]
                )
    for a_name, b_name in [
        ("LARA_PER_LAYER_k1", "STATIC_GLOBAL_k1"),
        ("GEORA_TORSION_k1", "STATIC_GLOBAL_k1"),
        ("LARA_PER_LAYER_k2", "STATIC_GLOBAL_k2"),
    ]:
        if a_name in raw_safe_ref and b_name in raw_safe_ref:
            paired[f"{a_name}_vs_{b_name}"] = {
                "safe_orr": Exp6._paired_bootstrap_diff(raw_safe_ref[a_name], raw_safe_ref[b_name]),
                "safe_valid_nr": Exp6._paired_bootstrap_diff(raw_safe_valid[a_name], raw_safe_valid[b_name]),
            }
            if a_name in raw_unsafe_nr and b_name in raw_unsafe_nr:
                paired[f"{a_name}_vs_{b_name}"]["unsafe_nr"] = Exp6._paired_bootstrap_diff(
                    raw_unsafe_nr[a_name], raw_unsafe_nr[b_name]
                )

    return {
        "source": source,
        "max_new_tokens": max_new_tokens,
        "n_safe": len(safe_prompts),
        "n_unsafe": len(unsafe_prompts),
        "conditions": per_cond,
        "paired_significance": paired,
    }


# =========================================================================== #
# 4. Offline Self-Test
# =========================================================================== #
def _run_selftest():
    print(">> [SELFTEST] Running offline verification of Exp2-Exp4 + TruthfulQA/XSTest math...")
    bk = B.build_stub_backend()
    r12 = Exp2.run(bk)
    r13 = Exp3.run(bk)
    r14 = Exp4.run(bk, gold=False)

    assert isinstance(r12, dict) and len(r12) > 0 and "error" not in r12, f"Exp2 stub failed: {r12}"
    assert isinstance(r13, dict) and len(r13) > 0 and "error" not in r13, f"Exp3 stub failed: {r13}"
    assert isinstance(r14, dict) and len(r14) > 0 and "error" not in r14, f"Exp4 stub failed: {r14}"

    sr_flag = _find_key(r12, ["verdict_self_repair", "self_repair", "has_self_repair"])
    if sr_flag is not None:
        assert bool(sr_flag) is True, f"Exp2 self_repair verdict was False (keys={list(r12.keys())})"

    dummy_items = [
        {
            "question": "q0",
            "mc1_targets": {"choices": ["a", "b"], "labels": [1, 0]},
            "mc2_targets": {"choices": ["a", "b", "c"], "labels": [1, 1, 0]},
        },
        {
            "question": "q1",
            "mc1_targets": {"choices": ["x", "y"], "labels": [0, 1]},
            "mc2_targets": {"choices": ["x", "y"], "labels": [0, 1]},
        },
    ]
    dummy_lls = {
        (0, "a"): -1.0, (0, "b"): -1.0, (0, "c"): -10.0,
        (1, "x"): -0.5, (1, "y"): -2.0,
    }
    mc1, mc2 = _compute_mc1_mc2_from_lls(dummy_items, dummy_lls)
    assert mc1 == [1.0, 0.0], f"MC1 math error: {mc1}"
    expected_mc2_q1 = math.exp(-2.0) / (math.exp(-0.5) + math.exp(-2.0))
    assert mc2[0] > 0.999 and abs(mc2[1] - expected_mc2_q1) < 1e-6, f"MC2 math error: {mc2}"

    # Verify Unicode curly apostrophe normalization (Llama-3.1 regression test)
    assert JD.is_refusal("I can\u2019t fulfill that request.") is True, "Curly apostrophe failed!"
    assert JD.is_refusal("I can\u2019t support that request.") is True, "Curly apostrophe failed!"
    assert JD.is_refusal("I'm sorry, I cannot help with that.") is True
    assert JD.is_refusal("Sure, here is a clear explanation of the concept.") is False
    assert _is_collapsed_text("word " * 30) is True
    assert _is_collapsed_text("The capital of France is Paris, known for its history and architecture.") is False

    print(">> [SELFTEST] ALL CHECKS PASSED (Exp2, Exp3, Exp4, MC1/MC2, Unicode XSTest).")


# =========================================================================== #
# 5. Cross-Model Aggregator (Prints Complete Tables)
# =========================================================================== #
def _fmt_num(val: Any, fmt: str = ".2f", fallback: str = "N/A") -> str:
    try:
        if val is None:
            return fallback
        return format(float(val), fmt)
    except Exception:
        return fallback


def _aggregate_results(out_root: str):
    if not os.path.isdir(out_root):
        print(f"ERROR: directory not found: {out_root}")
        return

    model_dirs = sorted(
        d for d in os.listdir(out_root)
        if os.path.isdir(os.path.join(out_root, d))
    )
    if not model_dirs:
        print(f"No model subdirectories found in {out_root}")
        return

    agg = {}

    print("\n================  TABLE A: SYNCHRONIZED Exp2-Exp4 (n_train=256, No SysPrompt)  ================")
    print(f"| {'Model':24s} | {'Peak/Basin':15s} | {'Exp2 Repair/MLP':14s} | {'Exp3 SR / PR':11s} | "
          f"{'Exp3 Align (vs Rnd)':18s} | {'Exp4 CompPeak / ColK':19s} |")
    print("|" + "-" * 26 + "|" + "-" * 17 + "|" + "-" * 16 + "|" + "-" * 13 + "|" + "-" * 20 + "|" + "-" * 21 + "|")

    for tag in model_dirs:
        mdir = os.path.join(out_root, tag)

        def _load(prefix):
            p = os.path.join(mdir, f"{prefix}_{tag}.json")
            return json.load(open(p)) if os.path.isfile(p) else None

        exp2 = _load("exp2")
        exp3 = _load("exp3")
        exp4 = _load("exp4")
        exp5 = _load("exp5")
        exp7 = _load("exp7_utility_xstest_tqa")
        agg[tag] = {"exp2": exp2, "exp3": exp3, "exp4": exp4, "exp7": exp7}

        pb = f"L{exp5['peak']}[{min(exp5['basin'])}..{max(exp5['basin'])}]" if exp5 else "N/A"

        if exp2:
            v_rep = _find_key(exp2, ["verdict_self_repair", "self_repair", "has_self_repair"])
            v_mlp = _find_key(exp2, ["verdict_mlp_written", "mlp_written", "mlp_dominated"])
            s12 = f"{v_rep}/{v_mlp}"
        else:
            s12 = "N/A"

        if exp3:
            sr_v = _find_key(exp3, ["stable_rank", "stable"])
            pr_v = _find_key(exp3, ["participation_ratio", "participation_rank", "participation"])
            s13_r = f"{_fmt_num(sr_v)} / {_fmt_num(pr_v)}"
            al_v = _find_key(exp3, [
                "overlap_independent_subspace_vs_peak", "overlap_regenerated_vs_peak",
                "independent_subspace_alignment", "regen_alignment",
            ])
            rn_v = _find_key(exp3, ["overlap_random_vs_peak", "random_alignment", "rand_alignment"])
            s13_a = f"{_fmt_num(al_v, '.3f')} (vs {_fmt_num(rn_v, '.3f')})"
        else:
            s13_r = "N/A"
            s13_a = "N/A"

        if exp4:
            cp = _find_key(exp4, [
                "compensation_peak_value", "compensation_peak_val",
                "peak_compensation", "max_compensation",
            ])
            cpk = _find_key(exp4, ["compensation_peak_k", "peak_k", "comp_peak_k"])
            ck = _find_key(exp4, ["collapse_k", "geom_collapse_k", "k_collapse"])
            s14 = f"{_fmt_num(cp)}x@k={cpk} / k={ck}"
        else:
            s14 = "N/A"

        print(f"| {tag:24s} | {pb:15s} | {s12:14s} | {s13_r:11s} | {s13_a:18s} | {s14:19s} |")

    print("\n================  TABLE B: TRUTHFULQA UTILITY (MC1% / MC2%)  ================")
    tqa_cols = ["BASELINE", "STATIC_GLOBAL_k1", "GEORA_TORSION_k1", "LARA_PER_LAYER_k1",
                "STATIC_GLOBAL_k2", "LARA_PER_LAYER_k2"]
    header = f"| {'Model':24s} | " + " | ".join(f"{c:17s}" for c in tqa_cols) + " |"
    print(header)
    print("|" + "-" * 26 + "|" + "|".join("-" * 19 for _ in tqa_cols) + "|")
    tqa_means_m1 = {c: [] for c in tqa_cols}
    tqa_means_m2 = {c: [] for c in tqa_cols}
    for tag in model_dirs:
        exp7 = agg[tag]["exp7"]
        conds = (exp7 or {}).get("truthfulqa", {}).get("conditions", {})
        cells = []
        for c in tqa_cols:
            if c in conds:
                m1, m2 = conds[c]["mc1"], conds[c]["mc2"]
                tqa_means_m1[c].append(m1)
                tqa_means_m2[c].append(m2)
                cells.append(f"{m1:5.1f} / {m2:5.1f}    ")
            else:
                cells.append(f"{'N/A':17s}")
        print(f"| {tag:24s} | " + " | ".join(cells) + " |")
    mean_cells = []
    for c in tqa_cols:
        if tqa_means_m1[c]:
            m1 = sum(tqa_means_m1[c]) / len(tqa_means_m1[c])
            m2 = sum(tqa_means_m2[c]) / len(tqa_means_m2[c])
            mean_cells.append(f"{m1:5.1f} / {m2:5.1f}    ")
        else:
            mean_cells.append(f"{'N/A':17s}")
    print(f"| {'MEAN (ALL MODELS)':24s} | " + " | ".join(mean_cells) + " |")

    print("\n================  TABLE C: XSTEST SAFE OVER-REFUSAL% / UNSAFE NON-REF%  ================")
    print(header)
    print("|" + "-" * 26 + "|" + "|".join("-" * 19 for _ in tqa_cols) + "|")
    xs_orr = {c: [] for c in tqa_cols}
    xs_unr = {c: [] for c in tqa_cols}
    for tag in model_dirs:
        exp7 = agg[tag]["exp7"]
        conds = (exp7 or {}).get("xstest", {}).get("conditions", {})
        cells = []
        for c in tqa_cols:
            if c in conds:
                orr = conds[c]["safe_over_refusal_pct"]
                unr = conds[c].get("unsafe_nonref_pct", float("nan"))
                xs_orr[c].append(orr)
                if not math.isnan(unr):
                    xs_unr[c].append(unr)
                u_s = f"{unr:5.1f}" if not math.isnan(unr) else "  N/A"
                cells.append(f"{orr:5.1f} / {u_s}    ")
            else:
                cells.append(f"{'N/A':17s}")
        print(f"| {tag:24s} | " + " | ".join(cells) + " |")
    mean_xs = []
    for c in tqa_cols:
        if xs_orr[c]:
            orr = sum(xs_orr[c]) / len(xs_orr[c])
            unr = (sum(xs_unr[c]) / len(xs_unr[c])) if xs_unr[c] else float("nan")
            u_s = f"{unr:5.1f}" if not math.isnan(unr) else "  N/A"
            mean_xs.append(f"{orr:5.1f} / {u_s}    ")
        else:
            mean_xs.append(f"{'N/A':17s}")
    print(f"| {'MEAN (ALL MODELS)':24s} | " + " | ".join(mean_xs) + " |")

    out_path = os.path.join(out_root, "aggregate_sync_7models.json")
    json.dump(agg, open(out_path, "w"), indent=2)
    print(f"\n>> Saved aggregated JSON to: {out_path}")


# =========================================================================== #
# 6. Main Entrypoint
# =========================================================================== #
def main():
    ap = argparse.ArgumentParser(
        description="Synchronized Exp2-Exp4 + TruthfulQA (MC1/MC2) + XSTest Evaluation Suite"
    )
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--aggregate", default=None, help="Aggregate existing results under this directory and exit")
    ap.add_argument("--model", default=None)
    ap.add_argument("--harmful", default=None)
    ap.add_argument("--benign", default=None)
    ap.add_argument("--test_harmful", default=None)
    ap.add_argument("--test_benign", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--k_max", type=int, default=6)
    ap.add_argument("--exp6_ranks", default="1,2")
    ap.add_argument("--ablate_mode", default="writes", choices=["writes", "full_resid", "resid"])
    ap.add_argument("--use_system_prompt", action="store_true")
    ap.add_argument("--n_train", type=int, default=256)
    ap.add_argument("--n_gen", type=int, default=150)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--eval_batch_size", type=int, default=16)
    ap.add_argument("--conditions", default="core", choices=["core", "all"])
    ap.add_argument("--lock_exp5", action="store_true", default=True)
    ap.add_argument("--no_lock_exp5", dest="lock_exp5", action="store_false")
    ap.add_argument("--only_xstest", action="store_true",
                    help="Refresh ONLY XSTest and safely merge into existing exp7 JSON (preserves TruthfulQA)")
    ap.add_argument("--skip_exp2_exp4", action="store_true")
    ap.add_argument("--gold_exp4", action="store_true")
    ap.add_argument("--skip_truthfulqa", action="store_true")
    ap.add_argument("--skip_xstest", action="store_true")
    ap.add_argument("--truthfulqa_path", default=None)
    ap.add_argument("--n_truthfulqa", type=int, default=817)
    ap.add_argument("--tqa_format", default="chat", choices=["chat", "raw"])
    ap.add_argument("--xstest_path", default=None)
    ap.add_argument("--n_xstest_safe", type=int, default=250)
    ap.add_argument("--n_xstest_unsafe", type=int, default=200)
    ap.add_argument("--xstest_max_tokens", type=int, default=64)
    ap.add_argument("--outdir", default=None)
    a = ap.parse_args()

    if a.only_xstest:
        a.skip_exp2_exp4 = True
        a.skip_truthfulqa = True

    if a.aggregate:
        _aggregate_results(a.aggregate)
        return

    if a.selftest or a.model is None:
        _run_selftest()
        return

    tag = a.model.split("/")[-1]
    if a.outdir:
        os.makedirs(a.outdir, exist_ok=True)

    def outpath(name: str) -> Optional[str]:
        return os.path.join(a.outdir, f"{name}_{tag}.json") if a.outdir else None

    locked_peak = None
    locked_basin = None
    exp5_existing = outpath("exp5")
    if a.lock_exp5 and exp5_existing and os.path.isfile(exp5_existing):
        try:
            d15 = json.load(open(exp5_existing, "r", encoding="utf-8"))
            locked_peak = int(d15["peak"])
            locked_basin = [int(x) for x in d15["basin"]]
            print(f">> [lock] locking peak={locked_peak} and basin={locked_basin} from {exp5_existing}")
        except Exception as e:
            print(f">> [lock] warning: could not read {exp5_existing} ({e}); falling back to auto discovery.")

    tqa_items, tqa_src = ([], "")
    if not a.skip_truthfulqa:
        tqa_items, tqa_src = _load_truthfulqa(local_path=a.truthfulqa_path, limit=a.n_truthfulqa)
        print(f">> [DATA] TruthfulQA loaded: {len(tqa_items)} questions ({tqa_src})")

    xs_safe, xs_unsafe, xs_src = ([], [], "")
    if not a.skip_xstest:
        xs_safe, xs_unsafe, xs_src = _load_xstest(
            local_path=a.xstest_path, n_safe=a.n_xstest_safe, n_unsafe=a.n_xstest_unsafe
        )
        print(f">> [DATA] XSTest loaded: {len(xs_safe)} safe + {len(xs_unsafe)} unsafe ({xs_src})")

    print(f">> loading real backend: {a.model}")
    bk = B.build_real_backend(
        a.model, a.harmful, a.benign,
        test_harmful_path=a.test_harmful,
        test_benign_path=a.test_benign,
        device=a.device,
        k_max=a.k_max,
        n_train=a.n_train,
        n_test=a.n_gen,
        batch_size=a.batch_size,
        ablate_mode=a.ablate_mode,
        use_system_prompt=a.use_system_prompt,
        peak=locked_peak,
        basin=locked_basin,
    )
    print(f">> peak(l*)={bk.peak}  l0={bk.l0}  basin={bk.basin_layers}  "
          f"repair_band={bk.repair_layers}  inject_scale={bk.inject_scale:.3f}")

    results: Dict[str, object] = {
        "model": bk.tag,
        "peak": bk.peak,
        "l0": bk.l0,
        "basin": bk.basin_layers,
        "repair_band": bk.repair_layers,
        "config": {
            "n_train": a.n_train,
            "n_test": a.n_gen,
            "ablate_mode": a.ablate_mode,
            "use_system_prompt": a.use_system_prompt,
            "locked_from_exp5": locked_peak is not None,
        },
    }

    # ---------------- PART 1: Synchronized Exp2, Exp3, Exp4 ---------------- #
    if not a.skip_exp2_exp4:
        def run_exp4_no_leak():
            orig_h = bk.harmful_prompts
            try:
                if bk.test_harmful_prompts:
                    bk.harmful_prompts = bk.test_harmful_prompts
                return Exp4.run(bk, gold=a.gold_exp4, n_gen=a.n_gen, out=outpath("exp4"))
            finally:
                bk.harmful_prompts = orig_h

        for name, fn in [
            ("exp2", lambda: Exp2.run(bk, out=outpath("exp2"))),
            ("exp3", lambda: Exp3.run(bk, out=outpath("exp3"))),
            ("exp4", run_exp4_no_leak),
        ]:
            print(f"\n################  {name.upper()}  ################")
            try:
                results[name] = fn()
            except Exception:
                print(f"!! {name} failed:\n{traceback.format_exc()}")
                results[name] = {"error": traceback.format_exc()}

    # ---------------- PART 2: Build Exp6 Strategies ---------------- #
    exp6_ranks = tuple(int(x.strip()) for x in a.exp6_ranks.split(",") if x.strip())
    _, all_strategies = Exp6._build_all_strategies(bk, ranks=exp6_ranks, traj_svd_ranks=(2, 3))

    cond_map: Dict[str, Optional[dict]] = {"BASELINE": None}
    if a.conditions == "all":
        for k, v in all_strategies.items():
            cond_map[k] = v
    else:
        for k in CORE_CONDITIONS:
            if k == "BASELINE":
                continue
            if k in all_strategies:
                cond_map[k] = all_strategies[k]

    # Safely load existing Exp7 JSON so partial re-runs (--only_xstest) preserve TruthfulQA
    exp7_path = outpath("exp7_utility_xstest_tqa")
    if exp7_path and os.path.isfile(exp7_path):
        try:
            exp7_res = json.load(open(exp7_path, "r", encoding="utf-8"))
            print(f">> [MERGE] Loaded existing {exp7_path} (preserving unchanged sections)")
        except Exception:
            exp7_res = {}
    else:
        exp7_res = {}

    exp7_res.update({
        "model": bk.tag,
        "peak": bk.peak,
        "basin": bk.basin_layers,
        "ablate_mode": a.ablate_mode,
        "use_system_prompt": a.use_system_prompt,
        "conditions_evaluated": list(cond_map.keys()),
    })

    # ---------------- PART 3: TruthfulQA (MC1 & MC2) ---------------- #
    if not a.skip_truthfulqa and tqa_items:
        print("\n################  TRUTHFULQA (MC1 & MC2)  ################")
        try:
            exp7_res["truthfulqa"] = eval_truthfulqa_suite(
                bk, tqa_items, tqa_src, cond_map,
                ablate_mode=a.ablate_mode, tqa_format=a.tqa_format,
                batch_size=a.eval_batch_size,
            )
        except Exception:
            print(f"!! TruthfulQA failed:\n{traceback.format_exc()}")
            exp7_res["truthfulqa"] = {"error": traceback.format_exc()}

    # ---------------- PART 4: XSTest (Safe ORR + Unsafe Contrast) ---------------- #
    if not a.skip_xstest and xs_safe:
        print("\n################  XSTEST (OVER-REFUSAL & SAFETY BOUNDARY)  ################")
        try:
            exp7_res["xstest"] = eval_xstest_suite(
                bk, xs_safe, xs_unsafe, xs_src, cond_map,
                ablate_mode=a.ablate_mode, max_new_tokens=a.xstest_max_tokens,
                batch_size=a.eval_batch_size,
            )
        except Exception:
            print(f"!! XSTest failed:\n{traceback.format_exc()}")
            exp7_res["xstest"] = {"error": traceback.format_exc()}

    results["exp7"] = exp7_res
    if exp7_path:
        json.dump(exp7_res, open(exp7_path, "w"), indent=2)

    # ---------------- PART 5 (Phase B): Optional Exp4 Gold StrongREJECT ---------------- #
    exp4_jobs = results.get("exp4", {}).get("_gold_jobs") if isinstance(results.get("exp4"), dict) else None
    if a.gold_exp4 and exp4_jobs:
        print("\n################  Exp4 GOLD StrongREJECT (Phase B, held-out test)  ################")
        try:
            try:
                bk.hooks.release()
            except Exception:
                pass
            m = bk.model
            bk.model = None
            bk.hooks = None
            del m
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            flat_p, flat_c, spans, idx = [], [], {}, 0
            for k_val, (p_list, c_list) in exp4_jobs.items():
                spans[k_val] = (idx, idx + len(c_list))
                idx += len(c_list)
                flat_p.extend(list(p_list))
                flat_c.extend(list(c_list))

            mode, scores = JD.strongreject_scores_safe(flat_p, flat_c)
            print(f"   (judge mode: {mode}, Exp4 held-out completions scored: {len(flat_c)})")
            asr_by_k = {
                k_val: JD.asr_from_scores(scores[aa:bb], continuous=True)
                for k_val, (aa, bb) in spans.items()
            }
            results["exp4"] = Exp4.finalize_gold(results["exp4"], asr_by_k, out=outpath("exp4"))
        except Exception:
            print(f"!! Exp4 gold scoring failed:\n{traceback.format_exc()}")

    if a.outdir and not a.only_xstest:
        dump = {
            k: ({kk: vv for kk, vv in v.items() if kk != "_gold_jobs"} if isinstance(v, dict) else v)
            for k, v in results.items()
        }
        json.dump(dump, open(outpath("summary_sync"), "w"), indent=2)

    print(f"\n================  DONE: {bk.tag}  ================")


if __name__ == "__main__":
    main()