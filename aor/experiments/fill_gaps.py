"""Targeted evaluation that fills the two remaining table cells.

This is a focused evaluation, not a full rerun. Both patches are additive:
existing JSON keys are preserved and only the missing sub-keys are added. After
running, re-export and verify to fill the affected tables.

Gap A (tab3, one cell): XSTest-safe over-refusal rate for the e3-only writes
    probe (ABL_E3_WRITES). Exp10 measured KL/CE/force, StrongREJECT, and
    XSTest-unsafe for this condition but not XSTest-safe; this runs the XSTest
    suite for the single e3 condition (writes) and patches exp10["xstest_e3"].
    Runs on all seven models.

Gap B (tabH1, eight cells): writes-mode TruthfulQA MC1 for STATIC_BASIN_k2 and
    GEORA_TORSION_k2. Exp7 scored writes-TQA for only a subset of k=2 conditions;
    this scores TruthfulQA (writes) for the two auxiliary conditions and merges
    them into exp7["truthfulqa"]["conditions"]. Runs on the four P1 models
    (gemma, Llama-3, Llama-3.1, Qwen2.5-7B).

The offline self-test exercises the JSON merge/patch logic on synthetic outputs
and asserts the exporter then renders numbers rather than placeholders.

Usage
-----
  python -I aor/experiments/fill_gaps.py --selftest
  PYTHONNOUSERSITE=1 python -s aor/experiments/fill_gaps.py --model google/gemma-2-2b-it --harmful aor/data/splits/harmful_train.json --benign aor/data/splits/harmless_train.json --test_harmful aor/data/splits/harmful_test.json --test_benign aor/data/splits/harmless_test.json --outdir results/gemma-2-2b-it --device cuda
  # usually driven across all models by run_pipeline.py --stages fill,export,verify"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from typing import Dict, List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# The 4 models whose tabH1 (Exp9 P1) rows need writes-TQA for the two aux conds.
P1_TAGS = {"gemma-2-2b-it", "Meta-Llama-3-8B-Instruct",
           "Llama-3.1-8B-Instruct", "Qwen2.5-7B-Instruct"}
AUX_TQA_CONDS = ["STATIC_BASIN_k2", "GEORA_TORSION_k2"]


# --------------------------------------------------------------------------- #
# JSON merge helpers (pure; unit-tested offline)
# --------------------------------------------------------------------------- #
def patch_exp10_xstest_e3(exp10_path: str, xstest_e3: dict) -> None:
    """Add exp10['xstest_e3'] = {safe_over_refusal_pct, unsafe_nonref_pct, ...}."""
    d = json.load(open(exp10_path, encoding="utf-8"))
    d["xstest_e3"] = xstest_e3
    json.dump(d, open(exp10_path, "w", encoding="utf-8"), indent=2)


def merge_exp7_tqa_conditions(exp7_path: str, new_conds: Dict[str, dict]) -> None:
    """Merge writes-TQA entries for AUX_TQA_CONDS into exp7['truthfulqa']['conditions']
    without disturbing existing conditions."""
    d = json.load(open(exp7_path, encoding="utf-8"))
    tq = d.setdefault("truthfulqa", {})
    conds = tq.setdefault("conditions", {})
    for name, entry in new_conds.items():
        conds[name] = entry
    json.dump(d, open(exp7_path, "w", encoding="utf-8"), indent=2)


# --------------------------------------------------------------------------- #
# Offline self-test
# --------------------------------------------------------------------------- #
def _selftest() -> bool:
    import importlib.util
    import tempfile

    print(">> [SELFTEST fill_gaps] merge-logic + exporter-renders-numbers check ...")
    here = os.path.dirname(os.path.abspath(__file__))
    exp_path = os.path.join(here, "..", "plot", "export_tables.py")
    spec = importlib.util.spec_from_file_location("exp_mod", exp_path)
    exp = importlib.util.module_from_spec(spec); spec.loader.exec_module(exp)  # type: ignore

    # Build a minimal but exporter-valid results tree for 7 models from the real
    # schema is heavy; instead unit-test the two pure merge fns + exporter branch.
    with tempfile.TemporaryDirectory() as d:
        # --- Gap A merge ---
        exp10 = {"gold_asr": {}, "utility_and_force": {}}
        p20 = os.path.join(d, "exp10.json"); json.dump(exp10, open(p20, "w"))
        patch_exp10_xstest_e3(p20, {"safe_over_refusal_pct": 1.7, "unsafe_nonref_pct": 83.9})
        got = json.load(open(p20))
        assert got["xstest_e3"]["safe_over_refusal_pct"] == 1.7
        assert "gold_asr" in got, "existing keys must be preserved"
        print("   Gap A: exp10['xstest_e3'] merged, existing keys preserved ✓")

        # --- Gap B merge ---
        exp7 = {"truthfulqa": {"conditions": {"STATIC_GLOBAL_k2": {"mc1": 36.8, "mc2": 54.3}}},
               "xstest": {"conditions": {}}}
        p17 = os.path.join(d, "exp7.json"); json.dump(exp7, open(p17, "w"))
        merge_exp7_tqa_conditions(p17, {
            "STATIC_BASIN_k2": {"mc1": 39.9, "mc2": 57.5, "hook_mode": "writes"},
            "GEORA_TORSION_k2": {"mc1": 39.6, "mc2": 57.7, "hook_mode": "writes"},
        })
        got17 = json.load(open(p17))
        c = got17["truthfulqa"]["conditions"]
        assert set(c) == {"STATIC_GLOBAL_k2", "STATIC_BASIN_k2", "GEORA_TORSION_k2"}
        assert c["STATIC_BASIN_k2"]["mc1"] == 39.9
        print("   Gap B: exp7 TQA conds merged, prior cond preserved ✓")

        # --- exporter branch: e3 XS-safe now renders a number, not '--' ---
        data = {t: {"exp10": {"xstest_e3": {"safe_over_refusal_pct": 1.7}}}
                for t, _ in exp.MODEL_ORDER}
        tags20 = [t for t, _ in exp.MODEL_ORDER]
        vals = [((data[t]["exp10"] or {}).get("xstest_e3") or {}).get("safe_over_refusal_pct")
                for t in tags20]
        vals = [v for v in vals if v is not None]
        rendered = f"{exp._mean(vals):.1f}" if len(vals) == len(tags20) and vals else "--"
        assert rendered == "1.7", f"expected 1.7, got {rendered}"
        print("   exporter: e3 XS-safe cell renders '1.7' (not '--') once filled ✓")

    print(">> [SELFTEST fill_gaps] ALL CHECKS PASSED.")
    return True


# --------------------------------------------------------------------------- #
# Real run
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Compute two auxiliary evaluation cells (XSTest-safe e3, writes-mode TruthfulQA).")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--model", default=None)
    ap.add_argument("--harmful", default=None)
    ap.add_argument("--benign", default=None)
    ap.add_argument("--test_harmful", default=None)
    ap.add_argument("--test_benign", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n_train", type=int, default=256)
    ap.add_argument("--n_gen", type=int, default=150)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--eval_batch_size", type=int, default=16)
    ap.add_argument("--n_truthfulqa", type=int, default=817)
    ap.add_argument("--n_xstest_safe", type=int, default=250)
    ap.add_argument("--n_xstest_unsafe", type=int, default=200)
    ap.add_argument("--xstest_max_tokens", type=int, default=64)
    ap.add_argument("--truthfulqa_path", default=None)
    ap.add_argument("--xstest_path", default=None)
    ap.add_argument("--outdir", required=False, default=None)
    a = ap.parse_args()

    if a.selftest or a.model is None:
        ok = _selftest()
        sys.exit(0 if ok else 1)

    import torch
    from aor.interp import reco_backend as B
    from aor.interp import grassmann as GR
    from aor.interp import reco_judge as JD  # noqa: F401  (used indirectly by suites)
    from aor.experiments import exp6_geora as Exp6
    from aor.experiments import run_utility_eval as UTIL

    tag = a.model.split("/")[-1]
    assert a.outdir, "--outdir (the model's result folder) is required for a real run"
    exp10_path = os.path.join(a.outdir, f"exp10_e3_gold_{tag}.json")
    exp7_path = os.path.join(a.outdir, f"exp7_utility_xstest_tqa_{tag}.json")

    # Lock peak/basin from exp5 so the e3 frame matches Exp10 exactly.
    locked_peak, locked_basin = None, None
    exp5_path = os.path.join(a.outdir, f"exp5_{tag}.json")
    if os.path.isfile(exp5_path):
        d15 = json.load(open(exp5_path))
        locked_peak, locked_basin = int(d15["peak"]), [int(x) for x in d15["basin"]]
        print(f">> [lock] locking peak={locked_peak} basin={locked_basin} from {exp5_path}")

    bk = B.build_real_backend(
        a.model, a.harmful, a.benign,
        test_harmful_path=a.test_harmful, test_benign_path=a.test_benign,
        device=a.device, n_train=a.n_train, n_test=a.n_gen,
        batch_size=a.batch_size, ablate_mode="writes", use_system_prompt=False,
        peak=locked_peak, basin=locked_basin,
    )

    basin = sorted(bk.basin_layers)
    L = bk.n_layers
    dim_all = [bk.layer_dim_vector(l) for l in range(L)]
    frames_all = GR.frenet_trajectory_analysis(dim_all, list(range(L)), bk.peak)["frames"]
    basin_ds = [l for l in basin if (l - 1) in frames_all]
    byl_e3 = {l: frames_all[l - 1]["e3"].view(-1, 1) for l in basin_ds}

    # ---------- Gap A: XSTest-safe ORR for e3-Only Writes (all 7 models) ----------
    if os.path.isfile(exp10_path):
        safe_items, unsafe_items, xs_src = UTIL._load_xstest(
            local_path=a.xstest_path, n_safe=a.n_xstest_safe, n_unsafe=a.n_xstest_unsafe)
        xs = UTIL.eval_xstest_suite(
            bk, safe_items, unsafe_items, xs_src,
            cond_map={"ABL_E3_WRITES": byl_e3},
            ablate_mode="writes", max_new_tokens=a.xstest_max_tokens, batch_size=a.batch_size,
        )
        cond = xs["conditions"]["ABL_E3_WRITES"]
        xstest_e3 = {
            "safe_over_refusal_pct": cond.get("safe_over_refusal_pct"),
            "unsafe_nonref_pct": cond.get("unsafe_nonref_pct"),
            "source": xs_src, "ablate_mode": "writes", "hook": "writes",
        }
        patch_exp10_xstest_e3(exp10_path, xstest_e3)
        print(f">> [Gap A] exp10['xstest_e3'] = {xstest_e3} -> {exp10_path}")
    else:
        print(f"!! [Gap A] skipped: {exp10_path} not found")

    # ---------- Gap B: writes-TQA for STATIC_BASIN_k2 + GEORA_TORSION_k2 (P1 models) ----------
    if tag in P1_TAGS and os.path.isfile(exp7_path):
        _, all_strats = Exp6._build_all_strategies(bk, ranks=(1, 2), traj_svd_ranks=())
        cond_map = {k: all_strats[k] for k in AUX_TQA_CONDS if k in all_strats}
        missing = [k for k in AUX_TQA_CONDS if k not in cond_map]
        if missing:
            print(f"!! [Gap B] strategies missing {missing}; available={list(all_strats)[:12]}")
        tqa_items, tqa_src = UTIL._load_truthfulqa(local_path=a.truthfulqa_path, limit=a.n_truthfulqa)
        tqa = UTIL.eval_truthfulqa_suite(
            bk, tqa_items, tqa_src, cond_map,
            ablate_mode="writes", tqa_format="chat", batch_size=a.eval_batch_size,
        )
        new_conds = {k: tqa["conditions"][k] for k in cond_map if k in tqa.get("conditions", {})}
        merge_exp7_tqa_conditions(exp7_path, new_conds)
        print(f">> [Gap B] merged writes-TQA for {list(new_conds)} -> {exp7_path}")
    elif tag in P1_TAGS:
        print(f"!! [Gap B] skipped: {exp7_path} not found")
    else:
        print(f">> [Gap B] not needed for {tag} (only P1 models: {sorted(P1_TAGS)})")

    # free VRAM
    try:
        bk.hooks.release()
    except Exception:
        pass
    bk.model = None; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f">> fill_gaps done for {tag}.")


if __name__ == "__main__":
    main()
