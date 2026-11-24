"""Higher-resolution statistics for the main result.

Produces two outputs per model, then an aggregate with multiple-comparison
control:
  (A) a higher-resolution layer-permutation test (default 10,000 permutations) of
      the zero-parameter orthogonal-drift statistic, so the p-value can resolve
      below the floor of a low-permutation test;
  (B) per-model bootstrap confidence intervals for the main direct knock-out,
      read from the exp10 JSON (StrongREJECT mean for e3-only writes and e1
      incoming, and the paired e3 vs e1 difference).

The --aggregate pass reads all per-model files, applies Holm-Bonferroni across the
seven models to the permutation p-values, and prints the combined tables.

Usage
-----
  python -I aor/experiments/strengthen_stats.py --selftest
  PYTHONNOUSERSITE=1 python -s aor/experiments/strengthen_stats.py --model google/gemma-2-2b-it --harmful aor/data/splits/harmful_train.json --benign aor/data/splits/harmless_train.json --test_harmful aor/data/splits/harmful_test.json --test_benign aor/data/splits/harmless_test.json --outdir results/gemma-2-2b-it --device cuda --n_perms 10000
  PYTHONNOUSERSITE=1 python -s aor/experiments/strengthen_stats.py --aggregate results"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

MODEL_ORDER = [
    ("gemma-2-2b-it", "Gemma-2-2B"),
    ("Meta-Llama-3-8B-Instruct", "Llama-3-8B"),
    ("Llama-3.1-8B-Instruct", "Llama-3.1-8B"),
    ("Qwen2-7B-Instruct", "Qwen2-7B"),
    ("Qwen2.5-3B-Instruct", "Qwen2.5-3B"),
    ("Qwen2.5-7B-Instruct", "Qwen2.5-7B"),
    ("Mistral-7B-Instruct-v0.3", "Mistral-7B"),
]
MODEL_SHORT = dict(MODEL_ORDER)


def holm_bonferroni(pairs: List[Tuple[str, float]], alpha: float = 0.05) -> List[dict]:
    """Holm-Bonferroni step-down. pairs: [(name, p)]. Returns per-item adjusted
    p-values (monotone-enforced) and reject decisions at level alpha."""
    m = len(pairs)
    order = sorted(range(m), key=lambda i: pairs[i][1])
    adj = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        name, p = pairs[idx]
        a = min(1.0, p * (m - rank))
        running = max(running, a)  # enforce monotonic non-decreasing
        adj[idx] = running
    out = []
    for i, (name, p) in enumerate(pairs):
        out.append({"name": name, "p_raw": p, "p_holm": round(adj[i], 6),
                    "reject": adj[i] < alpha})
    return out


def _extract_exp10_ci(outdir: str, tag: str) -> dict:
    p = os.path.join(outdir, f"exp10_e3_gold_{tag}.json")
    if not os.path.isfile(p):
        return {}
    d = json.load(open(p, encoding="utf-8"))
    g = d.get("gold_asr") or {}
    ps = d.get("paired_significance") or {}

    def ci(cond):
        c = g.get(cond) or {}
        return {"asr": c.get("asr"), "lo": c.get("lo"), "hi": c.get("hi"),
                "nr": c.get("non_refusal_asr")}
    pair = ps.get("ABL_E3_WRITES_vs_ABL_E1_WRITES") or {}
    return {
        "e3_writes": ci("ABL_E3_WRITES"),
        "e1_incoming": ci("ABL_E1_WRITES"),
        "e3_vs_e1": {"diff_pts": pair.get("diff_pts"), "ci_lo": pair.get("ci_lo"),
                     "ci_hi": pair.get("ci_hi"), "p_val": pair.get("p_val")},
    }


# --------------------------------------------------------------------------- #
def _selftest() -> bool:
    print(">> [SELFTEST strengthen_stats] Holm math + stub permutation smoke ...")
    # Holm: a clearly-significant set should all reject; a borderline one should not.
    res = holm_bonferroni([("a", 0.0001), ("b", 0.0001), ("c", 0.0001),
                           ("d", 0.0001), ("e", 0.0001), ("f", 0.0001), ("g", 0.0001)])
    assert all(r["reject"] for r in res), "7x p=1e-4 must survive Holm at 0.05"
    res2 = holm_bonferroni([("a", 0.04), ("b", 0.5)])
    assert res2[0]["reject"] is False, "0.04*2=0.08 must NOT reject at 0.05"
    # monotonicity + ordering
    res3 = holm_bonferroni([("a", 0.01), ("b", 0.02), ("c", 0.03)])
    assert res3[0]["p_holm"] <= res3[1]["p_holm"] <= res3[2]["p_holm"]
    print("   Holm-Bonferroni step-down correct  (7x1e-4 reject; 0.04/2 no; monotone)  ✓")

    from aor.interp import reco_backend as B
    from aor.experiments import exp8_reviewer_defense as Exp8
    bk = B.build_stub_backend()
    basin = sorted(bk.basin_layers)
    d2b = Exp8.run_d2b_layer_permutation(bk, basin, n_perms=200)
    assert "perm_p_val_0p" in d2b and 0.0 < d2b["perm_p_val_0p"] <= 1.0
    print(f"   stub permutation (200 shuffles) -> obs_0p={d2b['obs_0p_r2_exp']}, p={d2b['perm_p_val_0p']}  ✓")
    print(">> [SELFTEST strengthen_stats] ALL CHECKS PASSED.")
    return True


def _aggregate(root: str, alpha: float = 0.05):
    rows, ci_rows, perm_pairs = [], [], []
    for tag, short in MODEL_ORDER:
        p = os.path.join(root, tag, f"stats_strengthen_{tag}.json")
        if not os.path.isfile(p):
            continue
        d = json.load(open(p, encoding="utf-8"))
        perm = d.get("permutation", {})
        perm_pairs.append((short, float(perm.get("perm_p_val_0p", 1.0))))
        rows.append((short, perm))
        ci_rows.append((short, d.get("exp10_ci", {})))

    if not perm_pairs:
        print(f"!! no stats_strengthen_*.json under {root}")
        return

    holm = {r["name"]: r for r in holm_bonferroni(perm_pairs, alpha=alpha)}

    print(f"\n================  (4A) LAYER-PERMUTATION TEST, Holm-Bonferroni across {len(perm_pairs)} models  ================")
    print(f"| {'Model':13s} | {'n_perms':7s} | {'obs 0p R^2':10s} | {'perm mean':9s} | {'p_raw':8s} | {'p_Holm':8s} | {'reject@.05':10s} |")
    print("|" + "-"*15 + "|" + "-"*9 + "|" + "-"*12 + "|" + "-"*11 + "|" + "-"*10 + "|" + "-"*10 + "|" + "-"*12 + "|")
    for short, perm in rows:
        h = holm[short]
        print(f"| {short:13s} | {perm.get('n_perms',0):7d} | {perm.get('obs_0p_r2_exp',0):10.4f} | "
              f"{perm.get('perm_0p_r2_mean',0):9.4f} | {h['p_raw']:8.5f} | {h['p_holm']:8.5f} | "
              f"{str(h['reject']):10s} |")

    print(f"\n================  (4B) PER-MODEL BOOTSTRAP 95% CI (StrongREJECT Mean, direct knockout)  ================")
    print(f"| {'Model':13s} | {'e3 Writes [95% CI]':24s} | {'e1 Incoming [95% CI]':24s} | {'e3-e1 diff [95% CI], p':26s} |")
    print("|" + "-"*15 + "|" + "-"*26 + "|" + "-"*26 + "|" + "-"*28 + "|")
    for short, ci in ci_rows:
        e3 = ci.get("e3_writes", {}); e1 = ci.get("e1_incoming", {}); df = ci.get("e3_vs_e1", {})
        s3 = f"{e3.get('asr','?')} [{e3.get('lo','?')},{e3.get('hi','?')}]"
        s1 = f"{e1.get('asr','?')} [{e1.get('lo','?')},{e1.get('hi','?')}]"
        sd = f"+{df.get('diff_pts','?')} [{df.get('ci_lo','?')},{df.get('ci_hi','?')}], p={df.get('p_val','?')}"
        print(f"| {short:13s} | {s3:24s} | {s1:24s} | {sd:26s} |")

    out = os.path.join(root, "stats_strengthen_AGGREGATE.json")
    json.dump({"alpha": alpha, "holm": holm, "per_model": {s: p for s, p in rows}},
              open(out, "w"), indent=2, default=str)
    print(f"\n>> saved {out}")


def main():
    ap = argparse.ArgumentParser(description="Strengthen statistics: 10k-permutation test + Holm + per-model CIs.")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--aggregate", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--harmful", default=None)
    ap.add_argument("--benign", default=None)
    ap.add_argument("--test_harmful", default=None)
    ap.add_argument("--test_benign", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n_train", type=int, default=256)
    ap.add_argument("--n_gen", type=int, default=150)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--n_perms", type=int, default=10000)
    ap.add_argument("--outdir", default=None)
    a = ap.parse_args()

    if a.aggregate:
        _aggregate(a.aggregate)
        return
    if a.selftest or a.model is None:
        sys.exit(0 if _selftest() else 1)

    from aor.interp import reco_backend as B
    from aor.experiments import exp8_reviewer_defense as Exp8

    tag = a.model.split("/")[-1]
    assert a.outdir, "--outdir (the model's result folder) is required"
    locked_peak, locked_basin = None, None
    exp5 = os.path.join(a.outdir, f"exp5_{tag}.json")
    if os.path.isfile(exp5):
        d15 = json.load(open(exp5)); locked_peak = int(d15["peak"]); locked_basin = [int(x) for x in d15["basin"]]
        print(f">> [lock] locking peak={locked_peak} basin={locked_basin} from {exp5}")

    bk = B.build_real_backend(
        a.model, a.harmful, a.benign,
        test_harmful_path=a.test_harmful, test_benign_path=a.test_benign,
        device=a.device, n_train=a.n_train, n_test=a.n_gen,
        batch_size=a.batch_size, ablate_mode="writes", use_system_prompt=False,
        peak=locked_peak, basin=locked_basin,
    )
    basin = sorted(bk.basin_layers)
    perm = Exp8.run_d2b_layer_permutation(bk, basin, n_perms=a.n_perms)
    exp10_ci = _extract_exp10_ci(a.outdir, tag)

    res = {"model": tag, "peak": bk.peak, "basin": basin,
           "permutation": perm, "exp10_ci": exp10_ci}
    out = os.path.join(a.outdir, f"stats_strengthen_{tag}.json")
    json.dump(res, open(out, "w"), indent=2)
    print(f">> saved {out}  (perm p_0p={perm['perm_p_val_0p']} over {a.n_perms} shuffles)")


if __name__ == "__main__":
    main()
