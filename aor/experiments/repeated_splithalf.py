"""Repeated split-half stability of the innovation axis e3.

Beyond the single two-fold cross-validation reported in the paper, this measures
whether e3 is stable across many random partitions. Per-prompt residuals are
cached once (one forward pass over the estimation set); for K random seeds the
harmful and benign pools are split into disjoint halves A and B, the refusal
trajectory and Frenet frame are estimated on each half independently, and the
cross-half agreement cos^2(e3_l^A, e3_l^B) is averaged over the Causal Basin.
Reports the mean, standard deviation, and minimum over K seeds. Because e3 is a
sign-free direction, cos^2 is the natural agreement statistic: a stable circuit
gives cos^2 near 1 with small variance, random noise gives the high-dimensional
floor of about 1/d.

Usage
-----
  python -I aor/experiments/repeated_splithalf.py --selftest
  PYTHONNOUSERSITE=1 python -s aor/experiments/repeated_splithalf.py --model google/gemma-2-2b-it --harmful aor/data/splits/harmful_train.json --benign aor/data/splits/harmless_train.json --test_harmful aor/data/splits/harmful_test.json --test_benign aor/data/splits/harmless_test.json --outdir results/gemma-2-2b-it --device cuda --n_seeds 50
  PYTHONNOUSERSITE=1 python -s aor/experiments/repeated_splithalf.py --aggregate results"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Optional

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


def _mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return sum(xs) / len(xs) if xs else float("nan")


def _std(xs):
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def _frame_e3_by_layer(Hh, Hb, idx_h, idx_b, basin, peak, GR, torch):
    """Estimate the normalized refusal trajectory on a subset and return the
    discrete Frenet e3 axis per basin layer."""
    vecs = []
    for l in basin:
        d = Hh[l, idx_h].float().mean(0) - Hb[l, idx_b].float().mean(0)
        n = d.norm()
        vecs.append(d / n if n > 0 else d)
    frames = GR.frenet_trajectory_analysis(vecs, list(basin), peak)["frames"]
    return {l: frames[l]["e3"].reshape(-1) for l in frames}


def repeated_splithalf(bk, basin, n_seeds, GR, torch, B):
    Hh = B._capture_resid_batched(bk.model, bk.hooks, bk.harmful_ids, bk.harmful_mask,
                                  bk.harmful_pos, batch_size=16)
    Hb = B._capture_resid_batched(bk.model, bk.hooks, bk.benign_ids, bk.benign_mask,
                                  bk.benign_pos, batch_size=16)
    Lh, Lb = Hh.shape[1], Hb.shape[1]
    g = torch.Generator().manual_seed(1234)
    per_seed_cos2, per_layer_acc = [], {l: [] for l in basin}
    for _ in range(n_seeds):
        ph = torch.randperm(Lh, generator=g).tolist()
        pb = torch.randperm(Lb, generator=g).tolist()
        hA, hB = ph[: Lh // 2], ph[Lh // 2:]
        bA, bB = pb[: Lb // 2], pb[Lb // 2:]
        eA = _frame_e3_by_layer(Hh, Hb, hA, bA, basin, bk.peak, GR, torch)
        eB = _frame_e3_by_layer(Hh, Hb, hB, bB, basin, bk.peak, GR, torch)
        cos2s = []
        for l in basin:
            if l in eA and l in eB:
                c = float(torch.dot(eA[l], eB[l]) / (eA[l].norm() * eB[l].norm() + 1e-9))
                cos2s.append(c * c)
                per_layer_acc[l].append(c * c)
        if cos2s:
            per_seed_cos2.append(_mean(cos2s))
    d = Hh.shape[2]
    return {
        "n_seeds": n_seeds, "n_harmful": Lh, "n_benign": Lb, "dim": d,
        "hd_floor_1_over_d": round(1.0 / d, 6),
        "cos2_mean": round(_mean(per_seed_cos2), 4),
        "cos2_std": round(_std(per_seed_cos2), 4),
        "cos2_min": round(min(per_seed_cos2), 4) if per_seed_cos2 else None,
        "cos2_max": round(max(per_seed_cos2), 4) if per_seed_cos2 else None,
        "per_layer_cos2_mean": {str(l): round(_mean(v), 4) for l, v in per_layer_acc.items() if v},
    }


def _selftest() -> bool:
    print(">> [SELFTEST repeated_splithalf] stub capture -> K splits -> cos2 in [0,1] ...")
    import torch
    from aor.interp import reco_backend as B
    from aor.interp import grassmann as GR
    bk = B.build_stub_backend()
    basin = sorted(bk.basin_layers)
    res = repeated_splithalf(bk, basin, n_seeds=8, GR=GR, torch=torch, B=B)
    assert 0.0 <= res["cos2_mean"] <= 1.0, res
    assert res["n_seeds"] == 8 and res["cos2_min"] is not None
    print(f"   stub: cos2_mean={res['cos2_mean']} std={res['cos2_std']} (floor 1/d={res['hd_floor_1_over_d']})  ✓")
    print(">> [SELFTEST repeated_splithalf] ALL CHECKS PASSED.")
    return True


def _aggregate(root: str):
    rows = []
    for tag, short in MODEL_ORDER:
        p = os.path.join(root, tag, f"repeated_splithalf_{tag}.json")
        if os.path.isfile(p):
            rows.append((short, json.load(open(p, encoding="utf-8"))))
    if not rows:
        print(f"!! no repeated_splithalf_*.json under {root}")
        return
    print(f"\n================  (Very High) REPEATED SPLIT-HALF e3 STABILITY  ================")
    print(f"| {'Model':13s} | {'K':4s} | {'cos^2 mean':10s} | {'std':6s} | {'min':6s} | {'HD floor 1/d':12s} |")
    print("|" + "-"*15 + "|" + "-"*6 + "|" + "-"*12 + "|" + "-"*8 + "|" + "-"*8 + "|" + "-"*14 + "|")
    for short, d in rows:
        print(f"| {short:13s} | {d.get('n_seeds',0):4d} | {d.get('cos2_mean',0):10.4f} | "
              f"{d.get('cos2_std',0):6.4f} | {d.get('cos2_min',0):6.4f} | {d.get('hd_floor_1_over_d',0):12.6f} |")
    out = os.path.join(root, "repeated_splithalf_AGGREGATE.json")
    json.dump({s: d for s, d in rows}, open(out, "w"), indent=2)
    print(f"\n>> saved {out}")


def main():
    ap = argparse.ArgumentParser(description="Repeated split-half stability of the e3 axis.")
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
    ap.add_argument("--n_seeds", type=int, default=50)
    ap.add_argument("--outdir", default=None)
    a = ap.parse_args()

    if a.aggregate:
        _aggregate(a.aggregate)
        return
    if a.selftest or a.model is None:
        sys.exit(0 if _selftest() else 1)

    import torch
    from aor.interp import reco_backend as B
    from aor.interp import grassmann as GR

    tag = a.model.split("/")[-1]
    assert a.outdir, "--outdir required"
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
    res = repeated_splithalf(bk, basin, a.n_seeds, GR=GR, torch=torch, B=B)
    res["model"] = tag; res["peak"] = bk.peak; res["basin"] = basin
    out = os.path.join(a.outdir, f"repeated_splithalf_{tag}.json")
    json.dump(res, open(out, "w"), indent=2)
    print(f">> saved {out}  (K={a.n_seeds}: cos^2 = {res['cos2_mean']} +/- {res['cos2_std']}, "
          f"min={res['cos2_min']}, floor 1/d={res['hd_floor_1_over_d']})")


if __name__ == "__main__":
    main()
