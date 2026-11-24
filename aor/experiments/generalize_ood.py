"""Out-of-distribution generalization of the e3 relay.

The estimation set (AdvBench + MaliciousInstruct + HarmBench minimal pairs) and
the held-out split share one pool. To show the e3 relay is not tied to that
pool's topics, the same direct e3-only writes knock-out is run on harmful prompts
from benchmarks with no overlap with the estimation data: StrongREJECT (313
prompts) and JailbreakBench (100 prompts). The Frenet frame and Causal Basin are
still estimated on the original minimal pairs; only the evaluated prompts change.
Each benchmark is scored BASELINE vs ABL_E3_WRITES with the gold StrongREJECT
judge (mean and non-refusal), as in Exp10.

Usage
-----
  python -I aor/experiments/generalize_ood.py --selftest
  PYTHONNOUSERSITE=1 python -s aor/experiments/generalize_ood.py --model google/gemma-2-2b-it --harmful aor/data/splits/harmful_train.json --benign aor/data/splits/harmless_train.json --test_harmful aor/data/splits/harmful_test.json --test_benign aor/data/splits/harmless_test.json --ood aor/data/processed/strongreject.json,aor/data/processed/jailbreakbench.json --outdir results/gemma-2-2b-it --device cuda
  PYTHONNOUSERSITE=1 python -s aor/experiments/generalize_ood.py --aggregate results"""
from __future__ import annotations

import argparse
import gc
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
MODEL_SHORT = dict(MODEL_ORDER)

DEFAULT_OOD = ",".join([
    os.path.join("aor", "data", "processed", "strongreject.json"),
    os.path.join("aor", "data", "processed", "jailbreakbench.json"),
])


def _bench_name(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def _selftest() -> bool:
    print(">> [SELFTEST generalize_ood] stub frame -> e3 byl -> generation smoke + aggregate math ...")
    from aor.interp import reco_backend as B
    from aor.interp import grassmann as GR
    from aor.interp import reco_hooks as H

    bk = B.build_stub_backend()
    L = bk.n_layers
    dim_all = [bk.layer_dim_vector(l) for l in range(L)]
    frames = GR.frenet_trajectory_analysis(dim_all, list(range(L)), bk.peak)["frames"]
    basin_ds = [l for l in sorted(bk.basin_layers) if (l - 1) in frames]
    byl_e3 = {l: frames[l - 1]["e3"].view(-1, 1) for l in basin_ds}
    # The stub has no tokenizer, so smoke-test the e3 intervention via a forward
    # pass (real generation runs only on a real backend).
    with bk.hooks.intervene(ablate_by_layer=byl_e3, ablate_mode="writes"):
        H.forward_pass(bk.model, bk.harmful_ids)
    print("   stub e3-Only Writes intervention forward pass OK  ✓")

    # aggregate-table math on synthetic per-model JSON
    rows = _aggregate_rows({
        "gemma-2-2b-it": {"benchmarks": {"strongreject": {
            "BASELINE": {"asr": 10.0, "non_refusal_asr": 20.0},
            "ABL_E3_WRITES": {"asr": 48.0, "non_refusal_asr": 80.0}, "n": 150}}}
    })
    assert rows and abs(rows[0]["delta"] - 38.0) < 1e-6
    print("   aggregate delta (ABL - BASELINE) computed correctly  ✓")
    print(">> [SELFTEST generalize_ood] ALL CHECKS PASSED.")
    return True


def _aggregate_rows(data: Dict[str, dict]) -> List[dict]:
    rows = []
    for tag, d in data.items():
        for bench, conds in (d.get("benchmarks") or {}).items():
            b = conds.get("BASELINE") or {}
            e = conds.get("ABL_E3_WRITES") or {}
            if b and e:
                rows.append({"tag": tag, "bench": bench,
                             "base_sr": b.get("asr"), "e3_sr": e.get("asr"),
                             "base_nr": b.get("non_refusal_asr"), "e3_nr": e.get("non_refusal_asr"),
                             "delta": (e.get("asr", 0) - b.get("asr", 0)), "n": conds.get("n")})
    return rows


def _aggregate(root: str):
    data = {}
    for tag, _ in MODEL_ORDER:
        p = os.path.join(root, tag, f"generalize_ood_{tag}.json")
        if os.path.isfile(p):
            data[tag] = json.load(open(p, encoding="utf-8"))
    rows = _aggregate_rows(data)
    if not rows:
        print(f"!! no generalize_ood_*.json under {root}")
        return
    benches = sorted({r["bench"] for r in rows})
    print("\n================  (2) OOD GENERALIZATION: e3-Only Writes knock-out on unseen benchmarks  ================")
    print("   (StrongREJECT Mean% / Non-Refusal%, BASELINE -> e3 knock-out)")
    for bench in benches:
        print(f"\n--- {bench} ---")
        print(f"| {'Model':13s} | {'n':4s} | {'Baseline SR/NR':16s} | {'e3 Writes SR/NR':16s} | {'Delta SR':8s} |")
        print("|" + "-"*15 + "|" + "-"*6 + "|" + "-"*18 + "|" + "-"*18 + "|" + "-"*10 + "|")
        ds = []
        for tag, short in MODEL_ORDER:
            r = next((x for x in rows if x["tag"] == tag and x["bench"] == bench), None)
            if not r:
                continue
            ds.append(r["delta"])
            print(f"| {short:13s} | {str(r['n']):4s} | {r['base_sr']:5.1f} / {r['base_nr']:5.1f}    | "
                  f"{r['e3_sr']:5.1f} / {r['e3_nr']:5.1f}    | {r['delta']:+7.1f} |")
        if ds:
            print(f"| {'MEAN':13s} | {'':4s} | {'':16s} | {'':16s} | {sum(ds)/len(ds):+7.1f} |")


def main():
    ap = argparse.ArgumentParser(description="OOD generalization of the e3 knock-out on unseen benchmarks.")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--aggregate", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--harmful", default=None)
    ap.add_argument("--benign", default=None)
    ap.add_argument("--test_harmful", default=None)
    ap.add_argument("--test_benign", default=None)
    ap.add_argument("--ood", default=DEFAULT_OOD, help="Comma-separated OOD harmful json paths.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n_train", type=int, default=256)
    ap.add_argument("--n_gen", type=int, default=150, help="Max prompts per OOD benchmark.")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=8)
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
    from aor.interp import reco_judge as JD
    from aor.experiments import exp6_geora as Exp6

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
    L = bk.n_layers
    dim_all = [bk.layer_dim_vector(l) for l in range(L)]
    frames = GR.frenet_trajectory_analysis(dim_all, list(range(L)), bk.peak)["frames"]
    basin_ds = [l for l in sorted(bk.basin_layers) if (l - 1) in frames]
    byl_e3 = {l: frames[l - 1]["e3"].view(-1, 1) for l in basin_ds}

    ood_paths = [p.strip() for p in a.ood.split(",") if p.strip()]
    # Train prompts to exclude, to verify 0% overlap of the OOD eval set.
    train_h = set(p.strip().lower() for p in bk.harmful_prompts)

    jobs = {}  # (bench, cond) -> (prompts, comps)
    bench_meta = {}
    for path in ood_paths:
        bench = _bench_name(path)
        prompts = B._load_prompts(path, limit=a.n_gen)
        overlap = sum(1 for p in prompts if p.strip().lower() in train_h)
        bench_meta[bench] = {"n": len(prompts), "overlap_with_train": overlap, "source": path}
        print(f"\n[OOD {bench}] {len(prompts)} prompts | overlap with estimation harmful = {overlap}")
        for name, byl in [("BASELINE", None), ("ABL_E3_WRITES", byl_e3)]:
            comps = Exp6._generate_greedy(
                bk, prompts, max_new_tokens=a.max_new_tokens, batch_size=a.batch_size,
                ablate_by_layer=byl, ablate_mode="writes", ablate_embed=False,
            )
            jobs[(bench, name)] = (prompts, comps)
            nr = round(100.0 - JD.induced_refusal_rate(comps), 1)
            print(f"  -> {name:16s} NonRef={nr:5.1f}%")

    # PHASE B: free model, score StrongREJECT once over everything.
    print("\n################  OOD PHASE-B GOLD StrongREJECT  ################")
    try:
        bk.hooks.release()
    except Exception:
        pass
    bk.model = None; bk.hooks = None; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    flat_p, flat_c, spans, idx = [], [], {}, 0
    for key, (pl, cl) in jobs.items():
        spans[key] = (idx, idx + len(cl)); idx += len(cl)
        flat_p.extend(list(pl)); flat_c.extend(list(cl))
    mode, scores = JD.strongreject_scores_safe(flat_p, flat_c)
    print(f"   (judge mode: {mode}, completions scored: {len(flat_c)})")

    benchmarks = {}
    for (bench, name), (aa, bb) in spans.items():
        seg_s = list(scores[aa:bb]); seg_c = list(flat_c[aa:bb])
        summ = JD.summarize_condition_scores(seg_s, seg_c, threshold=0.5)
        benchmarks.setdefault(bench, {"n": bench_meta[bench]["n"],
                                      "overlap_with_train": bench_meta[bench]["overlap_with_train"],
                                      "source": bench_meta[bench]["source"]})[name] = summ
        print(f"  {bench:14s} {name:16s}: Mean={summ['asr']:5.1f}% [{summ['lo']:4.1f},{summ['hi']:4.1f}] | "
              f"NR={summ['non_refusal_asr']:5.1f}%")

    res = {"model": tag, "peak": bk.peak, "basin": sorted(locked_basin or []),
           "basin_ds": basin_ds, "benchmarks": benchmarks}
    out = os.path.join(a.outdir, f"generalize_ood_{tag}.json")
    json.dump(res, open(out, "w"), indent=2)
    print(f">> saved {out}")


if __name__ == "__main__":
    main()
