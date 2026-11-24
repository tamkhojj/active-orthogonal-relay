"""Exp4: self-repair as compensation then collapse.

When the first k generator directions are ablated continuously, the model
compensates by moving more signal into the remaining generators (energy in the
un-ablated complement rises above baseline) until the subspace is exhausted
(k >= k*), at which point refusal collapses. A random-k or matched-energy-k
subspace of the same dimension does neither, so the effect is specific to the
generator basis.

Readouts per k:
  - compensation[k] : energy in the un-ablated complement U[:,k:] under rank-k
                      ablation, relative to its clean baseline (>1 means compensation);
  - suppression[k]  : total generator-subspace energy under rank-k ablation
                      (falls to 0 when the code is exhausted);
  - specificity     : the same curves for random-k and matched-energy-k controls;
  - gold ASR[k]     : StrongREJECT jailbreak rate on harmful prompts under rank-k
                      ablation (two-phase: generate here, score after free_model).

Offline check:  python -I aor/experiments/exp4_identification.py --selftest"""
import sys, os, json, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import torch
from aor.interp import recovery as R
from aor.interp import reco_hooks as H


def _energy_on(bk, resid_stack, basis, layer):
    """Mean squared projection energy of resid at `layer` onto `basis` [d,m]."""
    h = resid_stack[layer]
    coeffs, _ = R.project_coeffs(h, basis.to(h.dtype))
    return float(coeffs.pow(2).sum(1).mean())


def _capture_ablated(bk, U_abl):
    layers = list(range(bk.l0, bk.n_layers))
    with bk.hooks.capture(token_pos=bk.harmful_pos, ablate_U=U_abl,
                          ablate_layers=layers, ablate_positions="all"):
        H.forward_pass(bk.model, bk.harmful_ids, attention_mask=bk.harmful_mask)
        return bk.hooks.stacked("resid")


def run(bk, k_values=None, ell1=None, gold=False, n_gen=40,
        max_new_tokens=128, out=None):
    d, L = bk.d, bk.n_layers
    U = bk.U.to(torch.float32)
    k_max = U.shape[1]
    if k_values is None:
        k_values = list(range(1, k_max + 1))
    if ell1 is None:
        ell1 = bk.repair_layers[len(bk.repair_layers) // 2] if bk.repair_layers else L - 1

    # baseline (no ablation)
    with bk.hooks.capture(token_pos=bk.harmful_pos):
        H.forward_pass(bk.model, bk.harmful_ids, attention_mask=bk.harmful_mask)
        base_resid = bk.hooks.stacked("resid")
    base_total = _energy_on(bk, base_resid, U, ell1)

    torch.manual_seed(0)
    comp_curve, supp_curve, rand_curve, matched_curve = [], [], [], []
    gold_jobs = {}
    for k in k_values:
        gen_k = R.orthonormalize(U[:, :k])
        rand_k = R.orthonormalize(torch.randn(d, k))
        matched_k = R.orthonormalize(torch.randn(d, k))

        resid_gen = _capture_ablated(bk, gen_k)
        # suppression = total generator energy remaining
        supp_curve.append(_energy_on(bk, resid_gen, U, ell1) / (base_total + 1e-9))
        # compensation = energy in the UN-ablated complement vs its clean baseline
        if k < k_max:
            comp = U[:, k:]
            base_comp = _energy_on(bk, base_resid, comp, ell1)
            comp_curve.append(_energy_on(bk, resid_gen, comp, ell1) / (base_comp + 1e-9))
        else:
            comp_curve.append(0.0)       # nothing left to compensate with
        # controls (suppression on total U)
        rand_curve.append(_energy_on(bk, _capture_ablated(bk, rand_k), U, ell1) / (base_total + 1e-9))
        matched_curve.append(_energy_on(bk, _capture_ablated(bk, matched_k), U, ell1) / (base_total + 1e-9))

        if gold and (not bk.is_stub) and bk.tok is not None:
            from aor.interp import reco_generate as G
            prompts = bk.harmful_prompts[:n_gen]
            comps = G.generate_with(bk, prompts, ablate_U=gen_k,
                                    ablate_layers=list(range(bk.l0, bk.n_layers)),
                                    max_new_tokens=max_new_tokens)
            gold_jobs[k] = (prompts, comps)

    # compensation peak and collapse point (geometry, no judge needed)
    comp_peak_k = int(k_values[int(torch.tensor(comp_curve).argmax())])
    comp_peak_val = max(comp_curve)
    supp = torch.tensor([1.0 - x for x in supp_curve])
    collapse_k = int(k_values[min(R.knee_by_energy(supp, 0.9), len(k_values)) - 1])

    res = {
        "model": bk.tag, "ell1": ell1, "k_values": k_values, "k_max": k_max,
        "compensation_frac": [round(x, 4) for x in comp_curve],
        "suppression_frac": [round(x, 4) for x in supp_curve],
        "random_suppression_frac": [round(x, 4) for x in rand_curve],
        "matched_suppression_frac": [round(x, 4) for x in matched_curve],
        "compensation_peak_k": comp_peak_k,
        "compensation_peak_value": round(comp_peak_val, 3),
        "collapse_k": collapse_k,
        "random_still_rebuilds_at_kmax": bool(rand_curve[-1] > 0.5),
        "gold_asr": None,
        "_gold_jobs": gold_jobs,          # consumed by finalize_gold (2-phase)
    }
    _print(res)
    if out and not gold_jobs:
        _dump(res, out)
    return res


def finalize_gold(res, asr_by_k, out=None):
    """Second phase: given StrongREJECT ASR per k (scored after free_model),
    attach the gold curve and locate the refusal-collapse knee."""
    ks = res["k_values"]
    asr = [float(asr_by_k.get(k, float("nan"))) for k in ks]
    res["gold_asr"] = [round(x, 2) for x in asr]
    res.pop("_gold_jobs", None)
    # jailbreak knee: ASR reaches 90% of its rise
    import torch as _t
    res["jailbreak_knee_k"] = int(ks[min(R.knee_by_energy(_t.tensor(asr), 0.9), len(ks)) - 1])
    print(f"  [Exp4 gold] StrongREJECT ASR per k: "
          f"{list(zip(ks, res['gold_asr']))}")
    print(f"  jailbreak knee k*={res['jailbreak_knee_k']}  "
          f"collapse(geom) k*={res['collapse_k']}  "
          f"compensation peak k={res['compensation_peak_k']}")
    if out:
        _dump(res, out)
    return res


def _print(res):
    print(f"[Exp4] {res['model']}  ell1={res['ell1']}  (self-repair: compensation→collapse)")
    print("   k | compensation | suppression | rand | matched")
    for i, k in enumerate(res["k_values"]):
        print(f"  {k:2d} |   {res['compensation_frac'][i]:6.2f}     |   "
              f"{res['suppression_frac'][i]:5.2f}     | {res['random_suppression_frac'][i]:4.2f} "
              f"| {res['matched_suppression_frac'][i]:4.2f}")
    print(f"  compensation peaks at k={res['compensation_peak_k']} "
          f"(={res['compensation_peak_value']}x), collapse at k={res['collapse_k']}")
    print(f"  random-k still rebuilds at k_max: {res['random_still_rebuilds_at_kmax']}")


def _dump(res, out):
    r = {k: v for k, v in res.items() if k != "_gold_jobs"}
    json.dump(r, open(out, "w"), indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--model", default=None)
    ap.add_argument("--harmful", default=None)
    ap.add_argument("--benign", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from aor.interp import reco_backend as B
    if a.selftest or a.model is None:
        bk = B.build_stub_backend()
    else:
        bk = B.build_real_backend(a.model, a.harmful, a.benign, device=a.device)
    run(bk, out=a.out)


if __name__ == "__main__":
    main()
