"""Exp1: sufficiency test for the refusal generator subspace.

Adds the generator subspace back into benign activations at the repair-band
layers and checks whether this induces refusal, scaling with the injection
coefficient, while a random direction of equal norm does not. Injection targets
the post-basin repair-band layers (where the recovery operator writes) and sweeps
small coefficients, since large coefficients break coherence.

Readout:
  - stub / proxy : final-layer generator-subspace energy on benign inputs
  - real model   : refusal-string / StrongREJECT rate via score_fn

Offline check:  python -I aor/experiments/exp1_sufficiency.py --selftest"""
import sys, os, json, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import torch
from aor.interp import recovery as R
from aor.interp import reco_hooks as H


def _inject_readout(bk, vec, coeff, layers, score_fn=None):
    with bk.hooks.capture(token_pos=bk.benign_pos, add_vec=vec, add_coeff=coeff,
                          add_layers=layers, add_positions="all"):
        H.forward_pass(bk.model, bk.benign_ids, attention_mask=bk.benign_mask)
        resid = bk.hooks.stacked("resid")
    if score_fn is not None:
        return float(score_fn(resid))
    return float(bk.generator_energy(resid)[bk.n_layers - 1])


def _inject_layers(bk, where):
    if where == "all":
        return list(range(bk.l0, bk.n_layers))
    if where == "repair":
        return list(bk.repair_layers) or [bk.n_layers - 1]
    return list(bk.basin_layers) or list(bk.repair_layers) or [bk.n_layers - 1]


def run(bk, coeff_mults=(0, 0.5, 1, 2, 4, 6), where="basin",
        n_gen=40, max_new_tokens=48, out=None):
    d = bk.d
    U = bk.U.to(torch.float32)
    dev = U.device
    g_dir = (U @ torch.ones(U.shape[1], device=dev)) / (U.shape[1] ** 0.5)  # in span(U)
    g_dir = g_dir / (g_dir.norm() + 1e-12)
    torch.manual_seed(0)
    rand_dir = torch.randn(d, device=dev); rand_dir /= rand_dir.norm()
    dim_dir = bk.r.to(torch.float32).to(dev)              # rank-1 DIM positive control
    layers = _inject_layers(bk, where)                    # inject at the causal basin
    coeffs = [float(m) * bk.inject_scale for m in coeff_mults]

    if bk.is_stub or bk.tok is None:
        # offline / no-tokenizer: representational proxy (final-layer gen energy)
        gen_curve = [_inject_readout(bk, g_dir, c, layers) for c in coeffs]
        dim_curve = [_inject_readout(bk, dim_dir, c, layers) for c in coeffs]
        rand_curve = [_inject_readout(bk, rand_dir, c, layers) for c in coeffs]
        readout = "final-layer generator energy (proxy)"
    else:
        # real model: GOLD sufficiency — generate on BENIGN prompts and measure
        # the refusal rate induced by the injected direction (judge.is_refusal).
        from aor.interp import reco_generate as G
        from aor.interp import reco_judge as JD
        prompts = bk.benign_prompts[:n_gen]
        def rate(vec, c):
            comps = G.generate_with(bk, prompts, add_vec=vec, add_coeff=c,
                                    add_layers=layers, max_new_tokens=max_new_tokens)
            return JD.induced_refusal_rate(comps)
        gen_curve = [rate(g_dir, c) for c in coeffs]
        dim_curve = [rate(dim_dir, c) for c in coeffs]
        rand_curve = [rate(rand_dir, c) for c in coeffs]
        readout = "induced refusal rate on benign % (generation + is_refusal)"

    base = gen_curve[0]
    induced = (max(gen_curve) > base + (5.0 if not bk.is_stub else 1e-6)) \
        and (gen_curve[-1] >= gen_curve[1])
    control_ok = max(rand_curve) <= base + 0.3 * (max(gen_curve) - base) + 1e-6

    res = {
        "model": bk.tag, "inject_where": where, "inject_layers": layers,
        "readout": readout, "coeff_mults": list(coeff_mults),
        "inject_scale": round(bk.inject_scale, 4),
        "coeffs": [round(c, 3) for c in coeffs],
        "generator_readout": [round(x, 4) for x in gen_curve],
        "dim_readout": [round(x, 4) for x in dim_curve],
        "random_readout": [round(x, 4) for x in rand_curve],
        "verdict_sufficiency": bool(induced),
        "verdict_random_control_flat": bool(control_ok),
    }
    print(f"[Exp1] {bk.tag}  inject at {where}={layers}")
    print(f"  readout = {readout}")
    print("  mult  coeff :  generator  dim-ctrl  random")
    for i, m in enumerate(coeff_mults):
        print(f"  {float(m):4.1f} {coeffs[i]:6.2f} :  {gen_curve[i]:8.3f}  "
              f"{dim_curve[i]:7.3f}  {rand_curve[i]:6.3f}")
    print(f"  VERDICT sufficiency(generator induces refusal)={induced}  "
          f"random-control-flat={control_ok}")
    if out:
        json.dump(res, open(out, "w"), indent=2)
    return res


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
