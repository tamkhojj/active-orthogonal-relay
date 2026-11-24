"""Exp2: recovery curve for refusal self-repair.

Tests whether ablating one refusal direction at an early layer is partially
undone downstream, and whether the repair is written by MLPs in a mid-to-late
band.

Procedure:
  1. one-shot ablate the rank-1 refusal direction r at layer l0 (basin start);
  2. per layer, measure generator-subspace energy for baseline vs ablated;
  3. recovery fraction rho_l = energy_abl / energy_base, and the layer where rho
     crosses 0.5;
  4. attribute the per-layer rebuild to attention (o_proj) vs MLP (down_proj) writes.

Outcomes:
  - self-repair present : rho climbs back above rec_thresh after l0
  - MLP-written         : summed MLP write-energy in the repair band exceeds attention

Offline check:  python -I aor/experiments/exp2_recovery_curve.py --selftest"""
import sys, os, json, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import torch
from aor.interp import recovery as R
from aor.interp import reco_hooks as H


def run(bk, rec_thresh=0.3, out=None):
    d, L = bk.d, bk.n_layers
    U = bk.U.to(torch.float32)
    r1 = U[:, 0:1]

    # baseline
    with bk.hooks.capture(token_pos=bk.harmful_pos):
        H.forward_pass(bk.model, bk.harmful_ids, attention_mask=bk.harmful_mask)
        resid_base = bk.hooks.stacked("resid")
    # one-shot rank-1 ablation at l0
    with bk.hooks.capture(token_pos=bk.harmful_pos, ablate_U=r1,
                          ablate_layers=[bk.l0], ablate_positions="all"):
        H.forward_pass(bk.model, bk.harmful_ids, attention_mask=bk.harmful_mask)
        resid_abl = bk.hooks.stacked("resid")
        attn_abl = bk.hooks.stacked("attn")
        mlp_abl = bk.hooks.stacked("mlp")

    e_base = bk.generator_energy(resid_base)        # [L]
    e_abl = bk.generator_energy(resid_abl)
    rho = R.recovery_fraction(e_base, e_abl)        # [L]
    rho_by_layer = {l: float(rho[l]) for l in range(L)}
    cross = R.crossing_layer({l: rho_by_layer[l] for l in range(bk.l0 + 1, L)}, rec_thresh)

    # component attribution of the rebuild (energy of writes inside U), per layer
    def write_energy(stack):
        c, _ = R.project_coeffs(stack.reshape(-1, d), U)
        return c.pow(2).sum(1).reshape(L, -1).mean(1)
    attn_e = write_energy(attn_abl)
    mlp_e = write_energy(mlp_abl)
    band = bk.repair_layers
    mlp_band = float(sum(mlp_e[l] for l in band))
    attn_band = float(sum(attn_e[l] for l in band))

    # dominant repair layers (top-3 by MLP write energy after l0)
    post = [(l, float(mlp_e[l])) for l in range(bk.l0 + 1, L)]
    top_mlp = sorted(post, key=lambda x: -x[1])[:3]

    self_repair = (cross is not None) and (max(rho[bk.l0 + 1:]) > rec_thresh)
    mlp_written = mlp_band > attn_band

    res = {
        "model": bk.tag, "l0": bk.l0, "peak": bk.peak, "repair_band": band,
        "energy_base": [round(float(x), 4) for x in e_base],
        "energy_ablated": [round(float(x), 4) for x in e_abl],
        "recovery_fraction": {l: round(v, 4) for l, v in rho_by_layer.items()},
        "crossing_layer": cross,
        "mlp_write_energy_band": round(mlp_band, 5),
        "attn_write_energy_band": round(attn_band, 5),
        "top_mlp_repair_layers": top_mlp,
        "verdict_self_repair": bool(self_repair),
        "verdict_mlp_written": bool(mlp_written),
    }
    print(f"[Exp2] {bk.tag}  l0={bk.l0}  repair_band={band}")
    print("  layer :   base    abl    rho")
    for l in range(L):
        mark = " <-" if l == bk.l0 else ""
        print(f"   {l:2d}   : {e_base[l]:7.3f} {e_abl[l]:7.3f}  {rho[l]:5.2f}{mark}")
    print(f"  crossing(rho>={rec_thresh}) at layer: {cross}")
    print(f"  repair writes  MLP={mlp_band:.4f}  attn={attn_band:.4f}  "
          f"top MLP layers={top_mlp}")
    print(f"  VERDICT self-repair={self_repair}  MLP-written={mlp_written}")
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
