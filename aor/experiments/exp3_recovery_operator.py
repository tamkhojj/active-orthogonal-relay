"""Exp3: the recovery operator and its rank.

Tests whether the downstream rebuild of the refusal signal is a low-rank linear
operator R mapping the damaged residual state to the regenerated refusal
component, with effective rank k*, and whether what it regenerates lies in the
generator subspace rather than a random one.

Procedure:
  1. one-shot ablate the rank-1 direction at l0; capture the damaged state
     X = residual at l0 (post-ablation) and the regenerated component
     Y = Pi_U(residual at ell1) for a repair-band layer ell1;
  2. fit a ridge operator R: Y ~ R X; report effective rank (stable /
     participation / energy);
  3. align the regenerated subspace top_k(Y) with the generators U and with a
     random subspace.

Outcomes:
  - low-rank operator     : effective rank small (~ k*), << d
  - generators, not noise : overlap(regenerated, U) >> overlap(regenerated, random)

Offline check:  python -I aor/experiments/exp3_recovery_operator.py --selftest"""
import sys, os, json, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import torch
from aor.interp import recovery as R
from aor.interp import reco_hooks as H


def run(bk, ell1=None, k_report=6, out=None):
    from aor.interp.reco_backend import local_healing_subspace, difference_in_means
    d, L = bk.d, bk.n_layers
    U = bk.U.to(torch.float32)                    # generators at the PEAK (clean)
    k_gen = U.shape[1]
    r1 = U[:, 0:1]
    if ell1 is None:
        ell1 = bk.repair_layers[len(bk.repair_layers) // 2] if bk.repair_layers else L - 1

    # clean residuals (for an INDEPENDENT refusal subspace at ell1)
    with bk.hooks.capture(token_pos=bk.harmful_pos):
        H.forward_pass(bk.model, bk.harmful_ids, attention_mask=bk.harmful_mask)
        clean_h = bk.hooks.stacked("resid")
    with bk.hooks.capture(token_pos=bk.benign_pos):
        H.forward_pass(bk.model, bk.benign_ids, attention_mask=bk.benign_mask)
        clean_b = bk.hooks.stacked("resid")
    # ablated run (one-shot rank-1 at l0) -> damaged state + regenerated signal
    with bk.hooks.capture(token_pos=bk.harmful_pos, ablate_U=r1,
                          ablate_layers=[bk.l0], ablate_positions="all"):
        H.forward_pass(bk.model, bk.harmful_ids, attention_mask=bk.harmful_mask)
        resid = bk.hooks.stacked("resid")

    # INDEPENDENT refusal subspace at ell1 (from clean harmful vs benign; does NOT
    # use U_peak) -> breaks the circularity of projecting onto U_peak.
    U_ind = local_healing_subspace(clean_h[ell1], clean_b[ell1], k=k_gen).to(torch.float32)

    # ROTATION diagnostic: cosine of the per-layer DIM anchor with the peak DIM.
    # A clean, rank-1, robust test of whether the refusal axis rotates across
    # layers (ties the result back to the LARA rotating-trajectory hypothesis).
    r_peak = bk.r.to(torch.float32)
    r_peak = r_peak / (r_peak.norm() + 1e-12)
    rot_cos = []
    for l in range(L):
        rl = difference_in_means(clean_h[l], clean_b[l]).to(torch.float32)
        rot_cos.append(round(float(torch.dot(rl, r_peak).abs()), 3))

    X = resid[bk.l0]                              # damaged state [B,d]
    _, Y = R.project_coeffs(resid[ell1], U_ind)   # regenerated component via U_ind

    Rop = R.fit_ridge_operator(X, Y, ridge=1e-2)  # [d,d]; output lives in U_ind (<=k_gen)
    er = R.effective_rank(Rop, energy=0.90)

    # NON-CIRCULAR alignment: does the regenerated signal (defined via U_ind) and
    # the independent subspace U_ind align with the PEAK generators U?
    U_regen = R.top_subspace(Y, k=k_gen, center=True)
    ov_regen_peak = R.subspace_overlap(U_regen, U)
    ov_ind_peak = R.subspace_overlap(U_ind, U)
    torch.manual_seed(0)
    ov_rand = sum(R.subspace_overlap(R.orthonormalize(torch.randn(d, k_gen)), U)
                  for _ in range(5)) / 5.0

    spec = er["spectrum"][:k_report]
    low_rank = er["stable_rank"] < max(2.0, 0.5 * k_gen)
    generators = ov_ind_peak > 3.0 * ov_rand + 1e-6

    res = {
        "model": bk.tag, "l0": bk.l0, "ell1": ell1, "k_gen": k_gen,
        "note": "rank bounded by k_gen (Y lives in a k_gen-dim subspace); "
                "report effective rank WITHIN the k_gen candidate generators",
        "stable_rank": round(er["stable_rank"], 3),
        "participation_ratio": round(er["participation_ratio"], 3),
        "energy_rank_0.90": er["energy_rank"],
        "singular_values_head": [round(float(s), 4) for s in spec],
        "overlap_independent_subspace_vs_peak": round(ov_ind_peak, 4),
        "overlap_regenerated_vs_peak": round(ov_regen_peak, 4),
        "overlap_random_vs_peak": round(ov_rand, 4),
        "rotation_cos_dim_vs_peak": rot_cos,
        "verdict_low_rank": bool(low_rank),
        "verdict_generators_persist_noncircular": bool(generators),
    }
    print(f"[Exp3] {bk.tag}  l0={bk.l0} -> ell1={ell1}  (d={d}, k_gen={k_gen})")
    print(f"  recovery-operator effective rank WITHIN {k_gen} candidate generators:")
    print(f"    stable={er['stable_rank']:.2f}  participation={er['participation_ratio']:.2f}  "
          f"energy_rank@0.90={er['energy_rank']}")
    print(f"  top singular values: {[round(float(s),3) for s in spec]}")
    print(f"  NON-CIRCULAR alignment vs peak generators:")
    print(f"    independent-subspace(ell1) vs peak = {ov_ind_peak:.3f}   "
          f"regenerated vs peak = {ov_regen_peak:.3f}   random = {ov_rand:.3f}")
    print(f"  rotation cos(DIM_l, DIM_peak) across layers: {rot_cos}")
    print(f"  VERDICT low-rank={low_rank}  generators-persist(non-circular)={generators}")
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
    ap.add_argument("--ell1", type=int, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from aor.interp import reco_backend as B
    if a.selftest or a.model is None:
        bk = B.build_stub_backend()
    else:
        bk = B.build_real_backend(a.model, a.harmful, a.benign, device=a.device)
    run(bk, ell1=a.ell1, out=a.out)


if __name__ == "__main__":
    main()
