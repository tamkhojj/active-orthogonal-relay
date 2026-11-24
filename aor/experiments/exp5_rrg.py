"""Exp5: discrete Frenet-Serret geometry and torsional self-repair coupling.

Exp5a (geometry and model selection):
  - exact discrete Frenet-Serret curvature (kappa_l) and torsion ratio (tau_l)
    inside the Causal Basin versus a disjoint out-of-basin control window;
  - zero- and one-parameter model comparison (planar geodesic cos vs incremental
    orthogonal drift gamma^m) with delta-AIC and a bootstrap p-value;
  - trajectory-span SVD energy spectrum (r = 1..4), linking Exp5 to TRAJ_SVD in Exp6;
  - three-anchor piecewise reconstruction error using a strictly interior middle
    anchor when the peak lies on a basin boundary.

Exp5b (torsional self-repair coupling):
  Block l transforms resid[l-1] -> resid[l], so its Frenet frame is
  frames_all[l-1], with e1 = v_{l-1} (incoming refusal axis), e2 the in-plane
  geodesic continuation in span(v_{l-2}, v_{l-1}), and e3 the out-of-plane axis
  introduced by block l. Clean contrastive writes (w_MLP, w_Attn) and
  post-ablation repair writes (Delta_MLP, Delta_Attn) are projected onto
  (e1_l0, e1, e2, e3) against a 200-draw isotropic null, giving enrichment and
  signed alignment."""
import sys, os, json, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import torch
from aor.interp import grassmann as GR
from aor.interp import reco_hooks as H


def _disjoint_control_window(n_layers: int, basin: list) -> list:
    """Return a contiguous out-of-basin control window strictly disjoint from
    `basin` (0% overlap guaranteed)."""
    b_lo, b_hi = min(basin), max(basin)
    w = len(basin)
    pre_avail = b_lo - 1
    post_avail = (n_layers - 1) - b_hi
    if post_avail >= w:
        ctrl = list(range(b_hi + 1, b_hi + 1 + w))
    elif pre_avail >= w:
        ctrl = list(range(b_lo - w, b_lo))
    elif post_avail >= pre_avail and post_avail >= 3:
        ctrl = list(range(b_hi + 1, n_layers))
    elif pre_avail >= 3:
        ctrl = list(range(1, b_lo))
    else:
        ctrl = []
    assert set(ctrl).isdisjoint(set(basin)), "BUG: control window overlaps basin!"
    return ctrl


def _trajectory_svd_spectrum(unit_vectors: list) -> dict:
    """Compute cumulative energy of the basin trajectory matrix V_B = [v_l]_{l in B}
    captured by rank r = 1, 2, 3, 4 (directly motivates TRAJ_SVD in Exp6)."""
    V = torch.stack([v.float().squeeze() / (v.float().squeeze().norm() + 1e-12)
                     for v in unit_vectors], dim=1)  # [d, W]
    s = torch.linalg.svdvals(V)
    e = s ** 2
    tot = float(e.sum().item()) + 1e-12
    cum = torch.cumsum(e, dim=0) / tot
    stable_rank = float(tot / (float(e.max().item()) + 1e-12))
    part_rank = float((tot ** 2) / (float((e ** 2).sum().item()) + 1e-12))
    out = {
        "n_layers_in_basin": int(V.shape[1]),
        "singular_values": [round(float(x.item()), 4) for x in s[:6]],
        "stable_rank": round(stable_rank, 3),
        "participation_rank": round(part_rank, 3),
    }
    for r in (1, 2, 3, 4):
        idx = min(r, len(cum)) - 1
        out[f"cum_energy_r{r}"] = round(float(cum[idx].item()), 4)
    return out


@torch.no_grad()
def _capture_sublayer_writes(bk, ids, mask, pos, batch_size=16,
                             ablate_U=None, ablate_layers=None):
    """Mini-batched capture of mean MLP and Attn writes per layer [L, d]."""
    N = ids.shape[0]
    mlp_sum = None
    attn_sum = None
    for i in range(0, N, batch_size):
        ids_b = ids[i:i + batch_size]
        mask_b = mask[i:i + batch_size] if mask is not None else None
        pos_b = pos[i:i + batch_size] if isinstance(pos, torch.Tensor) else pos
        with bk.hooks.capture(token_pos=pos_b, ablate_U=ablate_U,
                              ablate_layers=ablate_layers, ablate_positions="all"):
            H.forward_pass(bk.model, ids_b, attention_mask=mask_b)
            m = bk.hooks.stacked("mlp").sum(dim=1)    # [L, d]
            a = bk.hooks.stacked("attn").sum(dim=1)   # [L, d]
        mlp_sum = m if mlp_sum is None else (mlp_sum + m)
        attn_sum = a if attn_sum is None else (attn_sum + a)
    return mlp_sum / max(1, N), attn_sum / max(1, N)


def _cos_signed_and_sq(vec: torch.Tensor, axis: torch.Tensor):
    vn = float(vec.norm().item())
    an = float(axis.norm().item())
    if vn < 1e-12 or an < 1e-12:
        return 0.0, 0.0
    c = float(torch.dot(vec / vn, axis / an).item())
    return c, c * c


def _torsional_self_repair(bk, frames_all: dict, dim_all: list,
                           basin: list, batch_size: int = 16) -> dict:
    """Exp5b (FIXED index alignment): Block `l` creates transition v_{l-1} -> v_l,
    whose exact Frenet triad is `frames_all[l - 1]` (e1=v_{l-1}, e2=in-plane,
    e3=new out-of-plane torsion axis created by block l)."""
    l0 = min(basin)
    e1_l0 = dim_all[l0]
    d = bk.d

    mlp_h, attn_h = _capture_sublayer_writes(
        bk, bk.harmful_ids, bk.harmful_mask, bk.harmful_pos, batch_size=batch_size
    )
    mlp_b, attn_b = _capture_sublayer_writes(
        bk, bk.benign_ids, bk.benign_mask, bk.benign_pos, batch_size=batch_size
    )
    mlp_abl, attn_abl = _capture_sublayer_writes(
        bk, bk.harmful_ids, bk.harmful_mask, bk.harmful_pos,
        batch_size=batch_size, ablate_U=e1_l0.view(-1, 1), ablate_layers=[l0]
    )

    # 200 random isotropic unit vectors for exact null expectation (~1/d)
    g = torch.Generator().manual_seed(123)
    rand_axes = torch.randn(200, d, generator=g)
    rand_axes = rand_axes / (rand_axes.norm(dim=-1, keepdim=True) + 1e-12)

    # Layer l requires frames_all[l - 1] (exists for 2 <= l <= L - 1)
    downstream = [l for l in range(max(2, l0 + 1), bk.n_layers) if (l - 1) in frames_all]
    basin_ds = [l for l in downstream if l in basin]
    post_ds = [l for l in downstream if l > max(basin)]

    def eval_layer_set(layer_list):
        if not layer_list:
            return None
        keys = [
            "clean_mlp_e1", "clean_mlp_e2", "clean_mlp_e3", "clean_mlp_e3_signed", "clean_mlp_norm",
            "clean_attn_e1", "clean_attn_e2", "clean_attn_e3", "clean_attn_e3_signed", "clean_attn_norm",
            "repair_mlp_e1_l0", "repair_mlp_e1", "repair_mlp_e2", "repair_mlp_e3",
            "repair_mlp_e3_signed", "repair_mlp_rand", "repair_mlp_norm",
            "repair_attn_e1_l0", "repair_attn_e1", "repair_attn_e2", "repair_attn_e3",
            "repair_attn_e3_signed", "repair_attn_rand", "repair_attn_norm",
        ]
        acc = {k: [] for k in keys}
        for l in layer_list:
            fr = frames_all[l - 1]   # EXACT frame for block l (v_{l-2} -> v_{l-1} -> v_l)
            e1, e2, e3 = fr["e1"], fr["e2"], fr["e3"]
            w_m = mlp_h[l] - mlp_b[l]
            w_a = attn_h[l] - attn_b[l]
            dm = mlp_abl[l] - mlp_h[l]
            da = attn_abl[l] - attn_h[l]

            _, cm1 = _cos_signed_and_sq(w_m, e1)
            _, cm2 = _cos_signed_and_sq(w_m, e2)
            cm3_s, cm3 = _cos_signed_and_sq(w_m, e3)
            acc["clean_mlp_e1"].append(cm1)
            acc["clean_mlp_e2"].append(cm2)
            acc["clean_mlp_e3"].append(cm3)
            acc["clean_mlp_e3_signed"].append(cm3_s)
            acc["clean_mlp_norm"].append(float(w_m.norm().item()))

            _, ca1 = _cos_signed_and_sq(w_a, e1)
            _, ca2 = _cos_signed_and_sq(w_a, e2)
            ca3_s, ca3 = _cos_signed_and_sq(w_a, e3)
            acc["clean_attn_e1"].append(ca1)
            acc["clean_attn_e2"].append(ca2)
            acc["clean_attn_e3"].append(ca3)
            acc["clean_attn_e3_signed"].append(ca3_s)
            acc["clean_attn_norm"].append(float(w_a.norm().item()))

            _, rm0 = _cos_signed_and_sq(dm, e1_l0)
            _, rm1 = _cos_signed_and_sq(dm, e1)
            _, rm2 = _cos_signed_and_sq(dm, e2)
            rm3_s, rm3 = _cos_signed_and_sq(dm, e3)
            acc["repair_mlp_e1_l0"].append(rm0)
            acc["repair_mlp_e1"].append(rm1)
            acc["repair_mlp_e2"].append(rm2)
            acc["repair_mlp_e3"].append(rm3)
            acc["repair_mlp_e3_signed"].append(rm3_s)
            dm_u = dm / (dm.norm() + 1e-12)
            acc["repair_mlp_rand"].append(float(((rand_axes @ dm_u) ** 2).mean().item()))
            acc["repair_mlp_norm"].append(float(dm.norm().item()))

            _, ra0 = _cos_signed_and_sq(da, e1_l0)
            _, ra1 = _cos_signed_and_sq(da, e1)
            _, ra2 = _cos_signed_and_sq(da, e2)
            ra3_s, ra3 = _cos_signed_and_sq(da, e3)
            acc["repair_attn_e1_l0"].append(ra0)
            acc["repair_attn_e1"].append(ra1)
            acc["repair_attn_e2"].append(ra2)
            acc["repair_attn_e3"].append(ra3)
            acc["repair_attn_e3_signed"].append(ra3_s)
            da_u = da / (da.norm() + 1e-12)
            acc["repair_attn_rand"].append(float(((rand_axes @ da_u) ** 2).mean().item()))
            acc["repair_attn_norm"].append(float(da.norm().item()))

        summary = {k: round(sum(v) / len(v), 5) for k, v in acc.items()}
        null_m = max(summary["repair_mlp_rand"], 1e-9)
        null_a = max(summary["repair_attn_rand"], 1e-9)
        summary["repair_mlp_e3_enrichment_x"] = round(summary["repair_mlp_e3"] / null_m, 1)
        summary["repair_mlp_e1_enrichment_x"] = round(summary["repair_mlp_e1"] / null_m, 1)
        summary["repair_attn_e3_enrichment_x"] = round(summary["repair_attn_e3"] / null_a, 1)
        summary["clean_mlp_e3_enrichment_x"] = round(summary["clean_mlp_e3"] / null_m, 1)
        summary["clean_attn_e3_enrichment_x"] = round(summary["clean_attn_e3"] / null_a, 1)
        summary["repair_mlp_over_attn_norm_ratio"] = round(
            summary["repair_mlp_norm"] / max(summary["repair_attn_norm"], 1e-9), 3
        )
        m_tor = summary["clean_mlp_e3"] * summary["clean_mlp_norm"]
        a_tor = summary["clean_attn_e3"] * summary["clean_attn_norm"]
        summary["clean_torsion_mlp_share"] = round(m_tor / max(m_tor + a_tor, 1e-9), 4)
        return summary

    return {
        "l0": l0,
        "basin_downstream_layers": basin_ds,
        "post_basin_layers": post_ds,
        "inside_basin": eval_layer_set(basin_ds),
        "post_basin": eval_layer_set(post_ds),
        "all_downstream": eval_layer_set(downstream),
    }


def run(bk, out=None, batch_size: int = 16):
    L = bk.n_layers
    basin = sorted(bk.basin_layers) or [bk.l0, bk.peak]
    ctrl_layers = _disjoint_control_window(L, basin)
    peak_idx = basin.index(bk.peak) if bk.peak in basin else len(basin) // 2
    mid_idx = peak_idx if (0 < peak_idx < len(basin) - 1) else (len(basin) // 2)

    # Rank-1 DIM unit vectors across ALL layers
    dim_all = [bk.layer_dim_vector(l) for l in range(L)]
    frenet_all = GR.frenet_trajectory_analysis(dim_all, list(range(L)), bk.peak)
    frenet_basin = GR.frenet_trajectory_analysis([dim_all[l] for l in basin], basin, bk.peak)
    frenet_ctrl = (
        GR.frenet_trajectory_analysis(
            [dim_all[l] for l in ctrl_layers], ctrl_layers, ctrl_layers[len(ctrl_layers) // 2]
        )
        if len(ctrl_layers) >= 3 else None
    )

    # Trajectory-span SVD energy inside basin (bridges directly to TRAJ_SVD in Exp6)
    traj_svd = _trajectory_svd_spectrum([dim_all[l] for l in basin])

    # Multi-rank Grassmann metrics: k in {1, 2, k_gen}
    ranks_to_test = sorted(set([1, min(2, bk.k_gen), bk.k_gen]))
    by_rank = {}
    for k in ranks_to_test:
        subs_b = [bk.layer_subspace(l, k=k) for l in basin]
        mb = GR.trajectory_metrics(subs_b, peak_idx=peak_idx)
        # Non-degenerate 3-anchor piecewise reconstruction error (always interior middle anchor)
        if len(subs_b) >= 3:
            pw_mid = GR.piecewise_grassmann_transport(
                [(0, subs_b[0]), (mid_idx, subs_b[mid_idx]), (len(subs_b) - 1, subs_b[-1])],
                list(range(len(subs_b))),
            )
            mb["piecewise_mid_recon_err"] = round(
                sum(GR.geodesic_distance(pw_mid[i], subs_b[i]) for i in range(len(subs_b))) / len(subs_b), 4
            )
        else:
            mb["piecewise_mid_recon_err"] = mb["piecewise_recon_err"]

        mc = None
        if len(ctrl_layers) >= 3:
            subs_c = [bk.layer_subspace(l, k=k) for l in ctrl_layers]
            mc = GR.trajectory_metrics(subs_c, peak_idx=len(ctrl_layers) // 2)
        by_rank[str(k)] = {"basin": mb, "control_disjoint": mc}

    # Exp5b: Torsional Self-Repair Coupling (exact l-1 frame alignment)
    tsr = _torsional_self_repair(bk, frenet_all["frames"], dim_all, basin, batch_size=batch_size)

    fb_clean = {k: v for k, v in frenet_basin.items() if k != "frames"}
    fc_clean = {k: v for k, v in frenet_ctrl.items() if k != "frames"} if frenet_ctrl else None

    m1 = by_rank["1"]["basin"]
    mc1 = by_rank["1"]["control_disjoint"]
    orth_wins_1p = fb_clean["fit_1p_r2_orthogonal_drift"] > fb_clean["fit_1p_r2_geodesic"]
    orth_wins_0p = fb_clean["zero_param_r2_orthogonal_drift"] > fb_clean["zero_param_r2_geodesic"]
    high_torsion = fb_clean["mean_tau"] >= 0.30
    piecewise_improves = m1["piecewise_mid_recon_err"] < m1["geodesic_recon_err"]

    res = {
        "model": bk.tag,
        "peak": bk.peak,
        "basin": basin,
        "control_layers_disjoint": ctrl_layers,
        "basin_info": bk.basin_info,
        "frenet_basin": fb_clean,
        "frenet_control_disjoint": fc_clean,
        "trajectory_svd_basin": traj_svd,
        "grassmann_by_rank": by_rank,
        "torsional_self_repair": tsr,
        # summary keys
        "median_rotation_rank": by_rank[str(bk.k_gen)]["basin"]["median_rotation_rank"],
        "velocity_cv": fb_clean["velocity_cv"],
        "geodesic_r2": fb_clean["fit_1p_r2_geodesic"],
        "exp_drift_r2": fb_clean["fit_1p_r2_orthogonal_drift"],
        "zero_param_geodesic_r2": fb_clean["zero_param_r2_geodesic"],
        "zero_param_exp_drift_r2": fb_clean["zero_param_r2_orthogonal_drift"],
        "mean_torsion_tau": fb_clean["mean_tau"],
        "theta_hat_rad_per_layer": fb_clean["fit_1p_theta_hat"],
        "gamma_hat_per_layer": fb_clean["fit_1p_gamma_hat"],
        "verdict_high_torsion": bool(high_torsion),
        "verdict_orth_drift_beats_geodesic_1p": bool(orth_wins_1p),
        "verdict_orth_drift_beats_geodesic_0p": bool(orth_wins_0p),
        "verdict_piecewise_beats_flat": bool(piecewise_improves),
        "verdict_basin_more_coherent_than_disjoint_ctrl": (
            bool(m1["geodesic_recon_err"] < mc1["geodesic_recon_err"]) if mc1 else None
        ),
    }

    print(f"[Exp5a FRENET] {bk.tag}  peak={bk.peak}  basin={basin}  disjoint_ctrl={ctrl_layers}")
    print(f"  Curvature kappa_bar={fb_clean['mean_kappa_rad']:.4f} rad/layer (cv={fb_clean['velocity_cv']:.3f}) | "
          f"Torsion tau_bar={fb_clean['mean_tau']:.4f} (median={fb_clean['median_tau']:.4f}, "
          f"c_in={fb_clean['mean_inplane_c_in']:.4f})")
    print(f"  0-PARAM pred R^2: Planar cos(m*k)={fb_clean['zero_param_r2_geodesic']:.4f}  vs  "
          f"Orthogonal Drift (cos k)^m={fb_clean['zero_param_r2_orthogonal_drift']:.4f}")
    print(f"  1-PARAM fit  R^2: Planar cos(m*th)={fb_clean['fit_1p_r2_geodesic']:.4f}  vs  "
          f"Orthogonal Drift gamma^m={fb_clean['fit_1p_r2_orthogonal_drift']:.4f} "
          f"(gamma={fb_clean['fit_1p_gamma_hat']:.4f}, dAIC={fb_clean['delta_aic_exp_over_geo']:+.2f}, "
          f"p={fb_clean['bootstrap_p_exp_better']:.4f})")
    print(f"  Traj-SVD basin energy: r1={traj_svd['cum_energy_r1']:.3f}  r2={traj_svd['cum_energy_r2']:.3f}  "
          f"r3={traj_svd['cum_energy_r3']:.3f}  r4={traj_svd['cum_energy_r4']:.3f}  "
          f"(stable_rank={traj_svd['stable_rank']:.2f})")
    print(f"  Recon err (k=1): flat_2anchor={m1['geodesic_recon_err']:.4f} -> "
          f"torsion_3anchor={m1['piecewise_mid_recon_err']:.4f} "
          f"(disjoint_ctrl={mc1['geodesic_recon_err'] if mc1 else None})")
    if tsr["inside_basin"]:
        ib = tsr["inside_basin"]
        print(f"[Exp5b TORSIONAL SELF-REPAIR (in-basin downstream, aligned frame l-1)]")
        print(f"  CLEAN  writes cos^2 on e3(torsion): MLP={ib['clean_mlp_e3']:.5f} ({ib['clean_mlp_e3_enrichment_x']}x null, signed={ib['clean_mlp_e3_signed']:+.3f}) | "
              f"Attn={ib['clean_attn_e3']:.5f} ({ib['clean_attn_e3_enrichment_x']}x null, signed={ib['clean_attn_e3_signed']:+.3f}) | "
              f"MLP torsion share={ib['clean_torsion_mlp_share']:.3f}")
        print(f"  REPAIR MLP  cos^2: e1_l0={ib['repair_mlp_e1_l0']:.5f}  e1={ib['repair_mlp_e1']:.5f}  "
              f"e2={ib['repair_mlp_e2']:.5f}  e3(torsion)={ib['repair_mlp_e3']:.5f} ({ib['repair_mlp_e3_enrichment_x']}x null, signed={ib['repair_mlp_e3_signed']:+.3f})  "
              f"rand={ib['repair_mlp_rand']:.5f}")
        print(f"  REPAIR Attn cos^2: e1_l0={ib['repair_attn_e1_l0']:.5f}  e1={ib['repair_attn_e1']:.5f}  "
              f"e2={ib['repair_attn_e2']:.5f}  e3(torsion)={ib['repair_attn_e3']:.5f} ({ib['repair_attn_e3_enrichment_x']}x null, signed={ib['repair_attn_e3_signed']:+.3f})  "
              f"MLP/Attn norm ratio={ib['repair_mlp_over_attn_norm_ratio']:.2f}x")
    if out:
        json.dump(res, open(out, "w"), indent=2)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--model", default=None)
    ap.add_argument("--harmful", default=None)
    ap.add_argument("--benign", default=None)
    ap.add_argument("--test_harmful", default=None)
    ap.add_argument("--test_benign", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from aor.interp import reco_backend as B
    bk = B.build_stub_backend() if (a.selftest or a.model is None) else \
        B.build_real_backend(a.model, a.harmful, a.benign,
                             test_harmful_path=a.test_harmful,
                             test_benign_path=a.test_benign, device=a.device)
    run(bk, out=a.out)


if __name__ == "__main__":
    main()