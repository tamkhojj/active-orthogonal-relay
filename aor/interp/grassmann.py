"""Differential geometry of refusal trajectories.

Provides:
  1. a discrete Frenet-Serret frame on the unit sphere that decomposes each step
     v_{l-1} -> v_l -> v_{l+1} into orthonormal axes (e1 the current axis, e2 the
     in-plane geodesic continuation, e3 the out-of-plane orthogonal innovation),
     with step curvature kappa_l and torsion ratio tau_l in [0, 1];
  2. a model comparison between a planar geodesic cos(m*theta) and an incremental
     orthogonal-drift law gamma^m = exp(-lambda*m), with both zero-parameter
     step-derived predictions and one-parameter fits, plus delta-AIC and a paired
     bootstrap p-value;
  3. Grassmann Gr(k, d) tools: principal angles, geodesic distance, two-anchor
     flat geodesic interpolation, and three-anchor torsion-aware piecewise
     transport."""
from __future__ import annotations
import math
from typing import Dict, List, Tuple
import torch


def _orthonormal(U: torch.Tensor) -> torch.Tensor:
    Q, _ = torch.linalg.qr(U.float())
    return Q[:, : U.shape[1]]


def principal_cosines(U: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Cosines of principal angles between span(U), span(V), descending."""
    M = U.float().t() @ V.float()
    return torch.linalg.svdvals(M).clamp(-1.0, 1.0)


def principal_angles(U: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Principal angles (radians), ascending."""
    c = principal_cosines(U, V)
    return torch.sort(torch.arccos(c.clamp(-1.0, 1.0))).values


def geodesic_distance(U: torch.Tensor, V: torch.Tensor) -> float:
    th = principal_angles(U, V)
    return float(torch.sqrt((th ** 2).sum()))


def rotation_spectrum(U: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Principal angles between consecutive subspaces, descending."""
    return torch.sort(principal_angles(U, V), descending=True).values


def rotation_rank(U: torch.Tensor, V: torch.Tensor, frac: float = 0.90) -> int:
    th = rotation_spectrum(U, V)
    e = th ** 2
    if float(e.sum()) < 1e-12:
        return 0
    c = torch.cumsum(e, 0) / e.sum()
    return int((c < frac).sum().item()) + 1


def grassmann_interp(U0: torch.Tensor, U1: torch.Tensor, t: float) -> torch.Tensor:
    """Point at fraction t in [0, 1] along the Grassmann geodesic from U0 to U1."""
    U0 = U0.float()
    U1 = U1.float()
    if t <= 0.0:
        return _orthonormal(U0)
    if t >= 1.0:
        return _orthonormal(U1)
    M = U0.t() @ U1
    A, cos, Bt = torch.linalg.svd(M)
    cos = cos.clamp(-1.0, 1.0)
    theta = torch.arccos(cos)
    P0 = U0 @ A
    P1 = U1 @ Bt.t()
    W = P1 - P0 * cos.unsqueeze(0)
    sin = torch.sin(theta)
    safe_sin = torch.where(sin.abs() < 1e-7, torch.ones_like(sin), sin)
    What = torch.where(
        (sin.abs() < 1e-7).unsqueeze(0),
        torch.zeros_like(W),
        W / safe_sin.unsqueeze(0),
    )
    Ut = P0 * torch.cos(theta * t).unsqueeze(0) + What * torch.sin(theta * t).unsqueeze(0)
    return _orthonormal(Ut)


def piecewise_grassmann_transport(anchors: List[Tuple[int, torch.Tensor]],
                                  target_layers: List[int]) -> Dict[int, torch.Tensor]:
    """Piecewise Grassmann geodesic transport through K ordered anchor layers
    [(l_0, U_0), (l_1, U_1), ..., (l_{K-1}, U_{K-1})].
    Deduplicates identical layer indices automatically.
    K=2 -> flat geodesic (zero torsion).
    K=3 (e.g. basin_lo, peak, basin_hi) -> torsion-aware piecewise geodesic."""
    dedup: Dict[int, torch.Tensor] = {}
    for l_idx, U_mat in anchors:
        dedup[int(l_idx)] = U_mat
    ordered = sorted(dedup.items(), key=lambda x: x[0])

    out: Dict[int, torch.Tensor] = {}
    for l in target_layers:
        if l <= ordered[0][0]:
            out[l] = _orthonormal(ordered[0][1])
            continue
        if l >= ordered[-1][0]:
            out[l] = _orthonormal(ordered[-1][1])
            continue
        for idx in range(len(ordered) - 1):
            la, Ua = ordered[idx]
            lb, Ub = ordered[idx + 1]
            if la <= l <= lb:
                t = (l - la) / max(1, lb - la)
                out[l] = grassmann_interp(Ua, Ub, t)
                break
    return out


def _fit_1p_cos_and_exp(dist: torch.Tensor, y: torch.Tensor, kappa_bar: float,
                        use_abs_cos: bool = False):
    """Exact 1-parameter fits for cos(theta*m) and gamma^m, guaranteed to be
    at least as good as the 0-parameter predictions (includes kappa_bar,
    cos(kappa_bar), closed-form log-linear gamma, plus local zoom refinement)."""
    ss_tot = float(((y - y.mean()) ** 2).sum().item()) + 1e-12

    def r2_of(pred: torch.Tensor) -> float:
        return float(1.0 - ((y - pred) ** 2).sum().item() / ss_tot)

    # --- 1) Planar geodesic: cos(theta * dist) ---
    cands_th = torch.cat([
        torch.linspace(0.0, 1.5, 600),
        torch.tensor([max(0.0, min(1.5, kappa_bar))], dtype=torch.float32),
    ])
    best_r2_geo, best_th = -1e18, max(0.0, min(1.5, kappa_bar))
    for th in cands_th:
        th_v = float(th.item())
        pred = torch.abs(torch.cos(th_v * dist)) if use_abs_cos else torch.cos(th_v * dist)
        r2 = r2_of(pred)
        if r2 > best_r2_geo:
            best_r2_geo, best_th = r2, th_v

    lo_t, hi_t = max(0.0, best_th - 0.01), min(1.5, best_th + 0.01)
    for th in torch.linspace(lo_t, hi_t, 400):
        th_v = float(th.item())
        pred = torch.abs(torch.cos(th_v * dist)) if use_abs_cos else torch.cos(th_v * dist)
        r2 = r2_of(pred)
        if r2 > best_r2_geo:
            best_r2_geo, best_th = r2, th_v
    best_pred_geo = torch.abs(torch.cos(best_th * dist)) if use_abs_cos else torch.cos(best_th * dist)

    # --- 2) Incremental orthogonal drift: gamma ** dist ---
    gamma_0p = max(0.05, min(1.0, math.cos(kappa_bar)))
    mask = (dist > 0) & (y > 1e-6)
    gamma_cf = gamma_0p
    if mask.any():
        d_m, y_m = dist[mask], y[mask]
        lam = -float((d_m * torch.log(y_m)).sum().item() / ((d_m ** 2).sum().item() + 1e-12))
        gamma_cf = max(0.05, min(1.0, math.exp(-max(0.0, lam))))

    cands_g = torch.cat([
        torch.linspace(0.10, 1.0, 600),
        torch.tensor([gamma_0p, gamma_cf], dtype=torch.float32),
    ])
    best_r2_exp, best_g = -1e18, gamma_0p
    for g_t in cands_g:
        gv = float(g_t.item())
        r2 = r2_of(gv ** dist)
        if r2 > best_r2_exp:
            best_r2_exp, best_g = r2, gv

    lo_g, hi_g = max(0.05, best_g - 0.005), min(1.0, best_g + 0.005)
    for g_t in torch.linspace(lo_g, hi_g, 500):
        gv = float(g_t.item())
        r2 = r2_of(gv ** dist)
        if r2 > best_r2_exp:
            best_r2_exp, best_g = r2, gv
    best_pred_exp = best_g ** dist

    return (best_r2_geo, best_th, best_pred_geo), (best_r2_exp, best_g, best_pred_exp)


# =========================================================================== #
# NEW APPARATUS: Discrete Frenet-Serret Frame on S^{d-1}
# =========================================================================== #
def discrete_frenet_frame(v_prev: torch.Tensor, v_curr: torch.Tensor,
                          v_next: torch.Tensor) -> Dict[str, object]:
    """Compute exact orthonormal Frenet triad (e1, e2, e3), step curvature kappa,
    in-plane continuation c_in, and torsion ratio tau in [0, 1] at v_curr.

    Mathematical invariant: c_in^2 + tau == 1.0 (exact on S^{d-1}).
      - tau = 0.0 -> pure planar geodesic in span(v_prev, v_curr)
      - tau = 1.0 -> pure orthogonal innovation (incremental orthogonal drift)
    """
    vp = v_prev.float().squeeze()
    vc = v_curr.float().squeeze()
    vn = v_next.float().squeeze()
    vp = vp / (vp.norm() + 1e-12)
    vc = vc / (vc.norm() + 1e-12)
    vn = vn / (vn.norm() + 1e-12)

    e1 = vc
    dot_prev = float(torch.dot(vc, vp).clamp(-1.0, 1.0).item())
    u2 = dot_prev * vc - vp
    u2_norm = float(u2.norm().item())
    if u2_norm < 1e-9:
        e2 = torch.zeros_like(vc)
    else:
        e2 = u2 / u2_norm

    dot_next = float(torch.dot(vn, e1).clamp(-1.0, 1.0).item())
    kappa = math.acos(max(-1.0, min(1.0, dot_next)))
    sin2_kappa = max(0.0, 1.0 - dot_next * dot_next)

    proj_e2 = float(torch.dot(vn, e2).item()) if u2_norm >= 1e-9 else 0.0
    r_perp = vn - dot_next * e1 - proj_e2 * e2
    r_norm = float(r_perp.norm().item())
    e3 = r_perp / r_norm if r_norm > 1e-9 else torch.zeros_like(vc)

    if sin2_kappa < 1e-10:
        tau = 0.0
        c_in = 1.0
        torsion_angle = 0.0
    else:
        tau = max(0.0, min(1.0, (r_norm * r_norm) / sin2_kappa))
        c_in = max(-1.0, min(1.0, proj_e2 / math.sqrt(sin2_kappa)))
        torsion_angle = math.atan2(r_norm, proj_e2)

    return {
        "e1": e1,
        "e2": e2,
        "e3": e3,
        "kappa": kappa,
        "tau": tau,
        "c_in": c_in,
        "torsion_angle": torsion_angle,
    }


def frenet_trajectory_analysis(unit_vectors: List[torch.Tensor],
                               layers: List[int],
                               peak_layer: int) -> Dict[str, object]:
    """Full Frenet-Serret curvature/torsion profile + head-to-head model
    selection between Planar Geodesic cos(m*theta) and Torsional/Orthogonal
    Drift gamma^m across `layers`."""
    m = len(unit_vectors)
    vecs = [v.float().squeeze() / (v.float().squeeze().norm() + 1e-12) for v in unit_vectors]
    peak_idx = layers.index(peak_layer) if peak_layer in layers else m // 2

    kappas = []
    for i in range(m - 1):
        c = float(torch.dot(vecs[i], vecs[i + 1]).clamp(-1.0, 1.0).item())
        kappas.append(math.acos(max(-1.0, min(1.0, c))))

    taus, c_ins, torsion_angles, frames = [], [], [], {}
    for i in range(1, m - 1):
        fr = discrete_frenet_frame(vecs[i - 1], vecs[i], vecs[i + 1])
        frames[layers[i]] = fr
        taus.append(fr["tau"])
        c_ins.append(fr["c_in"])
        torsion_angles.append(fr["torsion_angle"])

    v_peak = vecs[peak_idx]
    cos_signed = [float(torch.dot(v_peak, v).clamp(-1.0, 1.0).item()) for v in vecs]
    dist = torch.tensor([abs(l - layers[peak_idx]) for l in layers], dtype=torch.float32)
    y = torch.tensor(cos_signed, dtype=torch.float32)
    ss_tot = float(((y - y.mean()) ** 2).sum().item()) + 1e-12

    # ---- 0-parameter predictions directly from mean step angle kappa_bar ----
    kappa_bar = float(sum(kappas) / max(1, len(kappas)))
    pred_geo_0p = torch.cos(kappa_bar * dist)
    pred_exp_0p = (math.cos(kappa_bar) ** dist)
    r2_geo_0p = float(1.0 - ((y - pred_geo_0p) ** 2).sum().item() / ss_tot)
    r2_exp_0p = float(1.0 - ((y - pred_exp_0p) ** 2).sum().item() / ss_tot)
    rmse_geo_0p = float(torch.sqrt(((y - pred_geo_0p) ** 2).mean()).item())
    rmse_exp_0p = float(torch.sqrt(((y - pred_exp_0p) ** 2).mean()).item())

    # ---- 1-parameter exact fits: cos(theta*m) vs gamma^m ----
    (best_r2_geo, best_theta, best_pred_geo), (best_r2_exp, best_gamma, best_pred_exp) = \
        _fit_1p_cos_and_exp(dist, y, kappa_bar, use_abs_cos=False)

    rss_geo = float(((y - best_pred_geo) ** 2).sum().item()) + 1e-12
    rss_exp = float(((y - best_pred_exp) ** 2).sum().item()) + 1e-12
    rmse_geo_1p = math.sqrt(rss_geo / max(1, m))
    rmse_exp_1p = math.sqrt(rss_exp / max(1, m))
    delta_aic_exp_over_geo = m * math.log(rss_geo / rss_exp)

    # Paired bootstrap test on squared residuals (H0: geodesic == exp)
    diff_sq = ((y - best_pred_geo) ** 2 - (y - best_pred_exp) ** 2)
    g = torch.Generator().manual_seed(42)
    B = 2000
    wins = 0
    for _ in range(B):
        idx = torch.randint(0, m, (m,), generator=g)
        if float(diff_sq[idx].mean().item()) > 0.0:
            wins += 1
    p_exp_better = round(1.0 - wins / B, 4)

    kappa_t = torch.tensor(kappas) if kappas else torch.zeros(1)
    vel_cv = float(kappa_t.std(unbiased=False).item() / (kappa_t.mean().item() + 1e-9))

    return {
        "layers": layers,
        "peak_layer": layers[peak_idx],
        "cos_signed_to_peak": [round(c, 4) for c in cos_signed],
        "step_kappas_rad": [round(k, 4) for k in kappas],
        "mean_kappa_rad": round(kappa_bar, 4),
        "velocity_cv": round(vel_cv, 4),
        "torsion_ratios_tau": [round(t, 4) for t in taus],
        "mean_tau": round(sum(taus) / max(1, len(taus)), 4),
        "median_tau": round(float(torch.tensor(taus).median().item()), 4) if taus else 0.0,
        "mean_inplane_c_in": round(sum(c_ins) / max(1, len(c_ins)), 4),
        "mean_torsion_angle_rad": round(sum(torsion_angles) / max(1, len(torsion_angles)), 4),
        "zero_param_r2_geodesic": round(r2_geo_0p, 4),
        "zero_param_r2_orthogonal_drift": round(r2_exp_0p, 4),
        "zero_param_rmse_geodesic": round(rmse_geo_0p, 4),
        "zero_param_rmse_orthogonal_drift": round(rmse_exp_0p, 4),
        "fit_1p_r2_geodesic": round(best_r2_geo, 4),
        "fit_1p_theta_hat": round(best_theta, 4),
        "fit_1p_rmse_geodesic": round(rmse_geo_1p, 4),
        "fit_1p_r2_orthogonal_drift": round(best_r2_exp, 4),
        "fit_1p_gamma_hat": round(best_gamma, 4),
        "fit_1p_rmse_orthogonal_drift": round(rmse_exp_1p, 4),
        "delta_aic_exp_over_geo": round(delta_aic_exp_over_geo, 2),
        "bootstrap_p_exp_better": p_exp_better,
        "frames": frames,
    }


def trajectory_metrics(subspaces: List[torch.Tensor], peak_idx: int = 0) -> dict:
    """Rank-k Grassmann trajectory metrics (reports leading, mean, and trailing
    principal cosines + 2-anchor flat vs 3-anchor torsion recon error)."""
    m = len(subspaces)
    step_angles_max, step_angles_lead, rot_ranks = [], [], []
    for i in range(m - 1):
        sp = rotation_spectrum(subspaces[i], subspaces[i + 1])  # descending
        step_angles_max.append(float(sp[0].item()))
        step_angles_lead.append(float(sp[-1].item()))           # smallest angle (leading axis)
        rot_ranks.append(rotation_rank(subspaces[i], subspaces[i + 1]))

    sa_t = torch.tensor(step_angles_lead) if step_angles_lead else torch.zeros(1)
    mu = float(sa_t.mean().item())
    sd = float(sa_t.std(unbiased=False).item())
    velocity_cv = sd / (mu + 1e-9)

    Upeak = subspaces[peak_idx]
    cos_lead = [float(principal_cosines(Upeak, U)[0].item()) for U in subspaces]
    cos_mean = [float(principal_cosines(Upeak, U).mean().item()) for U in subspaces]
    cos_trail = [float(principal_cosines(Upeak, U)[-1].item()) for U in subspaces]

    dist = torch.tensor([abs(i - peak_idx) for i in range(m)], dtype=torch.float32)
    y = torch.tensor(cos_lead, dtype=torch.float32)

    (best_r2_geo, best_theta, _), (best_r2_exp, best_gamma, _) = \
        _fit_1p_cos_and_exp(dist, y, mu, use_abs_cos=True)

    recon_flat, recon_pw = [], []
    if m >= 3:
        U0, Um, U1 = subspaces[0], subspaces[peak_idx], subspaces[-1]
        pw_map = piecewise_grassmann_transport(
            [(0, U0), (peak_idx, Um), (m - 1, U1)], list(range(m))
        )
        for i in range(m):
            Upred_flat = grassmann_interp(U0, U1, i / (m - 1))
            recon_flat.append(geodesic_distance(Upred_flat, subspaces[i]))
            recon_pw.append(geodesic_distance(pw_map[i], subspaces[i]))

    return {
        "k": int(subspaces[0].shape[1]),
        "mean_step_angle_lead": round(mu, 4),
        "mean_step_angle_max": round(sum(step_angles_max) / max(1, len(step_angles_max)), 4),
        "velocity_cv": round(velocity_cv, 4),
        "rotation_ranks": rot_ranks,
        "median_rotation_rank": int(torch.tensor(rot_ranks).float().median().item()) if rot_ranks else 0,
        "cos_to_peak_lead": [round(c, 4) for c in cos_lead],
        "cos_to_peak_mean": [round(c, 4) for c in cos_mean],
        "cos_to_peak_trail": [round(c, 4) for c in cos_trail],
        "geodesic_r2": round(best_r2_geo, 4),
        "theta_hat": round(best_theta, 4),
        "exp_drift_r2": round(best_r2_exp, 4),
        "gamma_hat": round(best_gamma, 4),
        "geodesic_recon_err": round(float(torch.tensor(recon_flat).mean().item()), 4) if recon_flat else 0.0,
        "piecewise_recon_err": round(float(torch.tensor(recon_pw).mean().item()), 4) if recon_pw else 0.0,
    }