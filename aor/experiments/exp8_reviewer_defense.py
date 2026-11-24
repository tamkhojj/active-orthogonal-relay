"""Exp8: mechanistic controls for the refusal relay (group 1).

D1.  Split-half cross-validated Exp5b (non-circular two-fold CV) with both an
     isotropic null (1/d) and an empirical covariance-matched null (u = X_l^T z
     drawn from the benign write covariance, removing the anisotropic-cone
     artifact), plus absolute relay amplitude and knock-out fraction
     ||dm|| / ||w_mlp|| and |dm . e3| / |w_mlp . e3|.
D2a. Random-split null control (5 seeds on harmless_train) with covariance-matched
     enrichment and absolute relay amplitude.
D2b. Layer-permutation control (shuffles of basin order).
D2c. Factual-concept control (TruthfulQA true vs false) evaluated in both the
     refusal basin and its own truth-native basin.
D2d. Behavioral control (numbered list vs single flowing paragraph on identical
     benign questions) evaluated in both the refusal basin and its own
     format-native basin.
D3.  Direct knock-out of e3 vs e2 vs e1 vs a random orthogonal axis on both
     mlp_writes and writes (refusal log-odds, benign KL, XSTest unsafe).
D4.  Bidirectional steering evaluated strictly outside beh_basin and geo_basin."""
from __future__ import annotations

import sys
import os
import json
import math
import random
import argparse
import traceback
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch

from aor.interp import reco_backend as B
from aor.interp import reco_hooks as H
from aor.interp import grassmann as GR
from aor.interp import basin as BA
from aor.interp import reco_judge as JD
from aor.experiments import exp6_geora as Exp6
from aor.experiments import run_utility_eval as UTIL


# =========================================================================== #
# Helpers
# =========================================================================== #
def _slice_batch(ids: torch.Tensor, mask: Optional[torch.Tensor], pos, idx: List[int]):
    t_idx = torch.tensor(idx, dtype=torch.long, device=ids.device)
    s_ids = ids.index_select(0, t_idx)
    s_mask = mask.index_select(0, t_idx) if mask is not None else None
    if isinstance(pos, torch.Tensor):
        s_pos = pos.index_select(0, t_idx.to(pos.device))
    else:
        s_pos = pos
    return s_ids, s_mask, s_pos


def _cos_signed_and_sq(vec: torch.Tensor, axis: torch.Tensor) -> Tuple[float, float]:
    vn = float(vec.norm().item())
    an = float(axis.norm().item())
    if vn < 1e-12 or an < 1e-12:
        return 0.0, 0.0
    c = float(torch.dot(vec / vn, axis / an).item())
    return c, c * c


def _ensure_min_basin(basin_in: List[int], peak: int, n_layers: int, min_w: int = 5) -> List[int]:
    """Ensure basin has >= min_w contiguous layers inside [2, n_layers-2]
    so every layer l has a valid 3-point Frenet frame at l-1."""
    lo_lim, hi_lim = 2, max(2, n_layers - 2)
    b = sorted(set(int(x) for x in basin_in if lo_lim <= int(x) <= hi_lim))
    if not b:
        p_clamped = max(lo_lim, min(hi_lim, int(peak)))
        b = [p_clamped]
    while len(b) < min(min_w, hi_lim - lo_lim + 1):
        can_lo = b[0] > lo_lim
        can_hi = b[-1] < hi_lim
        if can_hi and (not can_lo or (b[-1] - peak) <= (peak - b[0])):
            b.append(b[-1] + 1)
        elif can_lo:
            b.insert(0, b[0] - 1)
        else:
            break
    return sorted(b)


@torch.no_grad()
def _capture_mean_writes(bk, ids: torch.Tensor, mask: Optional[torch.Tensor], pos,
                         batch_size: int = 16,
                         ablate_U: Optional[torch.Tensor] = None,
                         ablate_layers: Optional[List[int]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    N = ids.shape[0]
    mlp_sum, attn_sum = None, None
    for i in range(0, N, batch_size):
        ids_b = ids[i:i + batch_size]
        mask_b = mask[i:i + batch_size] if mask is not None else None
        pos_b = pos[i:i + batch_size] if isinstance(pos, torch.Tensor) else pos
        with bk.hooks.capture(token_pos=pos_b, ablate_U=ablate_U,
                              ablate_layers=ablate_layers, ablate_positions="all"):
            H.forward_pass(bk.model, ids_b, attention_mask=mask_b)
            m = bk.hooks.stacked("mlp").sum(dim=1)
            a = bk.hooks.stacked("attn").sum(dim=1)
        mlp_sum = m if mlp_sum is None else (mlp_sum + m)
        attn_sum = a if attn_sum is None else (attn_sum + a)
    return mlp_sum / max(1, N), attn_sum / max(1, N)


@torch.no_grad()
def _capture_per_sample_writes(bk, ids: torch.Tensor, mask: Optional[torch.Tensor], pos,
                               batch_size: int = 16) -> Tuple[torch.Tensor, torch.Tensor]:
    N = ids.shape[0]
    m_chunks, a_chunks = [], []
    for i in range(0, N, batch_size):
        ids_b = ids[i:i + batch_size]
        mask_b = mask[i:i + batch_size] if mask is not None else None
        pos_b = pos[i:i + batch_size] if isinstance(pos, torch.Tensor) else pos
        with bk.hooks.capture(token_pos=pos_b):
            H.forward_pass(bk.model, ids_b, attention_mask=mask_b)
            m_chunks.append(bk.hooks.stacked("mlp"))
            a_chunks.append(bk.hooks.stacked("attn"))
    return torch.cat(m_chunks, dim=1), torch.cat(a_chunks, dim=1)


def _rand_unit_axes(d: int, n_axes: int = 200, seed: int = 123) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    r = torch.randn(n_axes, d, generator=g)
    return r / r.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _cov_matched_unit_axes(samples_nd: torch.Tensor, n_axes: int = 200, seed: int = 777) -> torch.Tensor:
    """Draw unit vectors from the EXACT empirical covariance Sigma = (1/N) X^T X
    of centered benign representations X in R^{N x d}, without forming d x d."""
    X = samples_nd.float()
    X = X - X.mean(dim=0, keepdim=True)
    g = torch.Generator().manual_seed(seed)
    Z = torch.randn(n_axes, X.shape[0], generator=g)
    U = Z @ X  # [n_axes, d], exact Cov = X^T X
    norms = U.norm(dim=-1, keepdim=True)
    valid = norms.squeeze(-1) > 1e-10
    if not valid.any():
        return _rand_unit_axes(X.shape[1], n_axes=n_axes, seed=seed)
    U = U[valid]
    return U / U.norm(dim=-1, keepdim=True).clamp_min(1e-12)


# =========================================================================== #
# D1 Core: Non-Circular 2-Fold Split-Half + Cov-Matched Null + Relay Amplitude
# =========================================================================== #
def _eval_split_half_pair(bk,
                          Hh: torch.Tensor, Hb: torch.Tensor,
                          h_ids: torch.Tensor, h_mask: Optional[torch.Tensor], h_pos,
                          b_ids: torch.Tensor, b_mask: Optional[torch.Tensor], b_pos,
                          basin: List[int], peak: int,
                          batch_size: int = 16,
                          precap_h_writes: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                          precap_b_writes: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                          benign_cov_bank: Optional[Dict[str, torch.Tensor]] = None) -> dict:
    L, d = bk.n_layers, bk.d
    nh, nb = Hh.shape[1], Hb.shape[1]
    idx_ah, idx_bh = list(range(0, nh, 2)), list(range(1, nh, 2))
    idx_ab, idx_bb = list(range(0, nb, 2)), list(range(1, nb, 2))
    l0 = min(basin)

    dim_A = [B.difference_in_means(Hh[l, idx_ah], Hb[l, idx_ab]) for l in range(L)]
    dim_B = [B.difference_in_means(Hh[l, idx_bh], Hb[l, idx_bb]) for l in range(L)]
    fr_A = GR.frenet_trajectory_analysis(dim_A, list(range(L)), peak)["frames"]
    fr_B = GR.frenet_trajectory_analysis(dim_B, list(range(L)), peak)["frames"]

    basin_ds = [l for l in sorted(basin) if l > l0 and (l - 1) in fr_A and (l - 1) in fr_B]
    if not basin_ds:
        basin_ds = [l for l in range(max(2, l0 + 1), L) if (l - 1) in fr_A and (l - 1) in fr_B][:5]

    rand_axes = _rand_unit_axes(d, n_axes=200, seed=123)

    # Pre-build per-layer empirical covariance-matched null axes from benign bank
    cov_resid_axes, cov_mlp_axes, cov_attn_axes = {}, {}, {}
    if benign_cov_bank is not None:
        for l in basin_ds:
            cov_resid_axes[l] = _cov_matched_unit_axes(benign_cov_bank["resid"][l], n_axes=200, seed=700 + l)
            cov_mlp_axes[l] = _cov_matched_unit_axes(benign_cov_bank["mlp"][l], n_axes=200, seed=800 + l)
            cov_attn_axes[l] = _cov_matched_unit_axes(benign_cov_bank["attn"][l], n_axes=200, seed=900 + l)

    # 1. Cross-fold geometric stability of Frenet axes + Cov-Matched null for e3
    e1_cos_sq, e1_cos_s = [], []
    e2_cos_sq, e3_cos_sq, e3_cos_s, e3_cov_null = [], [], [], []
    for l in basin_ds:
        c1, c1sq = _cos_signed_and_sq(fr_A[l - 1]["e1"], fr_B[l - 1]["e1"])
        _, c2sq = _cos_signed_and_sq(fr_A[l - 1]["e2"], fr_B[l - 1]["e2"])
        c3, c3sq = _cos_signed_and_sq(fr_A[l - 1]["e3"], fr_B[l - 1]["e3"])
        e1_cos_s.append(c1); e1_cos_sq.append(c1sq)
        e2_cos_sq.append(c2sq)
        e3_cos_s.append(c3); e3_cos_sq.append(c3sq)
        if l in cov_resid_axes:
            e3u = fr_A[l - 1]["e3"] / (fr_A[l - 1]["e3"].norm() + 1e-12)
            e3_cov_null.append(float(((cov_resid_axes[l] @ e3u) ** 2).mean().item()))

    # 2. Fold A/B Clean writes
    if precap_h_writes is not None:
        mlp_h_A = precap_h_writes[0][:, idx_ah].mean(dim=1)
        attn_h_A = precap_h_writes[1][:, idx_ah].mean(dim=1)
        mlp_h_B = precap_h_writes[0][:, idx_bh].mean(dim=1)
        attn_h_B = precap_h_writes[1][:, idx_bh].mean(dim=1)
    else:
        mlp_h_A, attn_h_A = _capture_mean_writes(bk, *_slice_batch(h_ids, h_mask, h_pos, idx_ah), batch_size=batch_size)
        mlp_h_B, attn_h_B = _capture_mean_writes(bk, *_slice_batch(h_ids, h_mask, h_pos, idx_bh), batch_size=batch_size)

    if precap_b_writes is not None:
        mlp_b_A = precap_b_writes[0][:, idx_ab].mean(dim=1)
        attn_b_A = precap_b_writes[1][:, idx_ab].mean(dim=1)
        mlp_b_B = precap_b_writes[0][:, idx_bb].mean(dim=1)
        attn_b_B = precap_b_writes[1][:, idx_bb].mean(dim=1)
    else:
        mlp_b_A, attn_b_A = _capture_mean_writes(bk, *_slice_batch(b_ids, b_mask, b_pos, idx_ab), batch_size=batch_size)
        mlp_b_B, attn_b_B = _capture_mean_writes(bk, *_slice_batch(b_ids, b_mask, b_pos, idx_bb), batch_size=batch_size)

    # 3. Cross-fold Ablated writes
    mlp_abl_B, attn_abl_B = _capture_mean_writes(
        bk, *_slice_batch(h_ids, h_mask, h_pos, idx_bh),
        batch_size=batch_size, ablate_U=dim_A[l0].view(-1, 1), ablate_layers=[l0],
    )
    mlp_abl_A, attn_abl_A = _capture_mean_writes(
        bk, *_slice_batch(h_ids, h_mask, h_pos, idx_ah),
        batch_size=batch_size, ablate_U=dim_B[l0].view(-1, 1), ablate_layers=[l0],
    )

    def _project_one_dir(frames_train, e1_l0_train, mlp_h_val, attn_h_val,
                         mlp_b_val, attn_b_val, mlp_abl_val, attn_abl_val):
        keys = [
            "clean_mlp_e1", "clean_mlp_e2", "clean_mlp_e3", "clean_mlp_e3_signed",
            "clean_mlp_rand", "clean_mlp_cov_null", "clean_mlp_norm", "clean_mlp_e3_abs_proj",
            "clean_attn_e1", "clean_attn_e2", "clean_attn_e3", "clean_attn_e3_signed",
            "clean_attn_rand", "clean_attn_cov_null", "clean_attn_norm",
            "repair_mlp_e1_l0", "repair_mlp_e1", "repair_mlp_e2", "repair_mlp_e3",
            "repair_mlp_e3_signed", "repair_mlp_rand", "repair_mlp_cov_null",
            "repair_mlp_norm", "repair_mlp_e3_abs_proj",
            "repair_attn_e1_l0", "repair_attn_e1", "repair_attn_e2", "repair_attn_e3",
            "repair_attn_e3_signed", "repair_attn_rand", "repair_attn_cov_null", "repair_attn_norm",
        ]
        acc = {k: [] for k in keys}
        for l in basin_ds:
            fr = frames_train[l - 1]
            e1, e2, e3 = fr["e1"], fr["e2"], fr["e3"]
            w_m = mlp_h_val[l] - mlp_b_val[l]
            w_a = attn_h_val[l] - attn_b_val[l]
            dm = mlp_abl_val[l] - mlp_h_val[l]
            da = attn_abl_val[l] - attn_h_val[l]

            _, cm1 = _cos_signed_and_sq(w_m, e1)
            _, cm2 = _cos_signed_and_sq(w_m, e2)
            cm3_s, cm3 = _cos_signed_and_sq(w_m, e3)
            w_mu = w_m / (w_m.norm() + 1e-12)
            acc["clean_mlp_e1"].append(cm1); acc["clean_mlp_e2"].append(cm2)
            acc["clean_mlp_e3"].append(cm3); acc["clean_mlp_e3_signed"].append(cm3_s)
            acc["clean_mlp_rand"].append(float(((rand_axes @ w_mu) ** 2).mean().item()))
            acc["clean_mlp_cov_null"].append(
                float(((cov_mlp_axes[l] @ w_mu) ** 2).mean().item()) if l in cov_mlp_axes else (1.0 / d)
            )
            acc["clean_mlp_norm"].append(float(w_m.norm().item()))
            acc["clean_mlp_e3_abs_proj"].append(abs(float(torch.dot(w_m, e3).item())))

            _, ca1 = _cos_signed_and_sq(w_a, e1)
            _, ca2 = _cos_signed_and_sq(w_a, e2)
            ca3_s, ca3 = _cos_signed_and_sq(w_a, e3)
            w_au = w_a / (w_a.norm() + 1e-12)
            acc["clean_attn_e1"].append(ca1); acc["clean_attn_e2"].append(ca2)
            acc["clean_attn_e3"].append(ca3); acc["clean_attn_e3_signed"].append(ca3_s)
            acc["clean_attn_rand"].append(float(((rand_axes @ w_au) ** 2).mean().item()))
            acc["clean_attn_cov_null"].append(
                float(((cov_attn_axes[l] @ w_au) ** 2).mean().item()) if l in cov_attn_axes else (1.0 / d)
            )
            acc["clean_attn_norm"].append(float(w_a.norm().item()))

            _, rm0 = _cos_signed_and_sq(dm, e1_l0_train)
            _, rm1 = _cos_signed_and_sq(dm, e1)
            _, rm2 = _cos_signed_and_sq(dm, e2)
            rm3_s, rm3 = _cos_signed_and_sq(dm, e3)
            dmu = dm / (dm.norm() + 1e-12)
            acc["repair_mlp_e1_l0"].append(rm0); acc["repair_mlp_e1"].append(rm1)
            acc["repair_mlp_e2"].append(rm2); acc["repair_mlp_e3"].append(rm3)
            acc["repair_mlp_e3_signed"].append(rm3_s)
            acc["repair_mlp_rand"].append(float(((rand_axes @ dmu) ** 2).mean().item()))
            acc["repair_mlp_cov_null"].append(
                float(((cov_mlp_axes[l] @ dmu) ** 2).mean().item()) if l in cov_mlp_axes else (1.0 / d)
            )
            acc["repair_mlp_norm"].append(float(dm.norm().item()))
            acc["repair_mlp_e3_abs_proj"].append(abs(float(torch.dot(dm, e3).item())))

            _, ra0 = _cos_signed_and_sq(da, e1_l0_train)
            _, ra1 = _cos_signed_and_sq(da, e1)
            _, ra2 = _cos_signed_and_sq(da, e2)
            ra3_s, ra3 = _cos_signed_and_sq(da, e3)
            dau = da / (da.norm() + 1e-12)
            acc["repair_attn_e1_l0"].append(ra0); acc["repair_attn_e1"].append(ra1)
            acc["repair_attn_e2"].append(ra2); acc["repair_attn_e3"].append(ra3)
            acc["repair_attn_e3_signed"].append(ra3_s)
            acc["repair_attn_rand"].append(float(((rand_axes @ dau) ** 2).mean().item()))
            acc["repair_attn_cov_null"].append(
                float(((cov_attn_axes[l] @ dau) ** 2).mean().item()) if l in cov_attn_axes else (1.0 / d)
            )
            acc["repair_attn_norm"].append(float(da.norm().item()))
        return {k: sum(v) / max(1, len(v)) for k, v in acc.items()}

    d_AB = _project_one_dir(fr_A, dim_A[l0], mlp_h_B, attn_h_B, mlp_b_B, attn_b_B, mlp_abl_B, attn_abl_B)
    d_BA = _project_one_dir(fr_B, dim_B[l0], mlp_h_A, attn_h_A, mlp_b_A, attn_b_A, mlp_abl_A, attn_abl_A)
    avg = {k: round(0.5 * (d_AB[k] + d_BA[k]), 5) for k in d_AB}

    iso_null = 1.0 / max(1, d)
    e3_cons_mean = sum(e3_cos_sq) / max(1, len(e3_cos_sq))
    e3_cov_mean = (sum(e3_cov_null) / len(e3_cov_null)) if e3_cov_null else iso_null

    avg.update({
        "basin_downstream_layers": basin_ds,
        "iso_null_1_over_d": round(iso_null, 6),
        "cross_fold_e1_cos_signed": round(sum(e1_cos_s) / max(1, len(e1_cos_s)), 4),
        "cross_fold_e1_cos2": round(sum(e1_cos_sq) / max(1, len(e1_cos_sq)), 5),
        "cross_fold_e2_cos2": round(sum(e2_cos_sq) / max(1, len(e2_cos_sq)), 5),
        "cross_fold_e3_cos_signed": round(sum(e3_cos_s) / max(1, len(e3_cos_s)), 4),
        "cross_fold_e3_cos2": round(e3_cons_mean, 5),
        "cross_fold_e3_cov_null": round(e3_cov_mean, 5),
        "cross_fold_e3_enrichment_x": round(e3_cons_mean / iso_null, 1),
        "cross_fold_e3_cov_enrichment_x": round(e3_cons_mean / max(e3_cov_mean, 1e-9), 2),
        # Isotropic enrichments
        "clean_mlp_e3_enrichment_x": round(avg["clean_mlp_e3"] / max(avg["clean_mlp_rand"], 1e-9), 1),
        "clean_attn_e3_enrichment_x": round(avg["clean_attn_e3"] / max(avg["clean_attn_rand"], 1e-9), 1),
        "repair_mlp_e3_enrichment_x": round(avg["repair_mlp_e3"] / max(avg["repair_mlp_rand"], 1e-9), 1),
        "repair_attn_e3_enrichment_x": round(avg["repair_attn_e3"] / max(avg["repair_attn_rand"], 1e-9), 1),
        # Empirical Covariance-Matched enrichments (eliminates anisotropic cone illusion)
        "clean_mlp_e3_cov_enrichment_x": round(avg["clean_mlp_e3"] / max(avg["clean_mlp_cov_null"], 1e-9), 2),
        "repair_mlp_e3_cov_enrichment_x": round(avg["repair_mlp_e3"] / max(avg["repair_mlp_cov_null"], 1e-9), 2),
        # Absolute Causal Relay Amplitude Ratios
        "repair_to_clean_mlp_norm_ratio": round(
            avg["repair_mlp_norm"] / max(avg["clean_mlp_norm"], 1e-9), 4
        ),
        "repair_to_clean_e3_proj_ratio": round(
            avg["repair_mlp_e3_abs_proj"] / max(avg["clean_mlp_e3_abs_proj"], 1e-9), 4
        ),
        "repair_mlp_over_attn_norm_ratio": round(
            avg["repair_mlp_norm"] / max(avg["repair_attn_norm"], 1e-9), 3
        ),
    })
    return avg


def run_d1_split_half(bk, basin: List[int], benign_cov_bank: dict, batch_size: int = 16) -> dict:
    print(f"\n[D1 SPLIT-HALF Exp5b (Non-Circular 2-Fold CV + Cov-Matched Null)] {bk.tag} basin={basin}")
    res = _eval_split_half_pair(
        bk, bk.Hh_all, bk.Hb_all,
        bk.harmful_ids, bk.harmful_mask, bk.harmful_pos,
        bk.benign_ids, bk.benign_mask, bk.benign_pos,
        basin=basin, peak=bk.peak, batch_size=batch_size,
        precap_b_writes=(benign_cov_bank["mlp"], benign_cov_bank["attn"]),
        benign_cov_bank=benign_cov_bank,
    )
    print(f"  Cross-fold e3: cos={res['cross_fold_e3_cos_signed']:+.3f} | cos^2={res['cross_fold_e3_cos2']:.4f} "
          f"(iso={res['cross_fold_e3_enrichment_x']}x | cov={res['cross_fold_e3_cov_enrichment_x']:.1f}x)")
    print(f"  CV CLEAN  MLP e3: cos^2={res['clean_mlp_e3']:.4f} (iso={res['clean_mlp_e3_enrichment_x']}x | "
          f"cov={res['clean_mlp_e3_cov_enrichment_x']:.1f}x) | ||w_m||={res['clean_mlp_norm']:.2f}")
    print(f"  CV REPAIR MLP e3: cos^2={res['repair_mlp_e3']:.4f} (iso={res['repair_mlp_e3_enrichment_x']}x | "
          f"cov={res['repair_mlp_e3_cov_enrichment_x']:.1f}x, signed={res['repair_mlp_e3_signed']:+.3f}) | "
          f"||dm||={res['repair_mlp_norm']:.2f} | |dm.e3|/|w.e3|={res['repair_to_clean_e3_proj_ratio']:.3f}")
    return res


# =========================================================================== #
# D2a: Random-Split Null Control (with Cov-Matched Null & Amplitude Ratio)
# =========================================================================== #
def run_d2a_random_split_null(bk, basin: List[int], benign_cov_bank: dict,
                              n_seeds: int = 5, batch_size: int = 16) -> dict:
    print(f"\n[D2a RANDOM-SPLIT NULL CONTROL ({n_seeds} seeds + Cov-Matched Null)] {bk.tag}")
    L = bk.n_layers
    nb = bk.Hb_all.shape[1]
    half = nb // 2
    b_mlp_all, b_attn_all = benign_cov_bank["mlp"], benign_cov_bank["attn"]

    keys_avg = [
        "tau", "r2_0p", "r2_1p", "e3_cos2", "e3_cov_x",
        "clean_e3", "clean_iso_x", "clean_cov_x", "clean_norm",
        "rep_e3", "rep_iso_x", "rep_cov_x", "rep_norm", "rep_proj_ratio",
    ]
    acc = {k: [] for k in keys_avg}

    for seed in range(n_seeds):
        rng = random.Random(1000 + seed)
        perm = list(range(nb))
        rng.shuffle(perm)
        g1, g2 = perm[:half], perm[half:2 * half]

        Hh_null, Hb_null = bk.Hb_all[:, g1], bk.Hb_all[:, g2]
        dim_null = [B.difference_in_means(Hh_null[l], Hb_null[l]) for l in range(L)]
        fb = GR.frenet_trajectory_analysis([dim_null[l] for l in basin], basin, bk.peak)

        g1_ids, g1_mask, g1_pos = _slice_batch(bk.benign_ids, bk.benign_mask, bk.benign_pos, g1)
        g2_ids, g2_mask, g2_pos = _slice_batch(bk.benign_ids, bk.benign_mask, bk.benign_pos, g2)
        sh = _eval_split_half_pair(
            bk, Hh_null, Hb_null, g1_ids, g1_mask, g1_pos, g2_ids, g2_mask, g2_pos,
            basin=basin, peak=bk.peak, batch_size=batch_size,
            precap_h_writes=(b_mlp_all[:, g1], b_attn_all[:, g1]),
            precap_b_writes=(b_mlp_all[:, g2], b_attn_all[:, g2]),
            benign_cov_bank=benign_cov_bank,
        )
        acc["tau"].append(fb["mean_tau"])
        acc["r2_0p"].append(fb["zero_param_r2_orthogonal_drift"])
        acc["r2_1p"].append(fb["fit_1p_r2_orthogonal_drift"])
        acc["e3_cos2"].append(sh["cross_fold_e3_cos2"])
        acc["e3_cov_x"].append(sh["cross_fold_e3_cov_enrichment_x"])
        acc["clean_e3"].append(sh["clean_mlp_e3"])
        acc["clean_iso_x"].append(sh["clean_mlp_e3_enrichment_x"])
        acc["clean_cov_x"].append(sh["clean_mlp_e3_cov_enrichment_x"])
        acc["clean_norm"].append(sh["clean_mlp_norm"])
        acc["rep_e3"].append(sh["repair_mlp_e3"])
        acc["rep_iso_x"].append(sh["repair_mlp_e3_enrichment_x"])
        acc["rep_cov_x"].append(sh["repair_mlp_e3_cov_enrichment_x"])
        acc["rep_norm"].append(sh["repair_mlp_norm"])
        acc["rep_proj_ratio"].append(sh["repair_to_clean_e3_proj_ratio"])

    m = {k: sum(v) / max(1, len(v)) for k, v in acc.items()}
    res = {
        "n_seeds": n_seeds,
        "mean_tau": round(m["tau"], 4),
        "zero_param_r2_exp": round(m["r2_0p"], 4),
        "fit_1p_r2_exp": round(m["r2_1p"], 4),
        "cross_fold_e3_cos2": round(m["e3_cos2"], 5),
        "cross_fold_e3_enrichment_x": round(m["e3_cos2"] * bk.d, 1),
        "cross_fold_e3_cov_enrichment_x": round(m["e3_cov_x"], 2),
        "cv_clean_mlp_e3_cos2": round(m["clean_e3"], 5),
        "cv_clean_mlp_e3_enrichment_x": round(m["clean_iso_x"], 1),
        "cv_clean_mlp_e3_cov_enrichment_x": round(m["clean_cov_x"], 2),
        "clean_mlp_norm": round(m["clean_norm"], 4),
        "cv_repair_mlp_e3_cos2": round(m["rep_e3"], 5),
        "cv_repair_mlp_e3_enrichment_x": round(m["rep_iso_x"], 1),
        "cv_repair_mlp_e3_cov_enrichment_x": round(m["rep_cov_x"], 2),
        "repair_mlp_norm": round(m["rep_norm"], 4),
        "repair_to_clean_e3_proj_ratio": round(m["rep_proj_ratio"], 4),
    }
    print(f"  Random-Split Null: tau={res['mean_tau']:.4f} | 0p_R2={res['zero_param_r2_exp']:.4f} | "
          f"e3_cons={res['cross_fold_e3_cos2']:.4f} (cov={res['cross_fold_e3_cov_enrichment_x']:.2f}x) | "
          f"CV Clean={res['cv_clean_mlp_e3_cos2']:.4f} (cov={res['cv_clean_mlp_e3_cov_enrichment_x']:.2f}x, ||w||={res['clean_mlp_norm']:.2f}) | "
          f"CV Repair={res['cv_repair_mlp_e3_cos2']:.4f} (cov={res['cv_repair_mlp_e3_cov_enrichment_x']:.2f}x, ||dm||={res['repair_mlp_norm']:.2f})")
    return res


# =========================================================================== #
# D2b: Layer Permutation Control
# =========================================================================== #
def run_d2b_layer_permutation(bk, basin: List[int], n_perms: int = 100) -> dict:
    print(f"\n[D2b LAYER PERMUTATION CONTROL ({n_perms} shuffles)] {bk.tag} basin={basin}")
    vecs = [bk.layer_dim_vector(l) for l in basin]
    obs = GR.frenet_trajectory_analysis(vecs, basin, bk.peak)
    obs_0p = obs["zero_param_r2_orthogonal_drift"]
    obs_1p = obs["fit_1p_r2_orthogonal_drift"]
    obs_cv = obs["velocity_cv"]

    rng = random.Random(42)
    perm_0p, perm_1p, perm_cv = [], [], []
    w = len(vecs)
    for _ in range(n_perms):
        order = list(range(w))
        rng.shuffle(order)
        pvecs = [vecs[i] for i in order]
        fb = GR.frenet_trajectory_analysis(pvecs, basin, bk.peak)
        perm_0p.append(fb["zero_param_r2_orthogonal_drift"])
        perm_1p.append(fb["fit_1p_r2_orthogonal_drift"])
        perm_cv.append(fb["velocity_cv"])

    p_0p = (1 + sum(1 for x in perm_0p if x >= obs_0p)) / (n_perms + 1)
    p_1p = (1 + sum(1 for x in perm_1p if x >= obs_1p)) / (n_perms + 1)
    res = {
        "n_perms": n_perms,
        "obs_0p_r2_exp": round(obs_0p, 4),
        "perm_0p_r2_mean": round(sum(perm_0p) / n_perms, 4),
        "perm_0p_r2_p95": round(sorted(perm_0p)[int(0.95 * n_perms)], 4),
        "perm_p_val_0p": round(p_0p, 4),
        "obs_1p_r2_exp": round(obs_1p, 4),
        "perm_1p_r2_mean": round(sum(perm_1p) / n_perms, 4),
        "perm_p_val_1p": round(p_1p, 4),
        "obs_velocity_cv": round(obs_cv, 4),
        "perm_velocity_cv_mean": round(sum(perm_cv) / n_perms, 4),
    }
    print(f"  0p R^2: obs={res['obs_0p_r2_exp']:.4f} vs perm_mean={res['perm_0p_r2_mean']:.4f} "
          f"(p95={res['perm_0p_r2_p95']:.4f}, p={res['perm_p_val_0p']:.4f})")
    return res


# =========================================================================== #
# Shared Concept Control Runner (D2c TruthfulQA + D2d Behavioral Format)
# =========================================================================== #
def _run_concept_control(bk, label: str, pos_prompts: List[str], neg_prompts: List[str],
                         ref_basin: List[int], benign_cov_bank: dict,
                         batch_size: int = 16) -> dict:
    if bk.is_stub or bk.tok is None:
        return {"mode": "stub_skipped"}

    prev_side = bk.tok.padding_side
    bk.tok.padding_side = "right"
    p_enc = bk.tok([bk.format_chat(p) for p in pos_prompts], return_tensors="pt",
                   padding=True, truncation=True, max_length=512, add_special_tokens=False)
    n_enc = bk.tok([bk.format_chat(p) for p in neg_prompts], return_tensors="pt",
                   padding=True, truncation=True, max_length=512, add_special_tokens=False)
    bk.tok.padding_side = prev_side

    p_ids, p_mask = p_enc["input_ids"].to(bk.model.device), p_enc["attention_mask"].to(bk.model.device)
    n_ids, n_mask = n_enc["input_ids"].to(bk.model.device), n_enc["attention_mask"].to(bk.model.device)
    p_pos, n_pos = H.last_token_index(p_mask), H.last_token_index(n_mask)

    Hp = B._capture_resid_batched(bk.model, bk.hooks, p_ids, p_mask, p_pos, batch_size=batch_size)
    Hn = B._capture_resid_batched(bk.model, bk.hooks, n_ids, n_mask, n_pos, batch_size=batch_size)

    # Pre-capture clean writes once so both ref_basin & native_basin reuse them
    p_writes = _capture_per_sample_writes(bk, p_ids, p_mask, p_pos, batch_size=batch_size)
    n_writes = _capture_per_sample_writes(bk, n_ids, n_mask, n_pos, batch_size=batch_size)

    L = bk.n_layers
    dim_c = [B.difference_in_means(Hp[l], Hn[l]) for l in range(L)]
    raw_native_basin, native_peak, dp_c = BA.geometric_basin(Hp, Hn, frac=0.65)
    native_basin = _ensure_min_basin(raw_native_basin, native_peak, L, min_w=5)
    dp_ref = BA.dprime_profile(bk.Hh_all, bk.Hb_all)

    fb_ref = GR.frenet_trajectory_analysis([dim_c[l] for l in ref_basin], ref_basin, bk.peak)
    fb_nat = GR.frenet_trajectory_analysis([dim_c[l] for l in native_basin], native_basin, native_peak)

    sh_ref = _eval_split_half_pair(
        bk, Hp, Hn, p_ids, p_mask, p_pos, n_ids, n_mask, n_pos,
        basin=ref_basin, peak=bk.peak, batch_size=batch_size,
        precap_h_writes=p_writes, precap_b_writes=n_writes,
        benign_cov_bank=benign_cov_bank,
    )
    sh_nat = _eval_split_half_pair(
        bk, Hp, Hn, p_ids, p_mask, p_pos, n_ids, n_mask, n_pos,
        basin=native_basin, peak=native_peak, batch_size=batch_size,
        precap_h_writes=p_writes, precap_b_writes=n_writes,
        benign_cov_bank=benign_cov_bank,
    )

    ref_vs_c_cos = [abs(float(torch.dot(bk.layer_dim_vector(l), dim_c[l]).item())) for l in ref_basin]

    res = {
        "concept_label": label,
        "n_pairs": len(pos_prompts),
        "refusal_peak_dprime": round(float(dp_ref[bk.peak].item()), 3),
        "native_peak": int(native_peak),
        "native_basin": native_basin,
        "native_peak_dprime": round(float(dp_c[native_peak].item()), 3),
        "mean_abs_cos_vs_refusal_in_basin": round(sum(ref_vs_c_cos) / max(1, len(ref_vs_c_cos)), 4),
        "in_refusal_basin": {
            "mean_kappa": fb_ref["mean_kappa_rad"],
            "velocity_cv": fb_ref["velocity_cv"],
            "mean_tau": fb_ref["mean_tau"],
            "zero_param_r2_exp": fb_ref["zero_param_r2_orthogonal_drift"],
            "fit_1p_r2_exp": fb_ref["fit_1p_r2_orthogonal_drift"],
            "split_half": sh_ref,
        },
        "in_native_basin": {
            "mean_kappa": fb_nat["mean_kappa_rad"],
            "velocity_cv": fb_nat["velocity_cv"],
            "mean_tau": fb_nat["mean_tau"],
            "zero_param_r2_exp": fb_nat["zero_param_r2_orthogonal_drift"],
            "fit_1p_r2_exp": fb_nat["fit_1p_r2_orthogonal_drift"],
            "split_half": sh_nat,
        },
    }
    print(f"  [{label}] native_peak=L{native_peak} native_basin={native_basin} "
          f"d'={res['native_peak_dprime']:.2f} (ref d'={res['refusal_peak_dprime']:.2f}) | "
          f"|cos_vs_ref|={res['mean_abs_cos_vs_refusal_in_basin']:.3f}")
    print(f"    In refusal_basin : e3_cons={sh_ref['cross_fold_e3_cos2']:.4f} (cov={sh_ref['cross_fold_e3_cov_enrichment_x']:.1f}x) | "
          f"CleanMLP={sh_ref['clean_mlp_e3']:.4f} (cov={sh_ref['clean_mlp_e3_cov_enrichment_x']:.1f}x) | "
          f"RepairMLP={sh_ref['repair_mlp_e3']:.4f} (cov={sh_ref['repair_mlp_e3_cov_enrichment_x']:.1f}x, |dm.e3|/|w.e3|={sh_ref['repair_to_clean_e3_proj_ratio']:.3f})")
    print(f"    In native_basin  : e3_cons={sh_nat['cross_fold_e3_cos2']:.4f} (cov={sh_nat['cross_fold_e3_cov_enrichment_x']:.1f}x) | "
          f"CleanMLP={sh_nat['clean_mlp_e3']:.4f} (cov={sh_nat['clean_mlp_e3_cov_enrichment_x']:.1f}x) | "
          f"RepairMLP={sh_nat['repair_mlp_e3']:.4f} (cov={sh_nat['repair_mlp_e3_cov_enrichment_x']:.1f}x, |dm.e3|/|w.e3|={sh_nat['repair_to_clean_e3_proj_ratio']:.3f})")
    return res


def run_d2c_truthfulqa_control(bk, basin: List[int], tqa_items: List[dict],
                               benign_cov_bank: dict, n_pairs: int = 256,
                               batch_size: int = 16) -> dict:
    print(f"\n[D2c FACTUAL CONCEPT CONTROL (TruthfulQA True vs. False, ref + native basin)] {bk.tag}")
    true_p, false_p = [], []
    for row in tqa_items:
        q = row["question"].strip()
        c1, y1 = row["mc1_targets"]["choices"], row["mc1_targets"]["labels"]
        t_ch = next((c for c, y in zip(c1, y1) if y == 1), None)
        f_ch = next((c for c, y in zip(c1, y1) if y == 0), None)
        if t_ch and f_ch:
            true_p.append(
                f"Determine whether the following answer to the question is factually true.\n"
                f"Question: {q}\nProposed Answer: {t_ch.strip()}"
            )
            false_p.append(
                f"Determine whether the following answer to the question is factually true.\n"
                f"Question: {q}\nProposed Answer: {f_ch.strip()}"
            )
        if len(true_p) >= n_pairs:
            break
    return _run_concept_control(bk, "TruthfulQA_True_vs_False", true_p, false_p,
                                basin, benign_cov_bank, batch_size=batch_size)


def run_d2d_behavioral_format_control(bk, basin: List[int], benign_cov_bank: dict,
                                      n_pairs: int = 256, batch_size: int = 16) -> dict:
    """Non-safety active behavioral control circuit: identical benign prompts
    contrasting numbered-list output vs. continuous-paragraph output."""
    print(f"\n[D2d BEHAVIORAL CONTROL CIRCUIT (Format: Numbered List vs. Prose, ref + native basin)] {bk.tag}")
    if bk.is_stub or bk.tok is None or not bk.benign_prompts:
        return {"mode": "stub_skipped"}
    base_qs = bk.benign_prompts[:n_pairs]
    list_p = [f"Answer the following request using ONLY a numbered list.\n\n{q}" for q in base_qs]
    prose_p = [f"Answer the following request as a single continuous paragraph without any lists.\n\n{q}" for q in base_qs]
    return _run_concept_control(bk, "Behavioral_Format_List_vs_Prose", list_p, prose_p,
                                basin, benign_cov_bank, batch_size=batch_size)


# =========================================================================== #
# D3: Direct Causal Knock-Out of e3 vs. e2 vs. e1 vs. Rand-Orth
# =========================================================================== #
def _make_orth_random(e1: torch.Tensor, e2: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    d = e1.shape[0]
    v = torch.randn(d, generator=gen)
    v = v - torch.dot(v, e1) * e1
    v = v - torch.dot(v, e2) * e2
    return v / v.norm().clamp_min(1e-12)


@torch.no_grad()
def run_d3_direct_e3_knockout(bk, basin: List[int],
                              xs_unsafe_prompts: Optional[List[str]] = None,
                              n_eval: int = 64,
                              n_gen_xs: int = 100,
                              batch_size: int = 16) -> dict:
    print(f"\n[D3 DIRECT e3 vs e2 vs e1 vs RAND_ORTH KNOCK-OUT] {bk.tag} basin={basin}")
    L = bk.n_layers
    dim_all = [bk.layer_dim_vector(l) for l in range(L)]
    frames_all = GR.frenet_trajectory_analysis(dim_all, list(range(L)), bk.peak)["frames"]
    basin_ds = [l for l in sorted(basin) if (l - 1) in frames_all]

    if bk.is_stub or bk.tok is None:
        byl_e3 = {l: frames_all[l - 1]["e3"].view(-1, 1) for l in basin_ds}
        byl_e2 = {l: frames_all[l - 1]["e2"].view(-1, 1) for l in basin_ds}
        with bk.hooks.intervene(ablate_by_layer=byl_e3, ablate_mode="mlp_writes"):
            H.forward_pass(bk.model, bk.harmful_ids)
        with bk.hooks.intervene(ablate_by_layer=byl_e2, ablate_mode="writes"):
            H.forward_pass(bk.model, bk.harmful_ids)
        return {"mode": "stub_ok", "n_layers": len(basin_ds)}

    ref_ids = (bk.basin_info.get("refusal_token_ids") if isinstance(bk.basin_info, dict) else None)
    if not ref_ids:
        ref_ids, _ = BA.discover_refusal_token_ids(bk, n_calib=32, batch_size=batch_size)

    test_h = bk.test_harmful_prompts[:n_eval]
    test_b = bk.test_benign_prompts[:n_eval]
    prev_side = bk.tok.padding_side
    bk.tok.padding_side = "right"
    h_enc = bk.tok([bk.format_chat(p) for p in test_h], return_tensors="pt", padding=True,
                   truncation=True, max_length=512, add_special_tokens=False)
    b_enc = bk.tok([bk.format_chat(p) for p in test_b], return_tensors="pt", padding=True,
                   truncation=True, max_length=512, add_special_tokens=False)
    bk.tok.padding_side = prev_side

    h_ids, h_mask = h_enc["input_ids"].to(bk.model.device), h_enc["attention_mask"].to(bk.model.device)
    b_ids, b_mask = b_enc["input_ids"].to(bk.model.device), b_enc["attention_mask"].to(bk.model.device)
    h_pos, b_pos = H.last_token_index(h_mask), H.last_token_index(b_mask)

    base_h_logits = BA._last_token_logits_batched(bk.model, h_ids, h_mask, h_pos, batch_size=batch_size)
    base_b_logits = BA._last_token_logits_batched(bk.model, b_ids, b_mask, b_pos, batch_size=batch_size)
    base_h_lo_vec = BA.refusal_log_odds_from_logits(base_h_logits, ref_ids)
    base_h_lo = float(base_h_lo_vec.mean().item())

    byl_e1 = {l: frames_all[l - 1]["e1"].view(-1, 1) for l in basin_ds}
    byl_e2 = {l: frames_all[l - 1]["e2"].view(-1, 1) for l in basin_ds}
    byl_e3 = {l: frames_all[l - 1]["e3"].view(-1, 1) for l in basin_ds}
    byl_rands = []
    for seed in (101, 202, 303):
        g = torch.Generator().manual_seed(seed)
        byl_rands.append({
            l: _make_orth_random(frames_all[l - 1]["e1"], frames_all[l - 1]["e2"], g).view(-1, 1)
            for l in basin_ds
        })

    out_modes = {}
    for hook_mode in ("mlp_writes", "writes"):
        print(f"  -- Hook site: {hook_mode} (layers={basin_ds}) --")
        mode_res = {"base_harmful_log_odds": round(base_h_lo, 4)}
        per_sample_drops = {}

        for cond_name, byl_list in [
            ("ABL_E3_TORSION", [byl_e3]),
            ("ABL_E2_INPLANE", [byl_e2]),
            ("ABL_E1_INCOMING", [byl_e1]),
            ("ABL_RAND_ORTH", byl_rands),
        ]:
            drops_all, kls_all = [], []
            per_item_drop_sum = torch.zeros_like(base_h_lo_vec)
            for byl in byl_list:
                with bk.hooks.intervene(ablate_by_layer=byl, ablate_positions="all", ablate_mode=hook_mode):
                    h_abl = BA._last_token_logits_batched(bk.model, h_ids, h_mask, h_pos, batch_size=batch_size)
                    b_abl = BA._last_token_logits_batched(bk.model, b_ids, b_mask, b_pos, batch_size=batch_size)
                lo_abl_vec = BA.refusal_log_odds_from_logits(h_abl, ref_ids)
                drop_vec = base_h_lo_vec - lo_abl_vec
                per_item_drop_sum += drop_vec / len(byl_list)
                drops_all.append(float(drop_vec.mean().item()))
                kls_all.append(BA.kl_divergence_from_logits(base_b_logits, b_abl))

            per_sample_drops[cond_name] = per_item_drop_sum.tolist()
            mean_drop = sum(drops_all) / len(drops_all)
            mean_kl = sum(kls_all) / len(kls_all)
            entry = {"log_odds_drop": round(mean_drop, 4), "benign_kl": round(mean_kl, 5)}

            if xs_unsafe_prompts and n_gen_xs > 0:
                xs_sub = xs_unsafe_prompts[:n_gen_xs]
                comps = Exp6._generate_greedy(
                    bk, xs_sub, max_new_tokens=48, batch_size=batch_size,
                    ablate_by_layer=byl_list[0], ablate_mode=hook_mode,
                )
                nr_flags = [0.0 if JD.is_refusal(c) else 1.0 for c in comps]
                nr_m, nr_lo, nr_hi, _ = JD._bootstrap_mean_ci(nr_flags, n_boot=2000, seed=42)
                entry.update({
                    "xstest_unsafe_nonref_pct": round(nr_m, 2),
                    "xstest_unsafe_nr_ci": [round(nr_lo, 2), round(nr_hi, 2)],
                })
                nr_s = f" | XSTest_Unsafe_NR={nr_m:5.1f}% [{nr_lo:4.1f},{nr_hi:4.1f}]"
            else:
                nr_s = ""

            mode_res[cond_name] = entry
            print(f"    {cond_name:18s} | ΔLogOdds={mean_drop:+7.4f} | BenignKL={mean_kl:.5f}{nr_s}")

        def _boot_diff_raw(va, vb, n_boot=2000, seed=42):
            a = torch.tensor(va, dtype=torch.float32)
            b = torch.tensor(vb, dtype=torch.float32)
            d_v = a - b
            g_b = torch.Generator().manual_seed(seed)
            boots = [float(d_v[torch.randint(0, len(d_v), (len(d_v),), generator=g_b)].mean().item())
                     for _ in range(n_boot)]
            bt = torch.tensor(boots).sort().values
            p_pos = float((bt <= 0.0).float().mean().item())
            p_neg = float((bt >= 0.0).float().mean().item())
            return {
                "diff": round(float(d_v.mean().item()), 4),
                "ci_lo": round(float(bt[int(0.025 * n_boot)].item()), 4),
                "ci_hi": round(float(bt[int(0.975 * n_boot)].item()), 4),
                "p_val": round(min(1.0, 2.0 * min(p_pos, p_neg)), 4),
            }

        mode_res["paired_log_odds_sig"] = {
            "e3_vs_e2": _boot_diff_raw(per_sample_drops["ABL_E3_TORSION"], per_sample_drops["ABL_E2_INPLANE"]),
            "e3_vs_rand_orth": _boot_diff_raw(per_sample_drops["ABL_E3_TORSION"], per_sample_drops["ABL_RAND_ORTH"]),
            "e3_vs_e1": _boot_diff_raw(per_sample_drops["ABL_E3_TORSION"], per_sample_drops["ABL_E1_INCOMING"]),
        }
        p_e2 = mode_res["paired_log_odds_sig"]["e3_vs_e2"]
        p_e1 = mode_res["paired_log_odds_sig"]["e3_vs_e1"]
        print(f"    [SIG] e3 vs e2: {p_e2['diff']:+.3f} (p={p_e2['p_val']:.4f}) | "
              f"e3 vs e1: {p_e1['diff']:+.3f} (p={p_e1['p_val']:.4f})")
        out_modes[hook_mode] = mode_res

    return {"basin_layers_tested": basin_ds, "n_eval_logodds": len(test_h), "by_hook_mode": out_modes}


# =========================================================================== #
# D4: Fixed Bidirectional Steering Control
# =========================================================================== #
def run_d4_fixed_steering(bk, basin: List[int], batch_size: int = 8) -> dict:
    if bk.is_stub or bk.tok is None:
        return {"mode": "stub_ok"}
    print(f"\n[D4 FIXED STEERING (Strict out_layer outside beh U geo basin)] {bk.tag}")
    test_b = bk.test_benign_prompts[:min(48, len(bk.test_benign_prompts))]
    st = Exp6._eval_causal_sufficiency_steering(bk, test_b, basin, n_steer=len(test_b), batch_size=batch_size)
    print(f"  base={st['benign_refusal_base_pct']}% -> steer_peak(L{st['peak_layer']})="
          f"{st['benign_refusal_steer_peak_pct']}% vs steer_out(L{st['out_of_basin_layer']})="
          f"{st['benign_refusal_steer_outside_pct']}% (out_local_vec="
          f"{st['benign_refusal_steer_outside_localvec_pct']}%)")
    return st


# =========================================================================== #
# Self-Test & Cross-Model Aggregator
# =========================================================================== #
def _run_selftest():
    print(">> [SELFTEST Exp8] Offline verification of D1 (Cov-Matched), D2a, D2b, D3...")
    bk = B.build_stub_backend()
    basin = bk.basin_layers
    b_mlp, b_attn = _capture_per_sample_writes(bk, bk.benign_ids, bk.benign_mask, bk.benign_pos, batch_size=16)
    cov_bank = {"resid": bk.Hb_all, "mlp": b_mlp, "attn": b_attn}

    d1 = run_d1_split_half(bk, basin, benign_cov_bank=cov_bank, batch_size=16)
    assert "repair_mlp_e3_cov_enrichment_x" in d1 and "repair_to_clean_e3_proj_ratio" in d1
    d2a = run_d2a_random_split_null(bk, basin, benign_cov_bank=cov_bank, n_seeds=2, batch_size=16)
    assert "cv_repair_mlp_e3_cov_enrichment_x" in d2a
    d2b = run_d2b_layer_permutation(bk, basin, n_perms=20)
    assert "perm_p_val_0p" in d2b
    d3 = run_d3_direct_e3_knockout(bk, basin)
    assert d3.get("mode") == "stub_ok"
    print(">> [SELFTEST Exp8] ALL CHECKS PASSED.")


def _aggregate_exp8(out_root: str):
    model_dirs = sorted(
        d for d in os.listdir(out_root) if os.path.isdir(os.path.join(out_root, d))
    )
    print("\n================  Exp8 TABLE 1: NON-CIRCULAR SPLIT-HALF + COV-MATCHED ENRICHMENT (Refusal vs Null vs Truth vs Format)  ================")
    print(f"| {'Model':24s} | {'Ref e3Cons(cov)':15s} | {'Ref RepMLP(cov)':15s} | {'Null Rep(cov)':14s} | "
          f"{'TruthNat Rep':14s} | {'FmtNat Rep':14s} | {'Ref|dm|/|w|':11s} | {'Perm p':6s} |")
    print("|" + "-" * 26 + "|" + "|".join("-" * 17 for _ in range(2)) + "|" + "|".join("-" * 16 for _ in range(3)) + "|" + "-" * 13 + "|" + "-" * 8 + "|")
    for tag in model_dirs:
        p = os.path.join(out_root, tag, f"exp8_defense_{tag}.json")
        if not os.path.isfile(p):
            continue
        d = json.load(open(p))
        d1 = d.get("d1_split_half_exp5b", {})
        d2a = d.get("d2a_random_split_null", {})
        d2b = d.get("d2b_layer_permutation", {})
        d2c_nat = d.get("d2c_truthfulqa_concept", {}).get("in_native_basin", {}).get("split_half", {})
        d2d_nat = d.get("d2d_format_behavior", {}).get("in_native_basin", {}).get("split_half", {})
        print(
            f"| {tag:24s} | "
            f"{d1.get('cross_fold_e3_cos2', 0):.3f} ({d1.get('cross_fold_e3_cov_enrichment_x', 0):5.1f}x) | "
            f"{d1.get('repair_mlp_e3', 0):.3f} ({d1.get('repair_mlp_e3_cov_enrichment_x', 0):5.1f}x) | "
            f"{d2a.get('cv_repair_mlp_e3_cos2', 0):.3f} ({d2a.get('cv_repair_mlp_e3_cov_enrichment_x', 0):4.1f}x) | "
            f"{d2c_nat.get('repair_mlp_e3', 0):.3f} ({d2c_nat.get('repair_mlp_e3_cov_enrichment_x', 0):4.1f}x) | "
            f"{d2d_nat.get('repair_mlp_e3', 0):.3f} ({d2d_nat.get('repair_mlp_e3_cov_enrichment_x', 0):4.1f}x) | "
            f"{d1.get('repair_to_clean_e3_proj_ratio', 0):11.3f} | "
            f"{d2b.get('perm_p_val_0p', 1.0):6.4f} |"
        )

    print("\n================  Exp8 TABLE 2: DIRECT e3 vs e2 vs e1 vs RAND_ORTH KNOCK-OUT (Δ Log-Odds / XSTest Unsafe NR%)  ================")
    print(f"| {'Model':24s} | {'MLP: e3':14s} | {'MLP: e2':14s} | {'MLP: rand':14s} | "
          f"{'Wrt: e3':14s} | {'Wrt: e1':14s} | {'Wrt: e2':14s} | {'p(e3>e1)':8s} |")
    print("|" + "-" * 26 + "|" + "|".join("-" * 16 for _ in range(6)) + "|" + "-" * 10 + "|")
    for tag in model_dirs:
        p = os.path.join(out_root, tag, f"exp8_defense_{tag}.json")
        if not os.path.isfile(p):
            continue
        by_m = json.load(open(p)).get("d3_direct_e3_knockout", {}).get("by_hook_mode", {})
        m, w = by_m.get("mlp_writes", {}), by_m.get("writes", {})

        def _fmt_ko(sec, key):
            v = sec.get(key, {})
            lo, nr = v.get("log_odds_drop", 0.0), v.get("xstest_unsafe_nonref_pct", None)
            return f"{lo:+5.2f} / {nr:4.1f}%" if nr is not None else f"{lo:+5.2f} /  N/A "

        pe1 = w.get("paired_log_odds_sig", {}).get("e3_vs_e1", {}).get("p_val", 1.0)
        print(
            f"| {tag:24s} | {_fmt_ko(m, 'ABL_E3_TORSION'):14s} | {_fmt_ko(m, 'ABL_E2_INPLANE'):14s} | "
            f"{_fmt_ko(m, 'ABL_RAND_ORTH'):14s} | {_fmt_ko(w, 'ABL_E3_TORSION'):14s} | "
            f"{_fmt_ko(w, 'ABL_E1_INCOMING'):14s} | {_fmt_ko(w, 'ABL_E2_INPLANE'):14s} | {pe1:8.4f} |"
        )

    print("\n================  Exp8 TABLE 3: FIXED STEERING (Strict out_layer outside beh U geo basin)  ================")
    print(f"| {'Model':24s} | {'Base%':7s} | {'Peak (Layer)':14s} | {'Out PeakVec (L)':16s} | {'Out LocalVec':12s} |")
    print("|" + "-" * 26 + "|" + "-" * 9 + "|" + "-" * 16 + "|" + "-" * 18 + "|" + "-" * 14 + "|")
    for tag in model_dirs:
        p = os.path.join(out_root, tag, f"exp8_defense_{tag}.json")
        if not os.path.isfile(p):
            continue
        st = json.load(open(p)).get("d4_fixed_steering", {})
        if not st or "peak_layer" not in st:
            continue
        print(
            f"| {tag:24s} | {st['benign_refusal_base_pct']:6.1f}% | "
            f"{st['benign_refusal_steer_peak_pct']:5.1f}% (L{st['peak_layer']:2d})   | "
            f"{st['benign_refusal_steer_outside_pct']:5.1f}% (L{st['out_of_basin_layer']:2d})     | "
            f"{st['benign_refusal_steer_outside_localvec_pct']:11.1f}% |"
        )


def main():
    ap = argparse.ArgumentParser(description="Exp8 - Mechanistic defense suite")
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
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--n_null_seeds", type=int, default=5)
    ap.add_argument("--n_perms", type=int, default=100)
    ap.add_argument("--n_tqa_pairs", type=int, default=256)
    ap.add_argument("--n_d3_eval", type=int, default=64)
    ap.add_argument("--n_d3_xstest", type=int, default=100)
    ap.add_argument("--truthfulqa_path", default=None)
    ap.add_argument("--xstest_path", default=None)
    ap.add_argument("--outdir", default=None)
    a = ap.parse_args()

    if a.aggregate:
        _aggregate_exp8(a.aggregate)
        return
    if a.selftest or a.model is None:
        _run_selftest()
        return

    tag = a.model.split("/")[-1]
    if a.outdir:
        os.makedirs(a.outdir, exist_ok=True)
    exp5_existing = os.path.join(a.outdir, f"exp5_{tag}.json") if a.outdir else None
    locked_peak, locked_basin = None, None
    if exp5_existing and os.path.isfile(exp5_existing):
        d15 = json.load(open(exp5_existing))
        locked_peak = int(d15["peak"])
        locked_basin = [int(x) for x in d15["basin"]]
        print(f">> [lock] locking peak={locked_peak} basin={locked_basin} from {exp5_existing}")

    tqa_items, _ = UTIL._load_truthfulqa(local_path=a.truthfulqa_path, limit=max(a.n_tqa_pairs + 64, 320))
    _, xs_unsafe, _ = UTIL._load_xstest(local_path=a.xstest_path, n_safe=0, n_unsafe=a.n_d3_xstest)
    xs_unsafe_prompts = [x["prompt"] for x in xs_unsafe]

    bk = B.build_real_backend(
        a.model, a.harmful, a.benign,
        test_harmful_path=a.test_harmful, test_benign_path=a.test_benign,
        device=a.device, n_train=a.n_train, n_test=a.n_gen,
        batch_size=a.batch_size, ablate_mode="writes", use_system_prompt=False,
        peak=locked_peak, basin=locked_basin,
    )
    basin = sorted(bk.basin_layers)

    # Capture benign per-sample writes ONCE for Covariance-Matched Null & reuse across D1/D2a/D2c/D2d
    b_mlp_all, b_attn_all = _capture_per_sample_writes(
        bk, bk.benign_ids, bk.benign_mask, bk.benign_pos, batch_size=a.batch_size
    )
    benign_cov_bank = {"resid": bk.Hb_all, "mlp": b_mlp_all, "attn": b_attn_all}

    out_dict = {
        "model": bk.tag,
        "peak": bk.peak,
        "basin": basin,
        "d": bk.d,
        "n_layers": bk.n_layers,
    }
    for key, fn in [
        ("d1_split_half_exp5b", lambda: run_d1_split_half(bk, basin, benign_cov_bank, batch_size=a.batch_size)),
        ("d2a_random_split_null", lambda: run_d2a_random_split_null(bk, basin, benign_cov_bank, n_seeds=a.n_null_seeds, batch_size=a.batch_size)),
        ("d2b_layer_permutation", lambda: run_d2b_layer_permutation(bk, basin, n_perms=a.n_perms)),
        ("d2c_truthfulqa_concept", lambda: run_d2c_truthfulqa_control(bk, basin, tqa_items, benign_cov_bank, n_pairs=a.n_tqa_pairs, batch_size=a.batch_size)),
        ("d2d_format_behavior", lambda: run_d2d_behavioral_format_control(bk, basin, benign_cov_bank, n_pairs=a.n_tqa_pairs, batch_size=a.batch_size)),
        ("d3_direct_e3_knockout", lambda: run_d3_direct_e3_knockout(bk, basin, xs_unsafe_prompts=xs_unsafe_prompts, n_eval=a.n_d3_eval, n_gen_xs=a.n_d3_xstest, batch_size=a.batch_size)),
        ("d4_fixed_steering", lambda: run_d4_fixed_steering(bk, basin, batch_size=8)),
    ]:
        try:
            out_dict[key] = fn()
        except Exception:
            print(f"!! {key} failed:\n{traceback.format_exc()}")
            out_dict[key] = {"error": traceback.format_exc()}

    if a.outdir:
        out_path = os.path.join(a.outdir, f"exp8_defense_{tag}.json")
        json.dump(out_dict, open(out_path, "w"), indent=2)
        print(f"\n>> Saved Exp8 defense results to: {out_path}")


if __name__ == "__main__":
    main()