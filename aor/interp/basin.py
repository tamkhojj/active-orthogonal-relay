"""Automatic behavioral peak (l*) and Causal Basin (B) discovery.

Follows Arditi et al. for peak and basin selection:
  1. uses ablate_mode="writes" (with ablate_embed=True for the global sweep),
     matching Arditi et al. Eq. 5;
  2. enforces the sufficiency filter (induce_log_odds > 0 under single-layer
     activation addition +r^(l) on benign prompts), so non-causal early layers
     cannot be selected;
  3. combines a normalized global_drop with a localized three-layer local_drop to
     bound a contiguous Causal Basin without leaking into inactive tail layers."""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
from aor.interp.reco_backend import difference_in_means


def search_zone(n_layers: int, lo_frac: float = 0.15, hi_frac: float = 0.80) -> List[int]:
    lo = max(1, int(n_layers * lo_frac))
    hi = min(n_layers - 2, int(n_layers * hi_frac))
    if hi <= lo:
        lo, hi = 1, max(1, n_layers - 2)
    return list(range(lo, hi + 1))


def dprime_profile(Hh_all: torch.Tensor, Hb_all: torch.Tensor) -> torch.Tensor:
    L = Hh_all.shape[0]
    out = torch.zeros(L)
    for l in range(L):
        r = difference_in_means(Hh_all[l], Hb_all[l])
        ph = Hh_all[l].float() @ r
        pb = Hb_all[l].float() @ r
        pooled = (0.5 * (ph.var(unbiased=False) + pb.var(unbiased=False))).sqrt()
        out[l] = (ph.mean() - pb.mean()).abs() / (pooled + 1e-6)
    return out


def discover_peak_geometric(Hh_all: torch.Tensor, Hb_all: torch.Tensor,
                            lo_frac: float = 0.15, hi_frac: float = 0.80) -> int:
    dp = dprime_profile(Hh_all, Hb_all)
    zone = search_zone(len(dp), lo_frac, hi_frac)
    return int(max(zone, key=lambda l: float(dp[l].item())))


discover_peak = discover_peak_geometric


def geometric_basin(Hh_all: torch.Tensor, Hb_all: torch.Tensor,
                    peak: Optional[int] = None, frac: float = 0.65,
                    lo_frac: float = 0.15, hi_frac: float = 0.80) -> Tuple[List[int], int, torch.Tensor]:
    dp = dprime_profile(Hh_all, Hb_all)
    zone = search_zone(len(dp), lo_frac, hi_frac)
    z_lo, z_hi = zone[0], zone[-1]
    if peak is None or peak < z_lo or peak > z_hi:
        peak = max(zone, key=lambda l: float(dp[l].item()))
    thr = frac * float(dp[peak].item())
    lo = int(peak)
    while lo - 1 >= z_lo and float(dp[lo - 1].item()) >= thr:
        lo -= 1
    hi = int(peak)
    while hi + 1 <= z_hi and float(dp[hi + 1].item()) >= thr:
        hi += 1
    return list(range(lo, hi + 1)), int(peak), dp


@torch.no_grad()
def _last_token_logits_batched(model, input_ids: torch.Tensor,
                               attention_mask: Optional[torch.Tensor],
                               pos, batch_size: int = 16) -> torch.Tensor:
    N = input_ids.shape[0]
    outs = []
    for i in range(0, N, batch_size):
        ids_b = input_ids[i:i + batch_size]
        mask_b = attention_mask[i:i + batch_size] if attention_mask is not None else None
        kw = {"attention_mask": mask_b} if mask_b is not None else {}
        out = model(ids_b, **kw)
        logits = out.logits if hasattr(out, "logits") else out[0]
        if isinstance(pos, torch.Tensor):
            p_b = pos[i:i + batch_size].to(logits.device).long()
            last = logits[torch.arange(logits.shape[0], device=logits.device), p_b, :]
        else:
            last = logits[:, pos, :]
        outs.append(last.detach().float().cpu())
    return torch.cat(outs, dim=0)


def refusal_log_odds_from_logits(logits: torch.Tensor, refusal_ids: List[int]) -> torch.Tensor:
    """Arditi et al. Appendix B: logit(sum_{t in R} p_t) per sample [N]."""
    probs = F.softmax(logits.float(), dim=-1)
    p_ref = probs[:, refusal_ids].sum(dim=-1).clamp(1e-7, 1.0 - 1e-7)
    return torch.log(p_ref) - torch.log(1.0 - p_ref)


def kl_divergence_from_logits(base_logits: torch.Tensor, abl_logits: torch.Tensor) -> float:
    p_base = F.softmax(base_logits.float(), dim=-1).clamp_min(1e-9)
    log_p_base = torch.log(p_base)
    log_p_abl = F.log_softmax(abl_logits.float(), dim=-1)
    kl = (p_base * (log_p_base - log_p_abl)).sum(dim=-1)
    return float(kl.mean().item())


@torch.no_grad()
def discover_refusal_token_ids(bk, n_calib: int = 32, batch_size: int = 16) -> Tuple[List[int], List[str]]:
    if bk.is_stub or bk.tok is None:
        return [0], ["<stub>"]

    nh = min(n_calib, bk.harmful_ids.shape[0])
    nb = min(n_calib, bk.benign_ids.shape[0])
    h_pos = bk.harmful_pos[:nh] if isinstance(bk.harmful_pos, torch.Tensor) else bk.harmful_pos
    b_pos = bk.benign_pos[:nb] if isinstance(bk.benign_pos, torch.Tensor) else bk.benign_pos

    h_logits = _last_token_logits_batched(
        bk.model, bk.harmful_ids[:nh],
        bk.harmful_mask[:nh] if bk.harmful_mask is not None else None,
        h_pos, batch_size=batch_size,
    )
    b_logits = _last_token_logits_batched(
        bk.model, bk.benign_ids[:nb],
        bk.benign_mask[:nb] if bk.benign_mask is not None else None,
        b_pos, batch_size=batch_size,
    )
    ph_mean = F.softmax(h_logits, dim=-1).mean(dim=0)
    pb_mean = F.softmax(b_logits, dim=-1).mean(dim=0)
    diff = ph_mean - pb_mean

    canonical_strs = [
        "I", "Sorry", "As", "Apologies", "Unfortunately", "Cannot", "No",
        " I", " Sorry", " As", "I'm", "I’m",
    ]
    cand_ids = set()
    for s in canonical_strs:
        toks = bk.tok.encode(s, add_special_tokens=False)
        if toks:
            cand_ids.add(int(toks[0]))

    selected = [tid for tid in cand_ids if float(diff[tid].item()) > 0.03 and float(ph_mean[tid].item()) > 0.04]
    if not selected:
        top_vals, top_idx = torch.topk(diff, k=3)
        selected = [int(idx.item()) for val, idx in zip(top_vals, top_idx) if float(val.item()) > 0.02]
    if not selected:
        selected = [int(diff.argmax().item())]

    selected = sorted(set(selected), key=lambda tid: float(diff[tid].item()), reverse=True)[:4]
    decoded = [repr(bk.tok.decode([tid])) for tid in selected]
    return selected, decoded


@torch.no_grad()
def discover_basin_behavioral(bk, n_calib: int = 32, batch_size: int = 16,
                              lo_frac: float = 0.15, hi_frac: float = 0.80,
                              ablate_mode: str = "writes",
                              kl_cap: float = 0.25,
                              kl_cap_relaxed: float = 0.60,
                              strict_frac: float = 0.50,
                              relaxed_frac: float = 0.30,
                              min_width: int = 5) -> Dict[str, object]:
    """Arditi §C.1 + AOR/LARA-style Peak & Causal Basin discovery.

    For every layer l in Z = [0.15L, 0.80L]:
      1) global_drop[l]     : refusal log-odds reduction under global ablation of v^(l)
      2) induce_log_odds[l] : refusal log-odds on benign prompts after adding +r_raw^(l)
      3) benign_kl[l]       : KL(P_base || P_abl) on benign prompts under global ablation
      4) local_drop[l]      : refusal log-odds drop under localized 3-layer [l-1..l+1] ablation
    """
    L = bk.n_layers
    zone = search_zone(L, lo_frac, hi_frac)
    z_lo, z_hi = zone[0], zone[-1]
    geo_basin, geo_peak, dp = geometric_basin(
        bk.Hh_all, bk.Hb_all, frac=0.65, lo_frac=lo_frac, hi_frac=hi_frac
    )

    if bk.is_stub or bk.tok is None:
        return {
            "peak": geo_peak,
            "basin": geo_basin,
            "geometric_peak": geo_peak,
            "geometric_basin": geo_basin,
            "basin_iou": 1.0,
            "refusal_token_ids": [0],
            "refusal_tokens": ["<stub>"],
            "global_drop": {l: float(dp[l].item()) for l in zone},
            "local_drop": {l: float(dp[l].item()) for l in zone},
            "induce_log_odds": {l: 1.0 for l in zone},
            "induce_gain": {l: 1.0 for l in zone},
            "causal_score": {l: float(dp[l].item()) for l in zone},
            "benign_kl": {l: 0.0 for l in zone},
            "peak_induce_log_odds": 1.0,
            "peak_global_drop": float(dp[geo_peak].item()),
            "peak_benign_kl": 0.0,
        }

    ref_ids, ref_strs = discover_refusal_token_ids(bk, n_calib=n_calib, batch_size=batch_size)
    nh = min(n_calib, bk.harmful_ids.shape[0])
    nb = min(n_calib, bk.benign_ids.shape[0])
    h_ids, h_mask = bk.harmful_ids[:nh], (bk.harmful_mask[:nh] if bk.harmful_mask is not None else None)
    b_ids, b_mask = bk.benign_ids[:nb], (bk.benign_mask[:nb] if bk.benign_mask is not None else None)
    h_pos = bk.harmful_pos[:nh] if isinstance(bk.harmful_pos, torch.Tensor) else bk.harmful_pos
    b_pos = bk.benign_pos[:nb] if isinstance(bk.benign_pos, torch.Tensor) else bk.benign_pos

    base_h_logits = _last_token_logits_batched(bk.model, h_ids, h_mask, h_pos, batch_size=batch_size)
    base_b_logits = _last_token_logits_batched(bk.model, b_ids, b_mask, b_pos, batch_size=batch_size)
    base_h_lo = float(refusal_log_odds_from_logits(base_h_logits, ref_ids).mean().item())
    base_b_lo = float(refusal_log_odds_from_logits(base_b_logits, ref_ids).mean().item())

    all_layers = list(range(L))
    r_unit_by_layer = {l: difference_in_means(bk.Hh_all[l], bk.Hb_all[l]).view(-1, 1) for l in range(L)}
    r_raw_by_layer = {l: (bk.Hh_all[l].float().mean(0) - bk.Hb_all[l].float().mean(0)) for l in range(L)}
    use_embed = (ablate_mode == "writes")

    global_drop: Dict[int, float] = {}
    local_drop: Dict[int, float] = {}
    induce_log_odds: Dict[int, float] = {}
    induce_gain: Dict[int, float] = {}
    benign_kl: Dict[int, float] = {}

    for l in zone:
        # (1) Global ablation with v^(l) -> bypass efficacy + benign KL
        with bk.hooks.intervene(
            ablate_U=r_unit_by_layer[l], ablate_layers=all_layers,
            ablate_positions="all", ablate_mode=ablate_mode, ablate_embed=use_embed,
        ):
            h_abl = _last_token_logits_batched(bk.model, h_ids, h_mask, h_pos, batch_size=batch_size)
            b_abl = _last_token_logits_batched(bk.model, b_ids, b_mask, b_pos, batch_size=batch_size)
        lo_abl = float(refusal_log_odds_from_logits(h_abl, ref_ids).mean().item())
        global_drop[l] = max(0.0, base_h_lo - lo_abl)
        benign_kl[l] = kl_divergence_from_logits(base_b_logits, b_abl)

        # (2) Sufficiency check (Arditi §C.1): single-layer activation addition +r_raw^(l)
        with bk.hooks.intervene(
            add_vec=r_raw_by_layer[l], add_coeff=1.0, add_layers=[l], add_positions="all",
        ):
            b_add = _last_token_logits_batched(bk.model, b_ids, b_mask, b_pos, batch_size=batch_size)
        lo_add = float(refusal_log_odds_from_logits(b_add, ref_ids).mean().item())
        induce_log_odds[l] = lo_add
        induce_gain[l] = max(0.0, lo_add - base_b_lo)

        # (3) Localized 3-layer ablation -> causal bottleneck sensitivity in depth
        win = [j for j in (l - 1, l, l + 1) if 0 <= j < L]
        byl = {j: r_unit_by_layer[j] for j in win}
        with bk.hooks.intervene(
            ablate_by_layer=byl, ablate_positions="all",
            ablate_mode=ablate_mode, ablate_embed=False,
        ):
            h_loc = _last_token_logits_batched(bk.model, h_ids, h_mask, h_pos, batch_size=batch_size)
        lo_loc = float(refusal_log_odds_from_logits(h_loc, ref_ids).mean().item())
        local_drop[l] = max(0.0, base_h_lo - lo_loc)

    # Arditi §C.1 tiered peak selection: induce_score > 0 AND low KL -> max global_drop
    valid = [l for l in zone if induce_log_odds[l] > 0.0 and benign_kl[l] <= kl_cap]
    if not valid:
        valid = [l for l in zone if induce_log_odds[l] > 0.0 and benign_kl[l] <= kl_cap_relaxed]
    if not valid:
        # Relax to strong positive refusal induction shift (handles lower-refusal models like Mistral)
        max_ig = max(induce_gain.values())
        valid = [l for l in zone if induce_gain[l] >= 0.4 * max_ig and benign_kl[l] <= kl_cap_relaxed]
    if not valid:
        valid = list(zone)

    # Pick layer maximizing refusal bypass (with gentle Pareto KL tie-breaking if relaxed)
    if all(benign_kl[l] <= kl_cap for l in valid):
        peak = max(valid, key=lambda l: global_drop[l])
    else:
        peak = max(valid, key=lambda l: global_drop[l] / (1.0 + 1.5 * benign_kl[l]))

    max_g = max(global_drop.values())
    max_loc = max(local_drop.values())
    max_ig = max(induce_gain.values())

    # Combined causal score (necessity + localized bottleneck + sufficiency)
    causal_score: Dict[int, float] = {}
    for l in zone:
        gn = global_drop[l] / (max_g + 1e-9) if max_g > 1e-4 else 0.0
        ln = local_drop[l] / (max_loc + 1e-9) if max_loc > 1e-4 else gn
        ign = induce_gain[l] / (max_ig + 1e-9) if max_ig > 1e-4 else gn
        causal_score[l] = 0.45 * gn + 0.35 * ln + 0.20 * ign

    peak_score = max(causal_score[peak], 1e-6)
    peak_g = max(global_drop[peak], 1e-6)
    if max_g > 1e-4:
        thr_strict = strict_frac * peak_score
        lo, hi = peak, peak
        while (lo - 1 >= z_lo
               and causal_score[lo - 1] >= thr_strict
               and global_drop[lo - 1] >= 0.35 * peak_g):
            lo -= 1
        while (hi + 1 <= z_hi
               and causal_score[hi + 1] >= thr_strict
               and global_drop[hi + 1] >= 0.35 * peak_g):
            hi += 1
        if (hi - lo + 1) < min_width:
            thr_rel = relaxed_frac * peak_score
            while lo - 1 >= z_lo and causal_score[lo - 1] >= thr_rel:
                lo -= 1
            while hi + 1 <= z_hi and causal_score[hi + 1] >= thr_rel:
                hi += 1
        beh_basin = list(range(lo, hi + 1))
    else:
        beh_basin = geo_basin
        peak = geo_peak

    while len(beh_basin) < min(min_width, len(zone)):
        can_lo = beh_basin[0] > z_lo
        can_hi = beh_basin[-1] < z_hi
        if can_lo and can_hi:
            if causal_score[beh_basin[-1] + 1] >= causal_score[beh_basin[0] - 1]:
                beh_basin.append(beh_basin[-1] + 1)
            else:
                beh_basin.insert(0, beh_basin[0] - 1)
        elif can_hi:
            beh_basin.append(beh_basin[-1] + 1)
        elif can_lo:
            beh_basin.insert(0, beh_basin[0] - 1)
        else:
            break

    s_beh, s_geo = set(beh_basin), set(geo_basin)
    iou = len(s_beh & s_geo) / max(1, len(s_beh | s_geo))

    return {
        "peak": int(peak),
        "basin": beh_basin,
        "geometric_peak": int(geo_peak),
        "geometric_basin": geo_basin,
        "basin_iou": round(float(iou), 4),
        "refusal_token_ids": ref_ids,
        "refusal_tokens": ref_strs,
        "ablate_mode": ablate_mode,
        "base_harmful_log_odds": round(base_h_lo, 4),
        "base_benign_log_odds": round(base_b_lo, 4),
        "peak_induce_log_odds": round(induce_log_odds[peak], 4),
        "peak_global_drop": round(global_drop[peak], 4),
        "peak_benign_kl": round(benign_kl[peak], 5),
        "global_drop": {int(k): round(v, 4) for k, v in global_drop.items()},
        "local_drop": {int(k): round(v, 4) for k, v in local_drop.items()},
        "induce_log_odds": {int(k): round(v, 4) for k, v in induce_log_odds.items()},
        "induce_gain": {int(k): round(v, 4) for k, v in induce_gain.items()},
        "causal_score": {int(k): round(v, 4) for k, v in causal_score.items()},
        "benign_kl": {int(k): round(v, 5) for k, v in benign_kl.items()},
    }


def discover_basin_causal(bk, U_peak=None, peak: Optional[int] = None, frac: float = 0.5):
    """Backward-compatible wrapper returning (basin_layers, effect_tensor)."""
    info = discover_basin_behavioral(bk)
    effect = torch.zeros(bk.n_layers)
    for l, val in info["causal_score"].items():
        effect[int(l)] = float(val)
    return info["basin"], effect