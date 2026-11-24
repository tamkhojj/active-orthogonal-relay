"""LaTeX (booktabs) table exporter.

Reads the experiment JSONs from results/ and emits the 13 paper tables (3 main,
10 appendix) with standardized terminology.

Usage:
  PYTHONNOUSERSITE=1 python -s aor/plot/export_tables.py --root results --outdir tables"""
from __future__ import annotations

import os
import json
import math
import argparse
from typing import Any, Dict, List, Optional, Sequence

MODEL_ORDER = [
    ("gemma-2-2b-it",            "Gemma-2-2B"),
    ("Meta-Llama-3-8B-Instruct", "Llama-3-8B"),
    ("Llama-3.1-8B-Instruct",    "Llama-3.1-8B"),
    ("Qwen2-7B-Instruct",        "Qwen2-7B"),
    ("Qwen2.5-3B-Instruct",      "Qwen2.5-3B"),
    ("Qwen2.5-7B-Instruct",      "Qwen2.5-7B"),
    ("Mistral-7B-Instruct-v0.3", "Mistral-7B"),
]
MODEL_SHORT = dict(MODEL_ORDER)

# Standardized display names for the exported tables
METHOD_LABEL = {
    "BASELINE":             "Baseline",
    "STATIC_GLOBAL_k1":     r"Static Global ($k{=}1$)",
    "STATIC_GLOBAL_k2":     r"Static Global ($k{=}2$)",
    "STATIC_BASIN_k1":      r"Static Basin ($k{=}1$)",
    "STATIC_BASIN_k2":      r"Static Basin ($k{=}2$)",
    "GEORA_FLAT_k1":        r"Flat Geodesic ($k{=}1$)",
    "GEORA_FLAT_k2":        r"Flat Geodesic ($k{=}2$)",
    "GEORA_TORSION_k1":     r"Torsion Transport ($k{=}1$)",
    "GEORA_TORSION_k2":     r"Torsion Transport ($k{=}2$)",
    "LARA_PER_LAYER_k1":    r"Per-Layer Basin ($k{=}1$)",
    "LARA_PER_LAYER_k2":    r"Per-Layer Basin ($k{=}2$)",
    "TRAJ_SVD_r2":          r"Traj-SVD ($r{=}2$)",
    "TRAJ_SVD_r3":          r"Traj-SVD ($r{=}3$)",
    "ABL_E3_WRITES":        r"$e_3$-Only Writes",
    "ABL_E3_MLP_ONLY":      r"$e_3$-Only MLP",
    "ABL_E1_WRITES":        r"$e_1$ (Incoming)",
    "ABL_E2_WRITES":        r"$e_2$ (In-Plane)",
    "ABL_RAND_ORTH_WRITES": r"Rand $\perp(e_1,e_2)$",
}


# =========================================================================== #
# Helpers
# =========================================================================== #
def _pick(d: Optional[dict], *keys: str, default: Any = None) -> Any:
    if not isinstance(d, dict):
        return default
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def _mean(vals: Sequence[Optional[float]]) -> float:
    clean = [float(v) for v in vals if v is not None and not math.isnan(float(v))]
    return sum(clean) / len(clean) if clean else float("nan")


def _fmt(val: Optional[float], dec: int = 1, suffix: str = "", sign: bool = False) -> str:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "--"
    f = f"{float(val):+.{dec}f}" if sign else f"{float(val):.{dec}f}"
    return f"{f}{suffix}"


def _fmt_p(p: Optional[float]) -> str:
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "--"
    pv = float(p)
    if pv < 0.0005:
        return r"$<.001$"
    return f"{pv:.3f}"


def _basin_str(b: Optional[List[int]]) -> str:
    if not b:
        return "--"
    return f"[{min(b)}..{max(b)}]"


def load_all(root: str) -> Dict[str, dict]:
    data: Dict[str, dict] = {}
    agg_p = os.path.join(root, "aggregate_sync_7models.json")
    agg_sync = json.load(open(agg_p, encoding="utf-8")) if os.path.isfile(agg_p) else {}

    for tag, short in MODEL_ORDER:
        mdir = os.path.join(root, tag)
        if not os.path.isdir(mdir):
            continue

        def _j(prefix: str) -> Optional[dict]:
            p = os.path.join(mdir, f"{prefix}_{tag}.json")
            return json.load(open(p, encoding="utf-8")) if os.path.isfile(p) else None

        data[tag] = {
            "short": short,
            "exp2": _j("exp2"),
            "exp3": _j("exp3"),
            "exp4": _j("exp4"),
            "exp5": _j("exp5"),
            "exp6": _j("exp6"),
            "exp7": _j("exp7_utility_xstest_tqa"),
            "exp8": _j("exp8_defense"),
            "exp9": _j("exp9_protocol"),
            "exp10": _j("exp10_e3_gold"),
            "agg_sync": agg_sync.get(tag) if isinstance(agg_sync, dict) else None,
        }
    return data


def _tags(data: Dict[str, dict], key: str) -> List[str]:
    return [t for t, _ in MODEL_ORDER if t in data and data[t].get(key) is not None]


def _wrap_table(env: str, caption: str, label: str, colspec: str,
                header_lines: List[str], body_lines: List[str],
                size_cmd: str = r"\small", tabcolsep: str = "4pt",
                resize: bool = True) -> str:
    lines = [
        f"\\begin{{{env}}}[t]",
        r"\centering",
        size_cmd,
        f"\\setlength{{\\tabcolsep}}{{{tabcolsep}}}",
    ]
    if resize:
        lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(f"\\begin{{tabular}}{{{colspec}}}")
    lines.append(r"\toprule")
    lines.extend(header_lines)
    lines.append(r"\midrule")
    lines.extend(body_lines)
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    if resize:
        lines.append(r"}")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append(f"\\end{{{env}}}")
    return "\n".join(lines) + "\n"


# =========================================================================== #
# MAIN TABLE 1: 3-Tier Causal Funnel (Exp8 Split-Half + Cov-Matched Controls)
# =========================================================================== #
def build_tab1_main_three_tier(data: Dict[str, dict]) -> str:
    tags = _tags(data, "exp8")
    hdr = [
        r"\textbf{Model} & \textbf{Refusal $e_3$ Cons.} & \textbf{Refusal Relay} & "
        r"\textbf{Format (Native)} & \textbf{Truth (Native)} & \textbf{Random Null} & "
        r"\textbf{Knockout Frac.} & \textbf{Perm.} \\",
        r" & $\cos^2$ (cov-$\times$) & $\cos^2$ (cov-$\times$) & $\cos^2$ (cov-$\times$) & "
        r"$\cos^2$ (cov-$\times$) & $\cos^2$ (cov-$\times$) & $|\Delta m{\cdot}e_3|/|w{\cdot}e_3|$ & $p$-val \\",
    ]
    body = []
    acc = {k: [] for k in ("rc", "rc_x", "rr", "rr_x", "fr", "fr_x", "tr", "tr_x", "nr", "nr_x", "kf")}
    for t in tags:
        d = data[t]["exp8"]
        d1 = d["d1_split_half_exp5b"]
        d2a = d["d2a_random_split_null"]
        d2b = d["d2b_layer_permutation"]
        fn = d["d2d_format_behavior"]["in_native_basin"]["split_half"]
        tn = d["d2c_truthfulqa_concept"]["in_native_basin"]["split_half"]

        rc, rc_x = d1["cross_fold_e3_cos2"], d1["cross_fold_e3_cov_enrichment_x"]
        rr, rr_x = d1["repair_mlp_e3"], d1["repair_mlp_e3_cov_enrichment_x"]
        fr, fr_x = fn["repair_mlp_e3"], fn["repair_mlp_e3_cov_enrichment_x"]
        tr, tr_x = tn["repair_mlp_e3"], tn["repair_mlp_e3_cov_enrichment_x"]
        nr, nr_x = d2a["cv_repair_mlp_e3_cos2"], d2a["cv_repair_mlp_e3_cov_enrichment_x"]
        kf = d1["repair_to_clean_e3_proj_ratio"]
        pv = d2b["perm_p_val_0p"]

        for k, v in [("rc", rc), ("rc_x", rc_x), ("rr", rr), ("rr_x", rr_x),
                     ("fr", fr), ("fr_x", fr_x), ("tr", tr), ("tr_x", tr_x),
                     ("nr", nr), ("nr_x", nr_x), ("kf", kf)]:
            acc[k].append(v)

        body.append(
            f"{MODEL_SHORT[t]} & "
            f"{rc:.3f} ({rc_x:.1f}$\\times$) & "
            f"\\textbf{{{rr:.3f}}} ({rr_x:.1f}$\\times$) & "
            f"{fr:.3f} ({fr_x:.1f}$\\times$) & "
            f"{tr:.3f} ({tr_x:.1f}$\\times$) & "
            f"{nr:.3f} ({nr_x:.1f}$\\times$) & "
            f"{kf:.3f} & {pv:.4f} \\\\"
        )

    body.append(r"\midrule")
    body.append(
        f"\\textbf{{Mean (7 Models)}} & "
        f"\\textbf{{{_mean(acc['rc']):.3f}}} ({_mean(acc['rc_x']):.1f}$\\times$) & "
        f"\\textbf{{{_mean(acc['rr']):.3f}}} ({_mean(acc['rr_x']):.1f}$\\times$) & "
        f"{_mean(acc['fr']):.3f} ({_mean(acc['fr_x']):.1f}$\\times$) & "
        f"{_mean(acc['tr']):.3f} ({_mean(acc['tr_x']):.1f}$\\times$) & "
        f"{_mean(acc['nr']):.3f} ({_mean(acc['nr_x']):.1f}$\\times$) & "
        f"\\textbf{{{_mean(acc['kf']):.3f}}} & 0.0099 \\\\"
    )

    cap = (
        r"\textbf{Non-circular 2-fold cross-validated relay analysis across 7 models.} "
        r"Refusal exhibits deterministic cross-fold $e_3$ stability and a strong downstream MLP causal relay "
        r"$\cos^2(\Delta m_l, e_3^{(l-1)})$ (mean $0.526$, knocking out $56.3\%$ of clean $e_3$ MLP writes), "
        r"outpacing both Behavioral Format ($3.3\times$) and Factual Truth ($10.4\times$) evaluated in their own native basins, "
        r"as well as empirical covariance-matched Random-Split Null ($4.0\text{--}7.9\times$). "
        r"Layer permutation (100 shuffles) yields $p=0.0099$ on 7/7 models."
    )
    return _wrap_table("table*", cap, "tab:three_tier", "lccccccc", hdr, body, tabcolsep="4.5pt")


# =========================================================================== #
# MAIN TABLE 2: Direct Axis Knock-Out Gold StrongREJECT & TruthfulQA (Exp10)
# =========================================================================== #
def build_tab2_main_direct_e3_gold(data: Dict[str, dict]) -> str:
    tags = _tags(data, "exp10")
    hdr = [
        r"\textbf{Model} & \textbf{$e_3$ Writes} & \textbf{$e_3$ MLP-Only} & "
        r"\textbf{$e_1$ (Incom.)} & \textbf{$e_2$ (Plane)} & \textbf{Rand $\perp$} & "
        r"\textbf{Per-Layer ($k{=}1$)} & \textbf{$p(e_3{>}e_1)$} & \textbf{$\Delta$TQA MC1} & \textbf{KL Ratio} \\",
        r" & SR / NR (\%) & SR / NR (\%) & SR / NR (\%) & SR / NR (\%) & SR / NR (\%) & "
        r"SR / NR (\%) & (SR Mean) & ($e_3$ vs. Base) & (PL/$e_3$) \\",
    ]
    cols = ["ABL_E3_WRITES", "ABL_E3_MLP_ONLY", "ABL_E1_WRITES", "ABL_E2_WRITES",
            "ABL_RAND_ORTH_WRITES", "LARA_PER_LAYER_k1"]
    acc_sr = {c: [] for c in cols}
    acc_nr = {c: [] for c in cols}
    acc_dtqa, acc_klr = [], []
    body = []

    for t in tags:
        d = data[t]["exp10"]
        g = d["gold_asr"]
        tc = d["truthfulqa"]["conditions"]
        uf = d["utility_and_force"]
        ps = d["paired_significance"]

        cells = []
        for c in cols:
            sr_v, nr_v = g[c]["asr"], g[c]["non_refusal_asr"]
            acc_sr[c].append(sr_v); acc_nr[c].append(nr_v)
            if c == "ABL_E3_WRITES":
                cells.append(f"\\textbf{{{sr_v:.1f}}} / {nr_v:.1f}")
            else:
                cells.append(f"{sr_v:.1f} / {nr_v:.1f}")

        pe1 = ps["ABL_E3_WRITES_vs_ABL_E1_WRITES"]["p_val"]
        dtqa = tc["ABL_E3_WRITES"]["mc1"] - tc["BASELINE"]["mc1"]
        kl_e3 = uf["ABL_E3_WRITES"]["benign_kl"]
        kl_pl = uf["LARA_PER_LAYER_k1"]["benign_kl"]
        klr = kl_pl / max(kl_e3, 1e-6)
        acc_dtqa.append(dtqa); acc_klr.append(klr)

        body.append(
            f"{MODEL_SHORT[t]} & " + " & ".join(cells) +
            f" & {_fmt_p(pe1)} & {dtqa:+.2f} & {klr:.2f}$\\times$ \\\\"
        )

    body.append(r"\midrule")
    mcells = []
    for c in cols:
        if c == "ABL_E3_WRITES":
            mcells.append(f"\\textbf{{{_mean(acc_sr[c]):.1f}}} / \\textbf{{{_mean(acc_nr[c]):.1f}}}")
        else:
            mcells.append(f"{_mean(acc_sr[c]):.1f} / {_mean(acc_nr[c]):.1f}")
    body.append(
        r"\textbf{Mean (7 Models)} & " + " & ".join(mcells) +
        f" & -- & {_mean(acc_dtqa):+.2f} & {_mean(acc_klr):.2f}$\\times$ \\\\"
    )

    cap = (
        r"\textbf{Direct axis knock-out inside the Causal Basin on held-out prompts ($n{=}150$, 256 tokens).} "
        r"Ablating strictly the orthogonal innovation axis $e_3^{(l-1)}$ (orthogonal to the incoming refusal direction $v_{l-1}$) "
        r"at sublayer writes achieves $50.0\%$ StrongREJECT Mean / $83.4\%$ Non-Refusal---exceeding incoming-axis $e_1$ ablation "
        r"by $+35.3$ pts ($p{<}0.001$ on 6/7 models, 7/7 significant) while reducing Benign KL by $1.1\text{--}4.3\times$ "
        r"relative to Per-Layer Basin ($k{=}1$) with minimal TruthfulQA MC1 change ($-0.6$ pts)."
    )
    return _wrap_table("table*", cap, "tab:direct_e3_gold", "lccccccccc", hdr, body, tabcolsep="3.8pt")


# =========================================================================== #
# MAIN TABLE 3: 7-Axis Safety-Utility Pareto Summary (7-Model Mean)
# =========================================================================== #
def build_tab3_main_pareto_7axis(data: Dict[str, dict]) -> str:
    tags16 = _tags(data, "exp6")
    tags17 = _tags(data, "exp7")
    tags20 = _tags(data, "exp10")

    hdr = [
        r"\textbf{Intervention (Writes)} & \textbf{SR Mean} & \textbf{NonRef} & "
        r"\textbf{XS Unsafe} & \textbf{XS Safe ORR} & \textbf{TQA MC1/MC2} & "
        r"\textbf{Benign $\Delta$CE} & \textbf{Force} & \textbf{Col.} \\",
        r" & ($\uparrow$ \%) & ($\uparrow$ \%) & ($\uparrow$ \%) & ($\downarrow$ \%) & "
        r"($\uparrow$ \%, 817 Qs) & ($\downarrow$) & ($\downarrow$) & (\%) \\",
    ]

    # (key, source_for_sr_util, source_for_tqa, has_xstest)
    specs = [
        ("BASELINE",          "exp6", "exp7", True),
        ("STATIC_GLOBAL_k1",  "exp6", "exp7", True),
        ("STATIC_BASIN_k1",   "exp6", "exp7", True),
        ("GEORA_FLAT_k1",     "exp6", "exp7", True),
        ("GEORA_TORSION_k1",  "exp6", "exp7", True),
        ("ABL_E3_WRITES",     "exp10", "exp10", False),
        ("LARA_PER_LAYER_k1", "exp6", "exp7", True),
        ("STATIC_GLOBAL_k2",  "exp6", "exp7", True),
        ("LARA_PER_LAYER_k2", "exp6", "exp7", True),
    ]

    base_ce_16 = {t: data[t]["exp6"]["utility_and_force"]["BASELINE"]["benign_ce_loss"] for t in tags16}
    base_ce_20 = {t: data[t]["exp10"]["utility_and_force"]["BASELINE"]["benign_ce_loss"] for t in tags20}

    body = []
    for idx, (ck, s_sr, s_tq, has_xs) in enumerate(specs):
        if idx in (1, 7):
            body.append(r"\midrule")
        ts = tags16 if s_sr == "exp6" else tags20
        bce = base_ce_16 if s_sr == "exp6" else base_ce_20

        sr_m = _mean([data[t][s_sr]["gold_asr"][ck]["asr"] for t in ts])
        nr_m = _mean([data[t][s_sr]["gold_asr"][ck]["non_refusal_asr"] for t in ts])
        dce = _mean([data[t][s_sr]["utility_and_force"][ck]["benign_ce_loss"] - bce[t] for t in ts])
        frc = _mean([data[t][s_sr]["utility_and_force"][ck]["force"]["total_force_norm"] for t in ts])
        col = _mean([data[t][s_sr]["lexical_diagnostics"][ck]["collapse_rate_pct"] for t in ts])

        ts_tq = tags17 if s_tq == "exp7" else tags20
        mc1 = _mean([data[t][s_tq]["truthfulqa"]["conditions"][ck]["mc1"] for t in ts_tq])
        mc2 = _mean([data[t][s_tq]["truthfulqa"]["conditions"][ck]["mc2"] for t in ts_tq])

        if has_xs:
            xs_u = _mean([data[t]["exp7"]["xstest"]["conditions"][ck]["unsafe_nonref_pct"] for t in tags17])
            xs_s = _mean([data[t]["exp7"]["xstest"]["conditions"][ck]["safe_over_refusal_pct"] for t in tags17])
            xs_u_s, xs_s_s = f"{xs_u:.1f}", f"{xs_s:.1f}"
        else:
            # Exp8 D3 tested XSTest Unsafe (n=100) for ABL_E3_TORSION writes
            tags18 = _tags(data, "exp8")
            xs_u = _mean([data[t]["exp8"]["d3_direct_e3_knockout"]["by_hook_mode"]["writes"]["ABL_E3_TORSION"]["xstest_unsafe_nonref_pct"] for t in tags18])
            xs_u_s = f"{xs_u:.1f}$^\\dagger$"
            # XSTest-safe ORR for e3-Only is filled by fill_gaps.py into
            # exp10["xstest_e3"]["safe_over_refusal_pct"]; falls back to "--".
            e3_safe = [((data[t]["exp10"] or {}).get("xstest_e3") or {}).get("safe_over_refusal_pct")
                       for t in tags20]
            e3_safe = [v for v in e3_safe if v is not None]
            xs_s_s = f"{_mean(e3_safe):.1f}" if len(e3_safe) == len(tags20) and e3_safe else "--"

        lbl = METHOD_LABEL[ck]
        if ck in ("ABL_E3_WRITES", "LARA_PER_LAYER_k1", "LARA_PER_LAYER_k2"):
            lbl = f"\\textbf{{{lbl}}}"
        sr_s = f"\\textbf{{{sr_m:.1f}}}" if ck in ("LARA_PER_LAYER_k1", "LARA_PER_LAYER_k2") else f"{sr_m:.1f}"
        xs_s_fmt = f"\\textbf{{{xs_s_s}}}" if ck == "LARA_PER_LAYER_k1" else xs_s_s

        body.append(
            f"{lbl} & {sr_s} & {nr_m:.1f} & {xs_u_s} & {xs_s_fmt} & "
            f"{mc1:.1f} / {mc2:.1f} & {dce:+.3f} & {frc:.1f} & {col:.1f} \\\\"
        )

    cap = (
        r"\textbf{7-axis safety--utility Pareto summary averaged across all 7 models (sublayer writes).} "
        r"Per-Layer Basin dominates Static Global (Arditi et al., 2024) simultaneously across StrongREJECT Mean, "
        r"Non-Refusal, XSTest Unsafe Non-Refusal, XSTest Safe Over-Refusal (ORR), TruthfulQA MC1/MC2, Benign $\Delta$CE, "
        r"and Total Force Norm with $0.0\%$ semantic collapse. "
        r"$^\dagger$Evaluated on 100 XSTest Unsafe prompts in Exp8 D3."
    )
    return _wrap_table("table*", cap, "tab:pareto_7axis", "lcccccccc", hdr, body, tabcolsep="4.5pt")


# =========================================================================== #
# APPENDIX C: Automated Causal Basin Localization & Exp5a Frenet Geometry
# =========================================================================== #
def build_tabC_basin_and_frenet(data: Dict[str, dict]) -> str:
    tags = _tags(data, "exp5")
    hdr = [
        r"\textbf{Model} & \textbf{Peak / Basin} & \textbf{IoU} & \textbf{$\bar{\kappa}$ (CV)} & "
        r"\textbf{$\bar{\tau}$} & \textbf{0p $R^2$ (Geo / Exp)} & \textbf{1p $R^2$ (Exp, $p$)} & "
        r"\textbf{SVD SR} & \textbf{Recon (Flat $\to$ Tor.)} \\",
    ]
    body = []
    for t in tags:
        d15 = data[t]["exp5"]
        fb = d15.get("frenet_basin", {})
        bi = d15.get("basin_info", {})
        pk = d15["peak"]
        bs = _basin_str(d15["basin"])
        iou = _pick(bi, "basin_iou")
        kap = _pick(fb, "mean_kappa_rad")
        cv = _pick(fb, "velocity_cv")
        tau = _pick(d15, "mean_torsion_tau") or _pick(fb, "mean_tau")
        r2_0g = _pick(fb, "zero_param_r2_geodesic")
        r2_0e = _pick(d15, "zero_param_exp_drift_r2") or _pick(fb, "zero_param_r2_orthogonal_drift")
        r2_1e = _pick(d15, "exp_drift_r2") or _pick(fb, "fit_1p_r2_orthogonal_drift")
        pval = _pick(fb, "bootstrap_p_exp_better")
        # SVD stable rank lives under trajectory_svd_basin; k=1 recon errors live
        # under grassmann_by_rank["1"]["basin"] (geodesic = flat 2-anchor,
        # piecewise_mid = torsion 3-anchor). Keep legacy flat-key fallbacks.
        svd_basin = d15.get("trajectory_svd_basin", {}) or {}
        g1_basin = (((d15.get("grassmann_by_rank", {}) or {}).get("1", {}) or {}).get("basin", {}) or {})
        sr = (_pick(fb, "traj_svd_stable_rank", "stable_rank")
              or _pick(d15, "traj_svd_stable_rank")
              or _pick(svd_basin, "stable_rank"))
        re_f = (_pick(fb, "recon_err_flat_k1", "flat_2anchor_recon_err")
                or _pick(d15, "recon_err_flat_k1")
                or _pick(g1_basin, "geodesic_recon_err"))
        re_t = (_pick(fb, "recon_err_torsion_k1", "torsion_3anchor_recon_err")
                or _pick(d15, "recon_err_torsion_k1")
                or _pick(g1_basin, "piecewise_mid_recon_err"))

        body.append(
            f"{MODEL_SHORT[t]} & L{pk} {bs} & {_fmt(iou, 3)} & "
            f"{_fmt(kap, 3)} ({_fmt(cv, 2)}) & {_fmt(tau, 4)} & "
            f"{_fmt(r2_0g, 2)} / {_fmt(r2_0e, 3)} & {_fmt(r2_1e, 3)} ({_fmt_p(pval)}) & "
            f"{_fmt(sr, 2)} & {_fmt(re_f, 3)} $\\to$ {_fmt(re_t, 3)} \\\\"
        )

    cap = (
        r"\textbf{Appendix C: Automated Causal Basin localization and discrete Frenet--Serret geometry (Exp5a).} "
        r"Across all 7 models, behavioral and geometric basins strongly overlap ($\mathrm{IoU}{=}0.636\text{--}1.000$), "
        r"torsion approaches unity ($\bar{\tau}{=}0.982\text{--}0.998$), and zero-parameter orthogonal drift $(\cos\bar{\kappa})^m$ "
        r"decisively outperforms planar geodesic rotation $\cos(m\bar{\kappa})$ ($p{<}0.001$)."
    )
    return _wrap_table("table*", cap, "tab:appC_basin_frenet", "lcccccccc", hdr, body, tabcolsep="3.8pt")


# =========================================================================== #
# APPENDIX D: Synchronized Exp2-Exp4 Self-Repair & Exp4 Gold StrongREJECT per k
# =========================================================================== #
def build_tabD_self_repair_exp2_exp4(data: Dict[str, dict]) -> str:
    tags = _tags(data, "exp4")
    hdr = [
        r"\textbf{Model} & \textbf{Exp2 Repair} & \textbf{Exp3 SR / PR} & \textbf{Comp. Peak} & "
        r"\textbf{Col. $k^*$} & \multicolumn{6}{c}{\textbf{Exp4 Gold StrongREJECT Mean ASR (\%) by $k$}} \\",
        r"\cmidrule(lr){6-11}",
        r" & (Self / MLP) & (Rank) & (Max $\times$ @ $k$) & (Geom) & $k{=}1$ & $k{=}2$ & $k{=}3$ & $k{=}4$ & $k{=}5$ & $k{=}6$ \\",
    ]
    body = []
    for t in tags:
        d12 = data[t]["exp2"] or {}
        d13 = data[t]["exp3"] or {}
        d14 = data[t]["exp4"] or {}
        sr_ok = _pick(d12, "verdict_self_repair", "self_repair", default=True)
        mlp_ok = _pick(d12, "verdict_mlp_written", "mlp_written", default=True)
        sr_v = _pick(d13, "stable_rank", "recovery_stable_rank")
        pr_v = _pick(d13, "participation_ratio", "recovery_participation")
        c_list = d14.get("compensation_frac") or []
        if c_list:
            c_peak = max(c_list)
            c_k = c_list.index(c_peak) + 1
            comp_s = f"{c_peak:.2f}$\\times$@{c_k}"
        else:
            comp_s = "--"
        col_k = _pick(d14, "collapse_k", "collapse_geom_k")
        if col_k is None and c_list:
            sup = d14.get("suppression_frac") or []
            col_k = next((i + 1 for i, v in enumerate(sup) if v <= 0.10), len(c_list))

        # Extract Exp4 gold ASR per k=1..6 defensively. The Exp4 JSON stores this
        # as a plain list of per-k StrongREJECT Mean ASR under "gold_asr"
        # (NOT a dict of conditions like Exp10's "gold_asr"); older/alt layouts may
        # use "gold_asr_per_k" etc. or [k, val] pairs.
        gold_k = _pick(d14, "gold_asr", "gold_asr_per_k", "asr_per_k", "strongreject_per_k")
        k_cells = ["--"] * 6
        if isinstance(gold_k, dict):
            for ki in range(1, 7):
                v = gold_k.get(str(ki), gold_k.get(ki))
                if isinstance(v, dict):
                    v = v.get("asr")
                if v is not None:
                    k_cells[ki - 1] = f"{float(v):.1f}"
        elif isinstance(gold_k, list):
            for i, item in enumerate(gold_k):
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    ki = int(item[0])
                    if 1 <= ki <= 6:
                        k_cells[ki - 1] = f"{float(item[1]):.1f}"
                elif isinstance(item, (int, float)) and i < 6:
                    k_cells[i] = f"{float(item):.1f}"

        body.append(
            f"{MODEL_SHORT[t]} & {'Yes' if sr_ok else 'No'} / {'MLP' if mlp_ok else 'Attn'} & "
            f"{_fmt(sr_v, 2)} / {_fmt(pr_v, 2)} & {comp_s} & {col_k if col_k else '--'} & " +
            " & ".join(k_cells) + r" \\"
        )

    cap = (
        r"\textbf{Appendix D: Synchronized self-repair dynamics (Exp2--Exp4) and Exp4 Gold StrongREJECT ASR across ablation rank $k$.} "
        r"All 7 models exhibit low-rank recovery ($\mathrm{SR}{=}1.41\text{--}2.55$) and a compensation-to-collapse phase transition "
        r"(peaking at $1.36\times\text{--}7.41\times$ before collapsing at $k^*{\in}\{4,5,6\}$), with architectural divergence "
        r"on Gemma-2-2B and Mistral-7B where Attention carries larger post-basin recovery write norm."
    )
    return _wrap_table("table*", cap, "tab:appD_self_repair", "lcccccccccc", hdr, body, tabcolsep="3.8pt")


# =========================================================================== #
# APPENDIX E: Exhaustive Exp8 Controls (Refusal Basin vs Native Basin)
# =========================================================================== #
def build_tabE_controls_ref_vs_native(data: Dict[str, dict]) -> str:
    tags = _tags(data, "exp8")
    hdr = [
        r"\textbf{Model} & \textbf{Ref. Clean} & \textbf{Ref. Relay} & "
        r"\textbf{Fmt. $d'$ / $|\cos|$} & \textbf{Fmt. Relay} & \textbf{Tru. $d'$ / $|\cos|$} & "
        r"\textbf{Tru. Relay} & \textbf{Null Clean / Relay} \\",
        r" & $\cos^2$ (iso / cov) & $s$, $\|\Delta m\|$ & (Native Basin) & "
        r"RefB / NatB ($\cos^2$) & (Native Basin) & RefB / NatB ($\cos^2$) & $\|\Delta m\|$, cov-$\times$ \\",
    ]
    body = []
    for t in tags:
        d = data[t]["exp8"]
        d1 = d["d1_split_half_exp5b"]
        d2a = d["d2a_random_split_null"]
        d2c = d["d2c_truthfulqa_concept"]
        d2d = d["d2d_format_behavior"]

        f_ref = d2d["in_refusal_basin"]["split_half"]["repair_mlp_e3"]
        f_nat = d2d["in_native_basin"]["split_half"]["repair_mlp_e3"]
        t_ref = d2c["in_refusal_basin"]["split_half"]["repair_mlp_e3"]
        t_nat = d2c["in_native_basin"]["split_half"]["repair_mlp_e3"]

        body.append(
            f"{MODEL_SHORT[t]} & "
            f"{d1['clean_mlp_e3']:.3f} ({d1['clean_mlp_e3_enrichment_x']:.0f} / {d1['clean_mlp_e3_cov_enrichment_x']:.1f}$\\times$) & "
            f"{d1['repair_mlp_e3_signed']:+.2f}, {d1['repair_mlp_norm']:.2f} & "
            f"{d2d['native_peak_dprime']:.1f} / {d2d['mean_abs_cos_vs_refusal_in_basin']:.3f} & "
            f"{f_ref:.3f} / {f_nat:.3f} & "
            f"{d2c['native_peak_dprime']:.2f} / {d2c['mean_abs_cos_vs_refusal_in_basin']:.3f} & "
            f"{t_ref:.3f} / {t_nat:.3f} & "
            f"{d2a['cv_clean_mlp_e3_cov_enrichment_x']:.2f}$\\times$ / {d2a['repair_mlp_norm']:.2f} ({d2a['cv_repair_mlp_e3_cov_enrichment_x']:.1f}$\\times$) \\\\"
        )

    cap = (
        r"\textbf{Appendix E: Exhaustive non-circular split-half controls across Refusal, Behavioral Format, Factual Truth, and Random-Split Null.} "
        r"Both Format and Truth are nearly orthogonal to Refusal ($|\cos|{=}0.02\text{--}0.22$). Even though Format has $5\text{--}13\times$ higher linear "
        r"separability $d'$ than Refusal, its native-basin causal relay remains far weaker ($0.058\text{--}0.255$ vs. Refusal $0.318\text{--}0.789$). "
        r"Under empirical covariance-matched null, Random-Split Clean MLP drops below $0.7\times$ baseline."
    )
    return _wrap_table("table*", cap, "tab:appE_controls", "lccccccc", hdr, body, tabcolsep="3.5pt")


# =========================================================================== #
# APPENDIX F: Direct Axis Knock-Out Log-Odds (D3) & Steering Specificity (D4)
# =========================================================================== #
def build_tabF_d3_logodds_and_steering(data: Dict[str, dict]) -> str:
    tags = _tags(data, "exp8")
    hdr = [
        r"\textbf{Model} & \multicolumn{3}{c}{\textbf{D3 `mlp\_writes` ($\Delta$LO / XS NR\%)}} & "
        r"\multicolumn{3}{c}{\textbf{D3 `writes` ($\Delta$LO / XS NR\%)}} & "
        r"\multicolumn{3}{c}{\textbf{D4 Steering Benign Ref. (\%)}} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
        r" & $e_3$ & $e_2$ & Rand $\perp$ & $e_3$ & $e_1$ & $e_2$ & Base & Peak ($l^*$) & Out Local \\",
    ]
    body = []
    for t in tags:
        d = data[t]["exp8"]
        m = d["d3_direct_e3_knockout"]["by_hook_mode"]["mlp_writes"]
        w = d["d3_direct_e3_knockout"]["by_hook_mode"]["writes"]
        st = d["d4_fixed_steering"]

        def _c(sec, k):
            v = sec[k]
            return f"{v['log_odds_drop']:+.2f} / {v['xstest_unsafe_nonref_pct']:.0f}\\%"

        body.append(
            f"{MODEL_SHORT[t]} & "
            f"\\textbf{{{_c(m, 'ABL_E3_TORSION')}}} & {_c(m, 'ABL_E2_INPLANE')} & {_c(m, 'ABL_RAND_ORTH')} & "
            f"\\textbf{{{_c(w, 'ABL_E3_TORSION')}}} & {_c(w, 'ABL_E1_INCOMING')} & {_c(w, 'ABL_E2_INPLANE')} & "
            f"{st['benign_refusal_base_pct']:.1f} & {st['benign_refusal_steer_peak_pct']:.1f} (L{st['peak_layer']}) & "
            f"{st['benign_refusal_steer_outside_localvec_pct']:.1f} (L{st['out_of_basin_layer']}) \\\\"
        )

    cap = (
        r"\textbf{Appendix F: Direct axis knock-out refusal log-odds drop (D3) and strict out-of-basin steering control (D4).} "
        r"In D3, knocking out $e_3$ produces $+4.43$ to $+12.66$ log-odds drop ($p{<}0.0001$ vs. $e_1, e_2$ on 7/7 models), "
        r"whereas $e_2$ and random orthogonal axes have near-zero effect. In D4, injecting the local difference-in-means vector "
        r"strictly outside $\mathcal{B}_{\mathrm{beh}}\cup\mathcal{B}_{\mathrm{geo}}$ induces exactly $+0.0\%$ refusal on 7/7 models."
    )
    return _wrap_table("table*", cap, "tab:appF_d3_d4", "lccccccccc", hdr, body, tabcolsep="3.5pt")


# =========================================================================== #
# APPENDIX G.1: Exhaustive 13-Condition Exp6 Matrix Across All 7 Models
# =========================================================================== #
def build_tabG1_exp6_all13(data: Dict[str, dict]) -> str:
    tags = _tags(data, "exp6")
    conds = [
        "BASELINE",
        "STATIC_GLOBAL_k1", "STATIC_BASIN_k1", "GEORA_FLAT_k1", "GEORA_TORSION_k1", "LARA_PER_LAYER_k1",
        "STATIC_GLOBAL_k2", "STATIC_BASIN_k2", "GEORA_FLAT_k2", "GEORA_TORSION_k2", "LARA_PER_LAYER_k2",
        "TRAJ_SVD_r2", "TRAJ_SVD_r3",
    ]
    hdr = [
        r"\textbf{Strategy (Writes)} & " +
        " & ".join(f"\\textbf{{{MODEL_SHORT[t]}}}" for t in tags) +
        r" & \textbf{Mean SR/NR} & \textbf{KL / Force} \\",
    ]
    body = []
    for idx, c in enumerate(conds):
        if idx in (1, 6, 11):
            body.append(r"\midrule")
        cells, srs, nrs, kls, frcs = [], [], [], [], []
        for t in tags:
            g = data[t]["exp6"]["gold_asr"][c]
            u = data[t]["exp6"]["utility_and_force"][c]
            srs.append(g["asr"]); nrs.append(g["non_refusal_asr"])
            kls.append(u["benign_kl"]); frcs.append(u["force"]["total_force_norm"])
            cells.append(f"{g['asr']:.1f}/{g['non_refusal_asr']:.0f}")
        lbl = METHOD_LABEL[c]
        if c in ("LARA_PER_LAYER_k1", "LARA_PER_LAYER_k2"):
            lbl = f"\\textbf{{{lbl}}}"
        body.append(
            f"{lbl} & " + " & ".join(cells) +
            f" & {_mean(srs):.1f} / {_mean(nrs):.1f} & {_mean(kls):.2f} / {_mean(frcs):.1f} \\\\"
        )

    cap = (
        r"\textbf{Appendix G.1: Full 13-condition Exp6 benchmark across all 7 models (held-out $n{=}150$, 256 tokens, writes mode).} "
        r"Each model cell reports StrongREJECT Mean\% / Non-Refusal\%. All $91/91$ runs exhibit $0.0\%$ semantic collapse. "
        r"Torsion Transport outperforms Flat Geodesic on 7/7 models at $k{=}1$ ($p{\leq}0.026$), and Per-Layer Basin achieves "
        r"the highest mean ASR at $2.9\text{--}3.3\times$ lower Force Norm than Static Global."
    )
    return _wrap_table("table*", cap, "tab:appG1_exp6_all13", "l" + "c" * (len(tags) + 2), hdr, body, tabcolsep="3.2pt")


# =========================================================================== #
# APPENDIX G.2: Per-Model TruthfulQA (817) & XSTest (450) (Exp7)
# =========================================================================== #
def build_tabG2_exp7_tqa_xstest(data: Dict[str, dict]) -> str:
    tags = _tags(data, "exp7")
    conds = ["BASELINE", "STATIC_GLOBAL_k1", "GEORA_TORSION_k1", "LARA_PER_LAYER_k1", "STATIC_GLOBAL_k2", "LARA_PER_LAYER_k2"]
    hdr = [
        r"\textbf{Model} & \textbf{Metric} & " +
        " & ".join(f"\\textbf{{{METHOD_LABEL[c]}}}" for c in conds) + r" \\",
    ]
    body = []
    for idx, t in enumerate(tags):
        if idx > 0:
            body.append(r"\midrule")
        tqa = data[t]["exp7"]["truthfulqa"]["conditions"]
        xs = data[t]["exp7"]["xstest"]["conditions"]
        r_tqa = [f"{tqa[c]['mc1']:.1f} / {tqa[c]['mc2']:.1f}" for c in conds]
        r_xs = [f"{xs[c]['safe_over_refusal_pct']:.1f} / {xs[c]['unsafe_nonref_pct']:.1f}" for c in conds]
        body.append(f"\\multirow{{2}}{{*}}{{{MODEL_SHORT[t]}}} & TQA MC1/MC2 & " + " & ".join(r_tqa) + r" \\")
        body.append(r" & XS Safe/Unsafe & " + " & ".join(r_xs) + r" \\")

    body.append(r"\midrule")
    m_tqa = [
        f"{_mean([data[t]['exp7']['truthfulqa']['conditions'][c]['mc1'] for t in tags]):.1f} / "
        f"{_mean([data[t]['exp7']['truthfulqa']['conditions'][c]['mc2'] for t in tags]):.1f}"
        for c in conds
    ]
    m_xs = [
        f"{_mean([data[t]['exp7']['xstest']['conditions'][c]['safe_over_refusal_pct'] for t in tags]):.1f} / "
        f"{_mean([data[t]['exp7']['xstest']['conditions'][c]['unsafe_nonref_pct'] for t in tags]):.1f}"
        for c in conds
    ]
    body.append(r"\multirow{2}{*}{\textbf{Mean (7)}} & \textbf{TQA MC1/MC2} & " + " & ".join(m_tqa) + r" \\")
    body.append(r" & \textbf{XS Safe/Unsafe} & " + " & ".join(m_xs) + r" \\")

    cap = (
        r"\textbf{Appendix G.2: Per-model TruthfulQA (817 MC1/MC2) and XSTest (250 Safe ORR / 200 Unsafe Non-Refusal) utility.} "
        r"Static Global ($k{=}2$) severely degrades TruthfulQA on Qwen2.5-7B ($-13.6$ MC1 / $-15.4$ MC2) and Qwen2-7B ($-5.6$ / $-9.0$), "
        r"whereas Per-Layer Basin ($k{=}2$) preserves accuracy within $-1.8$ / $-1.0$ and $-1.5$ / $-2.4$ pts respectively."
    )
    return _wrap_table("table*", cap, "tab:appG2_exp7_utility", "llcccccc", hdr, body, tabcolsep="3.8pt")


# =========================================================================== #
# APPENDIX H.1: Apples-to-Apples `writes` vs `full_resid` (Exp9 P1)
# =========================================================================== #
def build_tabH1_exp9_p1(data: Dict[str, dict]) -> str:
    tags = _tags(data, "exp9")
    conds = ["STATIC_GLOBAL_k2", "STATIC_BASIN_k2", "GEORA_TORSION_k2", "LARA_PER_LAYER_k2"]
    hdr = [
        r"\textbf{Model} & \textbf{Strategy ($k{=}2$)} & \multicolumn{4}{c}{\textbf{Sublayer `writes`}} & "
        r"\multicolumn{4}{c}{\textbf{Residual Stream `full\_resid`}} \\",
        r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}",
        r" & & SR\% & KL & CE & TQA MC1 & SR\% & KL & CE & TQA MC1 \\",
    ]
    body = []
    for idx, t in enumerate(tags):
        if idx > 0:
            body.append(r"\midrule")
        d16 = data[t]["exp6"]
        d17 = data[t]["exp7"]
        p1 = data[t]["exp9"]["p1_full_resid"]
        for j, c in enumerate(conds):
            m_lbl = f"\\multirow{{4}}{{*}}{{{MODEL_SHORT[t]}}}" if j == 0 else ""
            w_sr = d16["gold_asr"][c]["asr"]
            w_kl = d16["utility_and_force"][c]["benign_kl"]
            w_ce = d16["utility_and_force"][c]["benign_ce_loss"]
            w_tq_dict = d17["truthfulqa"]["conditions"].get(c)
            w_tq = _fmt(w_tq_dict["mc1"], 1) if w_tq_dict else "--"

            f_sr = p1["gold_asr"][c]["asr"]
            f_kl = p1["utility_and_force"][c]["benign_kl"]
            f_ce = p1["utility_and_force"][c]["benign_ce_loss"]
            f_tq = _fmt(p1["truthfulqa"]["conditions"][c]["mc1"], 1)

            c_lbl = METHOD_LABEL[c].replace(r" ($k{=}2$)", "")
            if c == "LARA_PER_LAYER_k2":
                c_lbl = f"\\textbf{{{c_lbl}}}"
            body.append(
                f"{m_lbl} & {c_lbl} & {w_sr:.1f} & {w_kl:.2f} & {w_ce:.3f} & {w_tq} & "
                f"{f_sr:.1f} & {f_kl:.2f} & {f_ce:.3f} & {f_tq} \\\\"
            )

    cap = (
        r"\textbf{Appendix H.1: Apples-to-apples comparison of `writes` vs.\ `full\_resid` ablation (Exp9 P1, $n_{\mathrm{train}}{=}256$, no system prompt).} "
        r"Under high-torsion rotation, Per-Layer Basin ($k{=}2$) on `full\_resid` collapses on 4/4 models "
        r"(Benign KL inflates $3.6\text{--}17.4\times$), confirming the Torsion--Collateral Tradeoff."
    )
    return _wrap_table("table*", cap, "tab:appH1_p1", "llcccccccc", hdr, body, tabcolsep="4.2pt")


# =========================================================================== #
# APPENDIX H.2: Token-Position Invariance & Optimal p* Comparison (Exp9 P2)
# =========================================================================== #
def build_tabH2_exp9_p2(data: Dict[str, dict]) -> str:
    tags = _tags(data, "exp9")
    hdr = [
        r"\textbf{Model} & \textbf{$(p^*, l^*)$} & \textbf{$\mathcal{B}(p^*)$ (IoU)} & "
        r"\textbf{$\bar{\tau}$ across $p{\in}[-5,-1]$} & \textbf{Static Global $(p^*)$} & "
        r"\textbf{Per-Layer@$\mathcal{B}(p^*)$ $k{=}1$} & \textbf{Per-Layer@$\mathcal{B}(p^*)$ $k{=}2$} \\",
        r" & & & (min--max) & SR\% / KL / Force & SR\% / KL / Force & SR\% / KL / Force \\",
    ]
    body = []
    for t in tags:
        p2 = data[t]["exp9"]["p2_optimal_pos"]
        taus = [p2["frenet_by_position"][str(p)]["mean_tau"] for p in (-5, -4, -3, -2, -1)]
        g = p2["gold_asr"]
        u = p2["utility_and_force"]

        def _blk(k):
            return f"{g[k]['asr']:.1f} / {u[k]['benign_kl']:.3f} / {u[k]['force']['total_force_norm']:.1f}"

        body.append(
            f"{MODEL_SHORT[t]} & ({p2['best_pos']}, L{p2['best_layer']}) & "
            f"{_basin_str(p2['basin_pstar'])} ({p2['basin_iou_vs_m1']:.2f}) & "
            f"{min(taus):.3f}--{max(taus):.3f} & "
            f"{_blk('ARDITI_OPT_GLOBAL_k1')} & {_blk('LARA_AT_PSTAR_k1')} & {_blk('LARA_AT_PSTAR_k2')} \\\\"
        )

    cap = (
        r"\textbf{Appendix H.2: Post-instruction token-position invariance and optimal $p^*$ trade-off (Exp9 P2).} "
        r"Orthogonal drift geometry holds across all 20/20 model--position pairs ($\bar{\tau}{=}0.978\text{--}0.997$, $p{\leq}0.0015$). "
        r"While full-network Static Global at early positions ($p^*{\in}\{-5,-3\}$) achieves higher raw ASR by intercepting signal "
        r"at embeddings and early layers, Per-Layer Basin on $\mathcal{B}(p^*)$ cuts Benign KL by $2.1\text{--}7.0\times$."
    )
    return _wrap_table("table*", cap, "tab:appH2_p2", "lcccccc", hdr, body, tabcolsep="3.8pt")


# =========================================================================== #
# APPENDIX H.3: 512-Token Parity (P3) & System-Prompt Robustness (P4)
# =========================================================================== #
def build_tabH3_exp9_p3_p4(data: Dict[str, dict]) -> str:
    tags = _tags(data, "exp9")
    hdr = [
        r"\textbf{Model} & \multicolumn{4}{c}{\textbf{P3: 512-Token StrongREJECT Mean\%}} & "
        r"\multicolumn{4}{c}{\textbf{P4: With System Prompt}} \\",
        r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
        r" & StaticG & FlatGeo & TorTransp & Per-Layer $k{=}1/2$ & "
        r"$\bar{\tau}$ / $R^2_{0p}$ & Relay (cov) & Flat / Tor. & Per-Layer $k{=}1$ \\",
    ]
    body = []
    for t in tags:
        p3g = data[t]["exp9"]["p3_512tok"]["gold_asr"]
        p4 = data[t]["exp9"]["p4_sys_prompt"]
        fb = p4["frenet_basin"]
        sh = p4["split_half_exp5b"]
        p4g = p4["gold_asr"]

        body.append(
            f"{MODEL_SHORT[t]} & "
            f"{p3g['STATIC_GLOBAL_k1']['asr']:.1f} & {p3g['GEORA_FLAT_k1']['asr']:.1f} & "
            f"{p3g['GEORA_TORSION_k1']['asr']:.1f} & "
            f"\\textbf{{{p3g['LARA_PER_LAYER_k1']['asr']:.1f} / {p3g['LARA_PER_LAYER_k2']['asr']:.1f}}} & "
            f"{fb['mean_tau']:.3f} / {fb['zero_param_r2_orthogonal_drift']:.2f} & "
            f"{sh['repair_mlp_e3']:.3f} ({sh['repair_mlp_e3_cov_enrichment_x']:.1f}$\\times$) & "
            f"{p4g['GEORA_FLAT_k1']['asr']:.1f} / {p4g['GEORA_TORSION_k1']['asr']:.1f} & "
            f"{p4g['LARA_PER_LAYER_k1']['asr']:.1f} \\\\"
        )

    cap = (
        r"\textbf{Appendix H.3: Generation-length parity at 512 tokens (Exp9 P3) and system-prompt robustness (Exp9 P4).} "
        r"All Pareto rankings are preserved at 512 tokens (Per-Layer Basin $k{=}2$ reaches $54.6\%\text{--}74.0\%$ SR Mean). "
        r"Under safety/default system prompts, both orthogonal drift ($\bar{\tau}{>}0.991$) and non-circular MLP causal relay "
        r"($26.4\text{--}69.3\times$ cov-null) persist on 4/4 models, and Torsion Transport exceeds Flat Geodesic on 4/4 models."
    )
    return _wrap_table("table*", cap, "tab:appH3_p3_p4", "lcccccccc", hdr, body, tabcolsep="3.5pt")


# =========================================================================== #
# APPENDIX I: Baseline Unicode Apostrophe & Marker Audit (Exp10)
# =========================================================================== #
def build_tabI_unicode_audit(data: Dict[str, dict]) -> str:
    tags = _tags(data, "exp10")
    hdr = [
        r"\textbf{Model} & \textbf{Curly Apos.} & \textbf{Marker Ref.} & "
        r"\textbf{SR $< 0.10$} & \textbf{NR \& SR $<0.10$} & \textbf{Ref \& SR $\geq 0.50$} \\",
        r" & (count / 150) & (\%) & (\%) & (count) & (count) \\",
    ]
    body = []
    for t in tags:
        aud = data[t]["exp10"]["baseline_marker_audit"]
        body.append(
            f"{MODEL_SHORT[t]} & {aud['n_with_unicode_apostrophe']} & "
            f"{aud['marker_refusal_pct']:.1f}\\% & {aud['sr_below_0.10_pct']:.1f}\\% & "
            f"{aud['disagree_nonref_but_sr_low_count']} & {aud['disagree_ref_but_sr_high_count']} \\\\"
        )

    cap = (
        r"\textbf{Appendix I: Baseline Unicode apostrophe normalization and marker--judge agreement audit ($n{=}150$).} "
        r"Normalizing curly apostrophes (\texttt{U+2019}) resolves 19 cases on Llama-3.1-8B (yielding $0$ marker--judge "
        r"disagreements), 27 on Qwen2.5-3B, and 11 on Qwen2.5-7B."
    )
    return _wrap_table("table*", cap, "tab:appI_unicode_audit", "lccccc", hdr, body,
                       size_cmd=r"\scriptsize", tabcolsep="4pt", resize=True)


# =========================================================================== #
# Main driver
# =========================================================================== #
def main():
    ap = argparse.ArgumentParser(description="Export all LaTeX booktabs tables from JSON.")
    ap.add_argument("--root", default="results", help="Directory with model result folders")
    ap.add_argument("--outdir", default="tables", help="Output directory for .tex table files")
    a = ap.parse_args()

    data = load_all(a.root)
    os.makedirs(a.outdir, exist_ok=True)
    print(f">> Loaded {len(data)} models from {a.root}: {[data[k]['short'] for k in data]}")

    tables = [
        ("tab1_main_three_tier.tex",           build_tab1_main_three_tier),
        ("tab2_main_direct_e3_gold.tex",       build_tab2_main_direct_e3_gold),
        ("tab3_main_pareto_7axis.tex",         build_tab3_main_pareto_7axis),
        ("tabC_basin_and_frenet.tex",          build_tabC_basin_and_frenet),
        ("tabD_self_repair_exp2_exp4.tex",       build_tabD_self_repair_exp2_exp4),
        ("tabE_controls_ref_vs_native.tex",    build_tabE_controls_ref_vs_native),
        ("tabF_d3_logodds_and_steering.tex",   build_tabF_d3_logodds_and_steering),
        ("tabG1_exp6_all13_conditions.tex",     build_tabG1_exp6_all13),
        ("tabG2_exp7_tqa_xstest_per_model.tex", build_tabG2_exp7_tqa_xstest),
        ("tabH1_exp9_p1_writes_vs_resid.tex",   build_tabH1_exp9_p1),
        ("tabH2_exp9_p2_optimal_pos.tex",       build_tabH2_exp9_p2),
        ("tabH3_exp9_p3_p4_512_sysprompt.tex",  build_tabH3_exp9_p3_p4),
        ("tabI_exp10_unicode_audit.tex",         build_tabI_unicode_audit),
    ]

    combined = [
        "% Auto-generated LaTeX tables (booktabs).",
        "% Required packages: \\usepackage{booktabs,multirow,graphicx,amsmath}",
        "",
    ]
    for fname, builder in tables:
        tex = builder(data)
        out_p = os.path.join(a.outdir, fname)
        with open(out_p, "w", encoding="utf-8") as f:
            f.write(tex)
        combined.append(f"% ==================== {fname} ====================")
        combined.append(tex)
        print(f"  [SAVED] {out_p}")

    all_p = os.path.join(a.outdir, "ALL_TABLES.tex")
    with open(all_p, "w", encoding="utf-8") as f:
        f.write("\n".join(combined))
    print(f"  [SAVED] {all_p}")
    print(f">> Exported {len(tables)} tables to {a.outdir}/")


if __name__ == "__main__":
    main()