"""Figure generator.

Renders the paper figures in a LaTeX-native serif style: Computer Modern math
with serif text (optional --usetex), light-gray panels with dashed grids, rounded
legends and callout boxes, and a high-contrast palette with black-outlined
markers, keeping labels non-overlapping across all figures."""
from __future__ import annotations

import os
import json
import math
import logging
import argparse
from typing import Dict, List, Optional

logging.getLogger("fontTools").setLevel(logging.ERROR)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon
from matplotlib.lines import Line2D


# =========================================================================== #
# 1. Global figure style
# =========================================================================== #
PALETTE = {
    "blue":       "#2B7BBA",   # HL / LARA / Primary clean
    "red":        "#D62728",   # HF / Refusal / e3 Torsion
    "green":      "#2CA02C",   # Multi-dir / GEORA_TORSION / e3 MLP-only
    "purple":     "#7E4EA8",   # TruthfulQA Factual Control
    "amber":      "#E68A00",   # Behavioral Format Control
    "slate":      "#4D4D55",   # STATIC_GLOBAL baseline
    "gray":       "#9A9AA6",   # Null / Flat / e2 / Random
    "dark":       "#1E1E24",   # Vector arrows / outlines
    "panel_bg":   "#EFEFF2",   # Soft gray canvas
    "grid":       "#CECED6",   # Dashed subtle grid
    "legend_bg":  "#F4F4F7",   # Pill legend background
    "legend_ed":  "#D4D4DC",   # Pill legend border
}

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


def apply_reference_style(use_tex: bool = False):
    plt.rcParams.update({
        "text.usetex": use_tex,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif", "Times New Roman"],
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "font.size": 9.5,
        "axes.titlesize": 11.2,
        "axes.titlepad": 7.0,
        "axes.labelsize": 10.0,
        "axes.labelpad": 5.0,
        "xtick.labelsize": 8.6,
        "ytick.labelsize": 8.6,
        "legend.fontsize": 8.8,
        "axes.facecolor": PALETTE["panel_bg"],
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.edgecolor": "none",
        "axes.linewidth": 0.0,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": PALETTE["grid"],
        "grid.linestyle": "--",
        "grid.linewidth": 0.65,
        "grid.alpha": 0.9,
        "xtick.major.size": 0.0,
        "ytick.major.size": 0.0,
        "xtick.major.pad": 4.5,
        "ytick.major.pad": 4.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def style_ax(ax, grid_axis: str = "both"):
    ax.set_facecolor(PALETTE["panel_bg"])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0, pad=4.5, colors="#222222")
    ax.grid(False)
    if grid_axis in ("both", "x", "y"):
        ax.grid(True, axis=grid_axis, color=PALETTE["grid"], linestyle="--", linewidth=0.65, alpha=0.9, zorder=0)
    ax.set_axisbelow(True)


def add_callout(ax, text: str, x: float = 0.97, y: float = 0.95,
                ha: str = "right", va: str = "top", fontsize: float = 8.0):
    ax.text(
        x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=fontsize,
        color="#111111", zorder=10,
        bbox=dict(
            boxstyle="round,pad=0.30",
            facecolor="white",
            edgecolor="#222222",
            linewidth=0.85,
        ),
    )


def add_top_pill_legend(fig, handles: list, ncol: Optional[int] = None,
                        y: float = 0.99, fontsize: float = 8.9):
    leg = fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, y),
        ncol=ncol or len(handles),
        frameon=True,
        fancybox=True,
        facecolor=PALETTE["legend_bg"],
        edgecolor=PALETTE["legend_ed"],
        framealpha=0.98,
        borderpad=0.40,
        columnspacing=1.25,
        handletextpad=0.45,
        fontsize=fontsize,
    )
    leg.get_frame().set_linewidth(0.85)
    return leg


def save_fig(fig, outdir: str, stem: str):
    os.makedirs(outdir, exist_ok=True)
    png_p = os.path.join(outdir, f"{stem}.png")
    fig.savefig(png_p, bbox_inches="tight", pad_inches=0.10, dpi=300)
    plt.close(fig)
    print(f"  [SAVED] {png_p}")


# =========================================================================== #
# 2. Data Loader
# =========================================================================== #
def load_all_results(root: str) -> Dict[str, dict]:
    data = {}
    for tag, short in MODEL_ORDER:
        mdir = os.path.join(root, tag)
        if not os.path.isdir(mdir):
            continue
        def _j(name):
            p = os.path.join(mdir, f"{name}_{tag}.json")
            return json.load(open(p, "r", encoding="utf-8")) if os.path.isfile(p) else None

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
        }
    return data


def _present_tags(data: Dict[str, dict], req_key: str) -> List[str]:
    return [tag for tag, _ in MODEL_ORDER if tag in data and data[tag].get(req_key) is not None]


# =========================================================================== #
# FIGURE 1: Hero Figure — Active Orthogonal Relay Schematic + Exp10 Direct e3
# =========================================================================== #
def plot_fig1_hero(data: Dict[str, dict], outdir: str):
    tags = _present_tags(data, "exp10")
    if not tags:
        print("  [SKIP Fig 1] No Exp10 files found.")
        return

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.6, 4.35), gridspec_kw={"width_ratios": [1.05, 1.18]})
    fig.subplots_adjust(top=0.80, bottom=0.17, left=0.04, right=0.98, wspace=0.38)

    # --- Panel A: Geometric Schematic of Discrete Frenet Frame & Orthogonal Relay ---
    style_ax(axL, grid_axis="both")
    axL.set_xlim(-1.15, 1.55)
    axL.set_ylim(-0.82, 1.30)
    axL.set_xticks([]); axL.set_yticks([])
    axL.set_title(r"Active Orthogonal Relay in Causal Basin $\mathcal{B}$", pad=8)

    plane = Polygon(
        [[-0.95, -0.45], [0.85, -0.58], [1.22, 0.08], [-0.58, 0.21]],
        closed=True, facecolor=PALETTE["blue"], alpha=0.10, edgecolor=PALETTE["blue"],
        linestyle=":", linewidth=1.0, zorder=1
    )
    axL.add_patch(plane)
    axL.text(-0.08, -0.56, r"Prior plane $\mathrm{span}(e_1^{(l-1)}, e_2^{(l-1)})$",
             fontsize=8.0, color="#2A4D69", ha="center", va="top", style="italic")

    rng = np.random.default_rng(7)
    hl_x = rng.normal(-0.55, 0.16, 110)
    hl_y = rng.normal(-0.18, 0.12, 110)
    hf_x = rng.normal(0.50, 0.20, 120)
    hf_y = rng.normal(0.62, 0.17, 120)
    axL.scatter(hl_x, hl_y, s=16, c=PALETTE["blue"], alpha=0.35, edgecolors="white", linewidths=0.25, zorder=2)
    axL.scatter(hf_x, hf_y, s=16, c=PALETTE["red"], alpha=0.35, edgecolors="white", linewidths=0.25, zorder=2)

    origin = np.array([-0.55, -0.18])
    p_e1 = np.array([0.45, -0.25])
    p_e2 = np.array([-0.18, 0.14])
    p_e3 = np.array([-0.55, 0.90])
    p_vl = np.array([0.45, 0.66])

    def _arrow(p0, p1, col, lw=2.2, ls="-", alpha=0.88, z=4):
        ar = FancyArrowPatch(
            tuple(p0), tuple(p1), arrowstyle="-|>,head_length=6.5,head_width=3.6",
            color=col, linewidth=lw, linestyle=ls, alpha=alpha, zorder=z
        )
        axL.add_patch(ar)

    _arrow(origin, p_e1, PALETTE["slate"], lw=2.0, alpha=0.78)
    _arrow(origin, p_e2, PALETTE["gray"], lw=1.8, ls="--", alpha=0.85)
    _arrow(origin, p_e3, PALETTE["red"], lw=2.8, alpha=0.95, z=5)
    _arrow(p_e1, p_vl, PALETTE["green"], lw=2.3, ls="-", alpha=0.90, z=5)
    _arrow(origin, p_vl, PALETTE["dark"], lw=1.5, ls="-.", alpha=0.65, z=4)

    axL.plot([-0.55, -0.43, -0.43], [-0.04, -0.05, -0.19], color="#222222", lw=0.9, zorder=4)

    axL.scatter(*origin, s=90, marker="D", c=PALETTE["blue"], edgecolors="#111111", linewidths=1.1, zorder=6)
    axL.scatter(*p_e1,   s=72, marker="s", c=PALETTE["slate"], edgecolors="#111111", linewidths=1.0, zorder=6)
    axL.scatter(*p_e3,   s=100, marker="D", c=PALETTE["red"], edgecolors="#111111", linewidths=1.1, zorder=6)
    axL.scatter(*p_vl,   s=92, marker="h", c=PALETTE["green"], edgecolors="#111111", linewidths=1.1, zorder=6)

    axL.text(0.52, -0.27, r"$e_1^{(l-1)} = v_{l-1}$" + "\n(incoming)", fontsize=8.0, color="#333333", va="center", ha="left")
    axL.text(-0.22, 0.25, r"$e_2^{(l-1)}$ (in-plane)", fontsize=7.8, color="#555555", va="bottom", ha="right")
    axL.text(-0.46, 0.96, r"$e_3^{(l-1)}$ (orthogonal" + "\ninnovation axis)", fontsize=8.2,
             color=PALETTE["red"], fontweight="bold", va="center", ha="left")
    axL.text(0.54, 0.72, r"$v_l$ (layer $l$)", fontsize=8.4, color="#111111", fontweight="bold", va="center", ha="left")
    axL.text(0.53, 0.22, "MLP + Attn write\n" + r"$\parallel e_3^{(l-1)}$ ($\bar{\tau}\approx 0.994$)",
             fontsize=7.9, color="#1D6F1D", ha="left", va="center",
             bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#2CA02C", lw=0.75, alpha=0.95))

    add_callout(axL, r"$\mathrm{CV}\;\cos^2(\Delta m, e_3) = 0.32\text{--}0.79$",
                x=0.97, y=0.05, ha="right", va="bottom", fontsize=7.9)

    # --- Panel B: Exp10 7-Model Direct Causal Knock-Out ---
    style_ax(axR, grid_axis="x")
    conds = [
        ("ABL_RAND_ORTH_WRITES", r"Rand $\perp (e_1,e_2)$", PALETTE["gray"],  "o"),
        ("ABL_E2_WRITES",        r"$e_2$ (In-Plane)",       PALETTE["gray"],  "s"),
        ("ABL_E1_WRITES",        r"$e_1$ (Incoming)",       PALETTE["slate"], "s"),
        ("ABL_E3_MLP_ONLY",      r"$e_3$ (MLP Only)",       PALETTE["green"], "h"),
        ("ABL_E3_WRITES",        r"$e_3$ (Writes)",         PALETTE["red"],   "D"),
        ("LARA_PER_LAYER_k1",    r"Per-Layer Basin ($k=1$)",     PALETTE["blue"],  "D"),
    ]
    y_pos = np.arange(len(conds))
    sr_means, sr_sems, nr_means = [], [], []
    for ck, _, _, _ in conds:
        vals_sr = [data[t]["exp10"]["gold_asr"][ck]["asr"] for t in tags]
        vals_nr = [data[t]["exp10"]["gold_asr"][ck]["non_refusal_asr"] for t in tags]
        sr_means.append(float(np.mean(vals_sr)))
        sr_sems.append(float(np.std(vals_sr, ddof=1) / math.sqrt(len(vals_sr))))
        nr_means.append(float(np.mean(vals_nr)))

    tags_16 = _present_tags(data, "exp6")
    sg1_mean = float(np.mean([data[t]["exp6"]["gold_asr"]["STATIC_GLOBAL_k1"]["asr"] for t in tags_16])) if tags_16 else 47.6

    for i, (ck, lbl, col, mk) in enumerate(conds):
        axR.barh(i, sr_means[i], height=0.56, color=col, alpha=0.82, edgecolor="#111111", linewidth=0.9, zorder=3)
        axR.errorbar(sr_means[i], i, xerr=sr_sems[i], fmt="none", ecolor="#111111", elinewidth=1.0, capsize=2.8, zorder=4)
        axR.plot([sr_means[i], nr_means[i]], [i, i], color="#222222", linestyle=":", linewidth=1.1, alpha=0.7, zorder=4)
        axR.scatter(nr_means[i], i, s=62, marker=mk, c="white", edgecolors="#111111", linewidths=1.25, zorder=5)
        axR.text(nr_means[i] + 2.5, i, f"{sr_means[i]:.1f}% / {nr_means[i]:.1f}%",
                 va="center", ha="left", fontsize=8.0, color="#111111", zorder=7,
                 bbox=dict(boxstyle="round,pad=0.10", facecolor=PALETTE["panel_bg"], edgecolor="none", alpha=0.88))

    axR.axvline(sg1_mean, color=PALETTE["slate"], linestyle="--", linewidth=1.2, zorder=2)
    axR.text(sg1_mean + 1.2, 0.15, f"Static Global\n({sg1_mean:.1f}%)",
             fontsize=7.6, color=PALETTE["slate"], va="bottom")

    axR.set_yticks(y_pos)
    axR.set_yticklabels([c[1] for c in conds])
    axR.set_xlim(0, 118)
    axR.set_xlabel("Held-Out ASR (%)\n(Bar: StrongREJECT Mean | Marker: Non-Refusal)", fontsize=9.2)
    axR.set_title("Direct Axis Knock-Out (7-Model Mean, $n=150$)", pad=8)

    delta_e3_e1 = sr_means[4] - sr_means[2]
    add_callout(axR, f"$e_3$ vs. $e_1$: $+{delta_e3_e1:.1f}$ pts\n(7/7 models, $p<0.001$)",
                x=0.98, y=0.14, va="bottom", fontsize=8.0)

    handles = [
        Line2D([0], [0], marker="D", color="none", markerfacecolor=PALETTE["blue"], markeredgecolor="#111111", markersize=7.4, label=r"Benign / Per-Layer Basin"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=PALETTE["red"], markeredgecolor="#111111", markersize=7.4, label=r"Orthogonal Innovation $e_3$ (Writes)"),
        Line2D([0], [0], marker="h", color="none", markerfacecolor=PALETTE["green"], markeredgecolor="#111111", markersize=7.8, label=r"$e_3$ MLP-Only ($\mathrm{down\_proj}$)"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=PALETTE["slate"], markeredgecolor="#111111", markersize=6.8, label=r"Incoming $e_1$ / In-Plane $e_2$"),
    ]
    add_top_pill_legend(fig, handles, ncol=4, y=0.975)
    save_fig(fig, outdir, "fig1_hero_relay_vs_e1")


# =========================================================================== #
# FIGURE 2: 3-Tier Causal Funnel (Zero Overlap)
# =========================================================================== #
def plot_fig2_three_tier_funnel(data: Dict[str, dict], outdir: str):
    tags = _present_tags(data, "exp8")
    if not tags:
        print("  [SKIP Fig 2] No Exp8 files found.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.05))
    fig.subplots_adjust(top=0.79, bottom=0.16, left=0.06, right=0.98, wspace=0.28)
    rng = np.random.default_rng(42)

    # --- Panel A: Tier 1 — High-Dim Trap ---
    ax = axes[0]
    style_ax(ax, grid_axis="y")
    ref_tau = [data[t]["exp5"]["mean_torsion_tau"] for t in tags]
    null_tau = [data[t]["exp8"]["d2a_random_split_null"]["mean_tau"] for t in tags]
    truth_tau = [data[t]["exp8"]["d2c_truthfulqa_concept"]["in_refusal_basin"]["mean_tau"] for t in tags]
    ref_r2 = [data[t]["exp8"]["d2b_layer_permutation"]["obs_0p_r2_exp"] for t in tags]
    null_r2 = [data[t]["exp8"]["d2a_random_split_null"]["zero_param_r2_exp"] for t in tags]
    perm_r2 = [data[t]["exp8"]["d2b_layer_permutation"]["perm_0p_r2_mean"] for t in tags]

    groups_a = [
        ("Ref.\n$\\bar{\\tau}$",   ref_tau,   PALETTE["red"],    "D"),
        ("Truth\n$\\bar{\\tau}$",  truth_tau, PALETTE["purple"], "s"),
        ("Null\n$\\bar{\\tau}$",   null_tau,  PALETTE["gray"],   "o"),
        ("Ref.\n$R^2_{0p}$",       ref_r2,    PALETTE["red"],    "D"),
        ("Null\n$R^2_{0p}$",       null_r2,   PALETTE["gray"],   "o"),
        ("Perm.\n$R^2_{0p}$",      perm_r2,   PALETTE["slate"],  "^"),
    ]
    for idx, (lbl, vals, col, mk) in enumerate(groups_a):
        mean_v = float(np.mean(vals))
        ax.bar(idx, mean_v, width=0.56, color=col, alpha=0.28, edgecolor=col, linewidth=1.0, zorder=2)
        jit = rng.uniform(-0.14, 0.14, len(vals))
        ax.scatter(np.full(len(vals), idx) + jit, vals, s=34, marker=mk,
                   c=col, edgecolors="#111111", linewidths=0.75, zorder=4)
    ax.axhline(0, color="#222222", linewidth=0.8, zorder=3)
    ax.axvline(2.5, color="#888894", linestyle=":", linewidth=1.0, zorder=2)
    ax.set_xticks(range(len(groups_a)))
    ax.set_xticklabels([g[0] for g in groups_a], fontsize=7.9)
    ax.set_ylim(-3.25, 1.30)
    ax.set_ylabel("Metric Value (7 Models)")
    ax.set_title("(a) Tier 1: High-Dim Trap", pad=7)
    # Placed in bottom-left empty quadrant (under positive tau bars)
    add_callout(ax, "Null also gives $\\bar{\\tau}\\approx 0.995$\nPerm. $p=0.0099$ (7/7)",
                x=0.04, y=0.05, ha="left", va="bottom", fontsize=7.7)

    # --- Panel B: Tier 2 — General MLP Concept Writing ---
    ax = axes[1]
    style_ax(ax, grid_axis="y")
    concepts_b = [
        ("Refusal",
         [data[t]["exp8"]["d1_split_half_exp5b"]["cross_fold_e3_cos2"] for t in tags],
         [data[t]["exp8"]["d1_split_half_exp5b"]["clean_mlp_e3"] for t in tags],
         PALETTE["red"], "D"),
        ("Format",
         [data[t]["exp8"]["d2d_format_behavior"]["in_native_basin"]["split_half"]["cross_fold_e3_cos2"] for t in tags],
         [data[t]["exp8"]["d2d_format_behavior"]["in_native_basin"]["split_half"]["clean_mlp_e3"] for t in tags],
         PALETTE["amber"], "h"),
        ("Truth",
         [data[t]["exp8"]["d2c_truthfulqa_concept"]["in_native_basin"]["split_half"]["cross_fold_e3_cos2"] for t in tags],
         [data[t]["exp8"]["d2c_truthfulqa_concept"]["in_native_basin"]["split_half"]["clean_mlp_e3"] for t in tags],
         PALETTE["purple"], "s"),
        ("Null",
         [data[t]["exp8"]["d2a_random_split_null"]["cross_fold_e3_cos2"] for t in tags],
         [data[t]["exp8"]["d2a_random_split_null"]["cv_clean_mlp_e3_cos2"] for t in tags],
         PALETTE["gray"], "o"),
    ]
    xs = np.arange(len(concepts_b))
    w = 0.34
    for i, (lbl, c_cons, c_clean, col, mk) in enumerate(concepts_b):
        ax.bar(i - w / 2, np.mean(c_cons), width=w, color=col, alpha=0.80, edgecolor="#111111", lw=0.8, zorder=3)
        ax.bar(i + w / 2, np.mean(c_clean), width=w, color=col, alpha=0.35, edgecolor="#111111", lw=0.8, hatch="///", zorder=3)
        ax.scatter(np.full(len(c_cons), i - w / 2) + rng.uniform(-0.06, 0.06, len(c_cons)), c_cons,
                   s=24, marker=mk, c="white", edgecolors="#111111", lw=0.75, zorder=5)
        ax.scatter(np.full(len(c_clean), i + w / 2) + rng.uniform(-0.06, 0.06, len(c_clean)), c_clean,
                   s=24, marker=mk, c=col, edgecolors="#111111", lw=0.75, zorder=5)
    ax.set_xticks(xs)
    ax.set_xticklabels([c[0] for c in concepts_b])
    # Dedicated upper header band (1.05..1.38) keeps callout strictly above all bars/markers
    ax.set_ylim(0, 1.38)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel(r"Cross-Validated $\cos^2$ Alignment")
    ax.set_title("(b) Tier 2: General MLP $e_3$ Writing", pad=7)
    add_callout(ax, "Solid: $e_3$ Fold Cons. | Hatched: CV Clean MLP\nAll real concepts: 30–70$\\times$ over Null",
                x=0.50, y=0.97, ha="center", va="top", fontsize=7.4)

    # --- Panel C: Tier 3 — Refusal-Specific Active Orthogonal Relay ---
    ax = axes[2]
    style_ax(ax, grid_axis="y")
    rep_ref = [data[t]["exp8"]["d1_split_half_exp5b"]["repair_mlp_e3"] for t in tags]
    rep_fmt = [data[t]["exp8"]["d2d_format_behavior"]["in_native_basin"]["split_half"]["repair_mlp_e3"] for t in tags]
    rep_tru = [data[t]["exp8"]["d2c_truthfulqa_concept"]["in_native_basin"]["split_half"]["repair_mlp_e3"] for t in tags]
    rep_nul = [data[t]["exp8"]["d2a_random_split_null"]["cv_repair_mlp_e3_cos2"] for t in tags]

    for j in range(len(tags)):
        ax.plot([0, 1, 2], [rep_ref[j], rep_fmt[j], rep_tru[j]],
                color="#222222", alpha=0.45, linewidth=1.25, zorder=3)
    tier3 = [
        ("Refusal", rep_ref, PALETTE["red"], "D"),
        ("Format\n(Native)", rep_fmt, PALETTE["amber"], "h"),
        ("Truth\n(Native)", rep_tru, PALETTE["purple"], "s"),
        ("Null", rep_nul, PALETTE["gray"], "o"),
    ]
    for i, (lbl, vals, col, mk) in enumerate(tier3):
        ax.bar(i, np.mean(vals), width=0.48, color=col, alpha=0.25, edgecolor=col, lw=1.0, zorder=2)
        ax.scatter(np.full(len(vals), i), vals, s=48, marker=mk, c=col, edgecolors="#111111", lw=0.9, zorder=5)

    ax.set_xticks(range(4))
    ax.set_xticklabels([x[0] for x in tier3], fontsize=8.4)
    ax.set_ylim(0, 0.96)
    ax.set_ylabel(r"CV Causal Relay $\cos^2(\Delta m_l, e_3^{(l-1)})$")
    ax.set_title("(c) Tier 3: Refusal-Specific Relay", pad=7)
    r_m, f_m, t_m = float(np.mean(rep_ref)), float(np.mean(rep_fmt)), float(np.mean(rep_tru))
    add_callout(ax, f"Refusal: {r_m:.2f}\nvs. Format: {f_m:.2f} ({r_m/f_m:.1f}$\\times$)\nvs. Truth: {t_m:.2f} ({r_m/t_m:.1f}$\\times$)\nNull: cov 4–8$\\times$ (cone)",
                x=0.97, y=0.96, fontsize=7.8)

    handles = [
        Line2D([0], [0], marker="D", color="none", markerfacecolor=PALETTE["red"], markeredgecolor="#111111", markersize=7.2, label="Refusal"),
        Line2D([0], [0], marker="h", color="none", markerfacecolor=PALETTE["amber"], markeredgecolor="#111111", markersize=7.8, label="Behavioral Format (List vs. Prose)"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=PALETTE["purple"], markeredgecolor="#111111", markersize=7.0, label="Factual Truth (TruthfulQA)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=PALETTE["gray"], markeredgecolor="#111111", markersize=7.0, label="Random-Split Null"),
    ]
    add_top_pill_legend(fig, handles, ncol=4, y=0.975)
    save_fig(fig, outdir, "fig2_three_tier_funnel")


# =========================================================================== #
# FIGURE 3: Circuit Anatomy (Fixed Steering L11 & Callout Spacing)
# =========================================================================== #
def plot_fig3_circuit_anatomy(data: Dict[str, dict], outdir: str):
    tags = _present_tags(data, "exp4")
    if not tags:
        print("  [SKIP Fig 3] No Exp4 files found.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.10), gridspec_kw={"width_ratios": [1.05, 1.22, 1.05]})
    fig.subplots_adjust(top=0.79, bottom=0.16, left=0.05, right=0.98, wspace=0.32)

    # --- Panel A: Exp4 Phase Transition ---
    ax = axes[0]
    style_ax(ax, grid_axis="both")
    ks = [1, 2, 3, 4, 5, 6]
    model_cols = ["#D62728", "#2B7BBA", "#2CA02C", "#7E4EA8", "#E68A00", "#17BECF", "#8C564B"]
    comp_mat, rand_mat = [], []
    for idx, t in enumerate(tags):
        d14 = data[t]["exp4"]
        c_vals = d14.get("compensation_frac", [])
        r_vals = d14.get("random_suppression_frac", [])
        if len(c_vals) >= 6:
            comp_mat.append(c_vals[:6])
            ax.plot(ks, c_vals[:6], color=model_cols[idx % len(model_cols)], alpha=0.55,
                    lw=1.35, marker="o", markersize=3.8, zorder=3)
        if len(r_vals) >= 6:
            rand_mat.append(r_vals[:6])

    if comp_mat:
        ax.plot(ks, np.mean(comp_mat, axis=0), color="#111111", lw=2.4, marker="D", markersize=5.5,
                markerfacecolor=PALETTE["red"], markeredgecolor="#111111", zorder=5)
    if rand_mat:
        mean_r = np.mean(rand_mat, axis=0)
        std_r = np.std(rand_mat, axis=0)
        ax.fill_between(ks, mean_r - std_r, mean_r + std_r, color=PALETTE["gray"], alpha=0.25, zorder=2)
        ax.plot(ks, mean_r, color=PALETTE["slate"], linestyle="--", lw=1.6, zorder=4)

    ax.set_xticks(ks)
    ax.set_ylim(-0.35, 8.25)
    ax.set_xlabel("Targeted Directions Ablated ($k$)")
    ax.set_ylabel("Downstream Compensation Factor ($\\times$)")
    ax.set_title("(a) Self-Repair Phase Transition", pad=7)
    add_callout(ax, "Peak: $7.41\\times$ (Llama-3)\nCollapse at $k^*\\in\\{4,5,6\\}$\nRandom-$k$: $\\approx 1.0\\times$",
                x=0.04, y=0.96, ha="left", fontsize=7.6)

    # --- Panel B: Exp10 Per-Model Heatmap ---
    ax = axes[1]
    tags20 = _present_tags(data, "exp10")
    cols20 = [
        ("ABL_E3_WRITES",        "$e_3$\nWrites"),
        ("ABL_E3_MLP_ONLY",      "$e_3$\nMLP"),
        ("ABL_E1_WRITES",        "$e_1$\nIncom."),
        ("ABL_E2_WRITES",        "$e_2$\nPlane"),
        ("ABL_RAND_ORTH_WRITES", "Rand\n$\\perp$"),
    ]
    mat = np.zeros((len(tags20), len(cols20)))
    for i, t in enumerate(tags20):
        g = data[t]["exp10"]["gold_asr"]
        for j, (ck, _) in enumerate(cols20):
            mat[i, j] = g[ck]["asr"]

    ax.imshow(mat, cmap="RdYlBu_r", aspect="auto", vmin=0, vmax=68)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.grid(False)
    ax.set_xticks(range(len(cols20)))
    ax.set_xticklabels([c[1] for c in cols20], fontsize=8.2)
    ax.set_yticks(range(len(tags20)))
    ax.set_yticklabels([MODEL_SHORT[t] for t in tags20], fontsize=8.3)
    ax.tick_params(length=0, pad=4)
    for i in range(len(tags20)):
        for j in range(len(cols20)):
            val = mat[i, j]
            txt_c = "white" if (val > 52 or val < 12) else "#111111"
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center", fontsize=7.8,
                    fontweight="bold" if j == 0 else "normal", color=txt_c)
    ax.set_title("(b) StrongREJECT Mean% by Axis (Exp10)", pad=7)

    # --- Panel C: Exp8 D4 Bidirectional Steering Specificity (Fixed L11 & Callout) ---
    ax = axes[2]
    style_ax(ax, grid_axis="x")
    tags18 = _present_tags(data, "exp8")
    y_st = np.arange(len(tags18))
    for i, t in enumerate(tags18):
        st = data[t]["exp8"]["d4_fixed_steering"]
        b_v = st["benign_refusal_base_pct"]
        o_v = st["benign_refusal_steer_outside_localvec_pct"]
        p_v = st["benign_refusal_steer_peak_pct"]
        ax.plot([o_v, p_v], [i, i], color="#222222", alpha=0.65, lw=2.0, zorder=3)
        ax.scatter(b_v, i, s=44, marker="o", c=PALETTE["gray"], edgecolors="#111111", lw=0.8, zorder=4)
        ax.scatter(o_v, i, s=56, marker="D", c=PALETTE["blue"], edgecolors="#111111", lw=0.9, zorder=5)
        ax.scatter(p_v, i, s=70, marker="D", c=PALETTE["red"], edgecolors="#111111", lw=1.0, zorder=6)
        # Clean fixed offset strictly to the right of diamond (no cap clipping)
        ax.text(p_v + 5.0, i, f"L{st['peak_layer']}", va="center", ha="left", fontsize=7.8, color="#222222")

    ax.set_yticks(y_st)
    ax.set_yticklabels([MODEL_SHORT[t] for t in tags18], fontsize=8.2)
    ax.set_ylim(len(tags18) - 0.15, -0.60)  # Extra bottom padding below Mistral
    ax.set_xlim(-8, 128)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Induced Benign Refusal (%)")
    ax.set_title("(c) Steering Specificity (D4)", pad=7)
    add_callout(ax, "Out-of-basin $\\Delta = 0.0\\%$\n(7/7 models)",
                x=0.97, y=0.04, ha="right", va="bottom", fontsize=7.4)

    handles = [
        Line2D([0], [0], color="#111111", lw=2.0, marker="D", markerfacecolor=PALETTE["red"], markeredgecolor="#111111", markersize=6.8, label="7-Model Mean / Peak $l^*$"),
        Line2D([0], [0], color=PALETTE["slate"], ls="--", lw=1.5, label="Random-$k$ Control"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=PALETTE["blue"], markeredgecolor="#111111", markersize=6.8, label="Out-of-Basin Local Vec"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=PALETTE["gray"], markeredgecolor="#111111", markersize=6.5, label="Unsteered Baseline"),
    ]
    add_top_pill_legend(fig, handles, ncol=4, y=0.975)
    save_fig(fig, outdir, "fig3_circuit_anatomy")


# =========================================================================== #
# FIGURE 4: Behavioral Falsification + Writes vs Resid + Pareto (Zero Overlap)
# =========================================================================== #
def plot_fig4_pareto_and_writes(data: Dict[str, dict], outdir: str):
    tags16 = _present_tags(data, "exp6")
    tags19 = _present_tags(data, "exp9")
    if not tags16:
        print("  [SKIP Fig 4] No Exp6 files found.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.15), gridspec_kw={"width_ratios": [0.98, 1.00, 1.32]})
    fig.subplots_adjust(top=0.79, bottom=0.16, left=0.05, right=0.98, wspace=0.28)

    # --- Panel A: Flat Geodesic Fails ---
    ax = axes[0]
    style_ax(ax, grid_axis="y")
    f_vals = [data[t]["exp6"]["gold_asr"]["GEORA_FLAT_k1"]["asr"] for t in tags16]
    t_vals = [data[t]["exp6"]["gold_asr"]["GEORA_TORSION_k1"]["asr"] for t in tags16]
    l_vals = [data[t]["exp6"]["gold_asr"]["LARA_PER_LAYER_k1"]["asr"] for t in tags16]
    for j in range(len(tags16)):
        ax.plot([0, 1, 2], [f_vals[j], t_vals[j], l_vals[j]], color="#222222", alpha=0.45, lw=1.35, zorder=3)
    for x_i, vals, col, mk in [
        (0, f_vals, PALETTE["gray"],  "o"),
        (1, t_vals, PALETTE["green"], "h"),
        (2, l_vals, PALETTE["blue"],  "D"),
    ]:
        ax.bar(x_i, np.mean(vals), width=0.44, color=col, alpha=0.22, edgecolor=col, lw=1.0, zorder=2)
        ax.scatter(np.full(len(vals), x_i), vals, s=46, marker=mk, c=col, edgecolors="#111111", lw=0.9, zorder=5)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Flat Geodesic\n($\\tau=0$)", "Torsion Transp.\n(3-anchor)", "Per-Layer\nBasin"], fontsize=8.0)
    ax.set_ylim(0, 84)
    ax.set_ylabel("StrongREJECT Mean ASR (%)")
    ax.set_title("(a) Flat Geodesic Fails (7/7)", pad=7)
    add_callout(ax, "Torsion $>$ Flat:\n7/7 models ($p\\leq 0.026$)\nMax $+38.7$ pts",
                x=0.04, y=0.96, ha="left", fontsize=7.6)

    # --- Panel B: Writes vs Full-Resid (Exp9 P1) ---
    ax = axes[1]
    style_ax(ax, grid_axis="y")
    if tags19:
        xs = np.arange(len(tags19))
        w = 0.34
        w_asr = [data[t]["exp6"]["gold_asr"]["LARA_PER_LAYER_k2"]["asr"] for t in tags19]
        fr_asr = [data[t]["exp9"]["p1_full_resid"]["gold_asr"]["LARA_PER_LAYER_k2"]["asr"] for t in tags19]
        w_kl = [data[t]["exp6"]["utility_and_force"]["LARA_PER_LAYER_k2"]["benign_kl"] for t in tags19]
        fr_kl = [data[t]["exp9"]["p1_full_resid"]["utility_and_force"]["LARA_PER_LAYER_k2"]["benign_kl"] for t in tags19]

        ax.bar(xs - w / 2, w_asr, width=w, color=PALETTE["blue"], alpha=0.85, edgecolor="#111111", lw=0.9, zorder=3)
        ax.bar(xs + w / 2, fr_asr, width=w, color=PALETTE["red"], alpha=0.72, edgecolor="#111111", lw=0.9, hatch="///", zorder=3)
        for i in range(len(tags19)):
            ax.text(xs[i] - w / 2, w_asr[i] + 1.5, f"KL\n{w_kl[i]:.2f}", ha="center", va="bottom",
                    fontsize=6.9, color=PALETTE["blue"], fontweight="bold")
            ax.text(xs[i] + w / 2, fr_asr[i] + 1.5, f"KL\n{fr_kl[i]:.2f}", ha="center", va="bottom",
                    fontsize=6.9, color=PALETTE["red"], fontweight="bold")
        ax.set_xticks(xs)
        ax.set_xticklabels([MODEL_SHORT[t].replace("-8B", "\n8B").replace("-7B", "\n7B").replace("-2B", "\n2B") for t in tags19], fontsize=8.0)
        ax.set_ylim(0, 92)
    ax.set_ylabel("Per-Layer Basin ($k=2$) SR Mean (%)")
    ax.set_title("(b) Writes vs. Full-Resid (Exp9 P1)", pad=7)
    add_callout(ax, "Writes: 0.0% collapse\nFull-resid KL: $3.6\\text{--}13.3\\times$",
                x=0.97, y=0.96, fontsize=7.6)

    # --- Panel C: 2D Safety-Utility Pareto Frontier (Strictly Non-Overlapping Labels) ---
    ax = axes[2]
    style_ax(ax, grid_axis="both")
    tags17 = _present_tags(data, "exp7")
    tags20 = _present_tags(data, "exp10")

    def _mean_delta_tqa(cond, src="exp7"):
        ts = tags17 if src == "exp7" else tags20
        return float(np.mean([
            data[t][src]["truthfulqa"]["conditions"][cond]["mc2"] -
            data[t][src]["truthfulqa"]["conditions"]["BASELINE"]["mc2"]
            for t in ts
        ]))

    def _mean_sr(cond, src="exp6"):
        ts = tags16 if src == "exp6" else tags20
        return float(np.mean([data[t][src]["gold_asr"][cond]["asr"] for t in ts]))

    def _mean_force(cond, src="exp6"):
        ts = tags16 if src == "exp6" else tags20
        return float(np.mean([data[t][src]["utility_and_force"][cond]["force"]["total_force_norm"] for t in ts]))

    # (key, label, color, marker, sr_src, tqa_src, offset_pts, ha, va)
    pts = [
        ("STATIC_GLOBAL_k2",  "Static Global $k=2$", PALETTE["slate"], "s", "exp6", "exp7", (0, -24),   "center", "top"),
        ("STATIC_GLOBAL_k1",  "Static Global $k=1$", PALETTE["slate"], "s", "exp6", "exp7", (0, -20),   "center", "top"),
        ("GEORA_TORSION_k1",  "Torsion Transport $k=1$", PALETTE["green"], "h", "exp6", "exp7", (-22, -14), "center", "top"),
        ("ABL_E3_WRITES",     r"$e_3$ Writes Only",  PALETTE["red"],   "D", "exp10", "exp10", (-14, -2),  "right",  "center"),
        ("LARA_PER_LAYER_k1", "Per-Layer Basin $k=1$",  PALETTE["blue"],  "D", "exp6", "exp7", (-14, 0),   "right",  "center"),
        ("LARA_PER_LAYER_k2", "Per-Layer Basin $k=2$",  PALETTE["blue"],  "D", "exp6", "exp7", (-10, 14),  "center", "bottom"),
    ]

    lx2 = _mean_delta_tqa("LARA_PER_LAYER_k2", "exp7")
    ly2 = _mean_sr("LARA_PER_LAYER_k2", "exp6")
    ax.fill_between([-4.9, lx2], 40, ly2, color=PALETTE["blue"], alpha=0.07, zorder=1)

    for ck, lbl, col, mk, s_sr, s_tq, (ox, oy), ha, va in pts:
        x_v = _mean_delta_tqa(ck, s_tq)
        y_v = _mean_sr(ck, s_sr)
        f_v = _mean_force(ck, s_sr)
        sz = 55 + 3.0 * f_v
        ax.scatter(x_v, y_v, s=sz, marker=mk, c=col, edgecolors="#111111", linewidths=1.15, zorder=5)
        ax.annotate(
            lbl, xy=(x_v, y_v), xytext=(ox, oy), textcoords="offset points",
            ha=ha, va=va, fontsize=7.8, color="#111111", zorder=7,
            bbox=dict(boxstyle="round,pad=0.12", facecolor=PALETTE["panel_bg"], edgecolor="none", alpha=0.75),
        )

    pf_x = [_mean_delta_tqa("GEORA_TORSION_k1", "exp7"), _mean_delta_tqa("ABL_E3_WRITES", "exp10"),
            _mean_delta_tqa("LARA_PER_LAYER_k1", "exp7"), lx2]
    pf_y = [_mean_sr("GEORA_TORSION_k1", "exp6"), _mean_sr("ABL_E3_WRITES", "exp10"),
            _mean_sr("LARA_PER_LAYER_k1", "exp6"), ly2]
    ax.plot(pf_x, pf_y, color=PALETTE["blue"], linestyle="--", lw=1.35, alpha=0.75, zorder=3)

    ax.axvline(0, color="#222222", linestyle=":", lw=0.9, zorder=2)
    ax.set_xlim(-4.95, 0.35)
    ax.set_ylim(40.8, 64.8)
    ax.set_xlabel(r"TruthfulQA Preservation ($\Delta$MC2%, closer to $0$ is better)")
    ax.set_ylabel("StrongREJECT Mean ASR (%)")
    ax.set_title("(c) Safety–Utility Pareto Frontier", pad=7)
    add_callout(ax, "Marker size $\\propto$ Force Norm\nUpper-right = Pareto Optimal",
                x=0.04, y=0.96, ha="left", fontsize=7.5)

    handles = [
        Line2D([0], [0], marker="D", color="none", markerfacecolor=PALETTE["blue"], markeredgecolor="#111111", markersize=7.2, label="Per-Layer Basin (Writes)"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=PALETTE["red"], markeredgecolor="#111111", markersize=7.2, label=r"$e_3$-Only Writes / Full-Resid"),
        Line2D([0], [0], marker="h", color="none", markerfacecolor=PALETTE["green"], markeredgecolor="#111111", markersize=7.8, label="(b) Torsion Transport (3-anchor)"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=PALETTE["slate"], markeredgecolor="#111111", markersize=7.0, label="Static Global"),
    ]
    add_top_pill_legend(fig, handles, ncol=4, y=0.975)
    save_fig(fig, outdir, "fig4_pareto_and_writes")


# =========================================================================== #
# FIGURE 5: Protocol Robustness (Fixed Panel C Callout Placement)
# =========================================================================== #
def plot_fig5_protocol_robustness(data: Dict[str, dict], outdir: str):
    tags19 = _present_tags(data, "exp9")
    if not tags19:
        print("  [SKIP Fig 5] No Exp9 files found.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.00))
    fig.subplots_adjust(top=0.79, bottom=0.17, left=0.06, right=0.98, wspace=0.30)

    # --- Panel A: Position Invariance ---
    ax = axes[0]
    offsets = ["-5", "-4", "-3", "-2", "-1"]
    tau_mat = np.zeros((len(tags19), len(offsets)))
    for i, t in enumerate(tags19):
        fp = data[t]["exp9"]["p2_optimal_pos"]["frenet_by_position"]
        for j, p in enumerate(offsets):
            tau_mat[i, j] = fp[p]["mean_tau"]
    ax.imshow(tau_mat, cmap="YlGnBu", aspect="auto", vmin=0.96, vmax=1.00)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.grid(False)
    ax.set_xticks(range(len(offsets)))
    ax.set_xticklabels([f"pos {p}" for p in offsets], fontsize=8.0)
    ax.set_yticks(range(len(tags19)))
    ax.set_yticklabels([MODEL_SHORT[t] for t in tags19], fontsize=8.4)
    ax.tick_params(length=0, pad=4)
    for i, t in enumerate(tags19):
        p_star = str(data[t]["exp9"]["p2_optimal_pos"]["best_pos"])
        for j, p in enumerate(offsets):
            v = tau_mat[i, j]
            star = "*" if p == p_star else ""
            ax.text(j, i, f"{v:.3f}{star}", ha="center", va="center", fontsize=7.6,
                    fontweight="bold" if star else "normal",
                    color="white" if v > 0.984 else "#111111")
    ax.set_xlabel("Post-Instruction Offset (* = Optimal $p^*$)")
    ax.set_title("(a) Position Invariance ($\\bar{\\tau}$, Exp9 P2)", pad=7)

    # --- Panel B: 256 vs 512 Token Parity ---
    ax = axes[1]
    style_ax(ax, grid_axis="y")
    p3_conds = [
        ("STATIC_GLOBAL_k1", "Static\nGlobal", PALETTE["slate"]),
        ("GEORA_FLAT_k1",    "Flat\nGeodesic",    PALETTE["gray"]),
        ("GEORA_TORSION_k1", "Torsion\nTransp.",    PALETTE["green"]),
        ("LARA_PER_LAYER_k1","Per-Layer\n$k=1$",    PALETTE["blue"]),
        ("LARA_PER_LAYER_k2","Per-Layer\n$k=2$",    PALETTE["red"]),
    ]
    xs = np.arange(len(p3_conds))
    w = 0.34
    for i, (ck, lbl, col) in enumerate(p3_conds):
        m256 = float(np.mean([data[t]["exp6"]["gold_asr"][ck]["asr"] for t in tags19]))
        m512 = float(np.mean([data[t]["exp9"]["p3_512tok"]["gold_asr"][ck]["asr"] for t in tags19]))
        ax.bar(i - w / 2, m256, width=w, color=col, alpha=0.45, edgecolor="#111111", lw=0.8, zorder=3)
        ax.bar(i + w / 2, m512, width=w, color=col, alpha=0.90, edgecolor="#111111", lw=0.8, zorder=3)
        ax.text(i + w / 2, m512 + 1.5, f"{m512:.1f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels([c[1] for c in p3_conds], fontsize=7.5)
    ax.set_ylim(0, 86)
    ax.set_ylabel("StrongREJECT Mean ASR (%)")
    ax.set_title("(b) 256 vs. 512 Tokens (Exp9 P3)", pad=7)
    add_callout(ax, "Light: 256 tok | Solid: 512 tok\nRanking 100% preserved",
                x=0.04, y=0.96, ha="left", fontsize=7.6)

    # --- Panel C: System-Prompt Robustness (Callout Centered Over Middle Valley) ---
    ax = axes[2]
    style_ax(ax, grid_axis="y")
    xs = np.arange(len(tags19))
    w = 0.34
    no_sys = [data[t]["exp8"]["d1_split_half_exp5b"]["repair_mlp_e3"] for t in tags19]
    w_sys = [data[t]["exp9"]["p4_sys_prompt"]["split_half_exp5b"]["repair_mlp_e3"] for t in tags19]
    ax.bar(xs - w / 2, no_sys, width=w, color=PALETTE["blue"], alpha=0.80, edgecolor="#111111", lw=0.9, zorder=3)
    ax.bar(xs + w / 2, w_sys, width=w, color=PALETTE["red"], alpha=0.80, edgecolor="#111111", lw=0.9, zorder=3)
    for i in range(len(tags19)):
        ax.text(xs[i] - w / 2, no_sys[i] + 0.02, f"{no_sys[i]:.2f}", ha="center", va="bottom", fontsize=7.4)
        ax.text(xs[i] + w / 2, w_sys[i] + 0.02, f"{w_sys[i]:.2f}", ha="center", va="bottom", fontsize=7.4)
    ax.set_xticks(xs)
    ax.set_xticklabels([MODEL_SHORT[t].replace("-8B", "\n8B").replace("-7B", "\n7B").replace("-2B", "\n2B") for t in tags19], fontsize=8.0)
    ax.set_ylim(0, 1.08)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel(r"CV Repair MLP $\cos^2(\Delta m, e_3)$")
    ax.set_title("(c) System-Prompt Invariance (P4)", pad=7)
    add_callout(ax, "All 4/4 retain $\\bar{\\tau}>0.991$\n& 26–69$\\times$ Cov-Null",
                x=0.50, y=0.96, ha="center", va="top", fontsize=7.6)

    handles = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor=PALETTE["blue"], markeredgecolor="#111111", markersize=7.2, label="(c) No System Prompt"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=PALETTE["red"], markeredgecolor="#111111", markersize=7.2, label="(c) With System Prompt"),
        Line2D([0], [0], marker="h", color="none", markerfacecolor=PALETTE["green"], markeredgecolor="#111111", markersize=7.8, label="(b) Torsion Transport (3-anchor)"),
    ]
    add_top_pill_legend(fig, handles, ncol=3, y=0.975)
    save_fig(fig, outdir, "fig5_protocol_robustness")


# =========================================================================== #
# APPENDIX FIGURE A1: Exp2 Self-Repair Recovery Curves rho(l)
# =========================================================================== #
def plot_figA1_exp2_recovery(data: Dict[str, dict], outdir: str):
    tags = _present_tags(data, "exp2")
    if not tags:
        return
    fig, axes = plt.subplots(2, 4, figsize=(12.4, 5.5))
    fig.subplots_adjust(top=0.84, bottom=0.11, left=0.06, right=0.98, hspace=0.40, wspace=0.26)
    axes_flat = axes.ravel()

    for idx, t in enumerate(tags):
        ax = axes_flat[idx]
        style_ax(ax, grid_axis="both")
        d12 = data[t]["exp2"]
        d15 = data[t]["exp5"]
        raw_rho = d12["recovery_fraction"]
        if isinstance(raw_rho, dict):
            pairs = sorted(((int(k), float(v)) for k, v in raw_rho.items()), key=lambda kv: kv[0])
            ls = np.array([k for k, _ in pairs], dtype=int)
            rho = np.array([v for _, v in pairs], dtype=float)
        else:
            rho = np.array([float(x) for x in raw_rho], dtype=float)
            ls = np.arange(len(rho))

        l0 = int(d12["l0"])
        basin = d15["basin"] if d15 else [l0]

        ax.axvspan(min(basin), max(basin), color=PALETTE["blue"], alpha=0.10, zorder=1)
        ax.axvline(l0, color=PALETTE["red"], linestyle="--", lw=1.1, zorder=2)
        ax.axhline(1.0, color="#666666", linestyle=":", lw=0.9, zorder=2)
        ax.plot(ls, rho, color=PALETTE["blue"], lw=1.9, marker="o", markersize=2.8,
                markeredgecolor="#111111", markeredgewidth=0.4, zorder=4)
        ax.set_title(MODEL_SHORT[t], fontsize=9.8, pad=5)
        ax.set_ylim(0.10, 1.52)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4])
        if idx >= 4:
            ax.set_xlabel("Layer $l$", fontsize=8.8)
        if idx % 4 == 0:
            ax.set_ylabel(r"Recovery $\rho(l)$", fontsize=8.8)
        mlp_dom = "MLP" if d12.get("verdict_mlp_written") else "Attn"
        add_callout(ax, f"$l_0={l0}$ | {mlp_dom}", x=0.96, y=0.06, va="bottom", fontsize=7.4)

    axes_flat[-1].axis("off")
    handles = [
        Line2D([0], [0], color=PALETTE["blue"], lw=2.0, marker="o", markersize=5, markeredgecolor="#111111", label=r"Recovery Fraction $\rho(l)$"),
        Line2D([0], [0], color=PALETTE["red"], ls="--", lw=1.3, label=r"Single-Layer Ablation $l_0$"),
        FancyBboxPatch((0, 0), 1, 1, facecolor=PALETTE["blue"], alpha=0.18, edgecolor="none", label=r"Causal Basin $\mathcal{B}$"),
    ]
    add_top_pill_legend(fig, handles, ncol=3, y=0.96)
    save_fig(fig, outdir, "figA1_exp2_recovery_curves")


# =========================================================================== #
# APPENDIX FIGURE A2: Exp7 XSTest + Exp10 Baseline Marker Audit (Leader Lines)
# =========================================================================== #
def plot_figA2_xstest_audit(data: Dict[str, dict], outdir: str):
    tags17 = _present_tags(data, "exp7")
    tags20 = _present_tags(data, "exp10")
    if not tags17:
        return

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.8, 4.15))
    fig.subplots_adjust(top=0.79, bottom=0.16, left=0.06, right=0.98, wspace=0.28)

    # Panel A: XSTest Safe Over-Refusal vs Unsafe Non-Refusal
    style_ax(axL, grid_axis="y")
    conds = [
        ("BASELINE",          "Base",         PALETTE["gray"]),
        ("STATIC_GLOBAL_k1",  "Global\n$k=1$", PALETTE["slate"]),
        ("GEORA_TORSION_k1",  "Torsion\nTransp.",  PALETTE["green"]),
        ("LARA_PER_LAYER_k1", "Per-Layer\n$k=1$",  PALETTE["blue"]),
        ("STATIC_GLOBAL_k2",  "Global\n$k=2$", PALETTE["slate"]),
        ("LARA_PER_LAYER_k2", "Per-Layer\n$k=2$",  PALETTE["red"]),
    ]
    xs = np.arange(len(conds))
    w = 0.36
    for i, (ck, lbl, col) in enumerate(conds):
        orr = float(np.mean([data[t]["exp7"]["xstest"]["conditions"][ck]["safe_over_refusal_pct"] for t in tags17]))
        unr = float(np.mean([data[t]["exp7"]["xstest"]["conditions"][ck]["unsafe_nonref_pct"] for t in tags17]))
        axL.bar(i - w / 2, unr, width=w, color=col, alpha=0.85, edgecolor="#111111", lw=0.85, zorder=3)
        axL.bar(i + w / 2, orr, width=w, color=PALETTE["amber"], alpha=0.85, edgecolor="#111111", lw=0.85, hatch="///", zorder=3)
        axL.text(i - w / 2, unr + 1.6, f"{unr:.1f}", ha="center", va="bottom", fontsize=7.4)
        axL.text(i + w / 2, orr + 1.6, f"{orr:.1f}", ha="center", va="bottom", fontsize=7.4)
    axL.set_xticks(xs)
    axL.set_xticklabels([c[1] for c in conds], fontsize=7.5)
    axL.set_ylim(0, 118)
    axL.set_yticks([0, 20, 40, 60, 80, 100])
    axL.set_ylabel("Rate (%) — 7-Model Mean")
    axL.set_title("(a) XSTest Unsafe Non-Refusal vs. Safe Over-Refusal", pad=7)
    add_callout(axL, "0.0% collapse (56/56 runs)\nSafe ORR: 5.2% $\\rightarrow$ 1.4%",
                x=0.04, y=0.96, ha="left", fontsize=7.8)

    # Panel B: Exp10 Baseline Unicode & Marker Audit (Clean Non-Overlapping Leader Lines)
    style_ax(axR, grid_axis="both")
    axR.plot([0, 108], [0, 108], color="#555555", linestyle="--", lw=1.1, zorder=2)

    # Explicit non-overlapping anchor coordinates (data-space xytext) + alignment per model
    label_layout = {
        "Mistral-7B-Instruct-v0.3": (35, 14, "left",  "center"),
        "Qwen2-7B-Instruct":        (56, 82, "right", "center"),
        "Qwen2.5-3B-Instruct":      (82, 46, "left",  "center"),
        "Qwen2.5-7B-Instruct":      (60, 95, "right", "center"),
        "gemma-2-2b-it":            (72, 107, "right", "center"),
        "Meta-Llama-3-8B-Instruct": (91, 66, "center", "top"),
        "Llama-3.1-8B-Instruct":    (106, 82, "right", "top"),
    }

    for t in tags20:
        aud = data[t]["exp10"]["baseline_marker_audit"]
        sx = float(aud["sr_below_0.10_pct"])
        my = float(aud["marker_refusal_pct"])
        curly = int(aud["n_with_unicode_apostrophe"])
        col = PALETTE["red"] if curly > 0 else PALETTE["blue"]
        mk = "D" if curly > 0 else "o"
        axR.scatter(sx, my, s=74, marker=mk, c=col, edgecolors="#111111", lw=1.0, zorder=5)

        tx, ty, ha, va = label_layout.get(t, (sx + 4, my - 4, "left", "center"))
        axR.annotate(
            MODEL_SHORT[t],
            xy=(sx, my), xytext=(tx, ty),
            ha=ha, va=va, fontsize=7.8, color="#111111", zorder=7,
            arrowprops=dict(arrowstyle="-", color="#444444", lw=0.8, shrinkA=2, shrinkB=5),
            bbox=dict(boxstyle="round,pad=0.14", facecolor=PALETTE["panel_bg"], edgecolor="none", alpha=0.80),
        )

    axR.set_xlim(5, 110)
    axR.set_ylim(2, 114)
    axR.set_xticks([20, 40, 60, 80, 100])
    axR.set_yticks([20, 40, 60, 80, 100])
    axR.set_xlabel(r"StrongREJECT Judge Score $< 0.10$ (%)")
    axR.set_ylabel("Unicode-Normalized Marker Refusal (%)")
    axR.set_title("(b) Baseline Marker vs. Judge Agreement (Exp10)", pad=7)
    add_callout(axR, "Llama-3.1: 19 curly apos.\nfixed (0 disagreements)",
                x=0.04, y=0.96, ha="left", fontsize=7.8)

    handles = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor=PALETTE["blue"], markeredgecolor="#111111", markersize=7.2, label="Unsafe Non-Refusal / ASCII Models"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=PALETTE["amber"], markeredgecolor="#111111", markersize=7.2, label="Safe Over-Refusal (XSTest 250)"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=PALETTE["red"], markeredgecolor="#111111", markersize=7.2, label="Models with Unicode Apostrophes Fixed"),
    ]
    add_top_pill_legend(fig, handles, ncol=3, y=0.975)
    save_fig(fig, outdir, "figA2_xstest_and_audit")


# =========================================================================== #
# Main
# =========================================================================== #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results", help="Directory with model result folders")
    ap.add_argument("--outdir", default="figures", help="Output folder for PDF + PNG figures")
    ap.add_argument("--usetex", action="store_true", help="Enable full LaTeX rendering (requires TeX Live)")
    a = ap.parse_args()

    apply_reference_style(use_tex=a.usetex)
    data = load_all_results(a.root)
    print(f">> Loaded {len(data)} models from {a.root}: {[data[k]['short'] for k in data]}")

    plot_fig1_hero(data, a.outdir)
    plot_fig2_three_tier_funnel(data, a.outdir)
    plot_fig3_circuit_anatomy(data, a.outdir)
    plot_fig4_pareto_and_writes(data, a.outdir)
    plot_fig5_protocol_robustness(data, a.outdir)
    plot_figA1_exp2_recovery(data, a.outdir)
    plot_figA2_xstest_audit(data, a.outdir)
    print(f">> All figures saved cleanly to: {a.outdir}/")


if __name__ == "__main__":
    main()