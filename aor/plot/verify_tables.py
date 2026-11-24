"""Verify that every LaTeX table on disk was regenerated from the current
experiment JSONs and that the manuscript's headline numbers match the JSONs.

Three independent guards, all of which must pass:
  [A] provenance   : for every (file, builder) pair in the exporter, rebuild the
                     table in memory from JSON and compare (whitespace-normalized)
                     with the .tex on disk; a missing source JSON or drifted
                     numbers fail here.
  [B] completeness : assert Exp10 JSON exist for all seven models and Exp9 JSON for
                     the four models the manuscript uses (configurable), listing
                     any missing.
  [C] manuscript   : recompute the abstract and Table 2 mean-row constants from
                     the Exp10 JSONs and assert they match the values stated in the
                     manuscript, within rounding tolerance.

Usage
-----
  PYTHONNOUSERSITE=1 python -s aor/plot/verify_tables.py --root results --tables tables
  python -I aor/plot/verify_tables.py --selftest

The exit code is non-zero if any guard fails, so it can gate CI."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import sys
import tempfile
from typing import Callable, Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXPORT_PATH = os.path.join(_HERE, "export_tables.py")

# Models the manuscript claims for each experiment (edit here if scope changes).
REQUIRED_Exp10 = [
    "gemma-2-2b-it", "Meta-Llama-3-8B-Instruct", "Llama-3.1-8B-Instruct",
    "Qwen2-7B-Instruct", "Qwen2.5-3B-Instruct", "Qwen2.5-7B-Instruct",
    "Mistral-7B-Instruct-v0.3",
]
REQUIRED_Exp9 = [
    "gemma-2-2b-it", "Meta-Llama-3-8B-Instruct",
    "Llama-3.1-8B-Instruct", "Qwen2.5-7B-Instruct",
]

# Manuscript headline constants (abstract + Table 2 mean row). Recomputed from
# Exp10 JSON and compared within TOL_PTS. Keep in sync with acl_latex.tex.
TOL_PTS = 0.15
MANUSCRIPT_CLAIMS = {
    "ABL_E3_WRITES.sr_mean": 50.0,
    "ABL_E3_WRITES.nr_mean": 83.4,
    "ABL_E3_MLP_ONLY.sr_mean": 33.2,
    "ABL_E3_MLP_ONLY.nr_mean": 65.0,
    "ABL_E1_WRITES.sr_mean": 14.7,
    "ABL_E1_WRITES.nr_mean": 29.9,
    "ABL_E2_WRITES.sr_mean": 12.4,
    "ABL_E2_WRITES.nr_mean": 24.9,
    "ABL_RAND_ORTH_WRITES.sr_mean": 11.2,
    "ABL_RAND_ORTH_WRITES.nr_mean": 22.6,
    "LARA_PER_LAYER_k1.sr_mean": 54.3,
}
# Integer claims checked exactly.
MANUSCRIPT_COUNT_CLAIMS = {
    "n_models_p_e3_gt_e1_lt_.001": 6,   # "p<0.001 on 6/7 models"
    "n_models_p_e3_gt_e1_lt_.05": 7,    # "significant on 7/7"
}


# --------------------------------------------------------------------------- #
# Import the exporter as a module (stdlib-only; safe without torch / GPU).
# --------------------------------------------------------------------------- #
def _load_exporter(path: str = _EXPORT_PATH):
    spec = importlib.util.spec_from_file_location("export_tables", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _norm_ws(s: str) -> str:
    """Collapse all whitespace so comparison ignores reflow but catches any
    change in tokens/numbers."""
    return re.sub(r"\s+", " ", s).strip()


# --------------------------------------------------------------------------- #
# [A] Provenance: table-on-disk == table-rebuilt-from-JSON
# --------------------------------------------------------------------------- #
def check_provenance(exporter, data: Dict[str, dict], tables_dir: str) -> List[dict]:
    rows = []
    for fname, builder in exporter_table_list(exporter):
        disk_p = os.path.join(tables_dir, fname)
        row = {"file": fname, "status": "PASS", "reason": ""}
        if not os.path.isfile(disk_p):
            row["status"], row["reason"] = "FAIL", "table .tex missing on disk"
            rows.append(row)
            continue
        try:
            rebuilt = builder(data)
        except Exception as e:  # missing source JSON surfaces as a build error
            row["status"], row["reason"] = "FAIL", f"cannot rebuild from JSON ({type(e).__name__}: {e})"
            rows.append(row)
            continue
        on_disk = open(disk_p, encoding="utf-8").read()
        if _norm_ws(rebuilt) == _norm_ws(on_disk):
            row["reason"] = "matches JSON"
        else:
            row["status"] = "FAIL"
            # Is it empty-because-no-source, or genuine drift?
            if _is_sourceless(exporter, data, fname):
                row["reason"] = "stale: on-disk numbers have NO backing JSON (source missing)"
            else:
                row["reason"] = "DRIFT: on-disk numbers != current JSON"
        rows.append(row)
    return rows


def exporter_table_list(exporter) -> List[Tuple[str, Callable]]:
    """Mirror the (filename, builder) list inside the exporter's main()."""
    b = exporter
    return [
        ("tab1_main_three_tier.tex",           b.build_tab1_main_three_tier),
        ("tab2_main_direct_e3_gold.tex",       b.build_tab2_main_direct_e3_gold),
        ("tab3_main_pareto_7axis.tex",         b.build_tab3_main_pareto_7axis),
        ("tabC_basin_and_frenet.tex",          b.build_tabC_basin_and_frenet),
        ("tabD_self_repair_exp2_exp4.tex",       b.build_tabD_self_repair_exp2_exp4),
        ("tabE_controls_ref_vs_native.tex",    b.build_tabE_controls_ref_vs_native),
        ("tabF_d3_logodds_and_steering.tex",   b.build_tabF_d3_logodds_and_steering),
        ("tabG1_exp6_all13_conditions.tex",     b.build_tabG1_exp6_all13),
        ("tabG2_exp7_tqa_xstest_per_model.tex", b.build_tabG2_exp7_tqa_xstest),
        ("tabH1_exp9_p1_writes_vs_resid.tex",   b.build_tabH1_exp9_p1),
        ("tabH2_exp9_p2_optimal_pos.tex",       b.build_tabH2_exp9_p2),
        ("tabH3_exp9_p3_p4_512_sysprompt.tex",  b.build_tabH3_exp9_p3_p4),
        ("tabI_exp10_unicode_audit.tex",         b.build_tabI_unicode_audit),
    ]


_FILE_SOURCE_KEY = {
    "tab2_main_direct_e3_gold.tex": ("exp10", REQUIRED_Exp10),
    "tabI_exp10_unicode_audit.tex": ("exp10", REQUIRED_Exp10),
    "tabH1_exp9_p1_writes_vs_resid.tex": ("exp9", REQUIRED_Exp9),
    "tabH2_exp9_p2_optimal_pos.tex": ("exp9", REQUIRED_Exp9),
    "tabH3_exp9_p3_p4_512_sysprompt.tex": ("exp9", REQUIRED_Exp9),
}


def _is_sourceless(exporter, data: Dict[str, dict], fname: str) -> bool:
    key_req = _FILE_SOURCE_KEY.get(fname)
    if not key_req:
        return False
    key, _ = key_req
    return len(exporter._tags(data, key)) == 0


# --------------------------------------------------------------------------- #
# [B] Completeness
# --------------------------------------------------------------------------- #
def check_completeness(data: Dict[str, dict]) -> List[dict]:
    rows = []
    for key, required in (("exp10", REQUIRED_Exp10), ("exp9", REQUIRED_Exp9)):
        have = [t for t in required if data.get(t, {}).get(key) is not None]
        missing = [t for t in required if t not in have]
        rows.append({
            "experiment": key.upper(),
            "have": len(have),
            "required": len(required),
            "status": "PASS" if not missing else "FAIL",
            "missing": missing,
        })
    return rows


# --------------------------------------------------------------------------- #
# [C] Manuscript constants recomputed from Exp10 JSON
# --------------------------------------------------------------------------- #
def _mean(xs: List[float]) -> float:
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return sum(xs) / len(xs) if xs else float("nan")


def recompute_manuscript_values(data: Dict[str, dict]) -> Dict[str, float]:
    tags = [t for t in REQUIRED_Exp10 if data.get(t, {}).get("exp10") is not None]
    out: Dict[str, float] = {}
    conds = ["ABL_E3_WRITES", "ABL_E3_MLP_ONLY", "ABL_E1_WRITES",
             "ABL_E2_WRITES", "ABL_RAND_ORTH_WRITES", "LARA_PER_LAYER_k1"]
    for c in conds:
        sr, nr = [], []
        for t in tags:
            g = (data[t]["exp10"].get("gold_asr") or {}).get(c)
            if g:
                sr.append(g.get("asr"))
                nr.append(g.get("non_refusal_asr"))
        out[f"{c}.sr_mean"] = round(_mean(sr), 1)
        out[f"{c}.nr_mean"] = round(_mean(nr), 1)

    n_001 = n_05 = 0
    for t in tags:
        ps = (data[t]["exp10"].get("paired_significance") or {})
        pv = ps.get("ABL_E3_WRITES_vs_ABL_E1_WRITES", {}).get("p_val")
        if pv is not None:
            n_001 += int(pv < 0.001)
            n_05 += int(pv < 0.05)
    out["n_models_p_e3_gt_e1_lt_.001"] = n_001
    out["n_models_p_e3_gt_e1_lt_.05"] = n_05
    out["_n_models_exp10"] = len(tags)
    return out


def check_manuscript(data: Dict[str, dict]) -> List[dict]:
    got = recompute_manuscript_values(data)
    rows = []
    for key, claimed in MANUSCRIPT_CLAIMS.items():
        val = got.get(key, float("nan"))
        ok = (not math.isnan(val)) and abs(val - claimed) <= TOL_PTS
        rows.append({"metric": key, "manuscript": claimed, "recomputed": val,
                     "status": "PASS" if ok else "FAIL"})
    for key, claimed in MANUSCRIPT_COUNT_CLAIMS.items():
        val = got.get(key)
        ok = (val == claimed)
        rows.append({"metric": key, "manuscript": claimed, "recomputed": val,
                     "status": "PASS" if ok else "FAIL"})
    return rows


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _print_section(title: str, rows: List[dict], cols: List[Tuple[str, str, int]]) -> bool:
    print(f"\n==================  {title}  ==================")
    header = " | ".join(f"{h:<{w}}" for _, h, w in cols)
    print(header)
    print("-" * len(header))
    all_pass = True
    for r in rows:
        if r.get("status") == "FAIL":
            all_pass = False
        cells = []
        for key, _, w in cols:
            v = r.get(key, "")
            if isinstance(v, float):
                v = f"{v:.1f}" if not math.isnan(v) else "nan"
            cells.append(f"{str(v):<{w}}")
        print(" | ".join(cells))
    return all_pass


def run_verification(root: str, tables_dir: str) -> bool:
    exporter = _load_exporter()
    data = exporter.load_all(root)
    if not data:
        print(f"!! No model result folders found under {root}")
        return False

    prov = check_provenance(exporter, data, tables_dir)
    comp = check_completeness(data)
    manu = check_manuscript(data)

    p1 = _print_section("[A] PROVENANCE  (table .tex == rebuilt-from-JSON)", prov,
                         [("status", "RESULT", 6), ("file", "TABLE", 38), ("reason", "DETAIL", 52)])
    p2 = _print_section("[B] COMPLETENESS  (required model JSONs present)", comp,
                         [("status", "RESULT", 6), ("experiment", "EXP", 5),
                          ("have", "HAVE", 5), ("required", "REQ", 4), ("missing", "MISSING", 50)])
    p3 = _print_section("[C] MANUSCRIPT  (abstract/Tab2 == recomputed from Exp10 JSON)", manu,
                         [("status", "RESULT", 6), ("metric", "METRIC", 34),
                          ("manuscript", "PAPER", 8), ("recomputed", "JSON", 8)])

    ok = p1 and p2 and p3
    print("\n" + "=" * 66)
    print(f">> VERIFICATION {'PASSED ✓' if ok else 'FAILED ✗'}  "
          f"(provenance={'ok' if p1 else 'FAIL'}, completeness={'ok' if p2 else 'FAIL'}, "
          f"manuscript={'ok' if p3 else 'FAIL'})")
    print("=" * 66)
    return ok


# --------------------------------------------------------------------------- #
# Offline self-test: synthesize a tiny results tree, export tables, verify PASS,
# then mutate one JSON number and confirm the verifier FAILS (guard is live).
# --------------------------------------------------------------------------- #
def _synth_exp10(tag: str, e3_sr: float) -> dict:
    conds = {
        "BASELINE": (0.0, 0.0), "ABL_E3_WRITES": (e3_sr, e3_sr + 30),
        "ABL_E3_MLP_ONLY": (e3_sr - 17, e3_sr + 15), "ABL_E2_WRITES": (12.4, 24.9),
        "ABL_E1_WRITES": (14.7, 29.9), "ABL_RAND_ORTH_WRITES": (11.2, 22.6),
        "LARA_PER_LAYER_k1": (54.3, 88.2),
    }
    gold = {k: {"asr": sr, "bin_asr": sr, "non_refusal_asr": nr, "lo": sr - 3, "hi": sr + 3}
            for k, (sr, nr) in conds.items()}
    tqa = {"source": "stub", "n_questions": 10, "conditions": {
        k: {"hook_mode": "writes", "mc1": 40.0, "mc2": 55.0, "mc1_norm": 40.0,
            "mc2_norm": 55.0, "mc1_lo": 38, "mc1_hi": 42, "mc2_lo": 53, "mc2_hi": 57, "n": 10}
        for k in ("BASELINE", "ABL_E3_WRITES", "ABL_E3_MLP_ONLY", "LARA_PER_LAYER_k1")},
        "paired_significance": {}}
    uf = {k: {"benign_kl": 0.05, "benign_ce_loss": 1.0,
              "force": {"total_force_norm": 14.0}} for k in conds}
    uf["BASELINE"] = {"benign_kl": 0.0, "benign_ce_loss": 1.0, "force": {"total_force_norm": 0.0}}
    return {
        "model": tag, "peak": 12, "basin": [11, 12, 13], "basin_ds": [11, 12, 13],
        "n_test_harmful": 150, "max_new_tokens": 256,
        "utility_and_force": uf, "lexical_diagnostics": {},
        "truthfulqa": tqa, "gold_asr": gold,
        "paired_significance": {
            "ABL_E3_WRITES_vs_ABL_E1_WRITES": {"diff_pts": 35.3, "ci_lo": 30, "ci_hi": 40,
                                               "p_val": 0.0001, "nr_diff_pts": 53.5, "nr_p_val": 0.0001},
            "ABL_E3_WRITES_vs_ABL_E2_WRITES": {"diff_pts": 37.6, "ci_lo": 32, "ci_hi": 42,
                                               "p_val": 0.0001, "nr_diff_pts": 58.5, "nr_p_val": 0.0001},
        },
        "baseline_marker_audit": {
            "n_prompts": 150, "n_with_unicode_apostrophe": 0, "marker_refusal_pct": 98.0,
            "marker_nonref_pct": 2.0, "sr_below_0.10_pct": 95.0,
            "disagree_nonref_but_sr_low_count": 0, "disagree_ref_but_sr_high_count": 0,
            "top_disagree_opening_4gram_counts": [],
        },
    }


def _selftest() -> bool:
    print(">> [SELFTEST verify] synthesizing a minimal Exp10-only results tree ...")
    exporter = _load_exporter()
    with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as tdir:
        # Mean of e3 sr across 7 tags set so mean == 50.0 exactly.
        e3_vals = [38.5, 64.9, 33.8, 53.0, 54.4, 52.7, 52.7]  # mean = 50.0
        for tag, v in zip(REQUIRED_Exp10, e3_vals):
            os.makedirs(os.path.join(root, tag), exist_ok=True)
            json.dump(_synth_exp10(tag, v),
                      open(os.path.join(root, tag, f"exp10_e3_gold_{tag}.json"), "w"), indent=2)
        data = exporter.load_all(root)

        # Rebuild just the two Exp10 tables and write them to disk (simulate exporter).
        for fname, builder in exporter_table_list(exporter):
            if fname in ("tab2_main_direct_e3_gold.tex", "tabI_exp10_unicode_audit.tex"):
                open(os.path.join(tdir, fname), "w").write(builder(data))

        prov = {r["file"]: r for r in check_provenance(exporter, data, tdir)}
        assert prov["tab2_main_direct_e3_gold.tex"]["status"] == "PASS", "clean provenance should PASS"
        manu = {r["metric"]: r for r in check_manuscript(data)}
        assert manu["ABL_E3_WRITES.sr_mean"]["status"] == "PASS", "mean 50.0 should PASS"
        assert manu["n_models_p_e3_gt_e1_lt_.05"]["status"] == "PASS", "7/7 sig should PASS"
        print("   clean tree -> provenance PASS, manuscript PASS  ✓")

        # Mutation 1: corrupt a JSON number -> provenance must FAIL for tab2.
        bad_tag = REQUIRED_Exp10[0]
        bp = os.path.join(root, bad_tag, f"exp10_e3_gold_{bad_tag}.json")
        d = json.load(open(bp)); d["gold_asr"]["ABL_E3_WRITES"]["asr"] = 99.9
        json.dump(d, open(bp, "w"))
        data2 = exporter.load_all(root)
        prov2 = {r["file"]: r for r in check_provenance(exporter, data2, tdir)}
        assert prov2["tab2_main_direct_e3_gold.tex"]["status"] == "FAIL", "number drift must FAIL"
        print("   mutated JSON -> provenance FAIL detected  ✓")

        # Mutation 2: delete a source JSON -> 'sourceless' detected.
        os.remove(bp)
        data3 = exporter.load_all(root)
        prov3 = {r["file"]: r for r in check_provenance(exporter, data3, tdir)}
        # still 6 tags -> not sourceless, but numbers now differ -> FAIL
        assert prov3["tab2_main_direct_e3_gold.tex"]["status"] == "FAIL"
        for t in REQUIRED_Exp10[1:]:
            os.remove(os.path.join(root, t, f"exp10_e3_gold_{t}.json"))
        data4 = exporter.load_all(root)
        prov4 = {r["file"]: r for r in check_provenance(exporter, data4, tdir)}
        assert "source missing" in prov4["tab2_main_direct_e3_gold.tex"]["reason"], "must flag missing source"
        print("   all source JSON removed -> 'source missing' flagged  ✓")

    print(">> [SELFTEST verify] ALL CHECKS PASSED.")
    return True


def main():
    ap = argparse.ArgumentParser(description="Verify tables are reproducible from JSON.")
    ap.add_argument("--root", default="results",
                    help="Results dir containing per-model folders with e*.json")
    ap.add_argument("--tables", default="tables",
                    help="Directory with the generated .tex table files")
    ap.add_argument("--selftest", action="store_true", help="Offline wiring + mutation self-test")
    a = ap.parse_args()

    if a.selftest:
        ok = _selftest()
        sys.exit(0 if ok else 1)

    ok = run_verification(a.root, a.tables)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
