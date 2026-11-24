"""Single-command pipeline: regenerate every experiment JSON with the current
code, rebuild all LaTeX tables, and verify that each table and the paper's
headline numbers are reproducible from the JSON.

Per model, in order (each stage locks peak and basin from exp5_{tag}.json):
  1. run_recovery           -> exp2, exp3, exp4, exp5, exp6
  2. run_utility_eval       -> exp7_utility_xstest_tqa
  3. exp8_reviewer_defense   -> exp8_defense  (D1 split-half, D2 nulls, D3, D4)
  4. exp9_protocol_ablations -> exp9_protocol (P1-P4, subset of models)
  5. exp10_e3_gold_and_audit  -> exp10_e3_gold  (e3 knock-out, TruthfulQA, audit)
Then once:
  6. export_tables.py -> tables/*.tex
  7. verify_tables.py -> provenance, completeness, manuscript audit

Examples
--------
  # Offline wiring check (no GPU, no downloads): runs each sub-script and the verifier self-test.
  python -I aor/experiments/run_pipeline.py --selftest

  # Print the exact commands without running them:
  python -I aor/experiments/run_pipeline.py --dry_run

  # Full reproduction run on one 24 GB GPU:
  PYTHONNOUSERSITE=1 python -s aor/experiments/run_pipeline.py --root results --tables tables --device cuda

  # Regenerate selected stages, then re-export and verify:
  PYTHONNOUSERSITE=1 python -s aor/experiments/run_pipeline.py --stages exp9,exp10,export,verify --root results"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import Dict, List, Optional

_THIS = os.path.abspath(__file__)
_EXP_DIR = os.path.dirname(_THIS)
_PKG_DIR = os.path.dirname(_EXP_DIR)
_REPO_ROOT = os.path.dirname(_PKG_DIR)   # dir that contains the `aor/` package

ALL_MODELS: List[str] = [
    "gemma-2-2b-it",
    "Meta-Llama-3-8B-Instruct",
    "Llama-3.1-8B-Instruct",
    "Qwen2-7B-Instruct",
    "Qwen2.5-3B-Instruct",
    "Qwen2.5-7B-Instruct",
    "Mistral-7B-Instruct-v0.3",
]
# Models the manuscript reports for the (expensive) Exp9 protocol suite.
DEFAULT_Exp9_MODELS: List[str] = [
    "gemma-2-2b-it", "Meta-Llama-3-8B-Instruct",
    "Llama-3.1-8B-Instruct", "Qwen2.5-7B-Instruct",
]

# tag -> HuggingFace hub id (used only when --models_json gives no local path).
DEFAULT_HF_ID: Dict[str, str] = {
    "gemma-2-2b-it":             "google/gemma-2-2b-it",
    "Meta-Llama-3-8B-Instruct":  "meta-llama/Meta-Llama-3-8B-Instruct",
    "Llama-3.1-8B-Instruct":     "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen2-7B-Instruct":         "Qwen/Qwen2-7B-Instruct",
    "Qwen2.5-3B-Instruct":       "Qwen/Qwen2.5-3B-Instruct",
    "Qwen2.5-7B-Instruct":       "Qwen/Qwen2.5-7B-Instruct",
    "Mistral-7B-Instruct-v0.3":  "mistralai/Mistral-7B-Instruct-v0.3",
}

ALL_STAGES = ["recovery", "sync", "exp8", "exp9", "exp10", "fill", "stats", "ood", "split", "export", "verify"]


def _resolve_model_path(tag: str, models_json: Dict[str, str], models_dir: Optional[str]) -> str:
    if tag in models_json:
        return models_json[tag]
    if models_dir:
        cand = os.path.join(models_dir, tag)
        if os.path.isdir(cand):
            return cand
    return DEFAULT_HF_ID.get(tag, tag)


def _data_paths(data_dir: str) -> Dict[str, str]:
    return {
        "harmful":      os.path.join(data_dir, "harmful_train.json"),
        "benign":       os.path.join(data_dir, "harmless_train.json"),
        "test_harmful": os.path.join(data_dir, "harmful_test.json"),
        "test_benign":  os.path.join(data_dir, "harmless_test.json"),
    }


def _py(mod_relpath: str) -> List[str]:
    # Run as a file path under the repo root; scripts insert sys.path themselves.
    return [sys.executable, "-s", os.path.join(_PKG_DIR, mod_relpath)]


def _common_data_args(dp: Dict[str, str]) -> List[str]:
    return [
        "--harmful", dp["harmful"], "--benign", dp["benign"],
        "--test_harmful", dp["test_harmful"], "--test_benign", dp["test_benign"],
    ]


def build_commands(tag: str, model_path: str, outdir: str, dp: Dict[str, str],
                   device: str, stages: List[str], run_exp9: bool,
                   n_train: int, n_gen: int, max_new_tokens: int,
                   batch_size: int) -> List[List[str]]:
    base = ["--model", model_path, *_common_data_args(dp),
            "--device", device, "--outdir", outdir]
    cmds: List[List[str]] = []
    if "recovery" in stages:
        cmds.append(_py("experiments/run_recovery.py") + base + [
            "--gold", "--ablate_mode", "writes", "--exp6_ranks", "1,2",
            "--n_train", str(n_train), "--n_gen", str(n_gen),
            "--max_new_tokens", str(max_new_tokens), "--batch_size", str(batch_size),
        ])
    if "sync" in stages:
        cmds.append(_py("experiments/run_utility_eval.py") + base + [
            "--n_gen", str(n_gen), "--batch_size", str(batch_size),
        ])
    if "exp8" in stages:
        cmds.append(_py("experiments/exp8_reviewer_defense.py") + base + [
            "--n_train", str(n_train), "--n_gen", str(n_gen), "--batch_size", str(batch_size),
        ])
    if "exp9" in stages and run_exp9:
        cmds.append(_py("experiments/exp9_protocol_ablations.py") + base + [
            "--n_train", str(n_train), "--n_gen", str(n_gen), "--batch_size", str(batch_size),
        ])
    if "exp10" in stages:
        cmds.append(_py("experiments/exp10_e3_gold_and_audit.py") + base + [
            "--n_train", str(n_train), "--n_gen", str(n_gen),
            "--max_new_tokens", str(max_new_tokens), "--batch_size", str(batch_size),
        ])
    if "fill" in stages:
        # Two auxiliary evaluation cells (XSTest-safe for e3-only, all models;
        # writes-TQA for 2 aux conds on P1 models — fill_gaps self-guards by tag).
        cmds.append(_py("experiments/fill_gaps.py") + base + [
            "--n_train", str(n_train), "--n_gen", str(n_gen), "--batch_size", str(batch_size),
        ])
    if "stats" in stages:
        # 10k-permutation test + per-model confidence intervals.
        cmds.append(_py("experiments/strengthen_stats.py") + base + [
            "--n_train", str(n_train), "--n_gen", str(n_gen),
            "--batch_size", str(batch_size), "--n_perms", "10000",
        ])
    if "ood" in stages:
        # e3 knock-out on unseen benchmarks (StrongREJECT + JailbreakBench).
        cmds.append(_py("experiments/generalize_ood.py") + base + [
            "--n_train", str(n_train), "--n_gen", str(n_gen),
            "--max_new_tokens", str(max_new_tokens), "--batch_size", str(batch_size),
        ])
    if "split" in stages:
        # repeated split-half stability of e3.
        cmds.append(_py("experiments/repeated_splithalf.py") + base + [
            "--n_train", str(n_train), "--n_gen", str(n_gen),
            "--batch_size", str(batch_size), "--n_seeds", "50",
        ])
    return cmds


def _run(cmd: List[str], env: Dict[str, str], dry: bool, cont: bool) -> bool:
    printable = " ".join(cmd)
    print(f"\n$ {printable}", flush=True)
    if dry:
        return True
    try:
        subprocess.run(cmd, cwd=_REPO_ROOT, env=env, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"!! command failed (exit {e.returncode})", flush=True)
        if not cont:
            raise
        return False


def run_selftest(env: Dict[str, str]) -> int:
    print(">> [SELFTEST run_finalize] offline wiring check of every sub-script ...")
    scripts = [
        "experiments/run_recovery.py",
        "experiments/exp8_reviewer_defense.py",
        "experiments/exp9_protocol_ablations.py",
        "experiments/exp10_e3_gold_and_audit.py",
        "experiments/fill_gaps.py",
        "experiments/strengthen_stats.py",
        "experiments/generalize_ood.py",
        "experiments/repeated_splithalf.py",
    ]
    ok = True
    for s in scripts:
        print(f"\n--- {s} --selftest ---", flush=True)
        r = subprocess.run(_py(s) + ["--selftest"], cwd=_REPO_ROOT, env=env)
        ok = ok and (r.returncode == 0)
    print("\n--- plot/verify_tables.py --selftest ---", flush=True)
    r = subprocess.run([sys.executable, "-I", os.path.join(_PKG_DIR, "plot/verify_tables.py"),
                        "--selftest"], cwd=_REPO_ROOT, env=env)
    ok = ok and (r.returncode == 0)
    print(f"\n>> [SELFTEST run_finalize] {'ALL PASSED ✓' if ok else 'FAILED ✗'}")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description="Regenerate + export + verify all results in one run.")
    ap.add_argument("--root", default="results",
                    help="Output results dir (per-model folders created inside).")
    ap.add_argument("--tables", default="tables", help="Output dir for .tex tables.")
    ap.add_argument("--data_dir", default=os.path.join("aor", "data", "splits"),
                    help="Dir with harmful_train/harmless_train/harmful_test/harmless_test .json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--models", default="all",
                    help="Comma-separated tags, or 'all'.")
    ap.add_argument("--exp9_models", default=",".join(DEFAULT_Exp9_MODELS),
                    help="Comma-separated tags to also run Exp9 on (expensive).")
    ap.add_argument("--models_dir", default=None,
                    help="Dir holding local HF snapshots named by tag (optional).")
    ap.add_argument("--models_json", default=None,
                    help="JSON file mapping tag -> local path or HF id (optional).")
    ap.add_argument("--stages", default=",".join(ALL_STAGES),
                    help=f"Subset of {ALL_STAGES}.")
    ap.add_argument("--n_train", type=int, default=256)
    ap.add_argument("--n_gen", type=int, default=150)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--dry_run", action="store_true", help="Print commands only.")
    ap.add_argument("--continue_on_error", action="store_true",
                    help="Keep going if one model/stage fails.")
    ap.add_argument("--selftest", action="store_true",
                    help="Offline: run every sub-script's --selftest + verifier --selftest.")
    a = ap.parse_args()

    env = dict(os.environ)
    env.setdefault("PYTHONNOUSERSITE", "1")

    if a.selftest:
        sys.exit(run_selftest(env))

    import json as _json
    models_json: Dict[str, str] = {}
    if a.models_json and os.path.isfile(a.models_json):
        models_json = _json.load(open(a.models_json))

    tags = ALL_MODELS if a.models.strip() == "all" else [t.strip() for t in a.models.split(",") if t.strip()]
    exp9_set = {t.strip() for t in a.exp9_models.split(",") if t.strip()}
    stages = [s.strip() for s in a.stages.split(",") if s.strip()]
    dp = _data_paths(a.data_dir)

    print("=" * 70)
    print("FINALIZE - regenerate | export | verify")
    print(f"  repo root : {_REPO_ROOT}")
    print(f"  results   : {a.root}")
    print(f"  tables    : {a.tables}")
    print(f"  models    : {tags}")
    print(f"  Exp9 on    : {sorted(t for t in tags if t in exp9_set)}")
    print(f"  stages    : {stages}")
    print(f"  data_dir  : {a.data_dir}")
    print("=" * 70)

    per_model_stages = [s for s in stages if s in ("recovery", "sync", "exp8", "exp9", "exp10", "fill", "stats", "ood", "split")]
    failures: List[str] = []

    if per_model_stages:
        for tag in tags:
            model_path = _resolve_model_path(tag, models_json, a.models_dir)
            outdir = os.path.join(a.root, tag)
            os.makedirs(outdir, exist_ok=True) if not a.dry_run else None
            print(f"\n########################  {tag}  ########################")
            print(f"  model_path = {model_path}")
            print(f"  outdir     = {outdir}")
            cmds = build_commands(
                tag, model_path, outdir, dp, a.device, per_model_stages,
                run_exp9=(tag in exp9_set), n_train=a.n_train, n_gen=a.n_gen,
                max_new_tokens=a.max_new_tokens, batch_size=a.batch_size,
            )
            for cmd in cmds:
                okc = _run(cmd, env, a.dry_run, a.continue_on_error)
                if not okc:
                    failures.append(f"{tag}: {cmd[2] if len(cmd) > 2 else cmd}")

    if "stats" in stages and not a.dry_run:
        _run(_py("experiments/strengthen_stats.py") + ["--aggregate", a.root], env, a.dry_run, True)
    if "ood" in stages and not a.dry_run:
        _run(_py("experiments/generalize_ood.py") + ["--aggregate", a.root], env, a.dry_run, True)
    if "split" in stages and not a.dry_run:
        _run(_py("experiments/repeated_splithalf.py") + ["--aggregate", a.root], env, a.dry_run, True)

    if "export" in stages:
        export_cmd = _py("plot/export_tables.py") + ["--root", a.root, "--outdir", a.tables]
        _run(export_cmd, env, a.dry_run, a.continue_on_error)

    verify_ok = True
    if "verify" in stages:
        verify_cmd = [sys.executable, "-s", os.path.join(_PKG_DIR, "plot/verify_tables.py"),
                      "--root", a.root, "--tables", a.tables]
        print(f"\n$ {' '.join(verify_cmd)}", flush=True)
        if not a.dry_run:
            r = subprocess.run(verify_cmd, cwd=_REPO_ROOT, env=env)
            verify_ok = (r.returncode == 0)

    print("\n" + "=" * 70)
    if failures:
        print(f"!! {len(failures)} stage failure(s):")
        for f in failures:
            print(f"   - {f}")
    if a.dry_run:
        print(">> dry run complete (no commands executed).")
        sys.exit(0)
    final_ok = (not failures) and verify_ok
    print(f">> FINALIZE {'COMPLETE ✓ — tables are reproducible from JSON' if final_ok else 'INCOMPLETE ✗ — see failures/verification above'}")
    print("=" * 70)
    sys.exit(0 if final_ok else 1)


if __name__ == "__main__":
    main()
