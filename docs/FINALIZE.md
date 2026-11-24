# Reproduction Pipeline

`run_pipeline.py` regenerates every experiment JSON with the current code,
rebuilds all LaTeX tables, and self-verifies that each table and the paper's
headline numbers are reproducible from the JSON. A successful run guarantees that
code, results, tables, and manuscript agree.

## 1. Offline wiring check (no GPU, no downloads)

```bash
python -I aor/experiments/run_pipeline.py --selftest
```

Runs each sub-script's stub `--selftest` (Exp2-Exp10) and the table verifier's
self-test, which synthesizes a small results tree, confirms a clean tree passes,
then mutates a JSON number and confirms the verifier fails. Expect
`>> [SELFTEST run_finalize] ALL PASSED`.

## 2. Full reproduction run (one command)

```bash
PYTHONNOUSERSITE=1 python -s aor/experiments/run_pipeline.py \
    --root results \
    --tables tables \
    --data_dir aor/data/splits \
    --device cuda
```

Per model, peak and basin are locked from `exp5_{tag}.json` so every stage shares
one basin:

1. `run_recovery.py` -> `exp2, exp3, exp4, exp5, exp6`
2. `run_utility_eval.py` -> `exp7_utility_xstest_tqa`
3. `exp8_reviewer_defense.py` -> `exp8_defense` (D1 split-half, D2 nulls, D3, D4)
4. `exp9_protocol_ablations.py` -> `exp9_protocol` (P1-P4; Exp9 models only)
5. `exp10_e3_gold_and_audit.py` -> `exp10_e3_gold` (e3 knock-out, TruthfulQA, audit)
6. `fill_gaps.py` -> two auxiliary evaluation cells (XSTest-safe for e3-only
   writes, and writes-mode TruthfulQA MC1 for the two auxiliary k=2 conditions)

Then once: `export_tables.py` -> `tables/*.tex`, and `verify_tables.py` ->
provenance, completeness, and manuscript audit. The run exits non-zero unless
verification passes, so it is safe in CI.

### Useful flags

- `--dry_run` — print the exact commands, run nothing.
- `--models gemma-2-2b-it,Llama-3.1-8B-Instruct` — subset of tags (`all` = 7).
- `--exp9_models ...` — which tags also get Exp9 (default 4).
- `--stages fill,export,verify` — run only part of the pipeline on an existing tree.
- `--models_json map.json` — `{tag: "/local/path/or/hf-id"}`; otherwise the Hub id.
- `--continue_on_error` — keep going if one model or stage fails.

## 3. Verify an existing tree on its own

```bash
python -s aor/plot/verify_tables.py --root results --tables tables
```

- **[A] Provenance** — rebuilds each `.tex` from JSON in memory and compares it
  (whitespace-normalized) with the file on disk.
- **[B] Completeness** — Exp10 for all seven models, Exp9 for the required four.
- **[C] Manuscript** — recomputes the abstract and Table 2 mean-row constants
  from the Exp10 JSON and checks them against the values in
  `verify_tables.py` (`MANUSCRIPT_CLAIMS`). If you change the paper's numbers,
  update that dictionary too.
