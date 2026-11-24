#!/usr/bin/env bash
# One command: regenerate all JSON -> export tables -> verify.
# Run from the repo root (the directory that CONTAINS the `aor/` folder).
#
#   bash scripts/run_pipeline.sh            # full 7-model run on cuda
#   bash scripts/run_pipeline.sh --selftest # offline wiring check (no GPU)
#   bash scripts/run_pipeline.sh --dry_run  # print commands only
#
# Any extra args are forwarded to run_pipeline.py.
set -euo pipefail
export PYTHONNOUSERSITE=1

ROOT="${ACL_ROOT:-results}"
TABLES="${ACL_TABLES:-tables}"
DEVICE="${ACL_DEVICE:-cuda}"
DATA_DIR="${ACL_DATA_DIR:-aor/data/splits}"

exec python -s aor/experiments/run_pipeline.py \
    --root "$ROOT" \
    --tables "$TABLES" \
    --data_dir "$DATA_DIR" \
    --device "$DEVICE" \
    "$@"
