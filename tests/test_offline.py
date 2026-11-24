"""Pytest entry point for the offline test suite (no GPU, no network).

- `test_package_imports` checks the package is importable and the stub backend
  builds.
- `test_orchestrator_selftest` runs every experiment script's `--selftest`
  through the orchestrator and asserts success.
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_package_imports():
    import aor  # noqa: F401
    from aor.interp import reco_backend as B

    bk = B.build_stub_backend()
    assert bk.is_stub and bk.n_layers == 10


def test_orchestrator_selftest():
    env = dict(os.environ)
    env.setdefault("PYTHONNOUSERSITE", "1")
    result = subprocess.run(
        [sys.executable, "-I", "aor/experiments/run_pipeline.py", "--selftest"],
        cwd=ROOT, env=env,
    )
    assert result.returncode == 0
