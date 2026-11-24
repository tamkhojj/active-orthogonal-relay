"""Make the repository root importable so `from aor... import ...` works
during `pytest` without an editable install."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
