"""Make `cafquant` importable for the test suite (code/ is not installed)."""
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent / "code"
sys.path.insert(0, str(CODE))

# worker subprocesses launched by the CLI inherit PYTHONPATH from here
import os  # noqa: E402

os.environ["PYTHONPATH"] = str(CODE) + os.pathsep + os.environ.get("PYTHONPATH", "")