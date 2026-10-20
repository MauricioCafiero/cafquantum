"""End-to-end test: full CLI run (six methods, all in their worker subprocesses)
on the water example, checking statuses and reference energies. One CLI run
serves every assertion here."""
import csv
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
PYTHON = REPO / ".venv" / "bin" / "python"
CLI = REPO / "code" / "cafquant.py"

# reference values on this machine (see README notes): PM7 heat of formation
# -57.8 kcal/mol for water; GFN2-xTB water ~ -137.97 eV
PM7_KCAL, GFN2_EV = -57.80, -137.97


@pytest.fixture(scope="module")
def rows(tmp_path_factory):
    out = tmp_path_factory.mktemp("e2e")
    cmd = [sys.executable, "-u", str(CLI), "examples/water.xyz", "-t", "sp",
           "-q", "-m", "uma,maceoff,egret,aimnet2,gfn2,pm7",
           "--out", str(out / "water")]
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, f"CLI exit {proc.returncode}\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
    summary_csv = out / "water" / "summary.csv"  # --out IS the rundir
    assert summary_csv.is_file(), f"no summary.csv:\n{proc.stdout[-1000:]}"
    with summary_csv.open() as fh:
        # "Energy_eV" -> "energy_eV": lower only the first letter of each key
        return {r["Method"].lower(): {k[:1].lower() + k[1:]: v for k, v in r.items()}
                for r in csv.DictReader(fh)}


def test_all_methods_present_and_ok(rows):
    assert set(rows) == {"uma", "maceoff", "egret", "aimnet2", "gfn2", "pm7"}
    for method, row in rows.items():
        assert row["status"] == "ok", f"{method} failed: {row.get('error', '')}"


def test_default_method_is_gfn2(tmp_path):
    out = tmp_path / "water_gfn2"
    cmd = [sys.executable, "-u", str(CLI), "examples/water.xyz", "-t", "sp",
           "-q", "--out", str(out)]
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    with (out / "summary.csv").open() as fh:
        rows = [r["Method"].lower() for r in csv.DictReader(fh)]
    assert rows == ["gfn2"], f"default must run gfn2 only, got {rows}"


def test_reference_energies(rows):
    pm7 = float(rows["pm7"]["energy_kcal"])
    gfn2 = float(rows["gfn2"]["energy_eV"])
    assert abs(pm7 - PM7_KCAL) < 0.05, f"PM7 {pm7} vs {PM7_KCAL}"
    assert abs(gfn2 - GFN2_EV) < 0.02, f"GFN2 {gfn2} vs {GFN2_EV}"
    # egret and maceoff are sibling MACE models: must agree to ~10 meV
    egret, maceoff = float(rows["egret"]["energy_eV"]), float(rows["maceoff"]["energy_eV"])
    assert abs(egret - maceoff) < 0.01
    # MLIPs cluster ~2079-2082 eV for water
    for m in ("uma", "maceoff", "egret", "aimnet2"):
        e = float(rows[m]["energy_eV"])
        assert -2082 <= e <= -2078, f"{m} energy {e} outside expected range"