"""Calculator factory: all five backends as ASE calculators.

A fresh calculator is built per (method, task) so no state leaks between jobs;
the Atoms object carries over. Charge/spin conventions:
  - uma (omol): atoms.info["charge"], atoms.info["spin"] = multiplicity
  - aimnet2:    passed at construction (charge, mult)
  - gfn2:       passed to TBLite (charge, multiplicity)
  - pm7:        MOPAC charge kwarg; multiplicity>1 via initial magmoms (set in inputs)
"""
from __future__ import annotations

import shutil
from pathlib import Path

# MACE-OFF23 element coverage: fails fast on unsupported elements instead of
# producing a garbage energy.
MACEOFF_ELEMENTS = {"H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Br", "I"}

# GFN2-xTB (tblite) parameter coverage runs to Z=86.
GFN2_MAX_Z = 86


def maceoff_supported(atoms) -> tuple[bool, str]:
    unsupported = sorted(set(atoms.get_chemical_symbols()) - MACEOFF_ELEMENTS)
    return (not unsupported, f"MACE-OFF23 does not cover: {', '.join(unsupported)}")


def element_guard(method: str, atoms) -> tuple[bool, str]:
    """(ok, reason) for methods with restricted element coverage.

    aimnet2 raises its own descriptive error from model metadata
    (implemented_species / charged-system checks); uma covers the full table;
    pm7 covers essentially everything H-Pu.
    """
    if method == "maceoff":
        return maceoff_supported(atoms)
    if method == "gfn2":
        from ase.data import atomic_numbers, chemical_symbols
        bad = sorted({s for s in set(atoms.get_chemical_symbols())
                      if atomic_numbers[s] > GFN2_MAX_Z})
        return (not bad,
                f"GFN2-xTB covers elements up to Z=86 (Rn); does not cover: {', '.join(bad)}"
                " (consider uma for heavy elements)" if bad else "")
    return True, ""


def make_calculator(method: str, *, charge: int = 0, multiplicity: int = 1,
                    label: str = "mol", workdir: str | Path = ".",
                    device: str = "cpu"):
    """Build the ASE calculator for ``method`` and return it."""
    if method == "uma":
        import cafquant.env
        cafquant.env.patch_torch_for_e3nn()
        from fairchem.core import FAIRChemCalculator
        from fairchem.core.calculate.pretrained_mlip import get_predict_unit
        predictor = get_predict_unit("uma-s-1p1", device=device)
        return FAIRChemCalculator(predictor, task_name="omol")  # charge/spin from atoms.info

    if method == "maceoff":
        import cafquant.env
        cafquant.env.patch_torch_for_e3nn()
        from mace.calculators.foundations_models import mace_off
        return mace_off(model="small", device=device, default_dtype="float64")

    if method == "aimnet2":
        from aimnet.calculators import AIMNet2ASE, AIMNet2Calculator
        base = AIMNet2Calculator("aimnet2-2025", device=device)
        return AIMNet2ASE(base, charge=charge, mult=multiplicity, validate_species=True)

    if method == "gfn2":
        from tblite.ase import TBLite
        return TBLite(method="GFN2-xTB", charge=float(charge),
                      multiplicity=multiplicity if multiplicity > 1 else None,
                      accuracy=1.0, verbosity=0)

    if method == "pm7":
        from ase.calculators.mopac import MOPAC
        exe = shutil.which("mopac") or str(Path.home() / ".local/bin/mopac")
        workdir = Path(workdir) / "pm7"
        workdir.mkdir(parents=True, exist_ok=True)
        return MOPAC(command=f"{exe} PREFIX.mop", directory=str(workdir),
                     label=label, method="PM7", task="1SCF GRADIENTS PRECISE",
                     charge=charge)

    raise ValueError(f"unknown method: {method}")