"""Calculator factory: all ten backends as ASE calculators.

A fresh calculator is built per (method, task) so no state leaks between jobs;
the Atoms object carries over. Charge/spin conventions:
  - uma / uma1p2 / omol / polar1-s / polar1-m (omol-style models):
        atoms.info["charge"], atoms.info["spin"] = multiplicity
  - maceoff/egret: MACECalculator(charges_key="charge") reads atoms.info["charge"]
  - aimnet2:    passed at construction (charge, mult)
  - gfn2:       passed to TBLite (charge, multiplicity)
  - pm7:        MOPAC charge kwarg; multiplicity>1 via initial magmoms (set in inputs)
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

# MACE-OFF23 element coverage: fails fast on unsupported elements instead of
# producing a garbage energy.
MACEOFF_ELEMENTS = {"H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Br", "I"}

# Egret-1 element coverage, read from the compiled model's own atomic_numbers
# (H, C, N, O, F, P, S, Cl, Br, I).
EGRET_ELEMENTS = {"H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"}

# Default location of the Egret-1 compiled MACE model: the repo-local
# weights/ dir (copy of EGRET_1.model; EGRET_1E/1T can be dropped there too).
# Override with the CAFQUANT_EGRET_MODEL env var.
EGRET_DEFAULT_MODEL = str(Path(__file__).resolve().parents[2] / "weights" / "EGRET_1.model")

# GFN2-xTB (tblite) parameter coverage runs to Z=86.
GFN2_MAX_Z = 86


def maceoff_supported(atoms) -> tuple[bool, str]:
    unsupported = sorted(set(atoms.get_chemical_symbols()) - MACEOFF_ELEMENTS)
    return (not unsupported, f"MACE-OFF23 does not cover: {', '.join(unsupported)}")


def egret_supported(atoms) -> tuple[bool, str]:
    unsupported = sorted(set(atoms.get_chemical_symbols()) - EGRET_ELEMENTS)
    return (not unsupported, f"Egret-1 does not cover: {', '.join(unsupported)}")


def _patch_polar_gto_kwarg() -> None:
    """mace 0.3.16's polar path passes ``force_pbc_evaluator`` into
    GTOElectrostaticFeatures.precompute_geometry, which the graph_longrange
    version mace 0.3.16 pairs with (v0.4.0) does accept -- but if a newer
    graph_longrange is installed instead, accept-and-drop the retired kwarg
    (mirrors external_source_features.py; default is False, no-op for our
    non-periodic frames). Idempotent."""
    import inspect
    from graph_longrange.features import GTOElectrostaticFeatures
    orig = GTOElectrostaticFeatures.precompute_geometry
    if "force_pbc_evaluator" in inspect.signature(orig).parameters:
        return

    def patched(self, *args, force_pbc_evaluator=False, **kwargs):
        return orig(self, *args, **kwargs)

    GTOElectrostaticFeatures.precompute_geometry = patched


class _PinnedNBList:
    """Guard around AIMNet2's AdaptiveNeighborList: resets the neighbour-buffer
    size before every call so the hysteresis shrink cannot leave one frame with
    a buffer too small for the next frame's neighbour count (which truncates
    silently and corrupts energies)."""


    def __init__(self, inner, n: int) -> None:
        self._inner = inner
        self._n = n

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def __call__(self, *args, **kwargs):
        self._inner.max_neighbors = self._n
        return self._inner(*args, **kwargs)


class _EgretForwardAdapter:
    """Adapter for compiled Egret-1 models (written against mace-torch 0.3.12)
    under the installed mace-torch 0.3.16 calculator, whose forward call passes
    compute_edge_forces/compute_atomic_stresses kwargs the older TorchScript
    schema does not accept. Keeps the shared mace install untouched; only the
    per-call kwargs entering the Egret module are filtered."""

    def __init__(self, module):
        self._module = module  # already dtype-cast by MACECalculator

    def __getattr__(self, name):  # r_max, atomic_numbers, heads, ...
        return getattr(self._module, name)

    def forward(self, data, training=False, **kwargs):
        return self._module(data, training=training)

    __call__ = forward

    def extra_repr(self):
        return "EgretForwardAdapter"


def element_guard(method: str, atoms) -> tuple[bool, str]:
    """(ok, reason) for methods with restricted element coverage.

    aimnet2 raises its own descriptive error from model metadata
    (implemented_species / charged-system checks); uma covers the full table;
    pm7 covers essentially everything H-Pu.
    """
    if method == "maceoff":
        return maceoff_supported(atoms)
    if method == "egret":
        return egret_supported(atoms)
    if method == "gfn2":
        from ase.data import atomic_numbers, chemical_symbols
        bad = sorted({s for s in set(atoms.get_chemical_symbols())
                      if atomic_numbers[s] > GFN2_MAX_Z})
        return (not bad,
                f"GFN2-xTB covers elements up to Z=86 (Rn); does not cover: {', '.join(bad)}"
                " (consider uma for heavy elements)" if bad else "")
    return True, ""


def find_omol_model() -> str | None:
    """MACE-OMOL-0 extra-large weights: repo weights/ first, then the mace cache."""
    local = Path(__file__).resolve().parents[2] / "weights" / "MACE-omol-0-extra-large-1024.model"
    if local.is_file():
        return str(local)
    cache = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "mace"
    hits = sorted(cache.glob("MACE-omol*"))
    return str(hits[0]) if hits else None


def make_calculator(method: str, *, charge: int = 0, multiplicity: int = 1,
                    label: str = "mol", workdir: str | Path = ".",
                    device: str = "cpu", mixer_damping: float = 0.4):
    """Build the ASE calculator for ``method`` and return it."""
    if method == "uma":
        import cafquant.env
        cafquant.env.patch_torch_for_e3nn()
        from fairchem.core import FAIRChemCalculator
        from fairchem.core.calculate.pretrained_mlip import get_predict_unit
        predictor = get_predict_unit("uma-s-1p1", device=device)
        return FAIRChemCalculator(predictor, task_name="omol")  # charge/spin from atoms.info

    if method == "uma1p2":
        import cafquant.env
        cafquant.env.patch_torch_for_e3nn()
        from fairchem.core import FAIRChemCalculator
        from fairchem.core.calculate.pretrained_mlip import get_predict_unit
        predictor = get_predict_unit("uma-s-1p2", device=device)
        return FAIRChemCalculator(predictor, task_name="omol")  # charge/spin from atoms.info

    if method in ("polar1-s", "polar1-m"):
        import cafquant.env
        cafquant.env.patch_torch_for_e3nn()
        _patch_polar_gto_kwarg()
        from mace.calculators import MACECalculator
        from mace.calculators.foundations_models import download_mace_polar_checkpoint
        name = {"polar1-s": "polar-1-s", "polar1-m": "polar-1-m"}[method]
        path = download_mace_polar_checkpoint(name)
        return MACECalculator(model_paths=[path], device=device,
                              default_dtype="float64", charges_key="charge")

    if method == "omol":
        import cafquant.env
        cafquant.env.patch_torch_for_e3nn()
        from mace.calculators.foundations_models import mace_omol
        w = find_omol_model()
        return mace_omol(model=w if w else "extra_large",
                         device=device, default_dtype="float64")

    if method == "maceoff":
        import cafquant.env
        cafquant.env.patch_torch_for_e3nn()
        from mace.calculators.foundations_models import mace_off
        return mace_off(model="small", device=device, default_dtype="float64")

    if method == "egret":
        import os
        import cafquant.env
        cafquant.env.patch_torch_for_e3nn()
        from mace.calculators import MACECalculator

        model_path = os.environ.get("CAFQUANT_EGRET_MODEL", EGRET_DEFAULT_MODEL)
        calc = MACECalculator(model_paths=[model_path], device=device,
                              default_dtype="float64", charges_key="charge")
        calc.models = [_EgretForwardAdapter(m) for m in calc.models]
        return calc

    if method == "aimnet2":
        from aimnet.calculators import AIMNet2ASE, AIMNet2Calculator
        base = AIMNet2Calculator("aimnet2-2025", device=device)
        calc = AIMNet2ASE(base, charge=charge, mult=multiplicity, validate_species=True)
        # AIMNet2ASE's adaptive neighbour list shrinks its buffer after small
        # frames (hysteresis) and then silently truncates the neighbour list on
        # denser frames -- energies shift by tens of eV depending on which
        # frames ran before. Verified converged at 64/128/256 on the sult
        # frames (and matching the other backends), so pin a fixed buffer.
        base._nblist = _PinnedNBList(base._nblist, 128)
        return calc

    if method == "gfn2":
        from tblite.ase import TBLite
        return TBLite(method="GFN2-xTB", charge=float(charge),
                      multiplicity=multiplicity if multiplicity > 1 else None,
                      accuracy=1.0, max_iterations=500,
                      mixer_damping=mixer_damping, verbosity=0)

    if method == "pm7":
        from ase.calculators.mopac import MOPAC
        exe = shutil.which("mopac") or str(Path.home() / ".local/bin/mopac")
        workdir = Path(workdir) / "pm7"
        workdir.mkdir(parents=True, exist_ok=True)
        return MOPAC(command=f"{exe} PREFIX.mop", directory=str(workdir),
                     label=label, method="PM7", task="1SCF GRADIENTS PRECISE",
                     charge=charge)

    raise ValueError(f"unknown method: {method}")