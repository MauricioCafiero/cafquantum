"""Tasks: single-point, geometry optimization, frequencies, thermochemistry.

All run through the ASE driver regardless of backend. Energies in eV (ASE
native), also reported in kcal/mol and Ha. MOPAC's "energy" is its heat of
formation, so energies are comparable within a method family, not between them.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from ase import Atoms, units
from ase.optimize import BFGS
from ase.thermochemistry import IdealGasThermo
from ase.vibrations import Vibrations

log = logging.getLogger("cafquant")

# Methods needing torch: run in their own worker process. tblite (gfn2) links
# libgomp, torch ships libomp -- the two OpenMP runtimes cannot be initialized
# in one process on macOS, even with KMP_DUPLICATE_LIB_OK=TRUE.
TORCH_METHODS = ("uma", "maceoff", "egret", "aimnet2")

KCAL = units.kcal / units.mol
HA = units.Ha


def singlepoint(atoms: Atoms, calc) -> dict:
    """Energy (eV) and max force (eV/A)."""
    atoms.calc = calc
    energy = float(atoms.get_potential_energy())
    fmax = float(np.linalg.norm(np.asarray(atoms.get_forces()), axis=1).max())
    return {"status": "ok", "energy_eV": energy, "energy_kcal": energy / KCAL,
            "energy_ha": energy / HA, "fmax": fmax}


def optimize(atoms: Atoms, calc, *, fmax: float = 0.01, steps: int = 1000,
             rundir: Path, method: str) -> dict:
    """Relax in place with BFGS; write opt_<method>.xyz + .traj."""
    traj = rundir / f"opt_{method}.traj"
    atoms.calc = calc
    dyn = BFGS(atoms, trajectory=str(traj), logfile=str(rundir / f"opt_{method}.log"))
    converged = bool(dyn.run(fmax=fmax, steps=steps))
    energy = float(atoms.get_potential_energy())
    fmax_reached = float(np.linalg.norm(np.asarray(atoms.get_forces()), axis=1).max())

    from ase.io import write as ase_write
    opt_xyz = rundir / f"opt_{method}.xyz"
    ase_write(str(opt_xyz), atoms, format="xyz",
              comment=f"{method} optimized, E={energy:.6f} eV, fmax={fmax_reached:.4f}")

    return {"status": "ok" if converged else "not_converged", "converged": converged,
            "energy_eV": energy, "energy_kcal": energy / KCAL, "energy_ha": energy / HA,
            "fmax": fmax_reached,
            "xyz": str(opt_xyz), "traj": str(traj)}


def frequencies(atoms: Atoms, calc, *, rundir: Path, method: str,
                delta: float = 0.01, nfree: int = 2) -> dict:
    """Harmonic frequencies (cm^-1, imaginary as negative) by finite differences.

    Run on a relaxed geometry. The displacement cache is wiped first -- stale
    caches from a killed run make ASE crash. Writes freq_<method>.dat.
    """
    atoms.calc = calc
    vib = Vibrations(atoms, nfree=nfree, delta=delta, name=str(rundir / f"freq_{method}"))
    vib.clean()
    vib.run()
    freqs = np.asarray(vib.get_frequencies()).real
    vib_energies = np.asarray(vib.get_energies())  # eV

    lines = [f"{method} harmonic frequencies, {len(freqs)} modes (cm^-1, imaginary < 0):",
             "  " + " ".join(f"{f:8.1f}" for f in freqs)]
    n_imag = int((freqs < -20).sum())
    n_soft = int(((freqs >= -20) & (freqs < 20)).sum())
    if n_imag:
        lines.append(f"WARNING: {n_imag} imaginary mode(s), min {freqs.min():.1f} cm^-1 "
                     "-- not a true minimum")
    if n_soft:
        lines.append(f"NOTE: {n_soft} near-zero mode(s) (0-20 cm^-1) -- conformational "
                     "soft mode or loose optimization; check geometry")
    text = "\n".join(lines)
    log.info("%s", text)
    (rundir / f"freq_{method}.dat").write_text(text + "\n")

    return {"status": "ok", "min_freq_cm1": float(freqs[freqs > 20].min())
            if (freqs > 20).any() else float(freqs.min()),
            "n_imag": n_imag, "n_soft": n_soft,
            "vib_energies_eV": vib_energies, "freqs_cm1": freqs}


def thermochemistry(atoms: Atoms, vib_energies_eV: np.ndarray, opt_energy_eV: float,
                    symmetrynumber: int = 1, T: float = 298.15) -> dict:
    """Ideal-gas thermochemistry at ``T`` (298.15 K default).

    Geometry class from moments of inertia (monatomic / linear / nonlinear).
    ``ignore_imag_modes=True``: imaginary modes above the 3N-6 cut are dropped
    (a genuine structure check is already reported by n_imag in the summary).
    """
    natoms = len(atoms)
    if natoms == 1:
        geometry = "monatomic"
    else:
        moments = sorted(atoms.get_moments_of_inertia())
        if moments[0] < 1e-4 and moments[1] < 1e-4:  # only Izz appreciable
            geometry = "linear"
        else:
            geometry = "nonlinear"
    spin = float(atoms.info.get("spin", 1) - 1) / 2.0
    P = 1e5  # pascal; 1 bar default (ASE 3.29 requires explicit pressure)
    thermo = IdealGasThermo(vib_energies_eV, geometry=geometry,
                            potentialenergy=opt_energy_eV, atoms=atoms,
                            symmetrynumber=symmetrynumber, spin=spin,
                            ignore_imag_modes=True)
    return {"zpe_eV": float(thermo.get_ZPE_correction()),
            "enthalpy_eV": float(thermo.get_enthalpy(T, verbose=False)),
            "entropy_eV_K": float(thermo.get_entropy(T, pressure=P, verbose=False)),
            "gibbs_eV": float(thermo.get_gibbs_energy(T, P, verbose=False))}


def run_method(method: str, atoms: Atoms, tasks: list[str], rundir: Path,
               *, charge: int = 0, multiplicity: int = 1, device: str = "cpu",
               fmax: float = 0.01, steps: int = 1000) -> dict:
    """Run sp (+ opt + freq on request) for one method; never raises.

    'opt' implies an sp first (the pre-optimization energy); 'freq' implies opt.
    Returns the summary row for this method.
    """
    from .mlips import element_guard, make_calculator

    row: dict = {"method": method, "status": "ok"}
    want_sp, want_opt, want_freq = ("sp" in tasks, "opt" in tasks, "freq" in tasks)
    if want_freq and not want_opt:
        want_opt = True

    try:
        ok, why = element_guard(method, atoms)
        if not ok:
            log.warning("%s skipped: %s", method, why)
            row.update({"status": "skipped", "error": why})
            return row

        def _calc():
            return make_calculator(method, charge=charge, multiplicity=multiplicity,
                                   label=f"{atoms.info.get('label', 'mol')}.pm7",
                                   workdir=rundir, device=device)

        if want_sp or want_opt:
            sp = singlepoint(atoms, _calc())
            row.update({k: sp[k] for k in ("energy_eV", "energy_kcal", "energy_ha", "fmax")})
            log.info("%s sp: E=%.6f eV (%.3f kcal/mol)", method,
                     sp["energy_eV"], sp["energy_kcal"])

        if want_opt:
            opt = optimize(atoms, _calc(), fmax=fmax, steps=steps,
                           rundir=rundir, method=method)
            row.update({k: opt[k] for k in ("converged", "energy_eV", "energy_kcal",
                                            "energy_ha", "fmax", "xyz", "traj")}
                       )
            row["status"] = opt["status"]
            log.info("%s opt: E=%.6f eV, fmax=%.4f, converged=%s",
                     method, opt["energy_eV"], opt["fmax"], opt["converged"])

        if want_freq:
            vib = frequencies(atoms, _calc(), rundir=rundir, method=method)
            thermo = thermochemistry(atoms, vib["vib_energies_eV"], row["energy_eV"])
            row.update({k: vib[k] for k in ("min_freq_cm1", "n_imag", "n_soft")})
            row.update({k: thermo[k] for k in ("zpe_eV", "enthalpy_eV",
                                               "entropy_eV_K", "gibbs_eV")})
            if vib["n_imag"]:
                row["status"] = "ok_imag_modes"
            log.info("%s freq: min=%.1f cm^-1, n_imag=%d, G=%.6f eV",
                     method, vib["min_freq_cm1"], vib["n_imag"], thermo["gibbs_eV"])

    except Exception as exc:  # one bad backend must not kill the run
        log.warning("%s failed: %s: %s", method, type(exc).__name__, exc)
        row.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
    return row