"""Input handling: XYZ file, SDF file, or SMILES string -> ASE Atoms.

SMILES pipeline (as specified): RDKit add Hs -> embed 5 conformers (ETKDGv3)
-> MMFF-optimize each -> keep the lowest-energy conformer. Ported from the
proven implementation in /Users/cafierom/python_mac/aimnet2/code/aimnet_calc.py.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read as ase_read


def load_structure(source: str, charge: int | None = None, multiplicity: int = 1,
                   n_confs: int = 5, seed: int = 42) -> tuple[Atoms, str]:
    """Dispatch on input type. Returns (atoms, label).

    ``source`` is treated as a file path if it ends with .xyz/.sdf (or exists
    as a file); otherwise as a SMILES string.
    """
    s = source.strip()
    if s.lower().endswith(".xyz") or (Path(s).is_file() and not _looks_like_smiles(s)):
        if not Path(s).is_file():
            raise FileNotFoundError(s)
        atoms = ase_read(s)
        label = Path(s).stem
        src = f"file:{s}"
    elif s.lower().endswith(".sdf"):
        atoms = read_sdf(s)
        label = Path(s).stem
        src = f"file:{s}"
    else:
        atoms, src = smiles_to_atoms(s, n_confs=n_confs, seed=seed)
        label = _sanitize_smiles(s)

    if charge is not None:
        atoms.info["charge"] = int(charge)
    atoms.info.setdefault("charge", 0)
    atoms.info["spin"] = int(multiplicity)
    if multiplicity > 1:
        # MOPAC (ase.calculators.mopac) reads unpaired electrons from initial magmoms;
        # UMA omol / AIMNet2 read atoms.info["spin"] (multiplicity).
        atoms.set_initial_magnetic_moments(np.full(len(atoms), multiplicity - 1.0))
    return atoms, label, src


def _looks_like_smiles(s: str) -> bool:
    """Rough check: bare strings without a dot-separated extension are SMILES."""
    return not Path(s).suffix and (" " not in s or "=" in s or "(" in s)


def _sanitize_smiles(smiles: str) -> str:
    """Compact filesystem-safe label from a SMILES string (e.g. CCO -> CCO)."""
    return re.sub(r"[^A-Za-z0-9]+", "", smiles)[:64] or "mol"


def smiles_to_atoms(smiles: str, n_confs: int = 5, seed: int = 42,
                    minimize_iters: int = 500) -> tuple[Atoms, str]:
    """SMILES -> add Hs -> embed n_confs (ETKDGv3) -> MMFF-optimize -> lowest.

    Returns (atoms, source-string). Formal charge is read off the SMILES and
    stored in atoms.info["charge"]. Falls back to UFF when MMFF has no
    parameters for the element set.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")
    charge = Chem.GetFormalCharge(mol)
    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    cids = AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params)
    if not cids:
        raise RuntimeError(f"RDKit failed to embed any conformers for {smiles!r}")

    energies: list[tuple[float, int, str]] = []  # (energy, conf_id, method)
    for cid in cids:
        props = AllChem.MMFFGetMoleculeProperties(mol)
        if props is None:
            ff = AllChem.UFFGetMoleculeForceField(mol, confId=cid)
            method_used = "uff"
        else:
            ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=cid)
            method_used = "mmff"
        ff.Minimize(maxIts=minimize_iters)
        energies.append((ff.CalcEnergy(), cid, method_used))

    best_e, best_cid, best_method = min(energies)
    conf = mol.GetConformer(best_cid)

    atoms = Atoms(
        [a.GetSymbol() for a in mol.GetAtoms()],
        positions=[list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())],
    )
    atoms.info["smiles"] = smiles
    atoms.info["charge"] = charge
    atoms.info["conformer_energy_kcal"] = float(best_e)
    atoms.info["forcefield"] = best_method
    source = (f"smiles:{smiles} | {n_confs} confs, best {best_method} E={best_e:.3f} kcal/mol")
    return atoms, source


def read_sdf(path: str) -> Atoms:
    """First molecule of an SDF. Uses the stored 3D conformer if present,
    otherwise embeds one and MMFF-minimizes."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.SDMolSupplier(str(path))[0]
    if mol is None:
        raise ValueError(f"RDKit could not read SDF: {path}")
    charge = Chem.GetFormalCharge(mol)
    conf = mol.GetConformer() if mol.GetNumConformers() and mol.GetConformer().Is3D() else None
    if conf is None:
        mol = Chem.AddHs(mol)
        props = AllChem.MMFFGetMoleculeProperties(mol)
        cid = AllChem.EmbedMolecule(mol, randomSeed=42)
        if cid != 0:
            raise RuntimeError(f"failed to embed 3D geometry for {path}")
        if props is not None:
            ff = AllChem.MMFFGetMoleculeForceField(mol, props)
            ff.Minimize(maxIts=500)
        conf = mol.GetConformer()
    atoms = Atoms(
        [a.GetSymbol() for a in mol.GetAtoms()],
        positions=[list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())],
    )
    atoms.info["charge"] = charge
    return atoms