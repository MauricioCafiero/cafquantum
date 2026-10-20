"""Unit tests: element guards, inputs, weight reporting. No model loading."""
import ase
import numpy as np
import pytest


def test_methods_tuple():
    from cafquant import METHODS
    assert METHODS == ("uma", "maceoff", "egret", "aimnet2", "gfn2", "pm7",
                       "polar1-s", "polar1-m", "omol", "uma1p2")


def test_element_guard_maceoff_rejects_metals():
    from cafquant.mlips import element_guard
    atoms = ase.Atoms("AuH", positions=[[0, 0, 0], [0, 0, 1.5]])
    ok, why = element_guard("maceoff", atoms)
    assert not ok and "Au" in why


def test_element_guard_egret_rejects_silicon():
    from cafquant.mlips import element_guard
    atoms = ase.Atoms("SiH4", positions=np.zeros((5, 3)))
    ok, why = element_guard("egret", atoms)
    assert not ok and "Si" in why


def test_element_guard_egret_accepted_set():
    from cafquant.mlips import element_guard, EGRET_ELEMENTS
    assert EGRET_ELEMENTS == {"H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"}


def test_element_guard_gfn2_rejects_heavy():
    from cafquant.mlips import element_guard
    atoms = ase.Atoms("U", positions=[[0, 0, 0]])
    ok, why = element_guard("gfn2", atoms)
    assert not ok and "consider uma" in why


def test_element_guard_uma_covers_everything():
    from cafquant.mlips import element_guard
    ok, _ = element_guard("uma", ase.Atoms("Au", positions=[[0, 0, 0]]))
    assert ok


def test_weights_report_shape():
    from cafquant import METHODS
    from cafquant.weights import check_weights
    report = check_weights()
    assert set(report) == set(METHODS)
    for rec in report.values():
        assert isinstance(rec["ok"], bool) and isinstance(rec["detail"], str)


def test_egret_weight_is_repo_local():
    """The egret model ships in the repo weights/ dir (or CAFQUANT_EGRET_MODEL)."""
    from cafquant.mlips import EGRET_DEFAULT_MODEL
    from cafquant.weights import check_weights
    report = check_weights()
    assert "weights/EGRET_1.model" in report["egret"]["detail"] or report["egret"]["ok"]
    pytest.importorskip("torch")  # model file sanity needs the loaders only via size
    import pathlib
    assert pathlib.Path(EGRET_DEFAULT_MODEL).stat().st_size > 1_000_000


def test_smiles_pipeline():
    from cafquant.inputs import load_structure
    atoms, label, src = load_structure("CCO", charge=0, multiplicity=1)
    assert label and src.startswith("smiles:CCO")
    assert sorted(atoms.get_chemical_symbols()) == sorted(["C", "C", "O"] + ["H"] * 6)
    assert atoms.info["charge"] == 0 and atoms.info["spin"] == 1


def test_xyz_input():
    from cafquant.inputs import load_structure
    import pathlib
    xyz = pathlib.Path(__file__).resolve().parent.parent / "examples" / "water.xyz"
    atoms, label, _ = load_structure(str(xyz), charge=0, multiplicity=1)
    assert sorted(atoms.get_chemical_symbols()) == ["H", "H", "O"]