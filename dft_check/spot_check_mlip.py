"""Spot-check charge-aware MLIPs against the dft_check DFT delta3 values.

Runs the chosen backends over the 35 dft_check frames (q/M parsed from
each frame's .inp `*xyzfile` line), stores per-frame energies as JSON,
then evaluates delta3 per spot triple next to the DFT column of
dft_delta3.tsv.

Run from the repo root:
    .venv/bin/python dft_check/spot_check_mlip.py polar1-s polar1-m omol uma-1p2

Idempotent/skip-if-done: existing per-(method, frame) energies in
mlip_check/<method>.sp.json are reused. Results: mlip_check/<method>.sp.json,
mlip_check/delta3_compare.tsv.
"""
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

REPO = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "code"))

EV_HARTREE = 27.211386245988
EH_KCAL = 627.509474

# stem -> stems playing each role (same as dft_check/eval.py)
_spec = importlib.util.spec_from_file_location("dft_eval", HERE / "eval.py")
dft_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dft_eval)
TRIPLES = dft_eval.TRIPLES

PATTERN = re.compile(r"\*xyzfile\s+(-?\d+)\s+(\d+)")


def parse_qm(stem: str) -> tuple[int, int]:
    inp = (HERE / stem.split("_")[0] / f"{stem}.inp").read_text(errors="replace")
    m = PATTERN.search(inp)
    if not m:
        raise ValueError(f"no *xyzfile line in {stem}.inp")
    return int(m.group(1)), int(m.group(2))


def _patch_gto_retired_kwarg() -> None:
    """mace 0.3.16's polar path passes force_pbc_evaluator into
    GTOElectrostaticFeatures.precompute_geometry, which no released
    graph_longrange accepts (its other class pops it as 'retired').
    Default is False, i.e. no-op for our non-periodic frames; patch to
    accept-and-drop, mirroring external_source_features.py."""
    import inspect
    from graph_longrange import features
    orig = features.GTOElectrostaticFeatures.precompute_geometry
    if "force_pbc_evaluator" in inspect.signature(orig).parameters:
        return

    def patched(self, *args, force_pbc_evaluator=False, **kwargs):
        return orig(self, *args, **kwargs)

    features.GTOElectrostaticFeatures.precompute_geometry = patched


def build_calc(method: str):
    from cafquant.env import patch_torch_for_e3nn
    patch_torch_for_e3nn()
    if method.startswith("polar"):
        _patch_gto_retired_kwarg()
        from mace.calculators import MACECalculator
        from mace.calculators.foundations_models import download_mace_polar_checkpoint
        name = {"polar1-s": "polar-1-s", "polar1-m": "polar-1-m"}[method]
        path = download_mace_polar_checkpoint(name)
        return MACECalculator(model_paths=[path], device="cpu",
                              default_dtype="float64", charges_key="charge")
    if method == "omol":
        from cafquant.mlips import find_omol_model
        from mace.calculators.foundations_models import mace_omol
        w = find_omol_model()
        if w is None:
            raise RuntimeError("no MACE-omol-0 model found; run with --download")
        return mace_omol(model=w, device="cpu", default_dtype="float64")
    if method.startswith("uma-"):
        from fairchem.core import FAIRChemCalculator
        from fairchem.core.calculate.pretrained_mlip import get_predict_unit
        name = {"uma-1p2": "uma-s-1p2", "uma-1p1": "uma-s-1p1"}[method]
        return FAIRChemCalculator(get_predict_unit(name, device="cpu"),
                                  task_name="omol")
    raise ValueError(f"unknown method: {method}")


def dft_column() -> dict[str, float]:
    tsv = (HERE / "dft_delta3.tsv").read_text().strip().splitlines()
    out = {}
    for line in tsv[1:]:
        parts = line.split("\t")
        if parts[1]:
            out[parts[0]] = float(parts[1].replace("−", "-"))
    return out


def main() -> None:
    methods = sys.argv[1:] or ["polar1-s"]
    outdir = HERE / "mlip_check"
    outdir.mkdir(exist_ok=True)
    stems = sorted(p.stem for sysdir in ("sult", "ddc", "comt")
                   for p in (HERE / sysdir).glob("*.xyz"))
    print(f"frames: {len(stems)}, methods: {methods}")

    for method in methods:
        jf = outdir / f"{method}.sp.json"
        cache = json.loads(jf.read_text()) if jf.exists() else {}
        calc = build_calc(method)
        for i, stem in enumerate(stems):
            if stem in cache:
                continue
            atoms = __import__("ase.io", fromlist=["read"]).read(
                HERE / stem.split("_")[0] / f"{stem}.xyz")
            q, m = parse_qm(stem)
            atoms.info["charge"] = q
            atoms.info["spin"] = m
            atoms.calc = calc
            e = atoms.get_potential_energy()  # eV
            cache[stem] = float(e)
            jf.write_text(json.dumps(cache, indent=1))
            print(f"[{method}] {i+1}/{len(stems)} {stem} q={q} M={m}: {e:.6f} eV",
                  flush=True)
        del calc

    dft = dft_column()
    methods = sorted(p.name[:-8] for p in outdir.glob("*.sp.json"))
    rows = ["triple\tdft\t" + "\t".join(f"{m}\t{m}_err" for m in methods)]
    for tname, parts in TRIPLES.items():
        vals = []
        for method in methods:
            jf = outdir / f"{method}.sp.json"
            cache = json.loads(jf.read_text()) if jf.exists() else {}
            if all(s in cache for s in parts.values()):
                d3 = (cache[parts["lab"]] - cache[parts["la"]] - cache[parts["lb"]]
                      - cache[parts["ab"]] + cache[parts["l"]] + cache[parts["a"]]
                      + cache[parts["b"]]) / EV_HARTREE * EH_KCAL
                vals.append((d3, d3 - dft.get(tname, float("nan"))))
            else:
                vals.append(None)
        line = f"{tname}\t{dft[tname]:.2f}"
        for v in vals:
            line += f"\t{v[0]:+.2f}\t{v[1]:+.2f}" if v else "\t\t"
        rows.append(line)
        print(line)
    (outdir / "delta3_compare.tsv").write_text("\n".join(rows) + "\n")
    print(f"[written {outdir/'delta3_compare.tsv'}]")


if __name__ == "__main__":
    main()