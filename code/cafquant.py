#!/usr/bin/env python
"""cafquant CLI: XYZ / SDF / SMILES input -> sp / opt / freq across five backends.

Usage:
    python code/cafquant.py <xyz|sdf|SMILES> [-m uma,maceoff,aimnet2,gfn2,pm7]
        [-t sp,opt,freq] [--charge N] [--multiplicity N] [--label L] [--out DIR]
        [--fmax F] [--steps N] [--conf N] [--seed N] [--device cpu|cuda]
        [--check] [--download] [-q]

Defaults: all five methods, tasks = sp,opt. '--t freq' implies opt. Energies
are printed per method; MOPAC/PM7 reports a heat of formation, so compare
methods only within their own family (see summary.csv notes).
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # make code/ importable

from cafquant import METHODS, __version__
from cafquant import env  # noqa: F401  (sets OMP guards before torch)
from cafquant import inputs, weights

log = logging.getLogger("cafquant")

SUMMARY_COLUMNS = ["Method", "Task", "Status", "Energy_eV", "Energy_kcal", "Energy_Ha",
                   "Fmax_eV_A", "Converged", "Min_freq_cm1", "N_imag", "N_soft",
                   "ZPE_eV", "H_eV", "G_eV", "Error"]


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", help="path to .xyz/.sdf file or a SMILES string")
    p.add_argument("-m", "--methods", default=",".join(METHODS),
                   help="comma list from: %s (default: all)" % ",".join(METHODS))
    p.add_argument("-t", "--tasks", default="sp,opt",
                   help="comma list from: sp,opt,freq (default: sp,opt; freq implies opt)")
    p.add_argument("--charge", type=int, default=None,
                   help="total charge (default: formal charge if SMILES, else 0)")
    p.add_argument("--multiplicity", "--spin", type=int, default=1, dest="multiplicity",
                   help="spin multiplicity 2S+1 (default: 1)")
    p.add_argument("--label", default=None, help="run label (default: derived from input)")
    p.add_argument("--out", default=None, help="output dir (default: out/<label>)")
    p.add_argument("--fmax", type=float, default=0.01, help="opt convergence in eV/A")
    p.add_argument("--steps", type=int, default=1000, help="max optimizer steps")
    p.add_argument("--conf", type=int, default=5, help="SMILES conformers to sample")
    p.add_argument("--seed", type=int, default=42, help="SMILES embedding seed")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--check", action="store_true", help="weights report only, no run")
    p.add_argument("--download", action="store_true",
                   help="download missing weights before running")
    p.add_argument("-q", "--quiet", action="store_true", help="INFO progress logging off")
    p.add_argument("--view", action="store_true",
                   help="open a 3Dmol.js structure view of the results when done")
    p.add_argument("--version", action="version", version=__version__)
    return p.parse_args(argv)


def write_summary(rundir: Path, rows: list[dict], meta: dict) -> None:
    with (rundir / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({
                "Method": r["method"], "Task": r.get("task", ""), "Status": r.get("status", ""),
                "Energy_eV": r.get("energy_eV"), "Energy_kcal": r.get("energy_kcal"),
                "Energy_Ha": r.get("energy_ha"), "Fmax_eV_A": r.get("fmax"),
                "Converged": r.get("converged"), "Min_freq_cm1": r.get("min_freq_cm1"),
                "N_imag": r.get("n_imag"), "N_soft": r.get("n_soft"),
                "ZPE_eV": r.get("zpe_eV"), "H_eV": r.get("enthalpy_eV"),
                "G_eV": r.get("gibbs_eV"), "Error": r.get("error", ""),
            })
    (rundir / "summary.json").write_text(
        json.dumps({"run": meta, "results": rows}, indent=2, default=str))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    report = weights.check_weights()
    print("\n".join(weights.format_report(report)))
    print("Energies are on each method's own scale; PM7 is a heat of formation "
          "(kcal/mol), the others absolute electronic energies.\n")
    if args.check:
        missing = [m for m, r in report.items() if not r["ok"]]
        return 0

    available = weights.check_available()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    for m in methods:
        if m not in METHODS:
            sys.exit(f"unknown method: {m}")
    missing = [m for m in methods if m not in available]
    if missing:
        if args.download:
            for m in missing:
                print(weights.download(m))
        else:
            print("skipping (weights missing, use --download):", ", ".join(missing))
    methods = [m for m in methods if m in available] if not args.download else methods
    if not methods:
        sys.exit("no runnable methods; run with --download first")
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    for t in tasks:
        if t not in ("sp", "opt", "freq"):
            sys.exit(f"unknown task: {t}")
    if not methods:
        sys.exit("no runnable methods; run --download first")

    atoms, label, src = inputs.load_structure(
        args.source, charge=args.charge, multiplicity=args.multiplicity,
        n_confs=args.conf, seed=args.seed)
    atoms.info["label"] = label
    rundir = Path(args.out) if args.out else Path("out") / label
    rundir.mkdir(parents=True, exist_ok=True)

    from ase.io import write as ase_write
    ase_write(str(rundir / f"input_{label}.xyz"), atoms, format="xyz",
              comment=f"input geometry from {src}; charge={atoms.info['charge']}")

    meta = {"source": src, "label": label, "charge": atoms.info["charge"],
            "multiplicity": int(args.multiplicity), "n_conformers": args.conf,
            "seed": args.seed, "conformer_energy_kcal": atoms.info.get("conformer_energy_kcal"),
            "forcefield": atoms.info.get("forcefield"), "natoms": len(atoms),
            "formula": atoms.get_chemical_formula()}

    print(f"{label}: {meta['formula']}, {len(atoms)} atoms, charge={meta['charge']}, "
          f"mult={meta['multiplicity']}, source={src.split(' |')[0]}")
    print(f"output -> {rundir}\n")

    from cafquant.tasks import TORCH_METHODS  # noqa: F401  (import check only)
    code_dir = Path(__file__).resolve().parent

    rows, t0 = [], time.time()
    groups = [m for m in methods if m in TORCH_METHODS], \
             [m for m in methods if m not in TORCH_METHODS]
    for group in groups:
        if not group or not group[0]:
            continue
        import subprocess
        env = {**os.environ, "PYTHONPATH": str(code_dir)}
        cmd = [sys.executable, str(code_dir / "cafquant" / "worker.py"),
               "--rundir", str(rundir), "--input", str(rundir / f"input_{label}.xyz"),
               "--methods", ",".join(group), "--tasks", ",".join(tasks),
               "--charge", str(meta["charge"]), "--multiplicity", str(meta["multiplicity"]),
               "--label", label, "--device", args.device,
               "--fmax", str(args.fmax), "--steps", str(args.steps)]
        if args.quiet:
            cmd.append("--quiet")
        proc = subprocess.run(cmd, env=env)
        if proc.returncode != 0:
            print(f"WARNING: worker for {','.join(group)} exited "
                  f"with code {proc.returncode}")
        for m in group:
            resfile = rundir / f"{m}_result.json"
            if resfile.is_file():
                rows.append(json.loads(resfile.read_text()))
            else:
                rows.append({"method": m, "task": ",".join(tasks), "status": "failed",
                             "error": f"worker died before writing {resfile.name}",
                             "wall_s": 0.0})

    meta["wall_s"] = round(time.time() - t0, 1)
    write_summary(rundir, rows, meta)

    # final table
    print(f"\n{'method':9} {'status':15} {'E_eV':>12} {'E_kcal':>10} {'fmax':>7}")
    for r in rows:
        e = f"{r['energy_eV']:12.6f}" if "energy_eV" in r else f"{'-':>12}"
        ek = f"{r['energy_kcal']:10.3f}" if "energy_kcal" in r else f"{'-':>10}"
        fm = f"{r['fmax']:7.4f}" if "fmax" in r else f"{'-':>7}"
        warn = " IMAG" if r.get("n_imag") else ""
        print(f"{r['method']:9} {r['status']:15} {e} {ek} {fm}{warn} "
              f"[{r['wall_s']}s]" + (f"  error: {r['error']}" if r.get("error") else ""))
    print(f"\nwrote {rundir / 'summary.csv'} ({meta['wall_s']}s total)")
    if args.view:
        import view
        view.main([str(rundir)])  # writes structure.html and opens the browser
    return 0


if __name__ == "__main__":
    sys.exit(main())