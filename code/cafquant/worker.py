"""cafquant worker: run a group of methods for one molecule in THIS process.

The CLI split methods into a torch-backed group (uma/maceoff/aimnet2) and a
non-torch group (gfn2/pm7) because torch's libomp and tblite's libgomp cannot
coexist in one process on macOS. Each group runs as a separate `python -
m cafquant.worker` subprocess; results land in rundir as <method>_result.json.

Usage (internal, called by code/cafquant.py):
    python worker.py --rundir DIR --input file.xyz --methods uma,maceoff \
        --tasks sp,opt [--charge N --multiplicity N --device cpu \
        --fmax F --steps N --fmax]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--rundir", required=True)
    p.add_argument("--input", required=True, help="structure file (from the parent CLI)")
    p.add_argument("--methods", required=True)
    p.add_argument("--tasks", required=True, help="comma list: sp,opt,freq")
    p.add_argument("--charge", type=int, default=0)
    p.add_argument("--multiplicity", type=int, default=1)
    p.add_argument("--label", default="mol")
    p.add_argument("--device", default="cpu")
    p.add_argument("--fmax", type=float, default=0.01)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    import logging
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    import cafquant.env  # noqa: F401  (guards before torch)
    from ase.io import read as ase_read

    from cafquant.tasks import run_method

    rundir = Path(args.rundir)
    inp = ase_read(args.input)
    inp.info["charge"] = args.charge
    inp.info["spin"] = args.multiplicity
    inp.info["label"] = args.label
    if args.multiplicity > 1:
        import numpy as np
        inp.set_initial_magnetic_moments(np.full(len(inp), args.multiplicity - 1.0))

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    for method in [m.strip() for m in args.methods.split(",") if m.strip()]:
        t0 = time.time()
        row = run_method(method, inp.copy(), tasks, rundir, charge=args.charge,
                         multiplicity=args.multiplicity, device=args.device,
                         fmax=args.fmax, steps=args.steps)
        row["wall_s"] = round(time.time() - t0, 1)
        row["task"] = ",".join(tasks)
        out = rundir / f"{method}_result.json"
        out.write_text(json.dumps(row, indent=2, default=str))
        print(f"[{method}] -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())