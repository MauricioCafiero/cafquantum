"""Generate ORCA input files for the DFT spot-check of the mbe 2B/3B terms.

Reads the (untracked) mbe workspace frames and emits one .inp + .xyz per
frame needed to evaluate the spotlight triples' delta3 = E(LAB) - E(LA)
- E(LB) - E(AB) + E(L) + E(A) + E(B).

Spot triples chosen from the cross-method disagreement analysis
(see dft_check/README.md):
  sult  72, 99   ddc  116, 158   comt  222, 210
(also built: sult 80/81 optional extras? -- keep the minimal decisive set.)

Each triple needs 4 new DFT frames (LAB, LA, LB, AB); monomers (LD and the
residues) are shared across triples and emitted once.

Run:  ../mbe/.venv equivalent python -- or any venv with ase; this script
only reads mbe/out and writes here:
    python3 make_inputs.py
"""
from __future__ import annotations

import json
from pathlib import Path

from ase.io import read

HERE = Path(__file__).resolve().parent
MBE = HERE.parent / "mbe"

# frame k == frame index in mbe/out/<sys>/frames.xyz
SPOT = {
    "sult": [72, 99],
    "ddc": [116, 158],
    "comt": [222, 210],
}

FUNC_BASIS = "! WB97X-D3 def2-SVP def2/J TightSCF RIJK"
NPROCS = 16  # adjust to the runner's core count (README)

SPOT_METHOD_VALUES = {  # delta3 (kcal/mol) as of the 2026-10-19 table
    ("sult", 72): "uma -66.3(art) maceoff -0.0 egret -0.4 aimnet2 +6.8 gfn2 -14.1 pm7 -11.8",
    ("sult", 99): "uma -538.7(art) maceoff 0.0 egret 0.0 aimnet2 +10.3 gfn2 +8.8 pm7 +3.8",
    ("ddc", 116): "uma -196.2(art) maceoff -0.3 egret -0.2 aimnet2 +3.4 gfn2 +4.4 pm7 +57.8",
    ("ddc", 158): "uma +8.0 maceoff -4.1 egret -4.1 aimnet2 +6.6 gfn2 +43.2 pm7 -18.2",
    ("comt", 222): "uma +72.6 gfn2 +209.1 pm7 +31.6 (maceoff/egret/aimnet2: Mg guard)",
    ("comt", 210): "uma +43.9 gfn2 +195.0 pm7 +23.9",
}


def gen_system(system: str) -> list[tuple[str, int, str, int, int]]:
    """Write inputs for `system`; returns (filename stem) job list rows."""
    out = MBE / "out" / system
    frames = read(out / "frames.xyz", index=":")
    meta = json.loads((out / "charges.json").read_text())
    man = json.loads((out / "manifest.json").read_text())
    name = {f["fid"]: f["name"] for f in man["fragments"]}
    lr = {(a, b): k for a, b, k in man["pairs_ld"]}
    rr = {(a, b): k for a, b, k in man["pairs_rr"]}
    by_k = {(L, A, B): k for L, A, B, k in man["triples"]}

    want: dict[int, str] = {}  # frame idx -> role label
    for L, A, B, k in man["triples"]:
        if k not in SPOT[system]:
            continue
        want[k] = f"triple_{name[L]}_{name[A]}_{name[B]}"
        want[lr[(L, A)]] = f"pair_{name[L]}_{name[A]}"
        want[lr[(L, B)]] = f"pair_{name[L]}_{name[B]}"
        want[rr[(A, B)]] = f"pair_{name[A]}_{name[B]}"
        for fid in (L, A, B):
            want[fid] = f"monomer_{name[fid]}"

    jobs = []
    for k, label in sorted(want.items()):
        atoms = frames[k]
        q, m = meta["charges"][k], meta["mults"][k]
        stem = f"{system}_{k:03d}_{label}"
        xyz = HERE / f"{system}" / f"{stem}.xyz"
        xyz.parent.mkdir(exist_ok=True)
        atoms.write(xyz, format="xyz")
        inp = (f"{FUNC_BASIS}\n"
               f"%maxcore 1000\n"
               f"%pal nprocs {NPROCS} end\n"
               f"*xyzfile {q} {m} {stem}.xyz\n")
        (xyz.parent / f"{stem}.inp").write_text(inp)
        jobs.append((f"{system}/{stem}", k, label, q, m))
    return jobs


def main() -> None:
    rows = []
    for system in SPOT:
        rows += gen_system(system)
    with (HERE / "jobs.tsv").open("w") as fh:
        fh.write("input\tframe_idx\trole\tq\tM\tmethod_values_delta3_kcal\n")
        for f, k, label, q, m in rows:
            if "triple" in label:
                sysname = f.split("/")[0]
                vals = SPOT_METHOD_VALUES.get((sysname, k), "")
                fh.write(f"{f}\t{k}\t{label}\t{q}\t{m}\t{vals}\n")
    # README target table is hand-written; here just report
    print(f"generated {len(rows)} frames")
    for f, k, label, q, m in rows:
        print(f"  {f}.inp  idx={k} q={q} M={m}")


if __name__ == "__main__":
    main()