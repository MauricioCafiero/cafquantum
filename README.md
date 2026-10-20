# cafquant

A computational-chemistry workbench that runs one molecule through a set of
independent backends and collects the results in a single table. Give it an
XYZ file, an SDF/MOL file, or a SMILES string; it computes single-point
energies, optimized geometries, and harmonic forces, and writes everything to
one output directory.

Every backend is used through the same **ASE** driver, so methods are directly
comparable in how they are invoked and what they produce:

| method | backend | elements | weights |
|---|---|---|---|
| `uma` | UMA small (`uma-s-1p1`, omol task) via fairchem-core | full periodic table | auto-download from HuggingFace |
| `uma1p2` | UMA small 1.2 (`uma-s-1p2`, omol task) via fairchem-core | full periodic table | auto-download from HuggingFace |
| `maceoff` | MACE-OFF23 (small), float64, mace-torch | H C N O F + B Si P S Cl Br I | auto-download |
| `egret` | Egret-1 (compiled MACE model, float64), mace-torch | H C N O F P S Cl Br I | bundled in `weights/` (see below) |
| `aimnet2` | AIMNet2 (`aimnet2-2025`, B97-3c + D3) via aimnet | ~H–At in its species list | auto-download |
| `gfn2` | GFN2-xTB via tblite | Z ≤ 86 | none (params embedded) |
| `pm7` | PM7 via ASE's MOPAC calculator + mopac binary | H–Pu (mopac limits) | install mopac |
| `polar1-s` | MACE-POLAR-1 small (polarisable electrostatics, omol-scale), mace-torch + graph_longrange | OMol25 set (83 elements, incl. metals) | auto-download, cached in `~/.cache/mace` |
| `polar1-m` | MACE-POLAR-1 medium (12 Å electrostatic field) | OMol25 set | auto-download, cached in `~/.cache/mace` |
| `omol` | MACE-OMOL-0 extra-large (omol task), mace-torch | OMol25 set | repo `weights/` or auto-download |

Restricted-coverage methods fail fast with a clear "does not cover: X" message
(checked before any weights are loaded) rather than producing garbage
energies; the element lists themselves are shown in the table above.

Charged and open-shell species work everywhere: pass `--charge` and
`--multiplicity` (2S+1) and they reach each backend through its own convention.

## Environment

One uv virtualenv holds everything, including the three torch-based packages
whose dependency pins conflict with each other if resolved blindly. Build it
with:

```sh
zsh setup.sh     # creates .venv, installs all backends, runs an import probe
```

Two version facts worth knowing if you extend the environment:

- **torch 2.8.0** is the compatible intersection (fairchem-core 2.21 requires
  `~=2.8.0`, aimnet 0.2.0 requires `>=2.8`); numpy is pinned `>=2.0,<2.5` by
  fairchem.
- **e3nn 0.4.4** is mace-torch 0.3.16's hard pin (0.5 breaks mace; fairchem
  runs fine on it). Under torch ≥ 2.6 e3nn calls `torch.load` without
  `weights_only`, which torch refuses — patched at import time in
  `code/cafquant/env.py`, so no package is modified.

`setup.sh` installs fairchem-core and mace-torch with `--no-deps` plus a
hand-listed dependency set to bypass the conflicting pins, then runs an import
probe over all backends to catch any drift immediately.

Torch ships libomp while tblite links libgomp; the two OpenMP runtimes cannot
initialize in one process on macOS. The CLI therefore dispatches runs across
separate worker subprocesses (torch group and torch-free group) and merges the
results — one command, one venv, no environment tricks on your side. This is
handled automatically; see `code/cafquant/worker.py`.

### The egret model

Egret-1 models (Rowan Scientific, MACE-architecture compiled TorchScript,
`.model` files) are obtained separately — there is no public download, hence
no `--download` entry. The default model, `EGRET_1.model`, ships in the
repo-local `weights/` directory and is found automatically. To use a different
variant, drop it in `weights/` (or anywhere) and point the environment
variable at it:

```sh
export CAFQUANT_EGRET_MODEL=weights/EGRET_1E.model     # default: weights/EGRET_1.model
```

EGRET_1, EGRET_1E and EGRET_1T variants generally follow the same element set;
the actual list is read from the model file itself and enforced by the guard.

## Capabilities

- **Inputs**: `.xyz`, `.sdf`/`.mol`, or a SMILES string. SMILES are run
  through RDKit: add Hs → embed 5 conformers (ETKDGv3) → MMFF-minimize each →
  keep the lowest. Conformer count/seed are configurable. Formal charges from
  SMILES are picked up automatically.
- **Tasks**: `sp` (single-point energy), `opt` (BFGS relaxation,
  fmax 0.01 eV/Å default), `freq` (finite-difference harmonic frequencies —
  implies opt, plus ideal-gas thermochemistry at 298.15 K: ZPE, H(T), S(T),
  G(T)). Default task set is `sp,opt`; opt results are written as
  `opt_<method>.xyz` with trajectory and log.
- **Frequencies sanity check**: imaginary (−20 cm⁻¹ and below) or near-zero
  (0–20 cm⁻¹) modes are counted and flagged in both the run log and the
  summary — a cheap proxy for "is this really a minimum".
- **Reporting**: `summary.csv` and `summary.json` collect per-method status,
  energies (eV / kcal/mol / Ha), fmax, convergence, frequency statistics and
  thermochemistry, with per-method error messages so one backend failing
  never aborts the run.
- **Weights check/download**: `--check` prints where each backend's model
  weights live and whether it's usable; `--download` fetches what can be
  fetched (UMA/UMA-1.2 and MACE-POLAR-1 from HuggingFace/GitHub releases,
  MACE-OFF23 from GitHub, AIMNet2 from its registry, MACE-OMOL-0 via mace's
  downloader; pm7 prints the mopac install instruction).
- **Structure viewer**: `--view` (or the standalone `code/view.py`) opens a
  self-contained 3Dmol.js page with INPUT and optimized-geometry tabs.

## Files

```
cafquant/
├── setup.sh                 # build .venv, install everything, import probe
├── run_cafquant.sh          # zsh wrapper: env guards, caffeinate, cli entry
├── weights/                 # local model weights (git-ignored; egret ships here)
│   └── EGRET_1.model
├── tests/                   # pytest suite: unit tests + one full CLI e2e run
├── examples/
│   ├── water.xyz            # minimal smoke-test input
│   └── acetone.sdf          # file-format example
├── code/
│   ├── cafquant.py          # CLI entry point
│   └── cafquant/
│       ├── env.py           # env guards set before any torch import
│       ├── inputs.py        # xyz/sdf/smiles → ASE Atoms (+ RDKit pipeline)
│       ├── weights.py       # weight availability + download
│       ├── mlips.py         # calculator factory for the ten backends
│       ├── tasks.py         # sp / opt / freq / thermochemistry via ASE
│       └── worker.py        # per-process-group worker (see isolation note)
└── code/view.py             # standalone 3Dmol.js structure viewer
```

## Examples

```sh
zsh setup.sh                                        # once

./run_cafquant.sh "CCO"                             # ethanol: gfn2 only (default), sp + opt
./run_cafquant.sh "CCO" --all                       # every backend, sp + opt
./run_cafquant.sh "CCO" -m uma,maceoff,aimnet2,gfn2,pm7   # torch-family six, sp + opt
./run_cafquant.sh "CCO" -t sp,opt,freq              # + frequencies & thermochemistry
./run_cafquant.sh mol.sdf                            # file input
./run_cafquant.sh "CC(=O)[O-]" -m uma,gfn2 --charge -1   # charged species, subset
./run_cafquant.sh mol.xyz --method polar1-s          # one charge-aware backend
./run_cafquant.sh mol.xyz -m egret --fmax 0.02 --steps 1500
./run_cafquant.sh "CCO" --view                       # run then open 3D viewer
./run_cafquant.sh --check                            # weights report only
./run_cafquant.sh --download -m maceoff,aimnet2      # fetch missing weights
```

Expected output shape (water, defaults):

```
method    status                  E_eV     E_kcal    fmax
gfn2      ok               -137.97...   -3181.8..  0.003

wrote out/water/summary.csv (…s total)
```

Full flags: `--method`/`-m/--methods` (one method or comma list; no flag =
`gfn2`; `--all` = every backend), `-t/--tasks`, `--charge`, `--multiplicity`
(aliased `--spin`), `--conf`, `--seed`, `--fmax`, `--steps`, `--device`,
`--label`, `--out`, `--check`, `--download`, `--view`, `-q`.

## Testing

```sh
.venv/bin/python -m pytest tests/ -q
```

The suite has two layers:

- `tests/test_units.py` — fast checks that don't load any model: method list,
  element-coverage guards, the SMILES → RDKit pipeline, XYZ input, weight
  reporting (including that the egret model sits in `weights/`).
- `tests/test_e2e.py` — one full CLI run of the water example through all six
  methods (exercising the real worker-subprocess split), asserting every
  method reports `ok` and energies match known values (PM7 water heat of
  formation −57.80 kcal/mol, GFN2-xTB −137.97 eV, egret ≈ maceoff within
  10 meV).

Needs `pytest` in the venv (`setup.sh` adds it). A full suite run takes ~15 s.

## Output

```
out/<label>/
├── input_<label>.xyz        # what was actually computed (incl. SMILES product)
├── opt_<method>.xyz/.traj/.log
├── freq_<method>.dat        # frequencies + warning lines
├── pm7/                     # MOPAC scratch (.mop/.out)
└── summary.csv / summary.json
```

## Notes

- **Energy scales differ**: eV (+ kcal/mol, Ha columns) for all, but PM7
  reports a heat of formation while the MLIPs/xtb report absolute electronic
  energies — compare within a method family, not across.
- The viewer needs internet the first time (3Dmol.js loads from a CDN); after
  that the page is cached by the browser. Everything else runs offline.
- Long runs: `run_cafquant.sh` holds the Mac awake (`caffeinate`) for the
  duration; for very long jobs launch it detached (see the comment inside).
- `dft_check/` holds the ORCA DFT spot-check inputs and runner. ORCA here
  must be launched through `dft_check/run_orca.sh` (it puts the conda
  `orca-mpi` OpenMPI 4.1.6 on PATH; the `~/.local` OpenMPI 4.1.8 has no
  Fortran support and large jobs die at Startup) — see
  `dft_check/README.md`, "Environment gotchas".
- Reference point sanity: PM7 water ΔHf ≈ −57.8 kcal/mol and GFN2-xTB water
  E ≈ −137.97 eV confirm a healthy install.