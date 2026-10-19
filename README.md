# cafquant

Multi-MLIP + semiempirical computational-chemistry workspace. Input: XYZ, SDF,
or SMILES (SMILES -> RDKit: add Hs, 5 conformers ETKDGv3 + MMFF-minimize,
lowest kept). All backends run through **ASE**: single-point energy, geometry
optimization (BFGS, 0.01 eV/A), and harmonic finite-difference frequencies +
ideal-gas thermochemistry (298.15 K).

| method | backend | weights |
|---|---|---|
| `uma` | fairchem-core 2.21, uma-s-1p1, omol task | `~/.cache/fairchem` (HF) |
| `maceoff` | mace-torch 0.3.16, MACE-OFF23 small, float64 | `~/.cache/mace/MACE-OFF23_small.model` |
| `aimnet2` | aimnet 0.2.0, `aimnet2-2025` (B97-3c + D3) | `~/.cache/aimnet` |
| `gfn2` | tblite 0.7.0 (GFN2-xTB, params embedded) | none |
| `pm7` | ASE MOPAC calculator + conda-forge mopac binary | `~/.local/bin/mopac` |

## Setup

```sh
zsh setup.sh          # creates .venv, installs everything, runs an import probe
```

e3nn is 0.4.4: mace-torch 0.3.16 hard-pins it (0.5 breaks mace's Activation
code), and fairchem-core — despite declaring >=0.5 — runs fine on it. The one
real breakage (e3nn 0.4.4 calls `torch.load` without `weights_only`, which
torch >= 2.6 rejects) is patched at import time in `code/cafquant/env.py` by
allow-listing `slice` in torch's safe globals. Both backends install with
`--no-deps` + hand-listed dependencies to bypass the conflicting pins. Verify
with `./run_cafquant.sh --check` and the import probe in `setup.sh`.

## Usage

```sh
./run_cafquant.sh "CCO"                          # all 5 methods, sp + opt
./run_cafquant.sh "CCO" -t sp,opt,freq           # + frequencies & thermochemistry
./run_cafquant.sh mol.xyz -m uma,gfn2            # subset
./run_cafquant.sh mol.sdf --charge -1            # charged species
./run_cafquant.sh --check                        # weights report only
./run_cafquant.sh --download -m maceoff,pm7       # fetch missing weights
```

## Structure viewing

Self-contained 3Dmol.js pages, generated with the *system* python (no venv):

```sh
python3 code/view.py out/CCO        # run dir: INPUT + opt_<method> tabs
python3 code/view.py mol.sdf        # any .xyz/.sdf/.mol/.mol2 file
./run_cafquant.sh "CCO" --view      # open the viewer when the run finishes
```

Tabs (left/right arrow keys to cycle), ball-and-stick. 3Dmol.js loads from a
CDN the first time the page opens, so it needs a one-time internet connection.

Flags: `-m` methods, `-t` tasks, `--charge`, `--multiplicity` (2S+1),
`--conf` (SMILES conformers, 5), `--seed`, `--fmax`, `--steps`, `--device`,
`--label`, `--out`.

## Output

`out/<label>/` — `input_<label>.xyz`, `opt_<method>.xyz` + `.traj` + `.log`,
`freq_<method>.dat` (frequencies + imaginary-mode warnings), `pm7/`
(MOPAC scratch), `summary.csv`, `summary.json` (full run metadata).

## Notes

- **Process isolation**: torch ships libomp and tblite links libgomp — the two
  OpenMP runtimes cannot be initialized in one process on macOS even with
  `KMP_DUPLICATE_LIB_OK=TRUE`. The CLI therefore runs the torch-backed group
  (uma/maceoff/aimnet2) and the torch-free group (gfn2/pm7) as separate worker
  subprocesses (`cafquant/worker.py`) and merges the results. One command,
  same venv.
- Energies: eV (+ kcal/mol, Ha columns). PM7 reports a heat of formation;
  absolute energies are **not** comparable across method families.
- Imaginary (freq < -20 cm^-1) or near-zero modes are flagged in the run log
  and summary.
- MOPAC frequencies launch one process per displaced geometry (6N calls) —
  fine for small/medium molecules.
- Long runs: launch detached (see comment in `run_cafquant.sh`).
- Env guards (`KMP_DUPLICATE_LIB_OK=TRUE`, single-thread OMP) are set in
  `code/cafquant/env.py`, import-first.