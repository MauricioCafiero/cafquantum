# dft_check — ORCA spot-check of the 2B/3B decomposition

Tracked counterpart to the (untracked) `mbe/` workspace. Contains ORCA
inputs for the frames that **break the methods**: the triples whose
cross-method 3-body IE (delta3) spread is the largest in each charged
system, plus the monomer/pair frames needed to evaluate each delta3:

    delta3 = E(LAB) - E(LA) - E(LB) - E(AB) + E(L) + E(A) + E(B)

All geometries were extracted from `mbe/out/<sys>/frames.xyz` (the same
frames the MLIP single-points ran on); regenerate with
`make_inputs.py` (needs python + ase; `jobs.tsv` lists every job).

## Level and settings

- `wB97M-V def2-TZVP def2/J RIJCOSX TightSCF defgrid3`, 1000 MB/core
  (`%maxcore`), `%pal nprocs 6` (this laptop: 4 physical cores /
  8 hwthreads — set nprocs per the runner; see *Running and
  evaluating*). wB97M-V is a meta-GGA hybrid, so the aux basis is
  `def2/J` with density-fitted J; exact exchange is fitted with the
  COSX semi-numerical scheme (`RIJCOSX`), not `RIJK` (restricted to
  non-meta functionals in ORCA — a bare `RIJ` is not a keyword at all;
  RI-J alone is the default when an aux basis is given). `defgrid3` is
  the tighter XC grid that keeps the COSX error small. Same syntax as
  the proven rotaxanes stacking jobs (`v2_cp_*.inp`).
- ORCA infers UHF from the multiplicity in the `*xyzfile` line (several
  sult/ddc frames are open-shell: the anchored-radical residues carry
  hand-set multiplicities from the original Gaussian work — the .inp files
  use exactly the q/M the MLIP runs used).
- History: the first sweep ran at `WB97X-D3 def2-SVP def2/JK RIJK`
  (all comt + ddc monomers/pairs finished on 2026-10-19) but was
  superseded by this level; those artifacts were deleted on 2026-10-20
  (user request — old-level energies are not used anywhere).

## What each spot-check arbitrates (delta3 in kcal/mol)

DFT column: wB97M-V/def2-TZVP/RIJCOSX values computed 2026-10-20
(`eval.py`, full results in `dft_delta3.tsv`); MLIP columns from the
2026-10-19 run

| system | frame | triple | **DFT** | uma | maceoff | egret | aimnet2 | gfn2 | pm7 |
|--------|------:|--------|----:|----:|--------:|------:|--------:|-----:|----:|
| comt | 222 | LD(-1)+Mg(2)+trp2 | **+42.2** | +72.6 | — | — | — | **+209.1** | +31.6 |
| comt | 210 | LD(-1)+lys(+1)+Mg(2) | **+37.2** | +43.9 | — | — | — | **+195.0** | +23.9 |
| ddc  | 116 | LD(-1)+his1+lys(+1) | **+0.5** | -196.2‡ | -0.3 | -0.2 | +3.4 | +4.4 | **+57.8** |
| ddc  | 158 | LD(-1)+phe1+pro | **-14.5** | +8.0 | -4.1 | -4.1 | +6.6 | **+43.2** | **-18.2** |
| sult | 72  | LD+ala+his2(+1) | **-13.5** | -66.3‡ | -0.0 | -0.4 | +6.8 | -14.1 | -11.8 |
| sult | 99  | LD+his2(+1)+lys(+1) | **+5.1** | -538.7‡ | 0.0 | 0.0 | +10.3 | +8.8 | +3.8 |

Verdicts of the 2026-10-20 DFT sweep: gfn2's ~+200 kcal Mg-chelation 3B
in comt is a ~5x overestimate (DFT +37…+42, significant but not explosive);
uma is surprisingly good on the Mg frames (~±15 kcal) though still carrying
its charge artifact elsewhere; pm7 tracks comt well (+13) but is the lone
outlier on ddc 116 (+57.8 vs DFT 0.5); ddc 158's gfn2-vs-pm7 sign flip is
real (DFT agrees with pm7's sign); on sult, gfn2 and pm7 stay within a few
kcal, while maceoff/egret flatten the -13 kcal three-body effect near zero.

uma `‡` values are the known charge-conditioning artifact (not to be
reconciled). comt maceoff/egret/aimnet2 cannot run Mg. Context: the whole
`sult` system has a DFT reference (paper wB97XD: 2B+3B = -104.13, net 3B
+10.65) to validate the protocol; `aldh` needs no check (max spread 1.5
kcal).

## Priority

1. comt 222 / 210 — decide whether comt's 2B+3B is driven by a real
   ~+200 kcal Mg-chelation 3B (gfn2) or an electrostatics exaggeration
   (pm7 says +24…+32, uma +44…+73).
2. ddc 116 — PM7-alone anomaly (+57.8 vs ~0 elsewhere).
3. ddc 158 — the gfn2-vs-pm7 sign flip, representative of the LD+pro
   family (6 of ddc's top-7 mismatches).
4. sult 72/99 — the only system with an external DFT reference to
   validate the level against the paper.

## Running and evaluating

**Runner:** `./run_orca.sh` (from this directory). It is resume-safe:
a job whose `.out` already contains `FINAL SINGLE POINT ENERGY` is
skipped, work is appended to existing `.out`/`.err` files under a dated
`>>> restart <<<` marker, and progress goes to `restart.log`. If the
machine dies mid-queue, simply rerun it — also restart it under a sleep
inhibitor if it will run unattended longer than ~1 h (this laptop
auto-suspends on idle):

    systemd-inhibit --what=sleep:idle --who=orca-dft --why="dft_check" \
        --mode=block ./run_orca.sh > restart.log 2>&1 &

To evaluate a specific subset, edit the job list at the top of the
script (it lists every job by `system/stem`, in rough cost order).

### Environment gotchas (the reason for the env block in the script)

- This ORCA 6.1.1 install is the **shared_openmpi418** build: it shells
  out to `mpirun` for multi-process steps. The OpenMPI 4.1.8 in
  `~/.local` (built 2026-08-07) was configured **without Fortran**, and
  large jobs then die instantly in Startup with
  `MPI_Type_match_size ... MPI_ERR_ARG` (the routine can only return
  Fortran datatypes; small jobs just take the in-process path and look
  fine). Use the conda `orca-mpi` env's OpenMPI 4.1.6 instead — built
  with `--enable-mpi-fortran` — by prepending it to `PATH` and
  `LD_LIBRARY_PATH` (the `run_orca.sh` env block; same recipe as the
  rotaxanes stacking runner `run_orca_stacking.sh`).
- On this 4-physical-core/8-hwthread i7-1065G7, OpenMPI counts
  *physical cores* as slots, so `-np 6` also needs
  `OMPI_MCA_hwloc_base_use_hwthreads_as_cpus=1` +
  `OMPI_MCA_rmaps_base_oversubscribe=1` (+ the vader setting).
- If a future ORCA update is linked against a different OpenMPI series,
  rebuild/reinstall the matching MPI with Fortran enabled.

Each `.inp` references its `.xyz` by relative name; **the queue `cd`s
into each system directory before invoking ORCA** (running from the
parent directory fails with `Cannot open input file`).

To evaluate: sum the seven final-single-point energies per triple and
apply the delta3 formula at the top (energies in Eh; 1 Eh =
627.509 kcal/mol) — or just run `python3 eval.py` from this directory,
which does it and writes `dft_delta3.tsv`.

Needed frame sets per spot triple (file stems, see jobs.tsv for q/M):
`sult`: triple 072 → {000 LD, 001 ala, 005 his2, 014, 018, 027};
triple 099 → {000, 005, 006 lys, 018, 019, 054}.
`ddc`: triple 116 → {000 LD, 002 his1, 005 lys, 018, 021, 038};
triple 158 → {000, 006 phe1, 009 pro, 022, 025, 080}.
`comt`: triple 210 → {000 LD, 007 lys, 008 Mg, 027, 028, 105};
triple 222 → {000, 008, 013 trp2, 028, 033, 117}.

Monomers are shared between triples (35 inputs total, 4 new energy points
per triple). ~40–70 atoms per frame: monomers/pairs are minutes at
def2-TZVP; the four triples (56–70 atoms, meta-GGA + VV10) are the
long poles — allow a few hours each on this 6-rank laptop setup, so
use the sleep-inhibitor launch for the full queue.
When the `.out` files come back, put them next to the `.inp` in each
system directory and run `python3 eval.py` — it extracts the final
single-point energies, builds each triple's delta3 (Eh -> kcal/mol) and
prints it next to the MLIP values; results land in `dft_delta3.tsv`.
