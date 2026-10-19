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

- `WB97X-D3 def2-SVP def2/J TightSCF RIJK`, 1000 MB/core, 16 procs
  (`%pal nprocs 16 end`) — adjust nprocs/%maxcore to the runner.
- ORCA infers UHF from the multiplicity in the `*xyzfile` line (several
  sult/ddc frames are open-shell: the anchored-radical residues carry
  hand-set multiplicities from the original Gaussian work — the .inp files
  use exactly the q/M the MLIP runs used).
- If your ORCA version rejects the functional name, `WB97M-V`
  (def2/J + def2-SVP also fine) is the near-equivalent modern option.

## What each spot-check arbitrates (delta3 in kcal/mol, from the 2026-10-19 run)

| system | frame | triple | uma | maceoff | egret | aimnet2 | gfn2 | pm7 |
|--------|------:|--------|----:|--------:|------:|--------:|-----:|----:|
| comt | 222 | LD(-1)+Mg(2)+trp2 | +72.6 | — | — | — | +209.1 | +31.6 |
| comt | 210 | LD(-1)+lys(+1)+Mg(2) | +43.9 | — | — | — | +195.0 | +23.9 |
| ddc  | 116 | LD(-1)+his1+lys(+1) | -196.2‡ | -0.3 | -0.2 | +3.4 | +4.4 | **+57.8** |
| ddc  | 158 | LD(-1)+phe1+pro | +8.0 | -4.1 | -4.1 | +6.6 | **+43.2** | **-18.2** |
| sult | 72  | LD+ala+his2(+1) | -66.3‡ | -0.0 | -0.4 | +6.8 | -14.1 | -11.8 |
| sult | 99  | LD+his2(+1)+lys(+1) | -538.7‡ | 0.0 | 0.0 | +10.3 | +8.8 | +3.8 |

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

## Running

Each `.inp` references its `.xyz` by relative name; run inside each
system directory, e.g.:

    cd comt
    for f in comt_*.inp; do orca "$f" > "${f%.inp}.out"; done &

or submit one job per file on a scheduler. delta3 evaluation: sum the
seven ORCA final-single-point energies per triple (table below) and apply
the formula; energies are in Eh — 1 Eh = 627.509 kcal/mol.

Needed frame sets per spot triple (file stems, see jobs.tsv for q/M):
`sult`: triple 072 → {000 LD, 001 ala, 005 his2, 014, 018, 027};
triple 099 → {000, 005, 006 lys, 018, 019, 054}.
`ddc`: triple 116 → {000 LD, 002 his1, 005 lys, 018, 021, 038};
triple 158 → {000, 006 phe1, 009 pro, 022, 025, 080}.
`comt`: triple 210 → {000 LD, 007 lys, 008 Mg, 027, 028, 105};
triple 222 → {000, 008, 013 trp2, 028, 033, 117}.

Monomers are shared between triples (35 inputs total, 4 new energy points
per triple). Cost: ~40–55 atoms per frame at this level ≈ a few core-hours
each — well under a day of 16-core time for the full set.