#!/bin/bash
# ORCA queue runner for the dft_check spot-check jobs. Resumable by design:
# a job whose .out already contains "FINAL SINGLE POINT ENERGY" is skipped,
# so after any interruption/crash simply rerun it to continue where the
# queue left off. Launch it under a sleep inhibitor for unattended runs
# (> ~1 h) — see README.md "Running and evaluating" for the launch line
# and for why the OpenMPI environment block below is load-bearing.
# (Created 2026-10-19 to restart jobs killed by a machine crash;
# ORCA auto-restarted ddc_116's SCF from its pre-crash .gbw.)
#
# Job order: cheapest first, cheapest-system-first — all 35 inputs are run
# at the current level (wB97M-V/def2-TZVP/RIJCOSX). The superseded
# WB97X-D3/def2-SVP results were deleted 2026-10-20: a delta3 needs all
# consistent level. The skip-on-FINAL check makes finished jobs free.
set -u
cd "$(dirname "$0")"
ORCADIR=$HOME/.local/orca_6_1_1_linux_x86-64_shared_openmpi418_nodmrg
ORCA=$ORCADIR/orca
# Proven environment (see MD_openmm/rotaxanes/.../run_orca_stacking.sh):
# this ORCA is the shared_openmpi418 build and shells out to `mpirun`; the
# ~/.local/openmpi-4.1.8 on PATH was built WITHOUT Fortran support, so large
# jobs die in Startup on MPI_Type_match_size (MPI_ERR_ARG). The conda
# orca-mpi env has a 4.1.6 with --enable-mpi-fortran -- prepend it instead.
MPI=$HOME/miniforge3/envs/orca-mpi
export PATH="$MPI/bin:$ORCADIR:$PATH"
export LD_LIBRARY_PATH="$MPI/lib:$ORCADIR:${LD_LIBRARY_PATH:-}"
# 4 physical cores / 8 hwthreads: OpenMPI allocates slots by physical core,
# so -np 6 needs hwthreads declared usable plus oversubscribe.
export OMPI_MCA_hwloc_base_use_hwthreads_as_cpus=1
export OMPI_MCA_rmaps_base_oversubscribe=1
export OMPI_MCA_btl_vader_single_copy_mechanism=none

for stem in \
    sult/sult_000_monomer_LD \
    sult/sult_001_monomer_alanine \
    sult/sult_005_monomer_histidine2 \
    sult/sult_006_monomer_lysine \
    sult/sult_014_pair_LD_alanine \
    sult/sult_018_pair_LD_histidine2 \
    sult/sult_019_pair_LD_lysine \
    sult/sult_027_pair_alanine_histidine2 \
    sult/sult_054_pair_histidine2_lysine \
    sult/sult_072_triple_LD_alanine_histidine2 \
    sult/sult_099_triple_LD_histidine2_lysine \
    ddc/ddc_000_monomer_LD \
    ddc/ddc_002_monomer_histidine1 \
    ddc/ddc_005_monomer_lysine \
    ddc/ddc_006_monomer_phenylalanine1 \
    ddc/ddc_009_monomer_proline \
    ddc/ddc_018_pair_LD_histidine1 \
    ddc/ddc_021_pair_LD_lysine \
    ddc/ddc_022_pair_LD_phenylalanine1 \
    ddc/ddc_025_pair_LD_proline \
    ddc/ddc_038_pair_histidine1_lysine \
    ddc/ddc_080_pair_phenylalanine1_proline \
    ddc/ddc_116_triple_LD_histidine1_lysine \
    ddc/ddc_158_triple_LD_phenylalanine1_proline \
    comt/comt_000_monomer_LD \
    comt/comt_007_monomer_lysine \
    comt/comt_008_monomer_magnesiumw \
    comt/comt_013_monomer_tryptophan2 \
    comt/comt_027_pair_LD_lysine \
    comt/comt_028_pair_LD_magnesiumw \
    comt/comt_033_pair_LD_tryptophan2 \
    comt/comt_105_pair_lysine_magnesiumw \
    comt/comt_117_pair_magnesiumw_tryptophan2 \
    comt/comt_210_triple_LD_lysine_magnesiumw \
    comt/comt_222_triple_LD_magnesiumw_tryptophan2 \
    comt/comt_003_monomer_aspartic_acid1 \
    comt/comt_004_monomer_aspartic_acid2 \
    comt/comt_005_monomer_glutamic_acid \
    comt/comt_059_quint_LD_magnesiumw_aspartic_acid1_aspartic_acid2_glutamic_acid
do
  d=$(dirname "$stem"); b=$(basename "$stem")
  if grep -q "FINAL SINGLE POINT ENERGY" "$stem.out" 2>/dev/null; then
    echo "[skip] $(date '+%H:%M:%S') $b already done"
    continue
  fi
  echo "[run ] $(date '+%H:%M:%S') $b"
  printf '\n>>> restart %s after machine crash <<<\n' "$(date -Iseconds)" >> "$stem.out"
  ( cd "$d" && "$ORCA" "$b.inp" >> "$b.out" 2>> "$b.err")
  if grep -q "FINAL SINGLE POINT ENERGY" "$stem.out" 2>/dev/null; then
    echo "[done] $(date '+%H:%M:%S') $b"
  else
    echo "[FAIL] $(date '+%H:%M:%S') $b — see $stem.err and $stem.out"
  fi
done
echo "ALL QUEUE PROCESSED $(date -Iseconds)"