#!/bin/bash
#SBATCH --job-name=rtteam_gh200
#SBATCH -p gh
#SBATCH -A AST26021
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err

# Diagnose the sweep=tiled tile_size=0 regression at GH200 B=224 (nmb=1).
#
# tile_size=0 makes the tiled kernel do exactly what diagonal_compact does -- one tile, one
# launch, league = nmb*nang_tot, same inner h-loop -- yet it measured 32% slower at nmu=3 and
# 17% at nmu=6, while matching or beating diagonal_compact at every larger-league config, and
# showing no regression at all on B200.
#
# Hypothesis: Kokkos::AUTO resolves team size PER KERNEL from its register footprint, and the
# two kernels differ (tiled 88 regs, diagonal 94). At a single meshblock the league is only
# nmb*nang_tot -- 48 teams at nmu=3, 168 at nmu=6, on a 132-SM GPU -- so team size alone
# decides residency, and a different AUTO pick dominates. Note the regression is WORST at the
# SMALLEST league (48 teams, 0.68) and milder at 168 teams (0.83), which is the signature.
#
# Test: force both kernels to the SAME explicit team size. If the gap collapses, it was AUTO.
set -e
ATHENAK=${ATHENAK:-$HOME/athenak-rt-sc}
BUILD="$ATHENAK/build_gh200"
ml purge; ml use /home1/apps/nvidia/modulefiles; ml nvhpc-hpcx-cuda13/26.1
export PYTHONNOUSERSITE=1
export ATHENAK_BUILD="$BUILD/src" ATHENAK_LAUNCHER="ibrun" ATHENAK_DEVICE=gh200
export RT_PHASE=2 RT_N_STAR=224 RT_BLOCK_LIST=224 RT_NMU_LIST=3,6
export RT_N_REPEAT=3 RT_NLIM=50 RT_TILE_SIZES=0 RT_WARMUP=0

cd "$ATHENAK"
for TS in 0 64 128 256 512; do
  for SW in tiled diagonal_compact; do
    export RT_SWEEPS="$SW"
    export RT_EXTRA="diag_team_size=$TS"
    export RT_OUT="$ATHENAK/rt-profiling/gh200/teamsize/ts${TS}_${SW}"
    mkdir -p "$RT_OUT"
    echo "=== team_size=$TS (0=AUTO) sweep=$SW ==="
    python3 rt-profiling/run_sweep.py
  done
done
echo "ALL DONE"
