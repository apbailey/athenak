#!/bin/bash
#SBATCH --job-name=rtna_gh200
#SBATCH -p gh
#SBATCH -A AST26021
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
# The first composition attempt fixed tile_na=32, which collapses the league by 24-28x and
# starves the GPU -- it measured one bad operating point, not the hypothesis. tile_na trades
# coalescing WIDTH against TEAM COUNT: league = ntile*nmb*ceil(nang_tot/tile_na). Sweep it.
# Mesh reduced to N=112 so the angle_inner ir_normal companion (which DOUBLES ir) fits at
# nmu=6 -- the previous run OOMed at the normal-layout N*=224/256 for exactly that reason.
set -e
ATHENAK=${ATHENAK:-$HOME/athenak-rt-sc}
BUILD="$ATHENAK/build_gh200"
ml purge; ml use /home1/apps/nvidia/modulefiles; ml nvhpc-hpcx-cuda13/26.1
export PYTHONNOUSERSITE=1
export ATHENAK_BUILD="$BUILD/src" ATHENAK_LAUNCHER="ibrun" ATHENAK_DEVICE=gh200
export RT_PHASE=2 RT_N_STAR=112 RT_BLOCK_LIST=112 RT_NMU_LIST=3,6
export RT_N_REPEAT=3 RT_NLIM=50
# keep outputs off $HOME -- the last job died on an Errno 122 there
BASE=$SCRATCH/rt_tilena/gh200
cd "$ATHENAK"

for cfg in "normal:wavefront:0:" "normal:tiled:16:" "angle_inner:wavefront:0:" ; do
  IFS=: read L S T NA <<< "$cfg"
  export RT_IR_LAYOUT=$L RT_SWEEPS=$S RT_TILE_SIZES=$T RT_EXTRA=""
  export RT_OUT="$BASE/${L}_${S}"; mkdir -p "$RT_OUT"
  echo "=== $L $S ==="; python3 rt-profiling/run_sweep.py
done
for NA in 2 4 8 16 32; do
  export RT_IR_LAYOUT=angle_inner RT_SWEEPS=tiled RT_TILE_SIZES=16 RT_EXTRA="tile_na=$NA"
  export RT_OUT="$BASE/angle_inner_tiled_na$NA"; mkdir -p "$RT_OUT"
  echo "=== angle_inner tiled tile_na=$NA ==="; python3 rt-profiling/run_sweep.py
done
echo ALL DONE
