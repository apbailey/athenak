#!/bin/bash
#SBATCH --job-name=rtcomp_gh200
#SBATCH -p gh
#SBATCH -A AST26021
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 03:00:00
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
# Do I7 (tiled) and I2 (angle_inner) COMPOSE? Both win most at large blocks but fix different
# bottlenecks -- tiling supplies teams, angle-major lanes supply gather coalescing. Four corners
# per (B, nmu): {wavefront, tiled} x {normal, angle_inner}. angle_inner is uniform-mesh 3D only.
set -e
ATHENAK=${ATHENAK:-$HOME/athenak-rt-sc}
BUILD="$ATHENAK/build_gh200"
ml purge; ml use /home1/apps/nvidia/modulefiles; ml nvhpc-hpcx-cuda13/26.1
export PYTHONNOUSERSITE=1
export ATHENAK_BUILD="$BUILD/src" ATHENAK_LAUNCHER="ibrun" ATHENAK_DEVICE=gh200
export RT_PHASE=2 RT_N_STAR=224 RT_BLOCK_LIST=112,224 RT_NMU_LIST=3,6
export RT_N_REPEAT=3 RT_NLIM=50

cd "$ATHENAK"
echo "=== GPU determinism gate (all four corners must agree) ==="
G="$ATHENAK/rt-profiling/gh200/compose/_gate"; mkdir -p "$G"; hashes=""
for deck in wavefront tiled angle_inner tiled_angle_inner; do
  out=$(ibrun "$BUILD/src/athena" -i "$ATHENAK/tst/inputs/sc_determinism_${deck}.athinput" -d "$G" 2>&1) \
    || { echo "FATAL: $deck failed"; echo "$out" | tail -15; exit 1; }
  echo "$out" | grep -iE "determinism (PASS|FAIL)" || { echo "FATAL: no verdict $deck"; exit 1; }
  echo "$out" | grep -q "determinism PASS" || { echo "FATAL: $deck non-deterministic"; exit 1; }
  hashes="$hashes $(echo "$out" | grep -oE 'hash 0x[0-9a-f]+' | head -1 | cut -d' ' -f2)"
done
[ "$(echo $hashes | tr ' ' '\n' | sort -u | wc -l)" -eq 1 ] || { echo "FATAL: hashes differ:$hashes"; exit 1; }
echo "GATE PASSED -- all four corners hash$hashes"

for LAYOUT in normal angle_inner; do
  for SW in wavefront tiled; do
    export RT_IR_LAYOUT=$LAYOUT RT_SWEEPS=$SW
    if [ "$SW" = "tiled" ]; then export RT_TILE_SIZES=16,28; else export RT_TILE_SIZES=0; fi
    export RT_OUT="$ATHENAK/rt-profiling/gh200/compose/${LAYOUT}_${SW}"
    mkdir -p "$RT_OUT"
    echo "=== layout=$LAYOUT sweep=$SW tiles=$RT_TILE_SIZES ==="
    python3 rt-profiling/run_sweep.py
  done
done
echo ALL DONE
