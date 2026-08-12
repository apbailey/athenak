#!/bin/bash
#SBATCH --job-name=rttile_b200
#SBATCH -p gb
#SBATCH -A AST26021
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH -t 04:00:00
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err

# I7 step 3 -- tiled (KBA) sweep performance: tile_size x meshblock size, at the two angular
# orders that matter. Only the LARGE-block regime is scanned: tiling is a no-op at B=16 (one
# tile) and the diagonal already wins there. Reference sweeps run in the SAME job so the
# comparison is thermally paired. See rt-profiling/TILED_SWEEP_DESIGN.md.
set -e
ATHENAK=${ATHENAK:-$HOME/athenak-rt-sc}
DEVICE=b200
BUILD="$ATHENAK/build_$DEVICE"

ml purge
ml use /home1/apps/nvidia/modulefiles
ml nvhpc-hpcx-cuda13/26.1
export PYTHONNOUSERSITE=1

[ -x "$BUILD/src/athena" ] || { echo "ERROR: $BUILD/src/athena missing"; exit 1; }

export ATHENAK_BUILD="$BUILD/src"
export ATHENAK_LAUNCHER="ibrun"
export ATHENAK_DEVICE="$DEVICE"
export RT_OUT="${RT_OUT:-$ATHENAK/rt-profiling/$DEVICE/tile}"
export RT_PHASE=2
export RT_N_STAR=256
export RT_BLOCK_LIST=64,128,256
export RT_SWEEPS="tiled,wavefront,diagonal_compact"
export RT_TILE_SIZES=0,8,16,32,64
export RT_NMU_LIST=3,6
export RT_N_REPEAT=3
export RT_NLIM=50

if [ -z "$RT_DCGM_FIELDS" ] && command -v dcgmi >/dev/null 2>&1 && \
   timeout 10 dcgmi dmon -e 1002,1003 -c 2 >/dev/null 2>&1; then
  export RT_DCGM_FIELDS="1002,1003"
fi

mkdir -p "$RT_OUT"
cd "$ATHENAK"

# ---- GPU correctness gate ------------------------------------------------------------------
# The bit-identical claim for sweep=tiled was verified on a serial CPU build, where there is no
# concurrency to expose an ordering bug. Re-verify it HERE, on the real GPU with real
# parallelism, before spending the node on throughput numbers. Every sweep must produce the same
# XOR-hash; if tiled disagrees, the cross-tile ordering is wrong and the perf numbers would be
# meaningless, so abort.
echo "=== GPU determinism gate ==="
GATE_DIR="$RT_OUT/_gate"; mkdir -p "$GATE_DIR"
hashes=""
for deck in wavefront diagonal_compact tiled; do
  out=$(ibrun "$BUILD/src/athena" -i "$ATHENAK/tst/inputs/sc_determinism_${deck}.athinput" \
        -d "$GATE_DIR" 2>&1) || { echo "FATAL: $deck run failed"; echo "$out" | tail -20; exit 1; }
  echo "$out" | grep -iE "determinism (PASS|FAIL)" || { echo "FATAL: no verdict for $deck"; exit 1; }
  echo "$out" | grep -q "determinism PASS" || { echo "FATAL: $deck is non-deterministic"; exit 1; }
  hashes="$hashes $(echo "$out" | grep -oE 'hash 0x[0-9a-f]+' | head -1 | cut -d' ' -f2)"
done
uniq_count=$(echo $hashes | tr ' ' '\n' | sort -u | wc -l)
if [ "$uniq_count" -ne 1 ]; then
  echo "FATAL: sweeps disagree on the GPU -- hashes:$hashes"; exit 1
fi
echo "GPU determinism gate PASSED -- all sweeps hash$hashes"


echo "=== tile sweep (b200): N*=$RT_N_STAR B=$RT_BLOCK_LIST tiles=$RT_TILE_SIZES nmu=$RT_NMU_LIST ==="
python3 rt-profiling/run_sweep.py
echo "done -- results in $RT_OUT/"
