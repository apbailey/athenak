#!/bin/bash
#SBATCH --job-name=rtprof_sweep_b200
#SBATCH -p gb
#SBATCH -A AST22008
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH -t 06:00:00
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err

# Phase 2 -- 2D sweep at B200 memory-derived N*. Submit once per sweep:
#   RT_N_STAR=<N*> RT_SWEEPS=wavefront sbatch rt-profiling/b200/submit_sweep.sbatch
#   RT_N_STAR=<N*> RT_SWEEPS=diagonal sbatch rt-profiling/b200/submit_sweep.sbatch
#   RT_N_STAR=<N*> RT_SWEEPS=diagonal_compact sbatch rt-profiling/b200/submit_sweep.sbatch

set -e
ATHENAK=${ATHENAK:-$HOME/athenak-rt-sc}
DEVICE=b200
BUILD="$ATHENAK/build_$DEVICE"

ml purge
ml use /home1/apps/nvidia/modulefiles
ml nvhpc-hpcx-cuda13/26.1
export PYTHONNOUSERSITE=1

[ -x "$BUILD/src/athena" ] || { echo "ERROR: $BUILD/src/athena missing -- build on login node first"; exit 1; }
[ -n "$RT_N_STAR" ] || { echo "ERROR: set RT_N_STAR (read it from Phase 1's results_phase1.csv)"; exit 1; }

export ATHENAK_BUILD="$BUILD/src"
export ATHENAK_LAUNCHER="ibrun"
export ATHENAK_DEVICE="$DEVICE"
export RT_OUT="${RT_OUT:-$ATHENAK/rt-profiling/$DEVICE}"
export RT_PHASE=2
export RT_N_STAR
export RT_SWEEPS="${RT_SWEEPS:-wavefront}"
export RT_NMU_MAX="${RT_NMU_MAX:-6}"
export RT_N_REPEAT="${RT_N_REPEAT:-3}"
export RT_NLIM="${RT_NLIM:-50}"

if [ -z "$RT_DCGM_FIELDS" ] && command -v dcgmi >/dev/null 2>&1 && \
   timeout 10 dcgmi dmon -e 1002,1003 -c 2 >/dev/null 2>&1; then
  export RT_DCGM_FIELDS="1002,1003"
fi

cd "$ATHENAK"
echo "=== phase 2 sweep (B200): N*=$RT_N_STAR sweeps=$RT_SWEEPS nmu=1..$RT_NMU_MAX ==="
python3 rt-profiling/run_sweep.py
echo "done -- results_${RT_SWEEPS%,*}.csv in $RT_OUT/"
