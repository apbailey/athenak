#!/bin/bash
#SBATCH --job-name=rtprof_p1_b200
#SBATCH -p gb
#SBATCH -A AST22008
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH -t 00:45:00
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err

# Phase 1 -- memory-fill (mesh design) on Vista B200 (~180 GiB HBM).
# BUILD ON THE LOGIN NODE first:
#   cd $HOME/athenak-rt-sc && ml purge && ml use /home1/apps/nvidia/modulefiles && ml nvhpc-hpcx-cuda13/26.1
#   bash rt-profiling/vista/build_b200.sh

set -e
ATHENAK=${ATHENAK:-$HOME/athenak-rt-sc}
DEVICE=b200
BUILD="$ATHENAK/build_$DEVICE"

ml purge
ml use /home1/apps/nvidia/modulefiles
ml nvhpc-hpcx-cuda13/26.1
export PYTHONNOUSERSITE=1

[ -x "$BUILD/src/athena" ] || { echo "ERROR: $BUILD/src/athena missing -- build on login node first"; exit 1; }

export ATHENAK_BUILD="$BUILD/src"
export ATHENAK_LAUNCHER="ibrun"
export ATHENAK_DEVICE="$DEVICE"
export KR_ARCH=sm_100
export RT_OUT="${RT_OUT:-$ATHENAK/rt-profiling/$DEVICE}"
export RT_PHASE=1
export RT_N_LIST="${RT_N_LIST:-256,320,384,448,512,576}"
export RT_MEM_CEIL_GIB="${RT_MEM_CEIL_GIB:-160}"

if command -v dcgmi >/dev/null 2>&1 && timeout 10 dcgmi dmon -e 1002,1003 -c 2 >/dev/null 2>&1; then
  echo "DCGM profiling fields available -> enabling real-time SM occupancy/activity sampling"
  export RT_DCGM_FIELDS="1002,1003"
else
  echo "DCGM profiling fields NOT available -- nvidia-smi loading only"
fi

cd "$ATHENAK"
echo "=== GPU info ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo "=== static kernel resources (cuobjdump, sm_100) ==="
python3 rt-profiling/kernel_resources.py
echo "=== phase 1 memory-fill (B200) ==="
python3 rt-profiling/run_sweep.py
echo "done -- results in $RT_OUT/  (read N* from results_phase1.csv)"
