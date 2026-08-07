#!/bin/bash
#SBATCH --job-name=rtprof_p1_a100
#SBATCH --partition=apollo                   # IAS Apollo A100 (apollo[01-02])
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=00:45:00
#SBATCH --output=%x-%j.out

# Phase 1 -- memory-fill (mesh design) + Phase-0 DCGM capability probe + static kernel resources.
# Derives N* = the largest 16^3-block mesh that FILLS the A100's memory (no throughput knee).
# BUILD ON THE LOGIN NODE first (compute nodes lack CUDA headers):
#   cd $ATHENAK && source /usr/share/Modules/init/bash && module purge && source ~/.bash_scripts/athenak_a100
#   cmake3 -S . -B build_a100 -D Kokkos_ENABLE_CUDA=ON -D Kokkos_ARCH_AMPERE80=ON \
#          -D Athena_ENABLE_MPI=ON -D CMAKE_CXX_COMPILER=$PWD/kokkos/bin/nvcc_wrapper && cmake3 --build build_a100 -j12

# worktree root: honor an explicit $ATHENAK, else the sbatch submit dir, else the rt-sc worktree
ATHENAK=${ATHENAK:-${SLURM_SUBMIT_DIR:-/home/cfairbairn/Projects/AIthenaRT/athenak-rtsc}}
DEVICE=a100
BUILD="$ATHENAK/build_$DEVICE"

source /etc/profile.d/modules.sh 2>/dev/null || source /usr/share/Modules/init/bash 2>/dev/null || true
module purge
source "$HOME/.bash_scripts/athenak_$DEVICE"
module load anaconda3/2024.06 2>/dev/null || true
module load cuda 2>/dev/null || true        # for cuobjdump / dcgmi paths
export PYTHONNOUSERSITE=1

[ -x "$BUILD/src/athena" ] || { echo "ERROR: $BUILD/src/athena missing -- build on login node first"; exit 1; }

export ATHENAK_BUILD="$BUILD/src"
export ATHENAK_LAUNCHER="srun -n 1"
export ATHENAK_DEVICE="$DEVICE"
export RT_OUT="${RT_OUT:-$ATHENAK/rt-profiling/$DEVICE}"
export RT_PHASE=1
export RT_N_LIST="${RT_N_LIST:-128,144,160,176,192}"
export RT_MEM_CEIL_GIB="${RT_MEM_CEIL_GIB:-34}"      # A100 = 40 GiB; leave context/frag headroom

# Phase-0 DCGM capability probe (real-time SM occupancy/activity without ncu, if the host-engine is up)
if command -v dcgmi >/dev/null 2>&1 && timeout 10 dcgmi dmon -e 1002,1003 -c 2 >/dev/null 2>&1; then
  echo "DCGM profiling fields available -> enabling real-time SM occupancy/activity sampling"
  export RT_DCGM_FIELDS="1002,1003"
else
  echo "DCGM profiling fields NOT available (host-engine down or no user access) -- nvidia-smi loading only"
fi

cd "$ATHENAK"
echo "=== static kernel resources (cuobjdump) ==="
python3 rt-profiling/kernel_resources.py
echo "=== phase 1 memory-fill ==="
python3 rt-profiling/run_sweep.py
echo "done -- results in $RT_OUT/  (read N* from results_phase1.csv)"
