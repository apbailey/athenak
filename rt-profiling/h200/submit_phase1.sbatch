#!/bin/bash
#SBATCH --job-name=rtprof_p1_h200
#SBATCH --partition=h200                      # IAS Apollo H200 (apollo[03-04])
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=00:45:00
#SBATCH --output=%x-%j.out

# Phase 1 -- memory-fill (mesh design) on the H200 (~141 GiB => a MUCH larger N* than the A100).
# BUILD ON THE LOGIN NODE first with the Hopper arch:
#   cd $ATHENAK && source /usr/share/Modules/init/bash && module purge && source ~/.bash_scripts/athenak_h200
#   cmake3 -S . -B build_h200 -D Kokkos_ENABLE_CUDA=ON -D Kokkos_ARCH_HOPPER90=ON \
#          -D Athena_ENABLE_MPI=ON -D CMAKE_CXX_COMPILER=$PWD/kokkos/bin/nvcc_wrapper && cmake3 --build build_h200 -j12

ATHENAK=${ATHENAK:-/home/cfairbairn/Projects/AIthenaRT/athenak}
DEVICE=h200
BUILD="$ATHENAK/build_$DEVICE"

source /etc/profile.d/modules.sh 2>/dev/null || source /usr/share/Modules/init/bash 2>/dev/null || true
module purge
source "$HOME/.bash_scripts/athenak_$DEVICE"
module load anaconda3/2024.06 2>/dev/null || true
module load cuda 2>/dev/null || true
export PYTHONNOUSERSITE=1

[ -x "$BUILD/src/athena" ] || { echo "ERROR: $BUILD/src/athena missing -- build on login node first"; exit 1; }

export ATHENAK_BUILD="$BUILD/src"
export ATHENAK_LAUNCHER="srun -n 1"
export ATHENAK_DEVICE="$DEVICE"
export KR_ARCH=sm90
export RT_OUT="${RT_OUT:-$ATHENAK/rt-profiling/$DEVICE}"
export RT_PHASE=1
# H200 has ~141 GiB -> candidate meshes reach much higher; adjust to your card's memory (nvidia-smi).
export RT_N_LIST="${RT_N_LIST:-192,256,320,384,448,512}"
export RT_MEM_CEIL_GIB="${RT_MEM_CEIL_GIB:-128}"

if command -v dcgmi >/dev/null 2>&1 && timeout 10 dcgmi dmon -e 1002,1003 -c 2 >/dev/null 2>&1; then
  echo "DCGM profiling fields available -> enabling real-time SM occupancy/activity sampling"
  export RT_DCGM_FIELDS="1002,1003"
else
  echo "DCGM profiling fields NOT available -- nvidia-smi loading only"
fi

cd "$ATHENAK"
echo "=== static kernel resources (cuobjdump, sm90) ==="
python3 rt-profiling/kernel_resources.py
echo "=== phase 1 memory-fill (H200) ==="
python3 rt-profiling/run_sweep.py
echo "done -- results in $RT_OUT/  (read N* from results_phase1.csv)"
