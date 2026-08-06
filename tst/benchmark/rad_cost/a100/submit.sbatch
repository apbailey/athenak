#!/bin/bash
#SBATCH --job-name=rad_cost_a100
#SBATCH --partition=apollo                   # IAS Apollo A100 (apollo[01-02])
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out

# LTE radiation-cost benchmark (ZCPS radiation on/off + sweep-vs-hydro kernel split) on one H200.
# BUILD ON THE LOGIN NODE first (compute nodes lack CUDA headers), then submit this:
#   cd $ATHENAK && source /usr/share/Modules/init/bash && module purge && source ~/.bash_scripts/athenak_a100
#   cmake3 -S . -B build_a100 -D Kokkos_ENABLE_CUDA=ON -D Kokkos_ARCH_AMPERE80=ON \
#          -D Athena_ENABLE_MPI=ON -D CMAKE_CXX_COMPILER=$PWD/kokkos/bin/nvcc_wrapper && cmake3 --build build_a100 -j12

ATHENAK=${ATHENAK:-/home/cfairbairn/Projects/AIthenaRT/athenak}
DEVICE=a100
BUILD="$ATHENAK/build_$DEVICE"

source /etc/profile.d/modules.sh 2>/dev/null || source /usr/share/Modules/init/bash 2>/dev/null || true
module purge
source "$HOME/.bash_scripts/athenak_$DEVICE"
module load anaconda3/2024.06 2>/dev/null || true
export PYTHONNOUSERSITE=1

[ -x "$BUILD/src/athena" ] || { echo "ERROR: $BUILD/src/athena missing -- build on login node first"; exit 1; }

export ATHENAK_BUILD="$BUILD/src"
export ATHENAK_LAUNCHER="srun -n 1"
export ATHENAK_DEVICE="$DEVICE"
export RAD_COST_PROFILE="${RAD_COST_PROFILE:-full}"
export RAD_COST_OUT="$ATHENAK/tst/benchmark/rad_cost/$DEVICE"

cd "$ATHENAK/tst"
python3 benchmark/rad_cost/run_rad_cost.py
echo "done -- results in $RAD_COST_OUT/results.csv"
