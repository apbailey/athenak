#!/bin/bash
#SBATCH --job-name=rad_prof_a100
#SBATCH --partition=apollo                   # IAS Apollo A100 (apollo[01-02])
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out

# White-box GPU profiling (Nsight Compute roofline/occupancy/warp-efficiency + Nsight Systems
# launch-gap timeline) for ONE saturated LTE sweep. Companion to submit.sbatch (throughput).
# BUILD ON THE LOGIN NODE first (see submit.sbatch header). ncu counter collection may require
# profiling permissions on the node (NVreg_RestrictProfilingToAdminUsers=0); ask the admin if
# ncu reports "ERR_NVGPUCTRPERM".

ATHENAK=${ATHENAK:-/home/cfairbairn/Projects/AIthenaRT/athenak}
DEVICE=a100
BUILD="$ATHENAK/build_$DEVICE"

source /etc/profile.d/modules.sh 2>/dev/null || source /usr/share/Modules/init/bash 2>/dev/null || true
module purge
source "$HOME/.bash_scripts/athenak_$DEVICE"
module load anaconda3/2024.06 2>/dev/null || true
module load cuda 2>/dev/null || true          # provides ncu / nsys (Nsight)
export PYTHONNOUSERSITE=1

[ -x "$BUILD/src/athena" ] || { echo "ERROR: $BUILD/src/athena missing -- build on login node first"; exit 1; }

# Kokkos NVTX connector: makes par_for labels (e.g. sc_sweep3d) into NVTX ranges so ncu/nsys can
# filter by label (PROF_FILTER=nvtx, the default). If you do not have kokkos-tools built, leave
# KOKKOS_TOOLS_LIBS unset and set PROF_FILTER=window below to profile a launch-index window instead.
#   git clone https://github.com/kokkos/kokkos-tools && cd kokkos-tools && cmake -B build && cmake --build build
#   export KOKKOS_TOOLS_LIBS=$PWD/build/profiling/nvtx-connector/libkp_nvtx_connector.so
if [ -n "$KOKKOS_TOOLS_LIBS" ]; then export PROF_FILTER="${PROF_FILTER:-nvtx}";
else export PROF_FILTER="${PROF_FILTER:-window}"; fi

export ATHENAK_BUILD="$BUILD/src"
export ATHENAK_LAUNCHER="srun -n 1"
export ATHENAK_DEVICE="$DEVICE"
export PROFILE_OUT="${PROFILE_OUT:-$ATHENAK/tst/benchmark/rad_cost/$DEVICE/profile}"
# saturated wavefront headline point (override any PROF_* on the sbatch command line):
export PROF_B="${PROF_B:-32}"
export PROF_NMB="${PROF_NMB:-64}"
export PROF_NMU="${PROF_NMU:-3}"
export PROF_SWEEP="${PROF_SWEEP:-wavefront}"

cd "$ATHENAK/tst"
python3 benchmark/rad_cost/profile_sweep.py
echo "done -- profiler outputs in $PROFILE_OUT"
