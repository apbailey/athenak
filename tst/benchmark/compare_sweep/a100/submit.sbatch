#!/bin/bash
#SBATCH --job-name=compare_sweep_a100
#SBATCH --partition=apollo                 # IAS Apollo A100 partition (apollo[01-02], A100-PCIE-40GB)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:1                        # one GPU; the single rank sees exactly it (Kokkos picks it)
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=02:00:00
#SBATCH --output=/home/cfairbairn/Projects/AIthenaRT/athenak/tst/benchmark/compare_sweep/a100/slurm-%j.out

# Run the wavefront-vs-diagonal SC sweep benchmark on one IAS A100 and print the per-suite verdict.
# SAME config.py as the P100/V100/H200 runs (the GPU is the only variable), so results/analysis.txt
# drop into compare_sweep/a100/ next to p100/.
#
# Ported from Avery's p100/submit.sbatch (Ocelote) to the IAS Apollo cluster, following the local
# Magnetosphere runs/*.slurm convention: BUILD ON THE LOGIN NODE, run the prebuilt binary in the job.
# (Unlike Ocelote, the Apollo compute nodes do not carry the CUDA-toolkit headers, so an in-job cmake
# fails with "Could NOT find CUDAToolkit"; the login node has the full toolkit + an A100, so build there.)
#
#   PREREQUISITE -- build once on the login node (idempotent; rerun after any git pull):
#     cd /home/cfairbairn/Projects/AIthenaRT/athenak
#     source /usr/share/Modules/init/bash; module purge; source ~/.bash_scripts/athenak_a100
#     cmake3 -S . -B build_a100 -D Kokkos_ENABLE_CUDA=ON -D Kokkos_ARCH_AMPERE80=ON \
#            -D Athena_ENABLE_MPI=ON -D CMAKE_CXX_COMPILER=$PWD/kokkos/bin/nvcc_wrapper
#     # (use =ON, not =On: the driver's arch auto-detect matches the cache string ':BOOL=ON')
#     cmake3 --build build_a100 -j 12

# ---- IMPORTANT SHELL VARIABLES TO SET ----
ATHENAK=/home/cfairbairn/Projects/AIthenaRT/athenak
DEVICE=a100                                 # <-- the one per-device string; everything below derives from it
BENCH=benchmark/compare_sweep               # this benchmark's folder, relative to $ATHENAK/tst
# ---- END SHELL VARIABLES ----

case "$DEVICE" in                           # Kokkos GPU arch for $DEVICE (build-time; recorded here for reference)
  p100) ARCH=PASCAL60 ;;
  v100) ARCH=VOLTA70  ;;
  a100) ARCH=AMPERE80 ;;
  h200) ARCH=HOPPER90 ;;
  *) echo "unknown DEVICE=$DEVICE -- add its Kokkos arch here" >&2; exit 1 ;;
esac
BUILD="$ATHENAK/build_$DEVICE"              # arch-specific build dir (build_a100), populated on the login node
RUNDIR="$BENCH/$DEVICE"                      # the run dir (manifest.json + analysis.txt + raw/)

# ---- modules (IAS Apollo: nvhpc 24.5 + CUDA-aware OpenMPI 4.1.6 + CUDA 12.4 runtime; anaconda for the
#      driver's numpy). `module` is not defined in a non-login sbatch shell, so source its init first.
source /etc/profile.d/modules.sh 2>/dev/null || source /usr/share/Modules/init/bash 2>/dev/null || true
module purge
source "$HOME/.bash_scripts/athenak_$DEVICE"   # loads nvhpc/24.5, openmpi/cuda-12.4/..., cudatoolkit/12.4
module load anaconda3/2024.06                  # python3 + numpy (driver imports test_suite/athena_read)
export PYTHONNOUSERSITE=1                       # ignore ~/.local: its pytest is incompatible with anaconda's pluggy

# ---- the binary is built on the login node (see PREREQUISITE above); fail loudly if it is missing.
if [ ! -x "$BUILD/src/athena" ]; then
  echo "ERROR: $BUILD/src/athena not found -- build it on the LOGIN node first (see header)." >&2
  exit 1
fi

# ---- launch env: inside the allocation a plain `srun -n 1` runs the one GPU rank (the Magnetosphere
#      runs/*.slurm convention on IAS OpenMPI -- no mpirun/UCX vars, no --mpi flag needed here).
export ATHENAK_BUILD="$BUILD/src"           # driver runs ./athena from here
export ATHENAK_LAUNCHER="srun -n 1"         # driver launches: srun -n 1 ./athena -i <deck>
export ATHENAK_DEVICE="$DEVICE"             # partition results by hardware -> $RUNDIR (a100 auto-detects too)

# ---- run the sweep, then the analysis. (Avery's p100 run added --rerun-with-nsys for a second nsys
#      profiling pass; omitted here -- nsys-under-srun is unvalidated on Apollo and the reported verdict
#      comes only from the kernels+iteration probes. Add the flag back to also collect .nsys-rep files.)
cd "$ATHENAK/tst"
python3 benchmark/performance_driver.py "$BENCH/config.py"
python3 "$BENCH/analyze.py" "$RUNDIR"
echo "done -- verdict + raw perf in $ATHENAK/tst/$RUNDIR/"
