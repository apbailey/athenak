#!/bin/bash
#SBATCH --job-name=jacobi_nsys
#SBATCH --account=kkratter
#SBATCH --partition=gpu_standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:volta:1                 # Ocelote V100 GPU type is "volta". #SBATCH can't read shell vars
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00                    # first run builds (~30-40 min); reruns are incremental
#SBATCH --output=jacobi_nsys-%j.out

# Run the jacobi/wavefront/diagonal 20-sweep nsys attribution pass on one V100 -- SAME config.py as the P100/A100 runs
# (the GPU is the only variable), so results/analysis.txt drop into compare_sweep/v100/ next to p100/.
#
# Porting: copy this to ../<device>/submit.sbatch, set DEVICE below, and adjust the #SBATCH header and
# the module + launcher block for that cluster (SLURM parses #SBATCH before the shell runs).

# ---- IMPORTANT SHELL VARIABLES TO SET ----
ATHENAK=/home/u21/averybailey/athenak
DEVICE=v100                                # <-- the one per-device string; everything below derives from it
BENCH=benchmark/jacobi_nsys              # this benchmark's folder, relative to $ATHENAK/tst
# ---- END SHELL VARIABLES ----

case "$DEVICE" in                          # Kokkos GPU arch for $DEVICE
  p100) ARCH=PASCAL60 ;;
  v100) ARCH=VOLTA70  ;;
  a100) ARCH=AMPERE80 ;;
  *) echo "unknown DEVICE=$DEVICE -- add its Kokkos arch here" >&2; exit 1 ;;
esac
BUILD="$ATHENAK/build_$DEVICE"             # arch-specific build dir (build_v100)
RUNDIR="$BENCH/$DEVICE"                    # the run dir (manifest.json + analysis.txt + raw/)

# ---- modules (Ocelote V100: nvhpc + HPC-X with PMIx -- a different stack than P100's) ----
module purge
module load anaconda/2024.06               # python + numpy for the driver (drop/adjust if it clashes w/ nvhpc)
module load gnu13/13.2.0
module load nvidia-hpc-sdk/24.7
module load nvhpc-hpcx-cuda12/24.7         # CUDA-aware MPI with PMIx (enables srun --mpi=pmix)
export OMPI_CXX="$ATHENAK/kokkos/bin/nvcc_wrapper"   # Kokkos CUDA needs nvcc_wrapper as the host CXX
export OMPI_CC=gcc

# ---- build every run (incremental -- seconds when nothing changed; never runs a stale binary) ----
cmake -S "$ATHENAK" -B "$BUILD" \
  -D Kokkos_ENABLE_CUDA=ON \
  -D Kokkos_ARCH_${ARCH}=ON \
  -D Athena_ENABLE_MPI=ON \
  -D CMAKE_CXX_COMPILER=mpicxx \
  -D CMAKE_C_COMPILER=mpicc
cmake --build "$BUILD" -j 8

# ---- V100 launch: this toolchain has PMIx, so srun --mpi=pmix (NOT mpirun -np 1 + UCX, which the
#      older P100 HPC-X build needed). --gpu-bind=single:1 pins the one rank to the one GPU.
export ATHENAK_BUILD="$BUILD/src"          # driver runs ./athena from here
export ATHENAK_LAUNCHER="srun --gpu-bind=single:1 --mpi=pmix"
export ATHENAK_DEVICE="$DEVICE"            # partition results by hardware -> $RUNDIR

# ---- run the sweep, then the analysis. --rerun-with-nsys also profiles each config under nsys
#      (raw/<label>/<label>.nsys-rep + .nsys.txt); it ~doubles the sweep time -- drop the flag to skip.
cd "$ATHENAK/tst"
python3 benchmark/performance_driver.py "$BENCH/config.py" --rerun-with-nsys
python3 "$BENCH/analyze.py" "$RUNDIR"
echo "done -- verdict + raw perf + per-config nsys in $ATHENAK/tst/$RUNDIR/"
