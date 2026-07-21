#!/bin/bash
#SBATCH --job-name=compare_sweep
#SBATCH --account=kkratter
#SBATCH --partition=gpu_standard          # standard (NOT windfall/preemptible)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:pascal:1               # Ocelote GPU type "pascal" (P100). #SBATCH can't read shell vars
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00                   # first run builds (~30-40 min); reruns skip the build
#SBATCH --output=compare_sweep-%j.out

# Run the wavefront-vs-diagonal SC sweep benchmark on one P100 and print the per-suite verdict table.
# Reproduces the vet_bench suites (block count / block size / angular; see ~/Downloads/REPORT.md).
#
# Porting to another GPU: copy this to ../<device>/submit.sbatch and change DEVICE below -- the build
# dir, Kokkos arch, and all result paths follow from it. Only the #SBATCH header and the module block
# above/below are cluster-specific and edited by hand (SLURM parses #SBATCH before the shell runs).

# ---- IMPORTANT SHELL VARIABLES TO SET ----
ATHENAK=/home/u21/averybailey/athenak
DEVICE=p100                               # <-- the one per-device string; everything below derives from it
BENCH=benchmark/compare_sweep             # this benchmark's folder, relative to $ATHENAK/tst
# ---- END SHELL VARIABLES ----

case "$DEVICE" in                         # Kokkos GPU arch for $DEVICE
  p100) ARCH=PASCAL60 ;;
  v100) ARCH=VOLTA70  ;;
  a100) ARCH=AMPERE80 ;;
  *) echo "unknown DEVICE=$DEVICE -- add its Kokkos arch here" >&2; exit 1 ;;
esac
BUILD="$ATHENAK/build_$DEVICE"            # arch-specific build dir (build_p100)
RUNDIR="$BENCH/$DEVICE"                   # the run dir (manifest.json + analysis.txt + raw/)

# ---- modules (Ocelote P100) ----
module purge
module load anaconda/2024.06              # python + numpy (the driver imports test_suite/athena_read)
module load gnu8/8.3.0
module load cuda11/11.8
module load cmake/3.28.3
module use /opt/ohpc/pub/apps/nvidia-hpc-sdk/23.7/Linux_x86_64/23.7/comm_libs/11.8/hpcx/hpcx-2.14/modulefiles
module load hpcx                          # CUDA-aware HPC-X MPI
export OMPI_CXX="$ATHENAK/kokkos/bin/nvcc_wrapper"
export OMPI_CC=gcc

# ---- build into the arch-specific dir. Configure + build EVERY run: cmake is idempotent and
#      --build is incremental (seconds when nothing changed, a full ~40 min build only the first
#      time). Do NOT guard on "binary exists" -- that silently runs a stale binary after a git pull.
cmake -S "$ATHENAK" -B "$BUILD" \
  -D Kokkos_ENABLE_CUDA=ON \
  -D Kokkos_ARCH_${ARCH}=ON \
  -D Athena_ENABLE_MPI=ON \
  -D CMAKE_CXX_COMPILER=mpicxx \
  -D CMAKE_C_COMPILER=mpicc
cmake --build "$BUILD" -j 8

# ---- launch env: even a single GPU must go through mpirun -np 1 + these UCX vars.
#      A bare ./athena (or srun) aborts in MPI_Init ("direct launched using srun, OMPI without PMI").
export OMPI_MCA_pml=ucx
export OMPI_MCA_btl=^vader
export UCX_NET_DEVICES=mlx4_0:1           # harmless "not available" warning on nodes without mlx4
export UCX_WARN_UNUSED_ENV_VARS=n
export ATHENAK_BUILD="$BUILD/src"         # driver runs ./athena from here
export ATHENAK_LAUNCHER="mpirun -np 1"    # driver launches: mpirun -np 1 ./athena -i <deck>
export ATHENAK_DEVICE="$DEVICE"           # partition results by hardware -> $RUNDIR

# ---- run the sweep, then the analysis. --rerun-with-nsys also profiles each config under nsys
#      (raw/<label>/<label>.nsys-rep + .nsys.txt); it ~doubles the sweep time -- drop the flag to skip.
cd "$ATHENAK/tst"
python3 benchmark/performance_driver.py "$BENCH/config.py" --rerun-with-nsys
python3 "$BENCH/analyze.py" "$RUNDIR"
echo "done -- verdict + raw perf + per-config nsys in $ATHENAK/tst/$RUNDIR/"
