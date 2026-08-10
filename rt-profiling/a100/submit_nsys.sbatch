#!/bin/bash
#SBATCH --job-name=rtprof_nsys_a100
#SBATCH --partition=apollo                   # IAS Apollo A100 (apollo[01-02])
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out

# Lever-1 measure-first: size the wavefront sweep's inter-plane launch-bubble prize with
# fence-free nsys runs (no <output perf> block => async par_for, production-like), then
# gap_fraction = (span-active)/span per steady-state cycle. MEASUREMENT ONLY -- no solver
# change, no rebuild; uses the existing build_a100 binary.
#   gap_fraction > ~10% -> a fused single-launch sweep is worth building
#   gap_fraction < ~5%  -> retire Lever 1 (see rt-profiling/REPORT.md + plan)

# worktree root: honor an explicit $ATHENAK, else the sbatch submit dir, else the rt-sc worktree
ATHENAK=${ATHENAK:-${SLURM_SUBMIT_DIR:-/home/cfairbairn/Projects/AIthenaRT/athenak-rtsc}}
DEVICE=a100
BUILD="$ATHENAK/build_$DEVICE"

source /etc/profile.d/modules.sh 2>/dev/null || source /usr/share/Modules/init/bash 2>/dev/null || true
module purge
source "$HOME/.bash_scripts/athenak_$DEVICE"
module load anaconda3/2024.06 2>/dev/null || true
module load cuda 2>/dev/null || true        # for nsys on PATH
export PYTHONNOUSERSITE=1

[ -x "$BUILD/src/athena" ] || { echo "ERROR: $BUILD/src/athena missing -- build on login node first"; exit 1; }
command -v nsys >/dev/null 2>&1 || { echo "ERROR: nsys not on PATH (module load cuda?)"; exit 1; }

export ATHENAK_BUILD="$BUILD/src"
export ATHENAK_LAUNCHER="srun -n 1"
export ATHENAK_DEVICE="$DEVICE"
export RT_OUT="${RT_OUT:-$ATHENAK/rt-profiling/$DEVICE}"
export RT_NSYS_NLIM="${RT_NSYS_NLIM:-10}"
# optional: RT_NSYS_CONFIGS="176:176:6,88:176:6,16:176:6,176:176:3"  RT_NSYS_MATCH=...

cd "$ATHENAK"
echo "=== Lever-1 launch-overhead measurement (nsys, fence-free) ==="
nsys --version 2>/dev/null | head -1
python3 rt-profiling/launch_overhead/measure_nsys.py
echo "done -- results in $RT_OUT/launch_overhead/results_nsys.csv"
