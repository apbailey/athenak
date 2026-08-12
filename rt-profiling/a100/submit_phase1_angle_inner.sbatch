#!/bin/bash
#SBATCH --job-name=rtprof_p1_ai_a100
#SBATCH --partition=apollo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=00:45:00
#SBATCH --output=%x-%j.out

# Phase 1 memory-fill for ir_layout=angle_inner (I2): its ir_normal companion roughly doubles the
# ir footprint, so it needs its OWN N* -- cannot reuse the normal-layout N*=176 from the original
# report. Same methodology (hold worst corner B=16,nmu=6, grow N until OOM), same run_sweep.py,
# just RT_IR_LAYOUT=angle_inner + RT_SWEEPS=wavefront (the only sweep angle_inner supports) and a
# finer/lower N_LIST bracketing the expected (roughly halved) memory-fill point.

ATHENAK=${ATHENAK:-${SLURM_SUBMIT_DIR:-/home/cfairbairn/Projects/AIthenaRT/athenak-rtsc}}
DEVICE=a100
BUILD="$ATHENAK/build_$DEVICE"

source /etc/profile.d/modules.sh 2>/dev/null || source /usr/share/Modules/init/bash 2>/dev/null || true
module purge
source "$HOME/.bash_scripts/athenak_$DEVICE"
module load anaconda3/2024.06 2>/dev/null || true
module load cuda 2>/dev/null || true
export PYTHONNOUSERSITE=1

[ -x "$BUILD/src/athena" ] || { echo "ERROR: $BUILD/src/athena missing"; exit 1; }

export ATHENAK_BUILD="$BUILD/src"
export ATHENAK_LAUNCHER="srun -n 1"
export ATHENAK_DEVICE="$DEVICE"
export RT_OUT="${RT_OUT:-$ATHENAK/rt-profiling/$DEVICE/angle_inner_full}"
export RT_PHASE=1
export RT_SWEEPS=wavefront
export RT_IR_LAYOUT=angle_inner
export RT_N_LIST="${RT_N_LIST:-112,128,144,152,160,176}"
export RT_MEM_CEIL_GIB="${RT_MEM_CEIL_GIB:-34}"

if command -v dcgmi >/dev/null 2>&1 && timeout 10 dcgmi dmon -e 1002,1003 -c 2 >/dev/null 2>&1; then
  export RT_DCGM_FIELDS="1002,1003"
fi

cd "$ATHENAK"
echo "=== phase 1 memory-fill: ir_layout=angle_inner, sweep=wavefront ==="
python3 rt-profiling/run_sweep.py
echo "done -- results in $RT_OUT/results_phase1.csv (read N*_ai from it)"
