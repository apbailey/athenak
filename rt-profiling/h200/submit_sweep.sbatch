#!/bin/bash
#SBATCH --job-name=rtprof_sweep_h200
#SBATCH --partition=h200                      # IAS Apollo H200 (apollo[03-04])
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.out

# Phase 2 -- the 2D sweep at the H200 memory-derived mesh N*.  Submit once per sweep:
#   RT_N_STAR=<N*> RT_SWEEPS=wavefront sbatch rt-profiling/h200/submit_sweep.sbatch
#   RT_N_STAR=<N*> RT_SWEEPS=diagonal  sbatch rt-profiling/h200/submit_sweep.sbatch
# A larger N* than the A100 means more block rungs and a longer job -- split by nmu if needed
# (RT_NMU_MAX=4 then a 5-6 job).

ATHENAK=${ATHENAK:-/home/cfairbairn/Projects/AIthenaRT/athenak}
DEVICE=h200
BUILD="$ATHENAK/build_$DEVICE"

source /etc/profile.d/modules.sh 2>/dev/null || source /usr/share/Modules/init/bash 2>/dev/null || true
module purge
source "$HOME/.bash_scripts/athenak_$DEVICE"
module load anaconda3/2024.06 2>/dev/null || true
export PYTHONNOUSERSITE=1

[ -x "$BUILD/src/athena" ] || { echo "ERROR: $BUILD/src/athena missing -- build on login node first"; exit 1; }
[ -n "$RT_N_STAR" ] || { echo "ERROR: set RT_N_STAR (read it from Phase 1's results_phase1.csv)"; exit 1; }

export ATHENAK_BUILD="$BUILD/src"
export ATHENAK_LAUNCHER="srun -n 1"
export ATHENAK_DEVICE="$DEVICE"
export RT_OUT="${RT_OUT:-$ATHENAK/rt-profiling/$DEVICE}"
export RT_PHASE=2
export RT_N_STAR
export RT_SWEEPS="${RT_SWEEPS:-wavefront}"
export RT_NMU_MAX="${RT_NMU_MAX:-6}"
export RT_N_REPEAT="${RT_N_REPEAT:-3}"
export RT_NLIM="${RT_NLIM:-50}"

if [ -z "$RT_DCGM_FIELDS" ] && command -v dcgmi >/dev/null 2>&1 && \
   timeout 10 dcgmi dmon -e 1002,1003 -c 2 >/dev/null 2>&1; then
  export RT_DCGM_FIELDS="1002,1003"
fi

cd "$ATHENAK"
echo "=== phase 2 sweep (H200): N*=$RT_N_STAR sweeps=$RT_SWEEPS nmu=1..$RT_NMU_MAX ==="
python3 rt-profiling/run_sweep.py
echo "done -- results_${RT_SWEEPS%,*}.csv in $RT_OUT/"
