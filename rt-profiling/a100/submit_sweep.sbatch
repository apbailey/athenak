#!/bin/bash
#SBATCH --job-name=rtprof_sweep_a100
#SBATCH --partition=apollo                   # IAS Apollo A100 (apollo[01-02])
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out

# Phase 2 -- the 2D sweep (meshblock size x angular resolution) at the memory-derived mesh N*.
# Records ZCPS with and without RT for every (block, rays) cell + the sc_* vs hydro kernel split.
# Submit ONCE PER SWEEP to stay under the wall (each ~1.5-2.5 h):
#   RT_N_STAR=176 RT_SWEEPS=wavefront sbatch rt-profiling/a100/submit_sweep.sbatch
#   RT_N_STAR=176 RT_SWEEPS=diagonal  sbatch rt-profiling/a100/submit_sweep.sbatch
# (If a job nears the wall, split nmu too: RT_NMU_MAX=4 then a second job for 5-6.)

# worktree root: honor an explicit $ATHENAK, else the sbatch submit dir, else the rt-sc worktree
ATHENAK=${ATHENAK:-${SLURM_SUBMIT_DIR:-/home/cfairbairn/Projects/AIthenaRT/athenak-rtsc}}
DEVICE=a100
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

# reuse the DCGM decision from Phase 1 if the caller exported RT_DCGM_FIELDS; else best-effort re-probe
if [ -z "$RT_DCGM_FIELDS" ] && command -v dcgmi >/dev/null 2>&1 && \
   timeout 10 dcgmi dmon -e 1002,1003 -c 2 >/dev/null 2>&1; then
  export RT_DCGM_FIELDS="1002,1003"
fi

cd "$ATHENAK"
echo "=== phase 2 sweep: N*=$RT_N_STAR sweeps=$RT_SWEEPS nmu=1..$RT_NMU_MAX ==="
python3 rt-profiling/run_sweep.py
echo "done -- results_${RT_SWEEPS%,*}.csv in $RT_OUT/"
