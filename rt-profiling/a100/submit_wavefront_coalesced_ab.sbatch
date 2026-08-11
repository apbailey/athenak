#!/bin/bash
#SBATCH --job-name=rtprof_wfcoal_ab_a100
#SBATCH --partition=apollo                   # IAS Apollo A100 (apollo[01-02])
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=01:30:00
#SBATCH --output=%x-%j.out

# Ledger I2 measure-first A/B: sweep=wavefront_coalesced vs baseline wavefront.
#   1. GPU correctness gates (determinism PASS + coalesced hash == wavefront; sc_atmosphere)
#   2. nsys per-cycle sweep-kernel ACTIVE time: sc_sweep3d_coal vs sc_sweep3d = the GATHER PRIZE
#   3. whole-run ZCPS (NET, includes the transpose tax) via run_sweep
# MESH CAVEAT: ir_t doubles the ir footprint, so use a mesh where 2x ir fits (N=128 at nmu=6).
# Requires build_a100 rebuilt with the wavefront_coalesced changes.

ATHENAK=${ATHENAK:-${SLURM_SUBMIT_DIR:-/home/cfairbairn/Projects/AIthenaRT/athenak-rtsc}}
DEVICE=a100
BUILD="$ATHENAK/build_$DEVICE"

source /etc/profile.d/modules.sh 2>/dev/null || source /usr/share/Modules/init/bash 2>/dev/null || true
module purge
source "$HOME/.bash_scripts/athenak_$DEVICE"
module load anaconda3/2024.06 2>/dev/null || true
module load cuda 2>/dev/null || true        # nsys on PATH
export PYTHONNOUSERSITE=1

BIN="$BUILD/src/athena"
[ -x "$BIN" ] || { echo "ERROR: $BIN missing -- rebuild build_a100 on the login node first"; exit 1; }

export ATHENAK_BUILD="$BUILD/src"
export ATHENAK_LAUNCHER="srun -n 1"
export ATHENAK_DEVICE="$DEVICE"
export RT_OUT="${RT_OUT:-$ATHENAK/rt-profiling/$DEVICE/wavefront_coalesced}"
mkdir -p "$RT_OUT"
cd "$ATHENAK"

echo "############ 1. GPU CORRECTNESS GATES ############"
declare -A HASH
for sw in wavefront wavefront_coalesced; do
  line=$(srun -n 1 "$BIN" -i tst/inputs/sc_determinism_$sw.athinput 2>&1 | grep -E "determinism (PASS|FAIL)")
  echo "  $line"
  HASH[$sw]=$(echo "$line" | grep -oE "hash 0x[0-9a-fA-F]+" | awk '{print $2}')
done
if [ -n "${HASH[wavefront]}" ] && [ "${HASH[wavefront]}" = "${HASH[wavefront_coalesced]}" ]; then
  echo "  PASS: wavefront_coalesced hash ${HASH[wavefront_coalesced]} == wavefront (bit-identical)"
else
  echo "  FAIL: wavefront=${HASH[wavefront]} coalesced=${HASH[wavefront_coalesced]} (NOT bit-identical)"
fi
srun -n 1 "$BIN" -i tst/inputs/sc_atmosphere.athinput nr_radiation/sweep=wavefront_coalesced 2>&1 \
  | grep -iE "atmosphere test (PASSED|FAILED)"

echo "############ 2. nsys sweep-kernel ACTIVE time (gather prize) ############"
# N=128 (2x ir fits): block ladder 16/32/64/128 at nmu=6, plus 128 at nmu=3.
export RT_NSYS_NLIM=10
export RT_NSYS_CONFIGS="16:128:6,32:128:6,64:128:6,128:128:6,64:128:3"
if command -v nsys >/dev/null 2>&1; then
  RT_NSYS_SWEEP=wavefront            python3 rt-profiling/launch_overhead/measure_nsys.py
  RT_NSYS_SWEEP=wavefront_coalesced  python3 rt-profiling/launch_overhead/measure_nsys.py
  echo "  --- transpose tax (coalesced only) ---"
  RT_NSYS_SWEEP=wavefront_coalesced RT_NSYS_MATCH='sc_ir_transpose' RT_OUT="$RT_OUT/transpose" \
    python3 rt-profiling/launch_overhead/measure_nsys.py
else
  echo "  nsys not on PATH -- skipping"
fi

echo "############ 3. whole-run ZCPS (NET, includes transpose) ############"
RT_PHASE=2 RT_N_STAR=128 RT_SWEEPS=wavefront,wavefront_coalesced RT_NMU_MAX=6 \
  python3 rt-profiling/run_sweep.py

echo "############ DONE ############"
echo "results in $RT_OUT/ (launch_overhead/results_nsys_*.csv) and $RT_OUT/results_both.csv"
