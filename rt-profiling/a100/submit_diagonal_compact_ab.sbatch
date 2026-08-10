#!/bin/bash
#SBATCH --job-name=rtprof_dc_ab_a100
#SBATCH --partition=apollo                   # IAS Apollo A100 (apollo[01-02])
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out

# Ledger I1 A/B: sweep=diagonal_compact vs baseline diagonal vs wavefront.
#   1. GPU correctness gates (determinism PASS + diagonal_compact hash == diagonal; sc_atmosphere)
#   2. Throughput A/B: clean whole-run ZCPS across the block ladder at N*=176^3 (the golden rule)
#   3. nsys per-cycle sweep-kernel active time (diagonal vs diagonal_compact)
#   4. diag_team_size mini-sweep at 176^3 nmu=6
# Requires build_a100 rebuilt on the login node with the diagonal_compact changes.

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
export RT_OUT="${RT_OUT:-$ATHENAK/rt-profiling/$DEVICE/diagonal_compact}"
mkdir -p "$RT_OUT"
cd "$ATHENAK"

echo "############ 1. GPU CORRECTNESS GATES ############"
declare -A HASH
for sw in wavefront diagonal diagonal_compact jacobi; do
  line=$(srun -n 1 "$BIN" -i tst/inputs/sc_determinism_$sw.athinput 2>&1 | grep -E "determinism (PASS|FAIL)")
  echo "  $line"
  HASH[$sw]=$(echo "$line" | grep -oE "hash 0x[0-9a-fA-F]+" | awk '{print $2}')
done
echo "  --- bitwise equality: diagonal_compact vs diagonal ---"
if [ -n "${HASH[diagonal]}" ] && [ "${HASH[diagonal]}" = "${HASH[diagonal_compact]}" ]; then
  echo "  PASS: diagonal_compact hash ${HASH[diagonal_compact]} == diagonal (bit-identical)"
else
  echo "  FAIL: diagonal=${HASH[diagonal]} diagonal_compact=${HASH[diagonal_compact]} (NOT bit-identical)"
fi
echo "  --- sc_atmosphere (Davis Eq.30) with diagonal_compact ---"
srun -n 1 "$BIN" -i tst/inputs/sc_atmosphere.athinput nr_radiation/sweep=diagonal_compact 2>&1 \
  | grep -iE "atmosphere.*PASS|atmosphere.*FAIL|rms\|J-Jan"

echo "############ 2. THROUGHPUT A/B (clean whole-run ZCPS) ############"
RT_PHASE=2 RT_N_STAR=176 RT_SWEEPS=diagonal,diagonal_compact,wavefront RT_NMU_MAX=6 \
  python3 rt-profiling/run_sweep.py
echo "  -> $RT_OUT/results_both.csv"

echo "############ 3. nsys per-cycle sweep-kernel active time ############"
export RT_NSYS_NLIM=10
export RT_NSYS_CONFIGS="176:176:6,88:176:6,16:176:6"
if command -v nsys >/dev/null 2>&1; then
  RT_NSYS_SWEEP=diagonal          python3 rt-profiling/launch_overhead/measure_nsys.py
  RT_NSYS_SWEEP=diagonal_compact  python3 rt-profiling/launch_overhead/measure_nsys.py
else
  echo "  nsys not on PATH -- skipping active-time step"
fi

echo "############ 4. diag_team_size mini-sweep (176^3, nmu=6) ############"
python3 - "$RT_OUT" <<'PY'
import os, sys, subprocess, re
sys.path.insert(0, os.path.join(os.environ["ATHENAK_BUILD"], "..", "..", "tst", "benchmark", "rad_cost"))
import run_rad_cost as rc
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
BIN = os.path.join(os.environ["ATHENAK_BUILD"], "athena")
LAUNCH = os.environ.get("ATHENAK_LAUNCHER", "srun -n 1").split()
N = B = 176; nmu = 6; nlim = 50
rows = []
for ts in (0, 128, 256, 512):
    d = os.path.join(out, f"ts{ts}"); os.makedirs(d, exist_ok=True)
    text = rc.DECK.format(base=os.path.join(d, "ts"), N=N, B=B, nlim=nlim)
    text += rc.RAD_BLOCK.format(nmu=nmu, sweep="diagonal_compact")
    text += f"diag_team_size = {ts}\n"
    deck = os.path.join(d, "deck.athinput")
    open(deck, "w").write(text)
    r = subprocess.run(LAUNCH + [BIN, "-i", deck], capture_output=True, text=True, cwd=d)
    o = (r.stdout or "") + (r.stderr or "")
    m = re.search(r"zone-cycles/cpu_second\s*=\s*([-\d.eE+]+)", o)
    z = float(m.group(1)) if m else None
    rows.append((ts, z))
    print(f"  diag_team_size={ts:<4} ZCPS_on={z}")
open(os.path.join(out, "team_size_sweep.csv"), "w").write(
    "diag_team_size,zcps_on\n" + "\n".join(f"{ts},{z}" for ts, z in rows) + "\n")
print("  -> team_size_sweep.csv")
PY

echo "############ DONE ############"
echo "results in $RT_OUT/  (results_both.csv, launch_overhead/results_nsys_*.csv, team_size_sweep.csv)"
