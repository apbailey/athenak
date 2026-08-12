#!/bin/bash
#SBATCH --job-name=rtprof_hoist_ab_a100
#SBATCH --partition=apollo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out

# I3 (sc_hoist) A/B on A100:
#   1. GPU correctness gate: wavefront determinism hash, sc_hoist=false vs true, MUST MATCH bit-for-bit
#      (the precomputed per-ray invariant table must reproduce the recompute path exactly).
#   2. whole-run ZCPS: sweep=wavefront, sc_hoist=false(baseline) vs true, block ladder at N=176^3 (N*).
#      sc_hoist adds only a tiny (nang_tot x 10) table, no ir memory blow-up, so the full N* mesh fits.
#      This is the clean t_on - t_off number (affect_fluid=false, the ledger golden rule). Expected
#      ~neutral (memory-latency-bound kernel); reg-count check (section 3) is the leading indicator.

ATHENAK=${ATHENAK:-${SLURM_SUBMIT_DIR:-/home/cfairbairn/Projects/AIthenaRT/athenak-rtsc}}
DEVICE=a100
BUILD="$ATHENAK/build_$DEVICE"
source /etc/profile.d/modules.sh 2>/dev/null || source /usr/share/Modules/init/bash 2>/dev/null || true
module purge
source "$HOME/.bash_scripts/athenak_$DEVICE"
module load anaconda3/2024.06 2>/dev/null || true
export PYTHONNOUSERSITE=1
BIN="$BUILD/src/athena"
[ -x "$BIN" ] || { echo "ERROR: $BIN missing"; exit 1; }
export ATHENAK_BUILD="$BUILD/src"
RT_OUT="${RT_OUT:-$ATHENAK/rt-profiling/$DEVICE/hoist}"
mkdir -p "$RT_OUT"
cd "$ATHENAK"

echo "############ 1. GPU CORRECTNESS (sc_hoist=true == false, bit-for-bit) ############"
hb=$(srun -n 1 "$BIN" -i tst/inputs/sc_determinism_wavefront.athinput 2>&1 \
       | grep -oE "determinism PASS.*hash 0x[0-9a-f]+" | grep -oE "0x[0-9a-f]+")
hh=$(srun -n 1 "$BIN" -i tst/inputs/sc_determinism_hoist_wavefront.athinput 2>&1 \
       | grep -oE "determinism PASS.*hash 0x[0-9a-f]+" | grep -oE "0x[0-9a-f]+")
echo "  hash  baseline=[$hb]  sc_hoist=[$hh]  $([ "$hb" = "$hh" ] && echo MATCH || echo MISMATCH)"

echo "############ 2. WHOLE-RUN ZCPS A/B (wavefront: sc_hoist off vs on, N=176) ############"
python3 - "$RT_OUT" <<'PY'
import os, sys, subprocess, re
sys.path.insert(0, os.path.join(os.environ["ATHENAK_BUILD"], "..", "..", "tst", "benchmark", "rad_cost"))
import run_rad_cost as rc
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
BIN = os.path.join(os.environ["ATHENAK_BUILD"], "athena")
LAUNCH = ["srun", "-n", "1"]
N = 176; nlim = 50
rows = []
for B in (16, 22, 44, 88, 176):
    for nmu in (6, 3):
        z = {}
        for hoist in ("false", "true"):
            d = os.path.join(out, f"B{B}_nmu{nmu}_h{hoist}"); os.makedirs(d, exist_ok=True)
            text = rc.DECK.format(base=os.path.join(d, "hoist"), N=N, B=B, nlim=nlim)
            text += rc.RAD_BLOCK.format(nmu=nmu, sweep="wavefront")
            text += f"sc_hoist = {hoist}\n"
            deck = os.path.join(d, "deck.athinput"); open(deck, "w").write(text)
            r = subprocess.run(LAUNCH + [BIN, "-i", deck], capture_output=True, text=True, cwd=d)
            o = (r.stdout or "") + (r.stderr or "")
            m = re.search(r"zone-cycles/cpu_second\s*=\s*([-\d.eE+]+)", o)
            z[hoist] = float(m.group(1)) if m else None
        sp = (z["true"] / z["false"]) if (z["false"] and z["true"]) else None
        rows.append((B, nmu, z["false"], z["true"], sp))
        print(f"  B={B:<3} nmu={nmu} rays={8*nmu*(nmu+1)//2:<4} "
              f"baseline={z['false']}  sc_hoist={z['true']}  speedup={sp}")
open(os.path.join(out, "results_hoist.csv"), "w").write(
    "B,nmu,zcps_baseline,zcps_hoist,speedup\n" +
    "\n".join(f"{B},{nmu},{a},{b},{s}" for B,nmu,a,b,s in rows) + "\n")
print("  -> results_hoist.csv")
PY

echo "############ 3. REGISTER / OCCUPANCY (leading indicator) ############"
# The plausible mechanism for any I3 gain is lower reg count -> higher occupancy. Compare the
# hoisted sweep kernel (sc_sweep3d_hoist) vs the recompute kernel (sc_sweep3d) in the same binary.
if command -v cuobjdump >/dev/null 2>&1; then
  cuobjdump -res-usage "$BIN" 2>/dev/null \
    | grep -iE "sc_sweep3d(_hoist)?|Function|reg" | grep -iE "sc_sweep3d|REG:" | head -40
else
  echo "  cuobjdump not on PATH (module load cuda); skipped"
fi
echo "############ DONE ############"
