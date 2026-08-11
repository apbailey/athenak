#!/bin/bash
#SBATCH --job-name=rtprof_ai_ab_a100
#SBATCH --partition=apollo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out

# Native I2 (ir_layout=angle_inner) A/B on A100:
#   1. GPU correctness gates (jmean + boundary-exchange bridge, angle_inner == normal, bit-for-bit)
#   2. whole-run ZCPS: sweep=wavefront, ir_layout=normal vs angle_inner, block ladder at N=128^3
#      (angle_inner keeps ir + ir_normal = 2x ir, so N=128 to fit). This is the REAL speedup number
#      (native path has NO per-sweep transpose; only the O(surface) shell bridge).

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
RT_OUT="${RT_OUT:-$ATHENAK/rt-profiling/$DEVICE/angle_inner}"
mkdir -p "$RT_OUT"
cd "$ATHENAK"

echo "############ 1. GPU CORRECTNESS (bit-exact vs normal) ############"
jw=$(srun -n 1 "$BIN" -i tst/inputs/sc_determinism_wavefront.athinput 2>&1 | grep -oE "jmean 0x[0-9a-f]+")
ja=$(srun -n 1 "$BIN" -i tst/inputs/sc_determinism_angle_inner.athinput 2>&1 | grep -oE "jmean 0x[0-9a-f]+")
echo "  jmean  normal=[$jw]  angle_inner=[$ja]  $([ "$jw" = "$ja" ] && echo MATCH || echo MISMATCH)"
ew=$(srun -n 1 "$BIN" -i tst/inputs/sc_determinism_wavefront.athinput problem/test_exchange=true 2>&1 | grep -oE "exchange_test ir_layout=normal hash 0x[0-9a-f]+" | grep -oE "0x[0-9a-f]+")
ea=$(srun -n 1 "$BIN" -i tst/inputs/sc_determinism_angle_inner.athinput problem/test_exchange=true 2>&1 | grep -oE "hash 0x[0-9a-f]+" | grep -oE "0x[0-9a-f]+")
echo "  exch   normal=[$ew]  angle_inner=[$ea]  $([ "$ew" = "$ea" ] && echo MATCH || echo MISMATCH)"

echo "############ 2. WHOLE-RUN ZCPS A/B (wavefront: normal vs angle_inner, N=128) ############"
python3 - "$RT_OUT" <<'PY'
import os, sys, subprocess, re
sys.path.insert(0, os.path.join(os.environ["ATHENAK_BUILD"], "..", "..", "tst", "benchmark", "rad_cost"))
import run_rad_cost as rc
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
BIN = os.path.join(os.environ["ATHENAK_BUILD"], "athena")
LAUNCH = ["srun", "-n", "1"]
N = 128; nlim = 50
rows = []
for B in (16, 32, 64, 128):
    for nmu in (6, 3):
        z = {}
        for lay in ("normal", "angle_inner"):
            d = os.path.join(out, f"B{B}_nmu{nmu}_{lay}"); os.makedirs(d, exist_ok=True)
            text = rc.DECK.format(base=os.path.join(d, "ai"), N=N, B=B, nlim=nlim)
            text += rc.RAD_BLOCK.format(nmu=nmu, sweep="wavefront")
            text += f"ir_layout = {lay}\n"
            deck = os.path.join(d, "deck.athinput"); open(deck, "w").write(text)
            r = subprocess.run(LAUNCH + [BIN, "-i", deck], capture_output=True, text=True, cwd=d)
            o = (r.stdout or "") + (r.stderr or "")
            m = re.search(r"zone-cycles/cpu_second\s*=\s*([-\d.eE+]+)", o)
            z[lay] = float(m.group(1)) if m else None
        sp = (z["angle_inner"] / z["normal"]) if (z["normal"] and z["angle_inner"]) else None
        rows.append((B, nmu, z["normal"], z["angle_inner"], sp))
        print(f"  B={B:<3} nmu={nmu} rays={8*nmu*(nmu+1)//2:<4} "
              f"normal={z['normal']}  angle_inner={z['angle_inner']}  speedup={sp}")
open(os.path.join(out, "results_angle_inner.csv"), "w").write(
    "B,nmu,zcps_normal,zcps_angle_inner,speedup\n" +
    "\n".join(f"{B},{nmu},{n},{a},{s}" for B,nmu,n,a,s in rows) + "\n")
print("  -> results_angle_inner.csv")
PY
echo "############ DONE ############"
