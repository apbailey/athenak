#!/usr/bin/env python3
"""rad_cost -- LTE radiation-cost benchmark for the rt-sc solver.

Measures, in the simplest LTE regime (eps=1, ops=0 -> use_ali=false, ONE formal solution per
cycle, no in-meshblock iteration via iter_max=itermin=1):

  * whole-run ZCPS (zone-cycles/cpu_second, driver.cpp) with radiation OFF vs ON -> slowdown factor;
  * the per-kernel split radiation (sc_*) vs hydro (h_update,hflux_*) from the <output perf> kernels
    probe (real device time, fenced) -> completed-cell throughput of each and t(rad)/t(hydro);
  * how it scales with meshblock size / count (to saturate the GPU) and sweep mode.

Metric (headline): COMPLETED cell-updates/sec = zone-cycles / t.  zone-cycles = cells/block * nblocks
* ncycles (identical hydro & radiation & on/off; nang/niter are NOT multipliers). Completeness columns
nang_tot, niter(=1), and GCAUPS = zone-cycles*niter*nang / t(sweep) are also recorded.

Base problem: a periodic 3D hydro linear wave (nonzero rho/p so chi=opa*rho is meaningful).
affect_fluid=false -> hydro is bit-identical on/off (same cycles/dt) so ZCPS is directly comparable;
the radiation solver still runs a full formal solution every cycle (its cost is value-independent).

Env: ATHENAK_BUILD (dir with 'athena'), ATHENAK_LAUNCHER (e.g. 'srun -n 1'), ATHENAK_DEVICE (tag),
RAD_COST_PROFILE (validate|full), RAD_COST_OUT (output dir; default <here>/<device>).
Run from athenak/tst:  ATHENAK_BUILD=../build/src python3 benchmark/rad_cost/run_rad_cost.py
"""
import os
import re
import csv
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = os.environ.get("ATHENAK_BUILD", "build/src")
LAUNCHER = os.environ.get("ATHENAK_LAUNCHER", "").split()
DEVICE = os.environ.get("ATHENAK_DEVICE", "cpu")
PROFILE = os.environ.get("RAD_COST_PROFILE", "validate")
OUT = os.environ.get("RAD_COST_OUT", os.path.join(HERE, DEVICE))
BIN = os.path.abspath(os.path.join(BUILD, "athena"))

RAD_PREFIX = ("sc_", "gs_")           # all radiation-solver kernels
HYDRO_PREFIX = ("h_update", "hflux_")  # the standard hydro update + fluxes
SWEEP_PREFIX = ("sc_sweep", "gs_sweepa")  # the formal-solution sweep specifically

DECK = """<comment>
problem = rad_cost LTE benchmark
<job>
basename = {base}
<mesh>
nghost = 2
nx1 = {N}
x1min = 0.0
x1max = 3.0
ix1_bc = periodic
ox1_bc = periodic
nx2 = {N}
x2min = 0.0
x2max = 3.0
ix2_bc = periodic
ox2_bc = periodic
nx3 = {N}
x3min = 0.0
x3max = 3.0
ix3_bc = periodic
ox3_bc = periodic
<meshblock>
nx1 = {B}
nx2 = {B}
nx3 = {B}
<time>
evolution = dynamic
integrator = rk2
cfl_number = 0.3
nlim = {nlim}
tlim = 1.0e30
ndiag = 1000000
<hydro>
eos = ideal
reconstruct = plm
rsolver = llf
gamma = 1.66666666667
<problem>
pgen_name = linear_wave
wave_flag = 0
amp = 1.0e-3
dens = 1.0
pgas = 0.6
vx0 = 0.0
along_x1 = false
along_x2 = false
along_x3 = false
"""

RAD_BLOCK = """<nr_radiation>
nmu = {nmu}
opa = 1.0
ops = 0.0
prat = 1.0
crat = 1.0
affect_fluid = false
sweep = {sweep}
iter_max = 1
itermin = 1
ali_tol = 1.0e-6
"""

PERF_BLOCK = """<output77>
file_type = perf
perf_probe = kernels,iteration
kernels = sc_*,gs_*,h_update,hflux_*
dt = 1.0e30
"""


def build_grid(profile):
    if profile == "validate":
        nlim = 5
        g = [dict(suite="val", B=8, N=8 * k, nmu=2, sweep="wavefront") for k in (1, 2)]
        return g, nlim
    if profile == "b64":
        # 64^3-block saturation sweep: nmb=8/27/64 (mesh 128^3/192^3/256^3), nmu=3, to get a
        # SATURATED 64^3 headline (radiation parallelism = nblocks*nang needs enough blocks).
        nlim = 50
        g = [dict(suite="b64", B=64, N=64 * k, nmu=3, sweep=sw)
             for sw in ("wavefront", "diagonal") for k in (2, 3, 4)]
        return g, nlim
    nlim = 50
    g = []
    # A. saturation: fixed 32^3 blocks, grow block count -> nmb=k^3 (nmu=3)
    for sweep in ("wavefront", "diagonal"):
        for k in (1, 2, 3, 4, 5, 6):
            g.append(dict(suite="sat", B=32, N=32 * k, nmu=3, sweep=sweep))
    # B. block size: fixed nmb=8, grow block, all sweep modes
    for sweep in ("wavefront", "diagonal", "jacobi"):
        for B in (32, 48, 64):
            g.append(dict(suite="bsize", B=B, N=B * 2, nmu=3, sweep=sweep))
    # C. angular: fixed 64^3 mesh / 32^3 blocks (nmb=8), grow nmu
    for sweep in ("wavefront", "diagonal"):
        for nmu in (1, 2, 3, 4):
            g.append(dict(suite="ang", B=32, N=64, nmu=nmu, sweep=sweep))
    return g, nlim


def run_case(label, N, B, nlim, rad=None, perf=False):
    d = os.path.join(OUT, label)
    os.makedirs(d, exist_ok=True)
    text = DECK.format(base=os.path.join(d, "rc"), N=N, B=B, nlim=nlim)
    if rad is not None:
        text += RAD_BLOCK.format(nmu=rad["nmu"], sweep=rad["sweep"])
    if perf:
        text += PERF_BLOCK
    deck = os.path.join(d, "deck.athinput")
    with open(deck, "w") as f:
        f.write(text)
    r = subprocess.run(LAUNCHER + [BIN, "-i", deck], capture_output=True, text=True,
                       cwd=d, check=False)
    out = (r.stdout or "") + (r.stderr or "")
    with open(os.path.join(d, "stdout.txt"), "w") as f:
        f.write(out)
    return out, d


def f_search(pat, s):
    m = re.search(pat, s)
    return float(m.group(1)) if m else None


def parse_kernels(path):
    """Return {kernel: ms} = (last-cycle total_ms) - (cycle-0 total_ms)."""
    if not os.path.exists(path):
        return {}
    first, last = {}, {}
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            p = line.split()
            cyc, kern, ms = int(float(p[0])), p[2], float(p[4])
            if cyc not in (0,) and kern not in last:
                pass
            (first if cyc == 0 else last).setdefault(kern, ms)
            if cyc != 0:
                last[kern] = ms  # keep the largest cycle's value
    return {k: last.get(k, 0.0) - first.get(k, 0.0) for k in set(last) | set(first)}


def group_ms(kern, prefixes):
    return sum(v for k, v in kern.items() if k.lower().startswith(prefixes))


def main():
    grid, nlim = build_grid(PROFILE)
    os.makedirs(OUT, exist_ok=True)
    print(f"device={DEVICE} profile={PROFILE} bin={BIN} launcher={LAUNCHER or '(none)'} out={OUT}")
    off_cache = {}   # (N,B) -> (zcps_off, mbcyc)
    rows = []
    for cfg in grid:
        N, B = cfg["N"], cfg["B"]
        nmb = (N // B) ** 3
        cells = B ** 3
        tag = f"{cfg['suite']}_B{B}_nmb{nmb}_nmu{cfg['nmu']}_{cfg['sweep']}"
        # --- OFF (clean, once per geometry) ---
        if (N, B) not in off_cache:
            out, _ = run_case(f"off_B{B}_nmb{nmb}", N, B, nlim, rad=None, perf=False)
            off_cache[(N, B)] = (f_search(r"zone-cycles/cpu_second\s*=\s*([-\d.eE+]+)", out),
                                 f_search(r"MeshBlock-cycles\s*=\s*([-\d.eE+]+)", out))
        zcps_off, mbcyc = off_cache[(N, B)]
        # --- ON clean (ZCPS_on) ---
        out_on, _ = run_case("clean_" + tag, N, B, nlim, rad=cfg, perf=False)
        zcps_on = f_search(r"zone-cycles/cpu_second\s*=\s*([-\d.eE+]+)", out_on)
        # --- ON perf (kernel split) ---
        _, d = run_case("perf_" + tag, N, B, nlim, rad=cfg, perf=True)
        kern = parse_kernels(os.path.join(d, "rc.kernels"))
        t_rad = group_ms(kern, RAD_PREFIX)      # ms, all radiation kernels
        t_hyd = group_ms(kern, HYDRO_PREFIX)    # ms, hydro update + fluxes
        t_swp = group_ms(kern, SWEEP_PREFIX)    # ms, formal-solution sweep only
        # nang, niter from the iteration probe
        itr = os.path.join(d, "rc.iteration")
        nang = niter = None
        if os.path.exists(itr):
            with open(itr) as f:
                last = [ln.split() for ln in f if ln.strip() and not ln.startswith("#")][-1]
            # cols: cycle time n_solves cum_niter last_resid nblocks cells nang
            n_solves, cum_niter, nang = float(last[2]), float(last[3]), float(last[7])
            niter = cum_niter / n_solves if n_solves else None
        zc = cells * nmb * nlim                                    # zone-cycles (approx; nlim cycles)
        rad_tput = zc / (t_rad / 1e3) if t_rad else None          # completed cells/s (radiation)
        hyd_tput = zc / (t_hyd / 1e3) if t_hyd else None          # completed cells/s (hydro)
        gcaups = (zc * (niter or 1) * (nang or 0) / (t_swp / 1e3) / 1e9) if t_swp else None
        row = dict(suite=cfg["suite"], device=DEVICE, B=B, nmb=nmb, zones=cells * nmb,
                   nmu=cfg["nmu"], nang=nang, niter=niter, sweep=cfg["sweep"],
                   zcps_off=zcps_off, zcps_on=zcps_on,
                   slowdown=(zcps_off / zcps_on) if (zcps_off and zcps_on) else None,
                   t_rad_ms=t_rad, t_hydro_ms=t_hyd, t_sweep_ms=t_swp,
                   rad_over_hydro=(t_rad / t_hyd) if t_hyd else None,
                   sweep_over_hydro=(t_swp / t_hyd) if t_hyd else None,
                   rad_cells_per_s=rad_tput, hydro_cells_per_s=hyd_tput, gcaups=gcaups)
        rows.append(row)
        print(f"{tag:38s} nmb={nmb:<4} zones={cells*nmb:>10} "
              f"ZCPS off={zcps_off:.3g} on={zcps_on:.3g} slow={row['slowdown']:.2f}x "
              f"t_rad/t_hyd={row['rad_over_hydro']:.2f} sweep/hyd={row['sweep_over_hydro']:.2f}"
              if zcps_off and zcps_on and t_hyd else f"{tag}: (parse gap)")
    csvp = os.path.join(OUT, "results.csv")
    with open(csvp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", csvp)


if __name__ == "__main__":
    main()
