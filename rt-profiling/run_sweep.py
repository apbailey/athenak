#!/usr/bin/env python3
"""Controlled hydro vs hydro+RT (LTE short-characteristics) GPU speed study.

A more *controlled* successor to tst/benchmark/rad_cost's sweeps. Two phases:

  PHASE 1 -- memory-fill (mesh design).  Hold the finest meshblock (B=16) and the highest
    angular order (nmu=6, 168 rays) -- the memory-worst Phase-2 corner -- and grow the total
    mesh N until device memory is *filled* (largest N that runs without OOM).  That N* is the
    mesh for Phase 2.  This is a MEMORY measure of "fully occupying the GPU", not a throughput
    saturation knee.

  PHASE 2 -- the 2D sweep (meshblock size x angular resolution) at the single mesh N*.  Block
    size B sweeps over the divisors of N* that are >=16 (16^3 up to the full mesh = 1 block);
    for each B the angular order nmu sweeps 1..6 (reported as ray counts).  Records whole-run
    ZCPS with radiation OFF and ON for every cell, plus the sc_* vs hydro kernel split.

Physics/recipe (pure LTE, one formal solution per cycle, NO iteration): opa=1, ops=0, eps=1
(=> use_ali=false), iter_max=itermin=1, affect_fluid=false (hydro bit-identical on/off, so the
ZCPS delta is purely the SC sweep cost).  Base problem: the periodic 3D hydro linear_wave deck.

This script is a THIN orchestrator: it imports tst/benchmark/rad_cost/run_rad_cost.py as the
single source of truth for the deck (DECK/RAD_BLOCK/PERF_BLOCK) and the parsers
(f_search/parse_kernels/group_ms/run_case), exactly as profile_sweep.py does.  It adds:
  * memory-fill Phase 1 with a live nvidia-smi monitor + OOM detection;
  * N_REPEAT repeats -> median ZCPS (+min/max);
  * a real-time GPU-loading monitor (SM-util%, mem-util%, mem-used, clocks, power, temp) on
    every run, optionally DCGM SM-occupancy/activity if the Phase-0 probe found it;
  * ray-count columns (nmu -> rays total / per octant);
  * graceful CPU-smoke degradation (RT_SMOKE=1, no GPU) for off-cluster verification.

Angular quadrature (Bruls type-A, angular_grid.cpp): 3D total rays = 8*nmu(nmu+1)/2,
per octant = nmu(nmu+1)/2.  nmu is capped at 6 in the solver.

Env:
  ATHENAK_BUILD    dir with 'athena'            (default build/src)
  ATHENAK_LAUNCHER launcher, e.g. 'srun -n 1'   (default none)
  ATHENAK_DEVICE   tag for output dir/rows      (default cpu)
  RT_OUT           output dir                   (default <here>/<device>)
  RT_PHASE         1 | 2                         (default 1)
  RT_N_LIST        Phase-1 mesh candidates, csv (default 128,144,160,176,192)
  RT_N_STAR        Phase-2 fixed mesh N*         (required for phase 2; else read from phase1 csv)
  RT_SWEEPS        csv of wavefront|diagonal|jacobi (default wavefront,diagonal)
  RT_NMU_MAX       max angular order (<=6)       (default 6)
  RT_N_REPEAT      repeats for median ZCPS       (default 3)
  RT_NLIM          cycles per run                (default 50; phase1 uses RT_NLIM_MEMFILL)
  RT_NLIM_MEMFILL  cycles for the memfill probe  (default 3)
  RT_MEM_CEIL_GIB  memory-fill ceiling (GiB)     (default 34)
  RT_BLOCK_FLOOR   min meshblock size            (default 16)
  RT_DCGM_FIELDS   csv DCGM field ids to sample  (default empty; e.g. 1002,1003 if probe passed)
  RT_WARMUP        1 => discard a warmup run     (default 1)
  RT_SMOKE         1 => tiny CPU grid, no GPU    (default 0)

Run (Apollo, after building a CUDA build on the login node):
  cd $ATHENAK/tst && ATHENAK_BUILD=../build_a100/src ATHENAK_LAUNCHER='srun -n 1' \
    ATHENAK_DEVICE=a100 RT_PHASE=1 python3 ../rt-profiling/run_sweep.py
"""
import os
import sys
import csv
import time
import shutil
import subprocess
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
RAD_COST = os.path.normpath(os.path.join(HERE, "..", "tst", "benchmark", "rad_cost"))
sys.path.insert(0, RAD_COST)
import run_rad_cost as rc  # single source of truth for the LTE deck + parsers

# ---- config from env -------------------------------------------------------------------------
BUILD = os.environ.get("ATHENAK_BUILD", "build/src")
LAUNCHER = os.environ.get("ATHENAK_LAUNCHER", "").split()
DEVICE = os.environ.get("ATHENAK_DEVICE", "cpu")
BIN = os.path.abspath(os.path.join(BUILD, "athena"))
PHASE = os.environ.get("RT_PHASE", "1")
OUT = os.environ.get("RT_OUT", os.path.join(HERE, DEVICE))
SMOKE = os.environ.get("RT_SMOKE", "0") == "1"

N_REPEAT = int(os.environ.get("RT_N_REPEAT", "3"))
NLIM = int(os.environ.get("RT_NLIM", "50"))
NLIM_MEMFILL = int(os.environ.get("RT_NLIM_MEMFILL", "3"))
MEM_CEIL = float(os.environ.get("RT_MEM_CEIL_GIB", "34"))
BLOCK_FLOOR = int(os.environ.get("RT_BLOCK_FLOOR", "16"))
NMU_MAX = min(6, int(os.environ.get("RT_NMU_MAX", "6")))
SWEEPS = os.environ.get("RT_SWEEPS", "wavefront,diagonal").split(",")
N_LIST = [int(x) for x in os.environ.get("RT_N_LIST", "128,144,160,176,192").split(",")]
DCGM_FIELDS = [f for f in os.environ.get("RT_DCGM_FIELDS", "").split(",") if f.strip()]
WARMUP = os.environ.get("RT_WARMUP", "1") == "1"

if SMOKE:
    # tiny, GPU-free grid that still exercises every code path
    N_REPEAT, NLIM, NLIM_MEMFILL, WARMUP = 1, 2, 2, False
    BLOCK_FLOOR, NMU_MAX, SWEEPS = 8, 2, ["wavefront"]
    N_LIST = [16]
    LAUNCHER = []

# make run_rad_cost use OUR paths/device
rc.OUT = OUT
rc.BIN = BIN
rc.LAUNCHER = LAUNCHER
rc.DEVICE = DEVICE

ZCPS_RE = r"zone-cycles/cpu_second\s*=\s*([-\d.eE+]+)"
MBCYC_RE = r"MeshBlock-cycles\s*=\s*([-\d.eE+]+)"
OOM_MARKERS = ("out of memory", "cudaerrormemoryallocation", "failed to allocate",
               "cuda_error_out_of_memory", "allocation of size", "kokkos allocation")

HAVE_NVSMI = shutil.which("nvidia-smi") is not None and not SMOKE
HAVE_DCGM = shutil.which("dcgmi") is not None and bool(DCGM_FIELDS) and not SMOKE


# ---- small helpers ---------------------------------------------------------------------------
def divisors_ge(n, floor=16):
    """Cubic meshblock sizes B that divide n exactly with B >= floor (sorted ascending)."""
    return sorted(b for b in range(floor, n + 1) if n % b == 0)


def rays(nmu, ndim=3):
    """(rays_per_octant, rays_total) for the Bruls type-A grid (angular_grid.cpp)."""
    per_oct = nmu * (nmu + 1) // 2
    noct = {1: 2, 2: 4, 3: 8}[ndim]
    return per_oct, noct * per_oct


def detect_oom(out):
    lo = (out or "").lower()
    return any(m in lo for m in OOM_MARKERS)


def mem_total_gib():
    if not HAVE_NVSMI:
        return None
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.total",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=20, check=False)
        return float(r.stdout.strip().splitlines()[0]) / 1024.0  # MiB -> GiB
    except Exception:
        return None


MEM_TOTAL = mem_total_gib()


def _parse_sampler(path):
    """Parse nvidia-smi --query-gpu csv rows: util.gpu, util.mem, mem.used(MiB), sm.clk, pwr, temp."""
    smu, mmu, mem, clk, pwr, tmp = [], [], [], [], [], []
    try:
        with open(path) as f:
            for line in f:
                p = [x.strip() for x in line.split(",")]
                if len(p) < 6:
                    continue
                try:
                    smu.append(float(p[0])); mmu.append(float(p[1])); mem.append(float(p[2]))
                    clk.append(float(p[3])); pwr.append(float(p[4])); tmp.append(float(p[5]))
                except ValueError:
                    continue
    except FileNotFoundError:
        return {}
    if not mem:
        return {}
    mean = lambda a: sum(a) / len(a)
    return dict(peak_mem_gib=max(mem) / 1024.0,
                sm_util_mean=mean(smu), sm_util_peak=max(smu), mem_util_mean=mean(mmu),
                sm_clock_mhz=mean(clk), power_w=mean(pwr), gpu_temp_c=max(tmp))


def _parse_dcgm(path):
    """Parse `dcgmi dmon -e <fields>` output: mean of each numeric column after the GPU id."""
    vals = {}
    try:
        with open(path) as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#") or s.lower().startswith("id"):
                    continue
                p = s.split()
                # rows look like: GPU 0 <f1> <f2> ...
                nums = []
                for tok in p:
                    try:
                        nums.append(float(tok))
                    except ValueError:
                        pass
                if len(nums) >= 2:
                    vals.setdefault("cols", []).append(nums[1:])  # drop the GPU index
    except FileNotFoundError:
        return {}
    cols = vals.get("cols")
    if not cols:
        return {}
    ncol = min(len(r) for r in cols)
    means = [sum(r[i] for r in cols) / len(cols) for i in range(ncol)]
    out = {}
    # by convention field order 1002=SM active, 1003=SM occupancy (if requested in that order)
    names = {"1002": "dcgm_sm_active_mean", "1003": "dcgm_sm_occ_mean"}
    for i, fid in enumerate(DCGM_FIELDS):
        if i < len(means):
            out[names.get(fid.strip(), f"dcgm_{fid.strip()}_mean")] = means[i]
    return out


def run_with_monitor(run_fn, tag):
    """Run run_fn() while sampling GPU state; return (result, stats-dict)."""
    if not HAVE_NVSMI:
        return run_fn(), {}
    smp_path = os.path.join(OUT, f"_mon_{tag}.csv")
    os.makedirs(OUT, exist_ok=True)
    smp = open(smp_path, "w")
    cols = "utilization.gpu,utilization.memory,memory.used,clocks.sm,power.draw,temperature.gpu"
    p = subprocess.Popen(["nvidia-smi", f"--query-gpu={cols}",
                          "--format=csv,noheader,nounits", "-lms", "200"],
                         stdout=smp, stderr=subprocess.DEVNULL)
    dproc = dpath = None
    if HAVE_DCGM:
        dpath = os.path.join(OUT, f"_dcgm_{tag}.txt")
        df = open(dpath, "w")
        dproc = subprocess.Popen(["dcgmi", "dmon", "-e", ",".join(DCGM_FIELDS), "-d", "200"],
                                 stdout=df, stderr=subprocess.DEVNULL)
    try:
        res = run_fn()
    finally:
        for proc in (p, dproc):
            if proc is not None:
                proc.terminate()
                try:
                    proc.wait(3)
                except Exception:
                    proc.kill()
        smp.close()
    stats = _parse_sampler(smp_path)
    if dpath:
        stats.update(_parse_dcgm(dpath))
    return res, stats


def blank_row(**kw):
    """A CSV row with every column present (None where not applicable)."""
    cols = ["phase", "suite", "device", "sweep", "N", "B", "nmb", "zones", "nmu",
            "rays_per_octant", "rays_total", "nlim", "n_repeat", "peak_mem_gib", "pct_fill",
            "oom", "zcps_off_med", "zcps_on_med", "zcps_on_min", "zcps_on_max", "slowdown_med",
            "t_rad_ms", "t_hydro_ms", "t_sweep_ms", "rad_over_hydro", "sweep_over_hydro",
            "rad_cells_per_s", "hydro_cells_per_s", "gcaups", "nang", "niter",
            "sm_util_mean", "sm_util_peak", "mem_util_mean", "dcgm_sm_occ_mean",
            "dcgm_sm_active_mean", "sm_clock_mhz", "gpu_temp_c", "power_w"]
    row = {c: None for c in cols}
    row.update(kw)
    return row


def median_zcps(N, B, nlim, cfg, n_repeat, label):
    """Run n_repeat times, return (median, min, max, last_out, last_stats)."""
    zs, out, stats = [], "", {}
    for i in range(n_repeat):
        (out, _d), stats = run_with_monitor(
            lambda: rc.run_case(f"{label}_r{i}", N, B, nlim, rad=cfg, perf=False), f"{label}_r{i}")
        z = rc.f_search(ZCPS_RE, out)
        if z is not None:
            zs.append(z)
    if not zs:
        return None, None, None, out, stats
    return statistics.median(zs), min(zs), max(zs), out, stats


# ---- Phase 1: memory-fill --------------------------------------------------------------------
def phase1_memfill():
    B, nmu, sweep = BLOCK_FLOOR, NMU_MAX, SWEEPS[0]
    perocc, ntot = rays(nmu)
    print(f"[phase1] memory-fill: B={B} nmu={nmu} ({ntot} rays) sweep={sweep} "
          f"nlim={NLIM_MEMFILL} ceil={MEM_CEIL} GiB  device_mem={MEM_TOTAL} GiB")
    rows, N_star, consec_oom = [], None, 0
    for N in sorted(set(N_LIST)):
        if N % B != 0:
            print(f"  N={N} not divisible by B={B}; skipping"); continue
        cfg = dict(nmu=nmu, sweep=sweep)
        (out, _), stats = run_with_monitor(
            lambda: rc.run_case(f"memfill_N{N}", N, B, NLIM_MEMFILL, rad=cfg, perf=False),
            f"memfill_N{N}")
        z = rc.f_search(ZCPS_RE, out)
        oom = detect_oom(out) or z is None
        peak = stats.get("peak_mem_gib")
        pct = (100.0 * peak / MEM_TOTAL) if (peak and MEM_TOTAL) else None
        fits = (not oom) and (peak is None or peak <= MEM_CEIL)
        if fits:
            N_star = N
            consec_oom = 0
        else:
            consec_oom += 1
        rows.append(blank_row(phase="1", suite="memfill", device=DEVICE, sweep=sweep, N=N, B=B,
                              nmb=(N // B) ** 3, zones=N ** 3, nmu=nmu, rays_per_octant=perocc,
                              rays_total=ntot, nlim=NLIM_MEMFILL, n_repeat=1, peak_mem_gib=peak,
                              pct_fill=pct, oom=oom, zcps_on_med=z, **{k: stats.get(k) for k in
                              ("sm_util_mean", "sm_util_peak", "mem_util_mean", "dcgm_sm_occ_mean",
                               "dcgm_sm_active_mean", "sm_clock_mhz", "gpu_temp_c", "power_w")}))
        tag = "OOM/over-ceil" if not fits else "fits"
        print(f"  N={N:<4} zones={N**3:>11}  peak_mem={peak}  pct_fill={pct}  {tag}"
              f"  ZCPS_on={z}")
        if consec_oom >= 2:
            print("  two consecutive non-fits; stopping the climb"); break
    write_csv(rows, os.path.join(OUT, "results_phase1.csv"))
    print(f"[phase1] N* (largest mesh that fills memory <= {MEM_CEIL} GiB) = {N_star}")
    if N_star is None:
        print("[phase1] WARNING: no mesh fit; lower RT_N_LIST or raise RT_MEM_CEIL_GIB")
    return N_star, rows


# ---- Phase 2: the 2D sweep at N* -------------------------------------------------------------
def phase2_sweep(N_star):
    B_list = divisors_ge(N_star, BLOCK_FLOOR)
    print(f"[phase2] N*={N_star}  block ladder (>= {BLOCK_FLOOR}) = {B_list}")
    print(f"         sweeps={SWEEPS}  nmu=1..{NMU_MAX}  n_repeat={N_REPEAT}  nlim={NLIM}")
    rows = []
    if WARMUP and B_list:
        print("[phase2] warmup run (discarded)")
        run_with_monitor(lambda: rc.run_case("warmup", N_star, B_list[-1], 3,
                         rad=dict(nmu=1, sweep=SWEEPS[0]), perf=False), "warmup")
    for B in B_list:
        nmb = (N_star // B) ** 3
        cells = B ** 3
        # OFF once per B (nmu-independent), measured adjacent to this B's ON group
        off_med, off_lo, off_hi, _, _ = median_zcps(N_star, B, NLIM, None, N_REPEAT,
                                                     f"off_B{B}")
        for sweep in SWEEPS:
            for nmu in range(1, NMU_MAX + 1):
                perocc, ntot = rays(nmu)
                cfg = dict(nmu=nmu, sweep=sweep)
                label = f"B{B}_nmu{nmu}_{sweep}"
                on_med, on_lo, on_hi, out_on, stats = median_zcps(N_star, B, NLIM, cfg,
                                                                  N_REPEAT, "on_" + label)
                # one perf run for the kernel split
                (_out, d), _ = run_with_monitor(
                    lambda: rc.run_case("perf_" + label, N_star, B, NLIM, rad=cfg, perf=True),
                    "perf_" + label)
                kern = rc.parse_kernels(os.path.join(d, "rc.kernels"))
                t_rad = rc.group_ms(kern, rc.RAD_PREFIX)
                t_hyd = rc.group_ms(kern, rc.HYDRO_PREFIX)
                t_swp = rc.group_ms(kern, rc.SWEEP_PREFIX)
                nang = niter = None
                itr = os.path.join(d, "rc.iteration")
                if os.path.exists(itr):
                    with open(itr) as f:
                        last = [ln.split() for ln in f if ln.strip()
                                and not ln.startswith("#")][-1]
                    n_solves, cum_niter, nang = float(last[2]), float(last[3]), float(last[7])
                    niter = cum_niter / n_solves if n_solves else None
                zc = cells * nmb * NLIM
                rad_tput = zc / (t_rad / 1e3) if t_rad else None
                hyd_tput = zc / (t_hyd / 1e3) if t_hyd else None
                gcaups = (zc * (niter or 1) * (nang or 0) / (t_swp / 1e3) / 1e9) if t_swp else None
                peak = stats.get("peak_mem_gib")
                rows.append(blank_row(
                    phase="2", suite="sweep", device=DEVICE, sweep=sweep, N=N_star, B=B, nmb=nmb,
                    zones=cells * nmb, nmu=nmu, rays_per_octant=perocc, rays_total=ntot,
                    nlim=NLIM, n_repeat=N_REPEAT, peak_mem_gib=peak,
                    pct_fill=(100.0 * peak / MEM_TOTAL) if (peak and MEM_TOTAL) else None,
                    oom=detect_oom(out_on) or on_med is None,
                    zcps_off_med=off_med, zcps_on_med=on_med, zcps_on_min=on_lo, zcps_on_max=on_hi,
                    slowdown_med=(off_med / on_med) if (off_med and on_med) else None,
                    t_rad_ms=t_rad, t_hydro_ms=t_hyd, t_sweep_ms=t_swp,
                    rad_over_hydro=(t_rad / t_hyd) if t_hyd else None,
                    sweep_over_hydro=(t_swp / t_hyd) if t_hyd else None,
                    rad_cells_per_s=rad_tput, hydro_cells_per_s=hyd_tput, gcaups=gcaups,
                    nang=nang, niter=niter,
                    **{k: stats.get(k) for k in ("sm_util_mean", "sm_util_peak", "mem_util_mean",
                       "dcgm_sm_occ_mean", "dcgm_sm_active_mean", "sm_clock_mhz", "gpu_temp_c",
                       "power_w")}))
                sd = rows[-1]["slowdown_med"]
                print(f"  B={B:<4} nmb={nmb:<5} {sweep:<9} nmu={nmu} rays={ntot:<3} "
                      f"ZCPS off={off_med} on={on_med} slow={sd}")
    suffix = SWEEPS[0] if len(SWEEPS) == 1 else "both"
    write_csv(rows, os.path.join(OUT, f"results_{suffix}.csv"))
    return rows


def write_csv(rows, path):
    if not rows:
        print("(no rows to write)"); return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", path, f"({len(rows)} rows)")


def main():
    os.makedirs(OUT, exist_ok=True)
    print(f"device={DEVICE} phase={PHASE} bin={BIN} launcher={LAUNCHER or '(none)'} "
          f"smoke={SMOKE} out={OUT}")
    t0 = time.time()
    if PHASE == "1":
        phase1_memfill()
    elif PHASE == "2":
        N_star = os.environ.get("RT_N_STAR")
        if N_star:
            N_star = int(N_star)
        else:
            # fall back to the largest fitting mesh recorded by a prior phase 1
            p1 = os.path.join(OUT, "results_phase1.csv")
            if os.path.exists(p1):
                import csv as _csv
                fits = [int(r["N"]) for r in _csv.DictReader(open(p1))
                        if r.get("oom") in ("False", "0", "false") and r.get("N")]
                N_star = max(fits) if fits else None
            if N_star is None:
                sys.exit("RT_PHASE=2 needs RT_N_STAR (or a results_phase1.csv with a fitting mesh)")
        phase2_sweep(N_star)
    else:
        sys.exit(f"unknown RT_PHASE={PHASE}")
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
