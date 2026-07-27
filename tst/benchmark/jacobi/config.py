# Config: jacobi vs wavefront vs diagonal SC sweep -- kernel throughput AND full wall-time.
#
# Same fixed-effort design as compare_sweep (vet_uniform pgen, I==b steady state, driven by
# SolveTransfer with itermin=iter_max=NSWEEP forcing exactly NSWEEP full-cost FormalSolution sweeps),
# but three sweep modes, and NSWEEP is AUTO-SCALED per config so every run does ~the same amount of
# sweeping wall-time (a couple of minutes) instead of a fixed sweep count. That matters because the
# configs span ~400x in work per sweep (32^3/nmu=1 vs 128^3/nmu=3): a fixed NSWEEP would make the
# small ones sub-second (launch-noise-dominated) and the big ones minutes. Constant wall-time per
# config = robust, comparable timing everywhere. The deck is LTE (ops=0), so jacobi is a plain sweep
# option here -- no ALI, no guard.
#
# The "minutes" knob:
#   TARGET_SEC      wall-seconds of sweeping you want per config
#   NOMINAL_GCAUPS  your GPU's rough sweep throughput; NSWEEP is set from it. After the first run,
#                   check the printed "cpu time used" (or analyze's WALL table) and adjust this so the
#                   runs land on TARGET_SEC. P100/V100 ~ 0.5-0.9, A100 ~ 2-3.
# NSWEEP(config) = TARGET_SEC * NOMINAL_GCAUPS * 1e9 / (total_cells * nang)   (>= NSWEEP_FLOOR)
#
# analyze.py reports TWO metrics per problem size, using the MEASURED sweep count (self-consistent
# with the per-config NSWEEP):
#   - kernel gcaups: fenced vet_sweep* kernel time (perf.kernels)   -- pure sweep-kernel cost
#   - wall   gcaups: the run's "cpu time used" (run.log)            -- full driver wall time
# The gap is jacobi's real question: its flat par_for should win kernel throughput (no barriers /
# plane loop), but does per-iteration overhead erode that in wall time?
#
#   run:  cd tst && python benchmark/performance_driver.py benchmark/jacobi/config.py
#   then: python benchmark/jacobi/analyze.py benchmark/jacobi/<device>          # e.g. p100/
# ~30 sizes x 3 modes x TARGET_SEC each -> budget the SLURM --time accordingly (submit scripts).

deck    = "inputs/vet_sweep_bench.athinput"
probes  = ["kernels", "iteration"]       # iteration -> work (gcaups); kernels -> sweep-kernel time
cadence = "final"                        # cycle-0 (post-pgen) + final snapshot; analyze differences them
filter  = "vet_sweep*"                   # fence/time only the sweep kernels (less perturbation)

TARGET_SEC     = 120                      # ~2 min of sweeping per config (the "minutes" target)
NOMINAL_GCAUPS = 0.6                      # rough P100/V100 sweep throughput; tune after first run
NSWEEP_FLOOR   = 200                      # never fewer (timing stability for the biggest configs)


def nsweep_for(mesh, nmu):
    """Sweeps to make one config take ~TARGET_SEC at NOMINAL_GCAUPS. nang = 4*nmu*(nmu+1) (Carlson,
    3D totals 8/24/48/80/120/168); total cells = mesh^3 regardless of block subdivision."""
    nang = 4 * nmu * (nmu + 1)
    work_per_sweep = (mesh ** 3) * nang
    return max(NSWEEP_FLOOR, round(TARGET_SEC * NOMINAL_GCAUPS * 1e9 / work_per_sweep))


def cube(mesh, block, nmu):
    ns = nsweep_for(mesh, nmu)
    return {
        "mesh/nx1": mesh, "mesh/nx2": mesh, "mesh/nx3": mesh,
        "mesh/x2min": 0.0, "mesh/x2max": 1.0, "mesh/x3min": 0.0, "mesh/x3max": 1.0,
        "meshblock/nx1": block, "meshblock/nx2": block, "meshblock/nx3": block,
        "nr_radiation/nmu": nmu,
        "nr_radiation/iter_max": ns, "nr_radiation/itermin": ns,   # per-config: ~constant wall time
    }


configs = {}
for mode in ["wavefront", "diagonal", "jacobi"]:
    # Suite 1 -- block count: fixed 32^3 blocks, mesh grows -> nmb 1/8/27/64 (nmu=3).
    for mesh in [32, 64, 96, 128]:
        nblk = (mesh // 32) ** 3
        configs[f"blockcount_nmb{nblk}_{mode}"] = {**cube(mesh, 32, 3), "nr_radiation/sweep": mode}
    # Suite 2 -- block size: single block, grow it (32^3 is blockcount nmb=1).
    for n in [48, 64, 96]:
        configs[f"blocksize_{n}_{mode}"] = {**cube(n, n, 3), "nr_radiation/sweep": mode}
    # Suite 3 -- angular: nmb=8 (mesh 64^3, 32^3 blocks), grow nmu (nmu=3 is blockcount nmb=8).
    for nmu in [1, 2, 4, 6]:
        configs[f"angular_nmu{nmu}_{mode}"] = {**cube(64, 32, nmu), "nr_radiation/sweep": mode}
