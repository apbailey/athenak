# Config: SC sweep throughput, wavefront vs diagonal -- reproduces sc_bench's three suites.
#
# Matches the A100 sc_bench benchmark (~/Downloads/REPORT.md, suites A.1-A.3) through the file-based
# perf framework. The deck uses the sc_uniform pgen (constant chi, S=b, I=b steady state) driven by
# SolveTransfer; itermin=iter_max=NSWEEP forces exactly NSWEEP full-cost sweeps (sc_bench's fixed
# nbench effort). analyze.py groups by measured problem size and orders rows by config emission, so
# each suite prints as a contiguous block, one wavefront-vs-diagonal verdict per point.
#
# Two points are shared between suites and emitted once (in block-count): nmb=1 is the 32^3 single
# block; nmb=8 (mesh 64^3, 32^3 tiles) is the nmu=3 angular point.
#
#   run:  cd tst && python benchmark/performance_driver.py benchmark/compare_sweep/config.py
#   then: python benchmark/compare_sweep/analyze.py benchmark/compare_sweep/<device>  # e.g. cpu/

deck    = "inputs/sc_sweep_bench.athinput"
probes  = ["kernels", "iteration"]       # iteration -> gcaups; kernels -> sweep time
cadence = "final"                        # cycle-0 (post-pgen) + final snapshot; analyze differences them
filter  = "sc_sweep*"                   # only fence/time the sweep kernels (less perturbation)

NSWEEP  = 20                             # sc_bench nbench: force exactly this many sweeps


def cube(mesh, block, nmu):
    return {
        "mesh/nx1": mesh, "mesh/nx2": mesh, "mesh/nx3": mesh,
        "mesh/x2min": 0.0, "mesh/x2max": 1.0, "mesh/x3min": 0.0, "mesh/x3max": 1.0,
        "meshblock/nx1": block, "meshblock/nx2": block, "meshblock/nx3": block,
        "nr_radiation/nmu": nmu,
        "nr_radiation/iter_max": NSWEEP, "nr_radiation/itermin": NSWEEP,   # fixed-effort: N sweeps
    }


configs = {}
for mode in ["wavefront", "diagonal"]:
    # Suite A.1 -- block count: fixed 32^3 blocks, mesh grows -> nmb 1/8/27/64 (nmu=3).
    for mesh in [32, 64, 96, 128]:
        nblk = (mesh // 32) ** 3
        configs[f"blockcount_nmb{nblk}_{mode}"] = {**cube(mesh, 32, 3), "nr_radiation/sweep": mode}
    # Suite A.2 -- block size: single block, grow it (32^3 is blockcount nmb=1).
    for n in [48, 64, 96]:
        configs[f"blocksize_{n}_{mode}"] = {**cube(n, n, 3), "nr_radiation/sweep": mode}
    # Suite A.3 -- angular: nmb=8 (mesh 64^3, 32^3 blocks), grow nmu (nmu=3 is blockcount nmb=8).
    for nmu in [1, 2, 4, 6]:
        configs[f"angular_nmu{nmu}_{mode}"] = {**cube(64, 32, nmu), "nr_radiation/sweep": mode}
