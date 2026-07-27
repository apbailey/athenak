# Config: jacobi/wavefront/diagonal ATTRIBUTION pass -- fixed 20 sweeps, run WITH --rerun-with-nsys.
#
# Sibling of benchmark/jacobi/ (the throughput run). Purpose is different: a SHORT fixed-effort run
# (NSWEEP=20 for every config) whose two deliverables are
#   1. KERNEL throughput -- fenced vet_sweep* time is a warm, baseline-subtracted sum over 20 clean
#      per-sweep timings, so it converges fast; 20 sweeps is plenty (analyze.py's KERNEL table). The
#      WALL table from this run is NOT reliable (fixed per-cycle cost -- one hydro step, NewTimeStep,
#      task-list setup -- does not amortize over 20 iters); take WALL from benchmark/jacobi instead.
#   2. PER-KERNEL / API ATTRIBUTION -- run this with --rerun-with-nsys; each config also traces under
#      nsys. Read raw/<label>/<label>.nsys.txt: the GPU Kernel Summary splits vet_sweep_* (sweep) vs
#      vet_computeJ vs the bvals pack/unpack (exchange) vs vet_dj (residual); the CUDA API Summary
#      (cudaLaunchKernel, cudaDeviceSynchronize) shows the host/launch-bound share. That attribution
#      is what decides whether ComputeJ fusion or cutting the per-iteration exchange/launch count is
#      the lever for jacobi's kernel<->wall gap.
#
# 20 sweeps -> ~20 iters x ~20 launches = ~400 kernel events per trace: nsys finalizes instantly (no
# hang), so every config can be traced cheaply. OWN folder -> benchmark/jacobi_nsys/<device>/, does
# NOT clobber the throughput run's benchmark/jacobi/<device>/.
#
#   run:  cd tst && python benchmark/performance_driver.py benchmark/jacobi_nsys/config.py --rerun-with-nsys
#   kernel throughput:  python benchmark/jacobi_nsys/analyze.py benchmark/jacobi_nsys/<device>   # KERNEL table
#   attribution:        less benchmark/jacobi_nsys/<device>/raw/<label>/<label>.nsys.txt

deck    = "inputs/vet_sweep_bench.athinput"
probes  = ["kernels", "iteration"]       # kernel throughput; nsys (the --rerun) gives the attribution
cadence = "final"
filter  = "vet_sweep*"                   # perf side times the sweep only; nsys still traces ALL kernels

NSWEEP  = 20                             # fixed short effort -- enough for kernel gcaups + a clean trace


def cube(mesh, block, nmu):
    return {
        "mesh/nx1": mesh, "mesh/nx2": mesh, "mesh/nx3": mesh,
        "mesh/x2min": 0.0, "mesh/x2max": 1.0, "mesh/x3min": 0.0, "mesh/x3max": 1.0,
        "meshblock/nx1": block, "meshblock/nx2": block, "meshblock/nx3": block,
        "nr_radiation/nmu": nmu,
        "nr_radiation/iter_max": NSWEEP, "nr_radiation/itermin": NSWEEP,
    }


configs = {}
for mode in ["wavefront", "diagonal", "jacobi"]:
    for mesh in [32, 64, 96, 128]:                                   # block count (32^3 blocks)
        nblk = (mesh // 32) ** 3
        configs[f"blockcount_nmb{nblk}_{mode}"] = {**cube(mesh, 32, 3), "nr_radiation/sweep": mode}
    for n in [48, 64, 96]:                                           # block size (single block)
        configs[f"blocksize_{n}_{mode}"] = {**cube(n, n, 3), "nr_radiation/sweep": mode}
    for nmu in [1, 2, 4, 6]:                                         # angular (nmb=8)
        configs[f"angular_nmu{nmu}_{mode}"] = {**cube(64, 32, nmu), "nr_radiation/sweep": mode}
