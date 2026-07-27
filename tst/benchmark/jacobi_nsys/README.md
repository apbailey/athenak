# jacobi_nsys — 20-sweep attribution pass (kernel throughput + per-kernel/API breakdown)

Companion to `benchmark/jacobi/` (the throughput run). This is a **short, fixed 20-sweep** pass run
**with `--rerun-with-nsys`**, for two things the long run can't cheaply give:

1. **KERNEL throughput** — the fenced `vet_sweep*` time is a warm, baseline-subtracted sum over 20
   clean per-sweep timings, so it converges fast; 20 sweeps is plenty. (`analyze.py`'s KERNEL table.)
   The **WALL table from this run is unreliable** — the fixed per-cycle cost (one hydro step,
   NewTimeStep, task-list setup) doesn't amortize over 20 iters. Take WALL from `benchmark/jacobi`.
2. **Per-kernel / API attribution** — 20 sweeps ≈ 400 kernel events per trace, so nsys finalizes
   instantly and every config is traced. For each config read `raw/<label>/<label>.nsys.txt`:
   - **CUDA GPU Kernel Summary** → `vet_sweep_*` (sweep) vs `vet_computeJ` vs the `bvals` pack/unpack
     (exchange) vs `vet_dj` (residual) — which *kernel* owns the per-iteration overhead.
   - **CUDA API Summary** → `cudaLaunchKernel` / `cudaDeviceSynchronize` totals — the host/launch-bound
     share (large ⇒ overhead is launch/host-latency, not any one kernel).

   That attribution decides jacobi's kernel↔wall gap lever: ComputeJ fusion (if `vet_computeJ` is big,
   mainly at large blocks) vs cutting the per-iteration exchange/launch count (if `bvals` + launch API
   dominate, mainly at small blocks).

Own output dir (`benchmark/jacobi_nsys/<device>/`) — does **not** clobber the throughput run.

```bash
cd tst
python benchmark/performance_driver.py benchmark/jacobi_nsys/config.py --rerun-with-nsys
python benchmark/jacobi_nsys/analyze.py benchmark/jacobi_nsys/<device>        # KERNEL table
less benchmark/jacobi_nsys/<device>/raw/blockcount_nmb1_jacobi/*.nsys.txt      # attribution, per config
# or on the cluster:
sbatch benchmark/jacobi_nsys/p100/submit.sbatch      # or v100/  (has --rerun-with-nsys)
```
