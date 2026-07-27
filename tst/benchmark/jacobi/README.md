# jacobi — SC sweep performance: jacobi vs wavefront vs diagonal

Measures the throughput of the three `<nr_radiation>/sweep` kernels against each other, reporting
**both** the pure sweep-kernel cost **and** the full run wall time. The question it answers: jacobi's
sweep is a flat `par_for` (no team barriers, no per-plane host loop, fewer registers), so it should
win raw kernel throughput — but does the per-iteration overhead (exchange, ComputeJ, residual,
kernel launches, host↔device syncs) erode that advantage in wall-clock time?

## What it runs

`inputs/vet_sweep_bench.athinput` (the `vet_uniform` pgen: constant χ, S=b, I=b steady state) driven
by `SolveTransfer` with `itermin=iter_max=NSWEEP` forcing exactly NSWEEP full-cost `FormalSolution`
sweeps. **NSWEEP is auto-scaled per config** so every run does ~the same wall-time of sweeping
(target **~2 min**), rather than a fixed count — the configs span ~400× in work per sweep, so a fixed
count would make small configs sub-second (launch-noise) and big ones minutes. The `TARGET_SEC` /
`NOMINAL_GCAUPS` knobs in `config.py` set the target; after the first run, check the printed
`cpu time used` and tune `NOMINAL_GCAUPS` to your GPU (P100/V100 ~0.5–0.9, A100 ~2–3). The deck is
**LTE** (`ops=0`), so `jacobi` is a plain sweep option here: no ALI, no guard. Three suites:
- **block count** — fixed 32³ blocks, mesh grows → nmb 1/8/27/64 (nmu=3)
- **block size** — one block, grown 32³→96³
- **angular** — nmb=8, nmu 1/2/4/6

## Metrics (`analyze.py`)

Per problem size, per mode, `gcaups = cum_niter·cells·nblocks·nang / seconds`:
- **KERNEL** — from the fenced `vet_sweep*` kernel time (`perf.kernels`), device-only.
- **WALL** — from the run's `cpu time used` (`run.log`), host-inclusive: sweeps + all overhead.

`kernel ≫ wall` means a cheap sweep wrapped in large per-iteration overhead — the thing to watch for
jacobi. A wavefront KERNEL win that vanishes in the WALL table is its host plane-loop overhead.

## Run

```bash
# local CPU
cd tst
python benchmark/performance_driver.py benchmark/jacobi/config.py
python benchmark/jacobi/analyze.py benchmark/jacobi/cpu

# GPU cluster (edit ATHENAK + DEVICE at the top of the submit script first)
sbatch benchmark/jacobi/p100/submit.sbatch     # or v100/
```

Results land in `benchmark/jacobi/<device>/` (`manifest.json`, `analysis.txt`, `raw/<label>/`).
The full run is ~30 sizes × 3 modes × ~2 min ≈ 1 hr on a P100 (less on faster GPUs); the submit
scripts budget `--time=04:00:00` (incl. the ~40 min first build). Nsys is **not** run by default here
(it would double the time past the limit) — to profile one config, rerun it alone with
`--rerun-with-nsys` (see the note in the submit script).
