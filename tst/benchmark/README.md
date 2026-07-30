# Performance benchmarking

Diagnostics you attach to a run, plus a driver that sweeps them across configs and hardware. Four
pieces:

1. **A `perf` output block** you can add to *any* athinput to emit raw per-probe measurements.
2. **`performance_driver.py`** — runs a *sweep* of simulations from a config spec, each with perf
   outputs on, into a device-partitioned results tree.
3. **`analyze.py`** — a *problem-specific* aggregator that turns one run's raw files into an answer.
4. **`--rerun-with-nsys`** — an optional second pass that also traces each config under Nsight Systems.

Nothing here is gated by CI, and a deck with no `perf` block pays zero overhead.

---

## 1. The `perf` output block (works on any deck)

Perf is a normal athenak output type — add an `<output>` block with `file_type = perf` to any input
file and the binary writes raw time-series files, no code changes:

```
<output9>
file_type  = perf
perf_probe = kernels,iteration     # comma-list; OMIT the line to enable all four
kernels    = sc_sweep*            # optional glob: only these kernels are timed (kernels probe)
dcycle     = 50                    # cadence, like any output (or dt = 0.1; or dt = 1e30 for end-only)
```

Each named probe writes `<job/basename>.<probe>` as a wide, header-once, **cumulative** time-series
(`# cycle time <cols>`); the binary emits only raw reductions — averages / percentages / throughput
are derived downstream in `analyze.py`.

```
  probe       file columns                                    measures
  kernels     kernel  count total_ms                          per-kernel device time (per-kernel fence)
  memory      kind name  hwm_mb                                high-water-mark per memory space / label
  comms       bucket  wait_ms                                  exposed MPI wait (recv_wait, send_wait)
  iteration   n_solves cum_niter last_resid nblocks cells nang  solver convergence (VET SolveTransfer)
```

Parameters:

```
  file_type  = perf         required
  perf_probe = <list>        comma-list of {kernels,memory,comms,iteration}; omit = all four
  kernels    = <glob>        optional; restrict the kernels probe to matching labels (less perturbation)
  dcycle=N | dt=X            cadence (a probe in only ONE block; use extra blocks for other cadences)
```

Note: there is no `<perf>` block or `perf/` CLI parameter — perf is *only* an `<output file_type=perf>`
block. Put it in the deck, or let the driver inject it for a whole sweep (below).

---

## 2. Benchmark layout

A benchmark is a self-contained folder sharing the one driver:

```
  benchmark/
    performance_driver.py        shared runner (deck-agnostic)
    <name>/                      one benchmark, e.g. compare_sweep/
      config.py                  the sweep spec (section 3)
      analyze.py                 this benchmark's aggregator (section 4)
      <device>/                  p100/ v100/ a100/ cpu/  -- device = nvidia-smi or $ATHENAK_DEVICE
        submit.sbatch            committed: build + run on that cluster
        manifest.json            run metadata: device, arch, commit, configs   [gitignored]
        analysis.txt             the verdict                                    [gitignored]
        raw/<label>/             deck.athinput + perf.<probe> [+ nsys files]    [gitignored]
```

Only `config.py`, `analyze.py`, and `<device>/submit.sbatch` are committed; everything under a
run (`manifest.json`, `analysis.txt`, `raw/`) is regenerated and gitignored. Partitioning by
`<device>` keeps p100 / v100 / a100 runs of the same config side by side without overwriting, and
each `manifest.json` records the device + Kokkos arch + git commit, so you can `scp` runs into one
tree and compare.

---

## 3. The driver — sweep simulations from a config

`performance_driver.py` is pure orchestration: for each config it copies the deck, injects the `perf`
output block, points `job/basename` at that config's `raw/<label>/`, and runs the binary. It never
reads or reformats measurements — that's `analyze.py`'s job. Run it from `tst/`.

**Config spec** (`<name>/config.py`) — a plain Python module; the run's name is its *folder*:

```
  deck     str                       base athinput (required)
  configs  {label: {param: value}}   the sweep (required) -- Python loops build it, so covariation and
                                      multi-axis sweeps are just nested loops
  probes   list[str]                 default: all four probes
  cadence  str                       "final" (default) | "every N" | "dt X"
  filter   str                       kernel-name glob, e.g. "sc_sweep*"
```

```bash
cd tst
python benchmark/performance_driver.py benchmark/compare_sweep/config.py
```

**Quick one-off** (no spec needed; a comma-valued override becomes a small sweep, auto-labelled):

```bash
python benchmark/performance_driver.py inputs/mydeck.athinput \
    --kernels --iter --filter 'sc_sweep*' nr_radiation/sweep=wavefront,diagonal -o myrun
```

**Output:** `<name>/<device>/` with `manifest.json` + `analysis.txt` at top and per-config files under
`raw/<label>/` (a one-off lands in a central `benchmark/results/<name>/<device>/`). `<device>` is
auto-detected from `nvidia-smi` (`p100`/`v100`/`a100`), or `cpu`, or forced with `$ATHENAK_DEVICE`.

Cluster env: `ATHENAK_BUILD` (point at a prebuilt binary dir), `ATHENAK_LAUNCHER` (`"mpirun -np 1"` /
`"srun --mpi=pmix"`), `ATHENAK_DEVICE`.

---

## 4. `analyze.py` — a problem-specific aggregator

There is **no generic analyze** — each benchmark ships its own, because the question is specific to
the problem. `compare_sweep/analyze.py` answers "wavefront vs diagonal, which sweep is faster and by
how much": it reads one run dir (`manifest.json` + `raw/<label>/perf.*`), derives per-config
throughput

```
  gcaups = cum_niter * cells * nblocks * nang / sweep_seconds
```

from the `kernels` (summed `sc_sweep*` time) and `iteration` probes — differencing out the pgen's
cycle-0 setup sweep — groups by measured problem size, and prints one verdict per size, saved to
`analysis.txt`:

```bash
python benchmark/compare_sweep/analyze.py benchmark/compare_sweep/<device>   # e.g. .../cpu
```

A different benchmark (kernel breakdown, scaling, ...) is a new folder with its own `config.py` +
`analyze.py`, reusing the same driver.

---

## 5. nsys profiling (`--rerun-with-nsys`)

Add `--rerun-with-nsys` (spec or one-off) to rerun each config a *second* time under Nsight Systems,
on the base deck (no perf block) so the clean throughput run is left unperturbed:

```bash
python benchmark/performance_driver.py benchmark/compare_sweep/config.py --rerun-with-nsys
```

Per config it writes into `raw/<label>/`:

```
  <label>.nsys-rep    binary report -> open in the Nsight Systems GUI (timeline)
  <label>.nsys.txt    the nsys --stats summary tables (kernel / CUDA API / mem)
  <label>.kern.csv    untruncated per-kernel table (nsys stats --format csv) -- full symbol names,
                      e.g. FormalSolutionWavefront vs FormalSolutionDiagonal, with grid/block dims
```

Notes: it uses a lean trace `-t cuda,nvtx --sample=none --cpuctxsw=none` (GPU kernels only — the
default CPU/OSRT sampling's report finalization hangs under mpirun+UCX on Ocelote); a `timeout`
guards against any profile hang stalling the sweep; and it's graceful if `nsys` isn't on `PATH`.
`ncu` would be the same pattern but its counters are admin-blocked on Ocelote.
