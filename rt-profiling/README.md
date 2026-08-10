# rt-profiling — controlled hydro vs hydro+RT (LTE SC) GPU speed study

A controlled measurement of what the **LTE short-characteristics (SC) radiation sweep** costs on a
GPU, relative to pure hydro, as a function of **meshblock size** and **angular resolution**. It is a
tighter successor to `logs/validation/gpu_perf_lte_2026-08/` in the AIthenaRT workspace.

## What it measures

- **Problem:** the periodic 3D hydro `linear_wave` deck. Radiation is turned on purely by appending
  an `<nr_radiation>` block (no compile flag, no rebuild — the binary already has both sweeps).
- **Regime (pure LTE, no iteration):** `opa=1, ops=0, eps=1` (⇒ `use_ali=false`),
  `iter_max=itermin=1` (one formal solution per cycle), `affect_fluid=false` (hydro is bit-identical
  on/off, so the ZCPS delta is *purely* the SC sweep cost).
- **Metric:** whole-run **ZCPS** (`zone-cycles/cpu_second`) with radiation OFF and ON, plus the
  `sc_*`/`gs_*` (radiation) vs `h_update`/`hflux_*` (hydro) device-time split from the `perf` probe.
- **Angles:** input `nmu ∈ [1,6]`, **reported as ray counts**. 3D total rays = 8·nmu(nmu+1)/2:

  | nmu | 1 | 2 | 3 | 4 | 5 | 6 |
  |-----|---|---|---|---|---|---|
  | rays/octant | 1 | 3 | 6 | 10 | 15 | 21 |
  | **rays (3D total)** | **8** | **24** | **48** | **80** | **120** | **168** |

  (2D total = 4·nmu(nmu+1)/2 = half the 3D total.)

## Two phases

1. **Phase 1 — memory-fill (mesh design).** Hold the finest block (`B=16`) and highest angular order
   (`nmu=6`, 168 rays) — the memory-worst corner — and grow the mesh `N` until device memory is
   *filled*. **N★** = the largest mesh that runs without OOM. This is a **memory** measure of "fully
   occupying the GPU", not a throughput knee. Recorded in `results_phase1.csv` and the report.
2. **Phase 2 — the 2D sweep** at the single mesh `N★`: block size `B` over the divisors of `N★` that
   are ≥16 (16³ up to the full mesh = 1 block) × `nmu = 1..6`, for both `wavefront` and `diagonal`
   sweeps. ZCPS with/without RT per cell → `results_wavefront.csv`, `results_diagonal.csv`.

## Files

| file | role |
|---|---|
| `run_sweep.py` | orchestrator (imports `tst/benchmark/rad_cost/run_rad_cost.py` for the deck + parsers) |
| `kernel_resources.py` | static per-kernel registers/spills/shared-mem + theoretical occupancy (`cuobjdump`) |
| `analyze.py` | summary Markdown tables + plots from the CSVs |
| `a100/submit_phase1.sbatch`, `a100/submit_sweep.sbatch` | Apollo A100 Slurm jobs |
| `h200/submit_phase1.sbatch`, `h200/submit_sweep.sbatch` | Apollo H200 Slurm jobs |
| `H200_COLLABORATOR.md` | full instructions to reproduce on a (different-memory) H200 |
| `REPORT.md` | write-up of the A100 results (generated after the runs) |
| `OPTIMIZATION_LEDGER.md` | **living record of sweep optimizations tested + ideas to try** (start here for "what's next") |
| `launch_overhead/measure_nsys.py` | fence-free nsys launch-overhead (gap_fraction) probe for any sweep |
| `<device>/results_*.csv`, `<device>/*.png` | data + figures |

## Reproduce (Apollo A100)

```bash
# on Apollo, in the athenak-rtsc worktree (branch rt-sc), after a login-node build of build_a100:
sbatch rt-profiling/a100/submit_phase1.sbatch          # -> results_phase1.csv; read off N*
RT_N_STAR=<N*> RT_SWEEPS=wavefront sbatch rt-profiling/a100/submit_sweep.sbatch
RT_N_STAR=<N*> RT_SWEEPS=diagonal  sbatch rt-profiling/a100/submit_sweep.sbatch
# back on a workstation with matplotlib:
python3 rt-profiling/analyze.py rt-profiling/a100      # -> tables + PNGs
```

Off-cluster smoke test (no GPU, tiny grid — validates parsing/CSV/ray-count logic):
```bash
ATHENAK_BUILD=<serial build>/src RT_SMOKE=1 python3 rt-profiling/run_sweep.py
```

## Notes on control / caveats

- **N_REPEAT=3** repeats → median ZCPS (+min/max); a discarded **warmup** run per job.
- A live **`nvidia-smi dmon`** monitor records SM-util %, mem-util %, mem-used, clock, power, temp per
  run (real-time GPU *loading*). Real-time SM *occupancy/activity* is logged too **iff** DCGM
  profiling is accessible (Phase-0 probe). Achieved per-kernel occupancy/warp-efficiency needs Nsight
  Compute counters, which are **admin-blocked** on Apollo (`NVreg_RestrictProfilingToAdminUsers=1`).
- GPU clocks cannot be pinned without admin, so we log clocks/temp and report medians instead.
