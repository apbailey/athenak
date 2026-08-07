# Collaborator guide — run the controlled hydro-vs-RT speed study on an H200

This reproduces the A100 study (`rt-profiling/`) on an **H200**. The method is identical; the one
thing that genuinely changes is **memory**: an H200 has far more HBM than the A100 (Apollo's H200 ≈
**141 GiB** vs the A100's **40 GiB**), so the memory-fill mesh `N★` — and hence the block ladder and
the run time — are much larger. **You re-derive `N★` for your card; do not reuse the A100's.**

Everything is driven by the same two scripts (`run_sweep.py`, `kernel_resources.py`) and is fully
env-parameterized — no code edits needed.

---

## 0. What the study is (read once)

- **Base problem:** periodic 3D hydro `linear_wave`. Radiation is turned on purely by the presence of
  an `<nr_radiation>` block — no compile flag.
- **Regime — pure LTE, one formal solution per cycle, NO iteration:** `opa=1, ops=0, eps=1`
  (⇒ `use_ali=false`), `iter_max=itermin=1`, `affect_fluid=false` (hydro is bit-identical on/off, so
  the difference in ZCPS is purely the SC sweep cost). This is baked into the deck the script uses.
- **Metric:** whole-run **ZCPS** = `zone-cycles/cpu_second` from the driver, measured with radiation
  **OFF** and **ON**; plus the `sc_*`/`gs_*` (radiation) vs `h_update`/`hflux_*` (hydro) device-time
  split from a `perf` probe.
- **Angles:** `nmu ∈ [1,6]`, reported as **ray counts** (3D total = 8·nmu(nmu+1)/2):

  | nmu | 1 | 2 | 3 | 4 | 5 | 6 |
  |-----|---|---|---|---|---|---|
  | rays/octant | 1 | 3 | 6 | 10 | 15 | 21 |
  | **rays (3D total)** | **8** | **24** | **48** | **80** | **120** | **168** |

- **Two phases:** (1) *memory-fill* to find the mesh `N★` that fills your GPU's memory with 16³
  blocks; (2) a 2D sweep of **meshblock size × angular resolution** at that single `N★`, for both the
  `wavefront` and `diagonal` sweeps, recording ZCPS with and without RT.

---

## 1. Build on the login node (Hopper)

Compute nodes lack CUDA headers — build on the login node.

```bash
cd $ATHENAK                       # your athenak worktree on branch rt-sc
source /usr/share/Modules/init/bash
module purge
source ~/.bash_scripts/athenak_h200          # nvhpc + cuda toolchain for H200
cmake3 -S . -B build_h200 -D Kokkos_ENABLE_CUDA=ON -D Kokkos_ARCH_HOPPER90=ON \
       -D Athena_ENABLE_MPI=ON -D CMAKE_CXX_COMPILER=$PWD/kokkos/bin/nvcc_wrapper
cmake3 --build build_h200 -j12
```

> On a non-Apollo H200 cluster, adapt the module lines and the Slurm `--partition`, but keep
> `Kokkos_ARCH_HOPPER90=ON`. The rest is unchanged.

## 2. Check your device memory (this sets the mesh candidates)

```bash
nvidia-smi --query-gpu=name,memory.total --format=csv
```
Note the total GiB. You will fill it, leaving ~10 % headroom for CUDA/MPI context and fragmentation.
For a 141 GiB H200, `RT_MEM_CEIL_GIB≈128` is a good ceiling. Scale the memory-fill candidate meshes
`RT_N_LIST` so they bracket the ceiling: roughly, at the worst corner (B=16, nmu=6) the footprint is
~`168 · N³ · 8 bytes · O(1)`, so 141 GiB reaches into the **N ≈ 300–380** range (the `h200` sbatch
defaults `RT_N_LIST="192,256,320,384,448,512"` — the run picks the largest that fits).

## 3. Phase 1 — memory-fill (derive *your* N★)

```bash
sbatch rt-profiling/h200/submit_phase1.sbatch
# override candidates/ceiling for your card if needed, e.g.:
#   RT_N_LIST="256,320,384,448,512" RT_MEM_CEIL_GIB=128 sbatch rt-profiling/h200/submit_phase1.sbatch
```
When it finishes, read **N★** from the tail of the job log (`... N* ... = <N>`) and from
`rt-profiling/h200/results_phase1.csv` (largest `N` with `oom=False`). This is the single mesh for
Phase 2. The job also runs the static kernel-resource sidecar (`cuobjdump`, sm90) and a Phase-0 DCGM
capability probe (real-time SM occupancy/activity, if your cluster's DCGM host-engine allows it).

## 4. Phase 2 — the 2D sweep (once per sweep)

```bash
RT_N_STAR=<N*> RT_SWEEPS=wavefront sbatch rt-profiling/h200/submit_sweep.sbatch
RT_N_STAR=<N*> RT_SWEEPS=diagonal  sbatch rt-profiling/h200/submit_sweep.sbatch
```
The block ladder is derived automatically as the divisors of `N★` that are ≥16 (16³ up to the full
mesh). Because `N★` is large on an H200, expect **more block rungs and a longer job** than the A100 —
if a job nears its wall, split the angles:
```bash
RT_N_STAR=<N*> RT_SWEEPS=wavefront RT_NMU_MAX=4 sbatch rt-profiling/h200/submit_sweep.sbatch   # nmu 1-4
# then a second job with a trimmed RT_N_LIST/RT_NMU handling for 5-6 (or bump --time in the sbatch)
```
Useful knobs (all optional): `RT_N_REPEAT` (default 3, median ZCPS), `RT_NLIM` (default 50 cycles),
`RT_NMU_MAX` (default 6), `RT_BLOCK_FLOOR` (default 16).

## 5. Analyze + return results

```bash
python3 rt-profiling/analyze.py rt-profiling/h200      # Markdown tables + PNGs
```
Then either **commit** `rt-profiling/h200/results_*.csv` (+ `kernel_resources.csv`) on branch `rt-sc`
and push, or rsync them back to whoever is collating. Please include the H200 model + memory
(`nvidia-smi`) and the driver version so the numbers are comparable.

---

## What each output column means (`results_*.csv`)

- `N,B,nmb,zones` — mesh side, meshblock side, block count = (N/B)³, total cells.
- `nmu,rays_per_octant,rays_total` — angular order and its ray counts (report `rays_total`).
- `zcps_off_med` — hydro-only ZCPS (median of `RT_N_REPEAT`); nmu-independent.
- `zcps_on_med / _min / _max` — hydro+RT ZCPS (median / spread).
- `slowdown_med = zcps_off/zcps_on` — the RT cost factor (advisory; use absolute ZCPS as primary).
- `t_rad_ms,t_hydro_ms,t_sweep_ms` — device time for radiation / hydro / the formal-solution sweep.
- `rad_over_hydro,sweep_over_hydro` — per-cell cost of radiation / the sweep relative to hydro.
- `rad_cells_per_s,hydro_cells_per_s,gcaups` — completed cell-updates/s and the angle-weighted rate.
- `peak_mem_gib,pct_fill` — live peak device memory and % of total.
- `sm_util_mean/peak,mem_util_mean` — real-time GPU loading (nvidia-smi). `dcgm_sm_occ_mean,
  dcgm_sm_active_mean` — real-time SM occupancy/activity (only if DCGM was available).

## Gotchas

- **Re-derive N★ per device** — the A100's mesh does not fill an H200 and would under-load it.
- **GPU clocks** can't be pinned without admin; the script logs clocks/temp/power and uses medians.
- **Achieved per-kernel occupancy / warp-efficiency** needs Nsight Compute counters, which may be
  admin-blocked (`NVreg_RestrictProfilingToAdminUsers=1`); the static register + theoretical
  occupancy from `cuobjdump` needs no admin and is always produced.
- The 3D ray count is **double** the 2D count for the same `nmu` (8 octants vs 4) — this study is 3D.
