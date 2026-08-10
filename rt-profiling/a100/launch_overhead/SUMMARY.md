# Lever 1 — launch-overhead measurement (A100, fence-free nsys)

*Apollo A100-PCIE-40GB, branch `rt-sc` (`f7e96af3`), build_a100 (SC = compact-plane wavefront).
Run 2026-08-10 via `rt-profiling/a100/submit_nsys.sbatch` → `launch_overhead/measure_nsys.py`.
Data: `results_nsys.csv`.*

## Question

Before building a fused single-launch wavefront SC sweep ("Lever 1"), size the **only** thing such a
fusion can recover: the wall-time spent in **inter-plane launch/relaunch/ramp gaps** during a normal,
production-like (unfenced, async `par_for`) run. The current wavefront issues one kernel launch per
hyperplane — `3B−2` launches/cycle (**526** for a 176³ block, **46** for 16³).

**Why this needed measuring:** the earlier study's `rad_cells_per_s` (which showed a ~34% "loss" from
16³→176³ blocks) is computed from the **perf-probe** run, which calls `Kokkos::fence()` after *every*
kernel (`src/utils/perf.cpp` `KernelsProbe::OnEnd`) — fully exposing 526 vs 46 launch bubbles/cycle —
and **excludes radiation bvals**. Both biases penalise many-launch (big-block) configs. The clean
whole-run radiation cost (`t_on − t_off`, `affect_fluid=false`) is **U-shaped, not monotonic**
(16³:15.0 s, 44³:12.7 s, 176³:15.2 s at nmu=6). So the "loss" was largely an instrumentation artifact;
this run measures the real launch prize directly, with **no solver change and no rebuild**.

## Method

Same LTE deck as `run_rad_cost` (`opa=1, ops=0, eps=1 ⇒ use_ali=false`, `iter_max=itermin=1` ⇒ one
formal solution/cycle) but with **no `<output file_type=perf>` block**, so `par_for` stays async (no
fences). `nsys profile --trace=cuda` captures the CUDA timeline; per steady-state cycle for the sweep
kernel (`FormalSolutionWavefront`/`sc_sweep3d`):

- **active** = Σ kernel durations, **span** = last end − first start, **gap = span − active**.
- **gap_fraction = gap / span** = the recoverable-by-fusion upper bound.

Mesh fixed at N★=176³, `sweep=wavefront`, `nlim=10`, median-span full cycle.

## Results

| B | nmb | nmu | rays | launches/cyc | active (ms) | gap (ms) | **gap_fraction** | µs/plane |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 176 | 1 | 6 | 168 | 526 | 260.9 | 10.7 | **3.9 %** | 20.4 |
| 88 | 8 | 6 | 168 | 262 | 212.0 | 4.9 | **2.3 %** | 18.8 |
| 16 | 1331 | 6 | 168 | 46 | 169.5 | 0.8 | **0.5 %** | 17.9 |
| 176 | 1 | 3 | 48 | 526 | 86.4 | 12.2 | **12.4 %** | 23.2 |

## Reading

1. **The metric is real** — gap_fraction is monotonic in launch count (0.5 % → 2.3 % → 3.9 % as
   launches go 46 → 262 → 526 at nmu=6), exactly the sanity-anchor prediction.
2. **The inter-plane gap is a ~fixed ~18–23 µs/plane overhead** (launch latency + grid ramp/drain),
   essentially independent of angle count: gap_ms ≈ 10–12 ms/cycle at 176³ whether nmu=6 or nmu=3.
3. **At the study's headline nmu=6 the prize is small** — <4 % of the sweep even at the single-block
   extreme, <0.5 % in the production-like many-block regime (16³). As a fraction of the *whole app* it
   is ~1–2 % at most.
4. **The one >10 % cell is a non-production corner:** nmu=3 + single 176³ block — kernels *both* lean
   (few angles → small `active`) *and* maximally numerous (one giant block → 526 launches), so the
   fixed gap becomes a large fraction. Nobody runs a single 176³ block; production uses many blocks,
   where the wavefront already fuses all blocks into one launch/plane (the `m` loop dimension).
5. **Cross-check:** nsys unfenced `active` (260.9 ms/cycle at 176³) is ~8 % below the earlier fenced
   `t_sweep` (~285 ms/cycle) — quantifying the fence inflation that produced the artifactual "loss".

*(Caveat: at 176³ a few cycles show intra-cycle sub-bursts from occasional scheduling hiccups; the
analysis reports the median-span **clean** 526-launch cycle, the fair steady-state value.)*

## Verdict — retire Lever 1 as a general optimization

The 34 % motivation was a fenced-metric artifact. The true recoverable launch overhead is **<4 %** at
high angular resolution and **<0.5 %** in the many-block regime that matters — well below the ~5 % build
threshold. A `grid.sync()` fused kernel would also **not** touch the thin-corner-plane idle (the grid
still stalls at the barrier) nor the in-kernel latency (occupancy ~0.36, mem-interface ~35 % busy), which
is the actual ~16–20 % big-block gap. A **raw-CUDA cooperative-groups** kernel (non-portable, RDC/Rule-4
risk, invisible to the perf probe) is not justified to chase ~1–2 %.

**If** a fused sweep is ever wanted for the low-angle/single-giant-block corner (~12 %), the portable
route is **`Kokkos::Experimental::Graph`** (record/replay), never raw cooperative groups.

## Recommended pivot (portable, Rule-3-clean)

The clean data points at two real, in-kernel targets:
1. **De-starve the diagonal on big blocks** — `league = nmb·nang_tot` = 168 teams at 176³ starves;
   add a per-plane spatial tiling factor to the league (stays in Kokkos TeamPolicy).
2. **Latency lever** — `ir` layout / angle-ILP for the `UpdateCellSC` footpoint gather
   (`sc_interp.hpp`), the actual latency-bound cost (occupancy ~0.36, not bandwidth-bound).
