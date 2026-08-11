# SC radiation sweep — optimization ledger

A living record of what has been **tested** (with results + evidence) and what **ideas** remain, for the
non-relativistic short-characteristics (SC) solver (`nr_radiation::SC`, branch `rt-sc`). Scope: GPU
throughput of the formal-solution sweep (and directly related mesh/measurement choices). Keep this
current — add a row whenever an option is tried or an idea is opened/closed.

**Status legend:** ✅ landed · 📊 measured/characterised · ❌ retired (measured, not worth it) ·
💡 idea (not started) · 🔬 in progress.

**Golden rule (learned the hard way, 2026-08-10):** compare sweeps on **clean whole-run wall-time**
(`t_on − t_off`, `affect_fluid=false`), *not* the perf-probe `rad_cells_per_s`/`t_rad`. The perf probe
fences after every kernel (`src/utils/perf.cpp` `KernelsProbe::OnEnd`) and excludes radiation bvals, so
it **over-penalises many-launch (big-block) configs** and manufactured an artifactual "−34 % big-block
loss." See `REPORT.md` §3.6 and `a100/launch_overhead/SUMMARY.md`.

---

## 1. Tested / landed

| # | Option | Where | Status | Result | Evidence |
|--:|--------|-------|:------:|--------|----------|
| T1 | **Wavefront sweep, compact-plane** (exact per-plane interior cell list `wf_cell_`/`wf_plane_start_`; one `par_for` per hyperplane, `3B−2`/cycle) | `formal_solution.cpp` `FormalSolutionWavefront`/`BuildWavefrontIndex` | ✅ | Default sweep. Bit-identical to the earlier full-`B²`-grid-per-plane version; **+11–16 % (32³) / +10–11 % (64³)** A100 LTE vs it (removed the off-plane early-return waste). | HEAD `d776ee93` |
| T2 | **Diagonal sweep** (single `TeamPolicy` launch, internal `h`-loop + `team_barrier`, `league = nmb·nang_tot`) | `formal_solution.cpp` `FormalSolutionDiagonal` | ✅📊 | **Wins on small blocks** (single launch amortised: 16³ ≈ 3.7e7 vs wavefront 2.8e7 fenced) but **craters on big blocks** from team starvation (88³/176³: 1.0–1.1e7) — only `nmb·nang_tot` = 8·168 / 1·168 teams. Crossover ≈ 32³. 94 regs → theo. occ 0.25. | `REPORT.md` §3.3–3.4; `a100/results_diagonal.csv` |
| T3 | **Jacobi sweep** (unordered `par_for`, double-buffered `ir_prev`) | `formal_solution.cpp` `FormalSolutionJacobi` | ✅ | Race-free on GPU (double buffer); a reference/uncoupled variant. Not the throughput winner. | determinism suite |
| T4 | **Iteration-acceleration ladder** — Jacobi-ALI, SOR, center-out Gauss–Seidel ALI (`SweepUpdateGS`), GS local scatter (`gs_scatter`) | `formal_solution.cpp`, `iteration.cpp` | ✅ | Reduces *iteration count* (convergence), orthogonal to per-sweep throughput. GS-normalised scatter makes isotropic 3D stable at `ali_omega=1`. | `impl-athenak.md`; `iteration/` |
| T5 | **Memory-fill mesh design** (largest 16³-block mesh filling A100 memory at nmu=6) | `run_sweep.py` Phase 1 | 📊 | **N★ = 176³** (5.45 M zones, 35.8 GiB ≈ 90 % of 40 GB); 192³ OOMs. | `REPORT.md` §2; `a100/results_phase1.csv` |
| T6 | **Block × angle 2D throughput sweep** (both sweeps, nmu 1–6, block ladder {16,22,44,88,176} at N★) | `run_sweep.py` Phase 2 | 📊 | Full ZCPS-with/without-RT surface; the sweep crossover; live DCGM occupancy (~0.33–0.40) + static regs (wavefront 54–62, un-spilled). Sweep is **latency-bound**, not register/bandwidth-bound. | `REPORT.md` §3–5; `a100/results_{wavefront,diagonal}.csv` |
| T7 | **Lever 1 — fused single-launch wavefront** (cooperative-groups `grid.sync()` instead of `3B−2` launches) | *measurement only, no solver code* | ❌ | **Retired.** Fence-free nsys shows the recoverable inter-plane launch gap is a ~fixed **~18–23 µs/plane** → gap_fraction **3.9 % (176³) / 2.3 % (88³) / 0.5 % (16³)** at nmu=6 (monotonic in launch count). Only reaches **12 %** in the non-production corner (nmu=3 + single 176³ block). The big-block cost is **in-kernel latency**, not launches. A raw-CUDA cooperative kernel (non-portable, RDC/Rule-4 risk, perf-probe-blind) is unjustified for ~1–2 %. | `a100/launch_overhead/SUMMARY.md`; `results_nsys.csv`; HEAD `932e556f` |

---

## 2. Ideas — not yet implemented

Ranked by expected value (payoff ÷ risk/effort). Payoff = fraction of the **clean whole-run radiation
cost** plausibly recoverable; the real per-cell bottleneck is the latency-bound footpoint gather
(`UpdateCellSC`, `sc_interp.hpp`) at achieved occupancy ~0.36 with the memory interface only ~35 % busy.

| # | Idea | Target / rationale | Expected payoff | Portability / risk | Effort | Status |
|--:|------|--------------------|-----------------|--------------------|:------:|:------:|
| I1 | ~~De-starve the diagonal by tiling a plane across the league~~ → **shipped `sweep=diagonal_compact`** (compact-plane cell list + `diag_team_size` knob) | The literal tiling is a **DATA RACE** (1-cell footpoint reach; a plane's footpoints on h−1..h−3 are only `team_barrier`-synced → splitting a plane across teams needs a cross-team barrier = the retired Lever 1). So teams stay bounded at `nmb·nang_tot`; the shipped change instead removes the diagonal's ~⅔ off-plane early-return waste per team by reusing `wf_cell_`. | **Big win, bit-identical:** clean ZCPS **+5–11 % small/mid blocks, +39 % (88³) / +50 % (176³)** at nmu=6; nsys sweep-kernel −10 %/−31 %/−36 % (16³/88³/176³). Strict Pareto ≥ baseline diagonal everywhere; best sweep for B≤44; closes the 176³ gap to wavefront 58 %→87 %. Team-size knob (B): **no help, AUTO optimal.** Wavefront still wins B≥88. | **Portable** (pure Kokkos). Bit-identical (determinism hash == baseline). | Med | ✅📊 (`a100/diagonal_compact/SUMMARY.md`, HEAD post-`a7abcdbf`) |
| I2 | **Coalesce the footpoint gather via angle-innermost `ir`** — angle the contiguous dim, warp varies over angle. **SHIPPED native `ir_layout=angle_inner`** (opt-in; wavefront/3D/uniform). | The wavefront gather is memory-latency-bound; angle-innermost → coalesced `ir` + `chi`/`srad` broadcast. WAVEFRONT-ONLY (diagonal + `get_moments` pessimized). | **REAL whole-run speedup (native, N=128): 1.17× (64³) / 1.27× (128³) at nmu=6; ~break-even mid; 0.95× at 16³.** (Prototype measured the sweep-KERNEL prize 1.5–1.9×.) Bit-exact vs normal (jmean `0x0a5a…`, exchange `0xed93…`). | Native done ISOLATED: standard CC/AMR/hydro code **byte-unchanged**; SC-side shell-transpose bridge (`ir_normal` companion, O(surface)) around the unchanged exchange; SMR/AMR guarded off; `ir_normal` doubles ir memory. | High | ✅📊 (`a100/angle_inner/SUMMARY.md`, HEAD `bbfcc1f6`. Follow-up: reorder ComputeJ/moments to angle-warp to lift small/mid blocks; prototype seam in `a100/wavefront_coalesced/`.) |
| I3 | **Hoist angle-only invariants** out of the per-cell inner loop — precompute per-`(oct,a)` sign, dominant axis, and the 4 bilinear weights `c0..c3` into a small table | `sx/sy/sz`, `lx/ly/lz`, `lmin`, `c0..c3` depend only on the angle yet are recomputed per cell per plane (`sc_interp.hpp:172-224`). Removes `fabs`/`fmin`/divides + the axis branch from the hot loop. | Low (memory-bound ⇒ small), but trims instructions/registers | **Portable**, low-risk, self-contained | Low | 💡 (cheap warm-up) |
| I4 | **Mixed-precision `ir` transport** — carry intensity in `float`, keep moments/source in `double` | `ir` is the dominant memory object (35 GiB at N★); each cell reads 4 upwind `ir` + writes 1. Halving `ir` traffic is a large win for a memory-latency-bound kernel. | Potentially High (traffic halved) | **Physics/accuracy change (Rule 2)** — must validate vs Davis Eq. 30 + linear-wave; global. Flag-and-measure, not a "clear win." | High | 💡 (needs accuracy sign-off) |
| I5 | **`Kokkos::Experimental::Graph`** — record the `3B−2`-launch plane sequence once and replay | Same hypothesis as retired Lever 1 (cut launch overhead) but **portable**, no RDC, ~tens of lines. | Low (~1–2 %; nsys says the launch gap is small) — only meaningful in the low-angle/single-giant-block corner (T7). | Portable. Note: graph-replayed kernels may not fire the perf-probe callbacks. | Low–Med | 💡 (only if the launch corner ever matters; **preferred over raw CUDA**) |
| I6 | **Raw-CUDA cooperative-groups fused sweep** (`sweep=wavefront_coop`) | The faithful Lever 1. Design is fully worked out and preserved. | ~1–2 % (see T7) | **Non-portable** raw CUDA (only such kernel in the codebase), possible global RDC (**Rule-4** hydro risk), invisible to the perf probe. | High | ❌ (design archived in the plan file; do **not** build unless a measured prize > 10 % appears in a *production* config) |

---

## 3. Measurement infrastructure (reusable)

| Tool | Purpose |
|------|---------|
| `run_sweep.py` | Phase-1 memory-fill + Phase-2 block×angle ZCPS surface (imports `tst/benchmark/rad_cost/run_rad_cost.py` as the single deck source). |
| `launch_overhead/measure_nsys.py` | Fence-free nsys gap_fraction (launch-overhead) for any sweep; `RT_SMOKE=1` unit-checks the parser off-GPU. |
| `kernel_resources.py` | Static per-kernel regs/spills/shared-mem + theoretical occupancy (`cuobjdump`). |
| `analyze.py` | Tables + plots from the Phase-2 CSVs. |
| `a100/submit_*.sbatch`, `h200/…` | Slurm wrappers (partition `apollo`/`h200`); measurement-only, no rebuild. |

**Metric definitions** — clean whole-run ZCPS = driver `zone-cycles/cpu_second` (OFF vs ON, radiation cost
= `t_on − t_off`); gap_fraction = `(span − active)/span` per steady-state cycle (nsys, fence-free);
angles→rays (3D Bruls type-A): nmu {1..6} → {8,24,48,80,120,168} total.

---

## 4. Open questions / notes

- **Is a single sweep achievable that wins across the whole block ladder?** I1 (de-starved diagonal) is
  the most direct test; if it wins big blocks while keeping small-block launch efficiency, it could
  replace the wavefront/diagonal crossover.
- **Whole-run vs fenced:** always re-derive throughput from `zcps_on/zcps_off`; treat `rad_cells_per_s`
  as diagnostic only (§Golden rule).
- **Production regime reminder:** real runs use many meshblocks (16–32³), where the wavefront already
  fuses all blocks into one launch/plane. Optimisations should be judged there, not on the synthetic
  single-176³-block extreme.
- **Multifrequency** (separate track): staged multiband → binning → correlated-k; out of scope for this
  throughput ledger but will multiply per-cell cost — revisit I2/I4 (memory) when it lands.

---
*Last updated 2026-08-10 (`932e556f`). Update on every tried option / opened idea.*
