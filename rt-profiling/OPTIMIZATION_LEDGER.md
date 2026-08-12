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

**`REPORT2.md` (2026-08-12)** repeats the original `REPORT.md` A100 Phase-1/Phase-2 study with the
shipped I1 (`diagonal_compact`, same N★=176) and I2 (`angle_inner`, its own N★_ai=144) in place of the
old `diagonal`/baseline-only picture — read it for the updated crossover table and practical guidance.

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
| T8 | **Cross-architecture confirmation** (GH200 + B200 on Vista; all three sweeps, full block×ray grid) | *measurement only* | 📊 | `diagonal_compact` ≥ `diagonal` on **30/30** B200 points and **34/36** GH200 points, reproducing the A100 A/B on two more architectures. Crossover is invariant in **meshblock count** (dc wins nmb ≥ 64, wf wins nmb ≤ 8) on all three. The hydro/RT gap is **architecture-nearly-invariant**: `gap ≈ k·rays` with k = 2.99/2.50/2.25 % (A100/GH200/B200) at B=16 — block size moves k ~4×, architecture ~1.4×. Radiation scales A100→B200 by **2.67×** (B=16) / **3.52×** (big blocks) vs hydro's 1.93×/3.28×, i.e. **the gap does not widen on newer hardware**. One real arch effect: diagonal registers 94 → **122** on sm_100. | `ARCH_SCALING.md`; `{b200,gh200}/SUMMARY.md`; `arch/*.png` |
| T9 | **Boundary-lag measurement** — iterations to converge vs domain decomposition (`sc_lag` pgen + `lag_study.py`) | `src/pgen/unit_tests/sc_lag.cpp`, `tst/inputs/sc_lag.athinput` | 📊 | The sweep is meshblock-local, so light advances one block per iteration and **`niter = 3·(blocks per side) − 1`** exactly (measured 2/5/11/22 for 1/2/4/8). Independent of angular order **and** of `iter_tol` — it is a geometric count, not a decay rate. Saturates at ~4 when `τ_blk ≳ 3` (absorbed before crossing). Folding it into measured `ZCPS_rad`: at 16³ blocks an optically **thin** problem costs **28× (B200) / 17× (GH200) / 13× (A100)** more total radiation time than one big block — **inverting the small-block guidance**, which holds only in the thick regime. Every throughput row above was taken at `iter_max=1` and therefore does not include this. | `LAG_STUDY.md`; `lag/lag_results.csv` |
| T10 | **`Kokkos::AUTO` team-size audit** (GH200, single 224³ block, AUTO/64/128/256/512 × diagonal_compact and tiled) | *measurement only* | 📊 | Run to test whether AUTO explains the tiled `tile_size=0` regression. **It does not** — AUTO resolves to 128 for BOTH kernels (identical to 4 s.f.), and forcing them to match leaves the gap at 0.680. **But it found something bigger: AUTO is up to 2.6× suboptimal at starved configs.** At 48 rays/1 block, ts=512 beats AUTO by **1.91× (diagonal_compact)** and **2.61× (tiled)**; at 168 rays, 1.00× and 1.18×. So I1's "AUTO optimal" verdict holds only at the angular order it was measured at (nmu=6). The `tile_size=0` gap is a per-kernel team-size *sensitivity* difference, root cause still unknown (not registers: tiled 88 vs diagonal 94). | `TILING_REPORT.md` §7; `gh200/teamsize/`; job 904561 |
| T11 | **I7 × I2 composition** (`sweep=tiled` + `ir_layout=angle_inner`, `tile_na` sweep) | `formal_solution.cpp` `FormalSolutionTiledAngleInner` | 📊❌ | **They do not compose — they are alternatives.** Both spend the SAME resource: I7 turns angles into teams (`league = ntile·nmb·nang_tot`), I2 turns angles into lanes. `tile_na` is the exchange rate and the trade loses both ways — large `tile_na` collapses the league 24–28× (GH200 B=224: 24,696 → 882 teams; B200 B=256: 8,064 → **96**), small `tile_na` shrinks the coalesced run to nothing. GH200 B=112/168 rays vs baseline: I2 alone **1.22×**, I7 alone **1.40×**, composed best (`tile_na=2`) **1.21×**, degrading monotonically to 0.85× at `tile_na=8`. Kernel is bit-identical (`0x9a543797d2295dea`), just slower. **At large blocks prefer I7.** | `TILING_REPORT.md` §8; `gh200/tilena/`; jobs 907293/4, 907414 |

---

## 2. Ideas — not yet implemented

Ranked by expected value (payoff ÷ risk/effort). Payoff = fraction of the **clean whole-run radiation
cost** plausibly recoverable; the real per-cell bottleneck is the latency-bound footpoint gather
(`UpdateCellSC`, `sc_interp.hpp`) at achieved occupancy ~0.36 with the memory interface only ~35 % busy.

| # | Idea | Target / rationale | Expected payoff | Portability / risk | Effort | Status |
|--:|------|--------------------|-----------------|--------------------|:------:|:------:|
| I1 | ~~De-starve the diagonal by tiling **one cell-hyperplane** across the league~~ → **shipped `sweep=diagonal_compact`** (compact-plane cell list + `diag_team_size` knob) | Tiling a *plane* is a **DATA RACE** (1-cell footpoint reach; a plane's footpoints on h−1..h−3 are only `team_barrier`-synced → splitting a plane across teams needs a cross-team barrier = the retired Lever 1). **Scope note (2026-08-11): this verdict covers INTRA-PLANE tiling only.** Tiling the meshblock *volume* and running a wavefront over tiles, one launch per tile-plane, is a different algorithm whose cross-tile barrier is the kernel boundary — safe, and opened as **I7**. So within I1 teams stay bounded at `nmb·nang_tot`; the shipped change instead removes the diagonal's ~⅔ off-plane early-return waste per team by reusing `wf_cell_`. | **Big win, bit-identical:** clean ZCPS **+5–11 % small/mid blocks, +39 % (88³) / +50 % (176³)** at nmu=6; nsys sweep-kernel −10 %/−31 %/−36 % (16³/88³/176³). Strict Pareto ≥ baseline diagonal everywhere; best sweep for B≤44; closes the 176³ gap to wavefront 58 %→87 %. Team-size knob (B): no help **at 176³/nmu=6** — but that does NOT generalise (see T10: on GH200 at nmu=3 AUTO is 1.9× off). Wavefront still wins B≥88. | **Portable** (pure Kokkos). Bit-identical (determinism hash == baseline). | Med | ✅📊 (`a100/diagonal_compact/SUMMARY.md`, HEAD post-`a7abcdbf`) |
| I2 | **Coalesce the footpoint gather via angle-innermost `ir`** — angle the contiguous dim, warp varies over angle. **SHIPPED native `ir_layout=angle_inner`** (opt-in; wavefront/3D/uniform). | The wavefront gather is memory-latency-bound; angle-innermost → coalesced `ir` + `chi`/`srad` broadcast. WAVEFRONT-ONLY (diagonal + `get_moments` pessimized). | **REAL whole-run speedup (native, N=128): 1.17× (64³) / 1.27× (128³) at nmu=6; ~break-even mid; 0.95× at 16³.** (Prototype measured the sweep-KERNEL prize 1.5–1.9×.) Bit-exact vs normal (jmean `0x0a5a…`, exchange `0xed93…`). | Native done ISOLATED: standard CC/AMR/hydro code **byte-unchanged**; SC-side shell-transpose bridge (`ir_normal` companion, O(surface)) around the unchanged exchange; ❗**AMR/SMR unsupported — FATAL on any multilevel mesh** (needs `sweep=wavefront`, 3D, uniform single-level; guard `nr_radiation.cpp:100`; see §4); `ir_normal` doubles ir memory. | High | ✅📊 (`a100/angle_inner/SUMMARY.md`, HEAD `bbfcc1f6`. Follow-up: reorder ComputeJ/moments to angle-warp to lift small/mid blocks; prototype seam in `a100/wavefront_coalesced/`.) |
| I3 | **Hoist angle-only invariants** out of the per-cell inner loop → **shipped opt-in `<nr_radiation>/sc_hoist`** (precompute per-ray dominant-axis + weights `c0..c3` + path-length into `sc_inv_`; wavefront/3D/uniform/normal). | `sx/sy/sz`, `lx/ly/lz`, `lmin`, `c0..c3` depend only on the angle yet are recomputed per cell (`sc_interp.hpp` `UpdateCellSC`). Removes `fabs`/`fmin`/divides + the axis branch from the hot loop. Refactored `UpdateCellSC` → `ComputeSCAngleInv`+`GatherSolveSC` (the table reads the latter directly). | **MEASURED ~0 / slightly NEGATIVE, retired as a lever:** whole-run ZCPS `sc_hoist`/baseline **0.96–1.00× at nmu=6** (consistent small regression, worst 44³; break-even only at the single 176³ block); nmu=3 88³/176³ "1.04/1.07×" are single-shot noise (contradict nmu=6, no median). Confirms latency-bound: the per-cell table load *adds* memory traffic; the removed ALU already hid under gather latency; the axis branch is warp-**uniform** (not divergent). Bit-exact (GPU hash `0x9a54…` == baseline; refactor == pre-change HEAD). | **Portable**, low-risk, self-contained. The `ComputeSCAngleInv`/`GatherSolveSC` refactor is a clean DRY keeper (bit-exact; sets up the GS-scatter dedup) even though `sc_hoist` doesn't pay. | Low | ❌📊 (`a100/hoist/SUMMARY.md`, HEAD `e03c48fa`. Do NOT default-enable; opt-in kept as a documented negative. Diagonal in-register companion deferred — unbuilt.) |
| I4 | **Mixed-precision `ir` transport** — carry intensity in `float`, keep moments/source in `double` | `ir` is the dominant memory object (35 GiB at N★); each cell reads 4 upwind `ir` + writes 1. Halving `ir` traffic is a large win for a memory-latency-bound kernel. | Potentially High (traffic halved) | **Physics/accuracy change (Rule 2)** — must validate vs Davis Eq. 30 + linear-wave; global. Flag-and-measure, not a "clear win." | High | 💡 (needs accuracy sign-off) |
| I5 | **`Kokkos::Experimental::Graph`** — record the `3B−2`-launch plane sequence once and replay | Same hypothesis as retired Lever 1 (cut launch overhead) but **portable**, no RDC, ~tens of lines. | Low (~1–2 %; nsys says the launch gap is small) — only meaningful in the low-angle/single-giant-block corner (T7). | Portable. Note: graph-replayed kernels may not fire the perf-probe callbacks. | Low–Med | 💡 (only if the launch corner ever matters; **preferred over raw CUDA**) |
| I6 | **Raw-CUDA cooperative-groups fused sweep** (`sweep=wavefront_coop`) | The faithful Lever 1. Design is fully worked out and preserved. | ~1–2 % (see T7) | **Non-portable** raw CUDA (only such kernel in the codebase), possible global RDC (**Rule-4** hydro risk), invisible to the perf probe. | High | ❌ (design archived in the plan file; do **not** build unless a measured prize > 10 % appears in a *production* config) |
| I7 | **Tiled diagonal sweep (KBA)** — partition the meshblock into tiles, wavefront over tile-planes, one launch per tile-plane; each team sweeps one tile with today's inner `h`-loop | Fixes the diagonal's team starvation at large blocks *without* the I1 race: the cross-tile barrier is the kernel boundary. At B=176/nmu=6: **168 → 15,288 teams**, **31 launches** (vs wavefront's 526), and a tile-sized working set instead of a 176² plane — all three aimed at the latency-bound regime (mem interface only ~35 % busy). Degenerate setting `tile=block, Na=1` reproduces `diagonal_compact` **bit-for-bit**, so the kernel subsumes the current one and the determinism hash validates the refactor before any tiling is enabled. | **Potentially large at B ≥ 44**; no-op at B=16 by construction (that is where diagonal already wins) | **Portable** (pure Kokkos, host launch loop). Bit-identical by the upwind-corner argument. Needs exact tile/block division; **not applicable to `ali_mode=gauss_seidel`** (GS needs global center-out shells). Watch registers — diagonal is already 94 (122 on sm_100), so I3 may become a prerequisite. | Med | ✅📊 **LANDED AND MEASURED — wins on both devices.** Best sweep at every large-block point: **+26–63 % vs wavefront** (GH200/B200, 168 rays), and **1.27–1.40× better than the best config previously known anywhere on the ladder**. Optimal tile ≈16³ (GH200) / 32³ (B200) — the design's prediction, untuned. 8³ tiles lose (the predicted lower bound). New optimum sits at a LARGE block, which compounds with hydro throughput and with the `LAG_STUDY.md` iteration count. GPU determinism gate passed on both. `TILED_SWEEP_DESIGN.md`, `tile_study.py`, jobs 904039/904040 |
| I8 | **Angle-blocked sweep** (`Na` angles per thread/team) — load the 9+9 `chi`/`srad` footpoint stencil once per cell and reuse it across `Na` angles | `chi` and `srad` are **angle-independent**, yet the stencil is re-gathered for every one of the 8–168 rays: 18 loads/(cell,ray) where ~18/(cell,`Na` rays) would do. Attacks the same latency-bound gather as I2 but from the reuse side rather than the coalescing side. Two mappings trade oppositely: angle-per-thread keeps all 168× parallelism but the reuse group is capped at `nang` ≤ 21 (and warps straddle octants); angle-loop-per-thread gets full reuse and **zero warp divergence** (all lanes walk the same angle list) but cuts parallelism 168×. | Med–High at nmu ≥ 3; **nil or negative at nmu=1** | Portable. **Only pays once `ir` is angle-major (I2)** — under today's layout the `Na` angles of one cell are strided by `n1·n2·n3`. So I2 and I8 must land together; individually each is marginal. | Med | 💡 (knob specified in `TILED_SWEEP_DESIGN.md` §5) |
| I9 | **Globally ordered (pipelined KBA) sweep across meshblocks** — order blocks by the same upwind-hyperplane rule used *inside* a block, and pipeline the boundary exchange, instead of block-Jacobi with one exchange per iteration | Removes the T9 boundary-lag penalty **entirely**: converges in ONE iteration for any decomposition, paying pipeline fill/drain in message *latency* rather than in full sweeps of the whole domain. Since a rank owns ≥1 block, today's scheme makes `niter ∝ nranks^(1/3)` — a hard cap on strong scaling that no kernel optimisation can touch. | **Highest in the ledger for production at scale** — up to 28× on the measured thin case, and it is the only item that improves with rank count instead of degrading | Large: requires ordered/pipelined MPI in `sc_bvals` (today's exchange is unordered). Well-trodden in the transport literature (KBA). Same algorithmic family as I7, promoted from intra-block to inter-block. | High | 💡 (`LAG_STUDY.md` §7) |

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
- **I2 (`angle_inner`) is uniform-mesh only — AMR/SMR unsupported (by design).** The native path
  FATAL-errors at startup on any multilevel mesh (guard `nr_radiation.cpp:100`, checks
  `pmesh->multilevel`, true for both SMR *and* AMR). **Why:** the isolation that makes I2 safe (standard
  CC-exchange / AMR / hydro code byte-unchanged, Rule 4) only holds on a uniform mesh, where `ir` crosses
  into shared code *solely* through the per-cycle boundary exchange — bridged by the O(surface) shell
  transpose to the `ir_normal` companion. Under AMR the shared mesh machinery (`mesh_refinement.cpp`,
  `load_balance.cpp`) restricts / prolongs / refines / load-balances the *whole* `ir` array directly,
  interleaved with hydro — which angle-innermost `ir` cannot satisfy without either editing those shared
  routines (breaks isolation + Rule 4) or a full-array transpose every refinement (the exact tax that
  sank the transpose prototype). Refined meshes therefore fall back to the default `normal` layout.
  **Supporting AMR is a documented later increment** — it needs the shared restrict/prolong/refine/
  load-balance path bridged to angle-inner too (meaningfully larger, higher-risk work).
- **Is a single sweep achievable that wins across the whole block ladder?** **I7 (tiled/KBA diagonal)
  is now the most direct test** — it is the only proposal that raises team count at large blocks
  without either the I1 intra-plane race or the retired Lever-1 cooperative kernel, and it is a no-op
  at B=16 where the diagonal already wins. If it holds up, one sweep replaces the crossover.
- **I7 and I2 are ALTERNATIVES, not a sequence (T11, measured).** Both consume the single angle
  dimension — I7 as teams, I2 as lanes — so enabling both is strictly worse than either. At large
  blocks prefer I7 (1.40× vs 1.22× on GH200 B=112). The earlier "sequence I7 then I2" plan is void,
  as is the older suggestion that I3 would be a prerequisite (I3 measured ~0/negative, retired).
- **Whole-run vs fenced:** always re-derive throughput from `zcps_on/zcps_off`; treat `rad_cells_per_s`
  as diagnostic only (§Golden rule).
- **Production regime reminder — REVISED by T9.** The "real runs use many 16–32³ meshblocks" premise
  was itself a consequence of the `iter_max=1` throughput measurement. With iterations counted, an
  optically thin problem wants the LARGEST blocks it can afford (28× on B200), and only optically
  thick problems (`τ_blk ≳ 3`) want 16–32³. Judge optimisations in the regime your `τ_blk` puts you
  in — and note that the large-block regime is exactly where the diagonal collapses, which raises
  the value of I7.
- **`Kokkos::AUTO` is not to be trusted at starved configs (T10).** It leaves up to 2.6× on the
  table when the league is small (few blocks × low angular order). Any future sweep work should
  carry an explicit team size, or at minimum sweep it before quoting a number — and the existing
  A100 "AUTO is optimal" note applies only at nmu=6.
- **Cross-architecture guidance transfers.** T8 shows the gap law `gap ≈ k·rays` and the nmb-based
  sweep crossover hold on A100, GH200 and B200, so A100-derived tuning advice is portable. The
  corollary is that **newer silicon will not close the gap** — only the sweep work (I7, I2+I8, I3) will.
- **Multifrequency** (separate track): staged multiband → binning → correlated-k; out of scope for this
  throughput ledger but will multiply per-cell cost — revisit I2/I4 (memory) when it lands.

---

## 5. Consolidation to final version (how to reach the shipping solver)

**Principle.** Every experiment here is an **opt-in flag that is byte-identical when off** (`sweep`,
`ir_layout`, `sc_hoist`, `diag_team_size`, `ali_mode`, `gs_scatter`). So once the winners are chosen,
"cleanup" is mechanical and physics-free: (a) pick the default sweep + layout, (b) inline that path, (c)
delete the losing/prototype kernels + their flags + alloc branches, (d) keep only the surviving determinism
decks. Do it in one pass, gated by the determinism suite (hashes must not move for the kept paths).

**Per-option disposition** (fill in as evidence lands):

| Option | Disposition | Concrete cleanup target |
|--------|-------------|-------------------------|
| T1 wavefront (compact-plane) | **keep — default** | the production sweep; stays. |
| I2 `ir_layout=angle_inner` | **keep — opt-in** (real win, large blocks) | keep; `ir_normal` bridge + FATAL guards stay until AMR increment. |
| I1 `sweep=diagonal_compact` | **keep — opt-in** (wins small/mid blocks) | keep; `diag_team_size` knob = AUTO-optimal, could drop the knob. |
| I7 `sweep=tiled` | **keep — opt-in** (best sweep at every large block on GH200+B200) | keep; shares `BuildPlaneIndex`/`diag_team_size` with I1. Candidate to *replace* `diagonal_compact` once the `tile_size=0` gap is understood. |
| I7×I2 `FormalSolutionTiledAngleInner` | **delete-after-tests** (measured: does not compose, T11) | remove the kernel + `tile_na` + the `tiled` arm of the angle_inner validator; keep the finding in `TILING_REPORT.md` §8. |
| `sweep=jacobi` | **keep — opt-in** (reference/uncoupled) | keep (small). |
| T4 GS-ALI (`ali_mode=gauss_seidel`, `gs_scatter`) | **keep — opt-in** (convergence, orthogonal) | keep. |
| I3 `sc_hoist` | **delete-after-tests** (measured ~0/neg) | remove the flag + `sc_inv_` + `BuildAngleInvTable` + `sc_sweep3d_hoist` branch; **but KEEP** the `ComputeSCAngleInv`/`GatherSolveSC` refactor (bit-exact, DRY, enables the GS-scatter dedup). |
| `sweep=wavefront_coalesced` + `ir_t` | **delete** (superseded by native `angle_inner`) | remove `FormalSolutionWavefrontCoalesced`, `ir_t` member + its realloc, the coalesced deck. |
| Lever 1 / `wavefront_coop` (I6) | **already-retired** — do NOT build | design archived in the plan file only; no code. |
| I5 `Kokkos::Graph` | **not started** | only if the launch corner ever matters (T7). |
| I4 mixed-precision `ir` | **not started** (needs accuracy sign-off) | Rule-2 validation first. |

**Shared scaffolding to KEEP regardless** (permanent structure, not experiment-specific): `IrGet<>`,
`ComputeSCAngleInv`/`GatherSolveSC`, `BuildWavefrontIndex`/`wf_cell_`, the `sc_sweep_determinism` pgen +
`run_rad_cost`/`run_sweep.py` harness.

**Cleanup checklist when the call is made:** dispatcher in `formal_solution.cpp` (`FormalSolution()`); the flag
parses + FATAL guards in `nr_radiation.cpp`; the `ir`/`coarse_ir` alloc branches (`nr_radiation.cpp:238`);
the deleted kernels' method decls in `nr_radiation.hpp`; the retired decks in `tst/inputs/` + their pytest
rows; archive the corresponding `rt-profiling/a100/*` result dirs.

---
*Last updated 2026-08-12 — I3 (`sc_hoist`) measured on A100: ~0 / slightly negative, retired as a lever
(bit-exact refactor kept); added §5 consolidation-to-final-version plan. Merged the GH200/B200 branch:
T8 cross-architecture, T9 boundary-lag, T10 `Kokkos::AUTO` team-size audit, I7 `sweep=tiled` (measured,
wins), I8 angle blocking, I9 pipelined KBA across meshblocks. Update on every tried option / idea.*
