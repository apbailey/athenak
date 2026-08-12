# I3 — `sc_hoist` (precomputed per-ray interpolation invariants): measured NEUTRAL / slightly negative

*Apollo A100-PCIE-40GB, branch `rt-sc` (`e03c48fa`), `build_a100`. Run 2026-08-12 via
`a100/submit_hoist_ab.sbatch` (job 16953004). Data: `results_hoist.csv`. Single-shot per config
(nlim=50, no median) → ±few-% run-to-run noise.*

## What this is

The ledger's **I3**: hoist the angle-only SC interpolation invariants (dominant axis, bilinear weights
`c0..c3`, path-length `dx_dom/|mu_dom|`) out of the per-cell wavefront loop by precomputing them once per ray
into `sc_inv_` and reading the table instead of recomputing `lx/ly/lz/lmin` + the axis branch + weights per
cell. Opt-in `<nr_radiation>/sc_hoist=true` (default `false` byte-identical); wavefront/3D/uniform/normal only.
Implemented by refactoring `UpdateCellSC` → `ComputeSCAngleInv` + `GatherSolveSC` (`sc_interp.hpp`) and a
hoisted wavefront-3D sweep variant (`sc_sweep3d_hoist`) dispatched at host scope.

## Correctness (GPU, bit-exact)

`sc_hoist=true` wavefront determinism hash `0x9a543797d2295dea` **== baseline** (MATCH). The table is built
on-device by the same `ComputeSCAngleInv` the recompute path uses, so `GatherSolveSC` runs identical FP ops;
also verified bit-identical to **pre-refactor** HEAD on CPU (the refactor preserves the `(x*pdx)/pamu`
left-associative order of the path factor).

## Whole-run ZCPS — `sc_hoist=true` / baseline (sweep=wavefront, N=176³)

| block B | nmb | **nmu=6 (168 rays)** | nmu=3 (48 rays) |
|--:|--:|--:|--:|
| 16  | 1331 | 0.99× | 1.00× |
| 22  | 512  | 0.97× | 0.99× |
| 44  | 64   | **0.96×** | 0.99× |
| 88  | 8    | 0.97× | 1.04× |
| 176 | 1    | 1.00× | 1.07× |

**Verdict: no throughput win.** At the production-relevant **nmu=6** (many rays), `sc_hoist` is a **consistent
small regression** (0.96–1.00×, worst 0.96 at 44³, break-even only at the single 176³ block). The apparent
**nmu=3** gains at 88³/176³ (1.04/1.07×) are **not credible wins** — single-shot (no median), they contradict
the nmu=6 result at the *same* block, and 176³ is the single-giant-block synthetic extreme the ledger
explicitly says not to optimise for. The small blocks (16³–44³, the real production regime) are ≤1.00× at
both angle counts.

## Why (confirms the memory-latency-bound diagnosis)

The wavefront sweep is memory-latency-bound (occupancy ~0.36, memory ~35 % busy). The `fabs`/`fmin`/divides +
axis-branch arithmetic that I3 removes sits on threads already stalled on the footpoint `ir` gather, so it
was essentially **free** (hidden under that latency). Replacing it with a per-cell **global-memory table
load** (10 Reals/cell) *adds* dependent memory traffic to a memory-bound kernel — a net small cost, largest
where the sweep does the most per-cell work (nmu=6). The one mechanism that could have helped (lower registers
→ higher occupancy) did not materialise as a whole-run gain. (The axis branch is *uniform* across the warp in
the normal layout — not divergent — so removing arithmetic inside it never helped divergence anyway.)

## Disposition

- **Do NOT enable by default; `sc_hoist` retired as a throughput lever** (kept opt-in + documented, Lever-1
  style — a measured, honest negative). I3 confirms the latency-bound bottleneck: attack memory traffic (I4
  mixed-precision) or coalescing (I2), not per-cell ALU.
- **Keep the refactor.** `ComputeSCAngleInv` / `GatherSolveSC` are bit-exact, DRY, and set up the deferred
  GS-scatter dedup (Kernel A recomputes the same `lx/ly/lz/lmin`). The refactor is a clean win independent of
  the `sc_hoist` verdict.
- **Deferred (unbuilt):** the diagonal in-register hoist companion (would isolate the pure-ALU question with
  no table-load confound) — not worth building given the wavefront table already shows the ALU removal is
  worthless here; only revisit if a future kernel becomes compute-bound.

*A median (RT_N_REPEAT) re-run would firm the noisy nmu=3 large-block tail but cannot change the verdict — the
nmu=6 trend (consistently ≤1.0×) is the clear signal.*
