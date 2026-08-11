# I2 — wavefront footpoint-gather coalescing prize (A100, measured)

*Apollo A100-PCIE-40GB, branch `rt-sc` (`2889f282`), `build_a100`. Run 2026-08-11 via
`a100/submit_wavefront_coalesced_ab.sbatch`. Mesh N=128³ (so the 2×`ir` scratch fits). Data:
`sweep_prize.csv`, `launch_overhead/results_nsys*.csv`, `results_both.csv`.*

## What was measured

Whether coalescing the wavefront's per-cell footpoint gather is worth a (very invasive, wavefront-only)
angle-innermost `ir` layout. To size the prize *without* the global bvals/AMR rewrite, the opt-in
`sweep=wavefront_coalesced` transposes `ir` into an angle-innermost scratch `ir_t`, runs the wavefront
plane loop **reordered so the warp varies over angle** (`par_for(m,c,angg)`, `UpdateCellSC<true>`) → the
`ir` footpoint reads coalesce and the angle-independent `chi`/`srad` become broadcasts, then transposes
back. The transpose + 2×`ir` are measurement-only; a **native** angle-innermost `ir` would have neither.

## Correctness (GPU)

`sc_determinism_wavefront_coalesced` PASS, hash `0x9a543797d2295dea` **== baseline wavefront**
(bit-for-bit; new gate `test_sc_wavefront_coalesced_matches_wavefront`); `sc_atmosphere` PASSED.

## The gather prize — sweep-kernel active time, coalesced vs baseline (nsys, per 10 cycles)

*(sweep kernel isolated from the transposes by kernel name in the CUDA trace; `results_nsys_*.csv`
`active_ms` for the coalesced run lumps in the transposes and is NOT the prize — use `sweep_prize.csv`.)*

| block B | nmb | rays | wavefront sweep (ms) | **coalesced sweep (ms)** | **speedup** |
|--:|--:|--:|--:|--:|--:|
| 16 | 512 | 168 | 680 | 446 | **1.52×** |
| 32 | 64 | 168 | 746 | 467 | **1.60×** |
| 64 | 8 | 168 | 819 | 460 | **1.78×** |
| 128 | 1 | 168 | 967 | 498 | **1.94×** |
| 64 | 8 | 48 (nmu=3) | 275 | 179 | **1.54×** |

**Coalescing the gather makes the wavefront sweep kernel 1.5×–1.9× faster** — a large, consistent win
that grows with block size (more fat middle planes). This directly confirms the report's diagnosis that
the sweep is **memory-latency-bound on the footpoint gather** (occ ~0.36, mem-util ~35%): coalescing the
`ir` reads + broadcasting `chi`/`srad` roughly halves the memory-latency stall. (The earlier worry that
angle-in-warp divergence — octant sign, `lmin` axis branch — would cancel the win did not materialize;
the coalescing dominates.)

## The transpose tax (prototype-only) and the whole-run NET

The isolating transpose is expensive — two full-array copies of `ir` per cycle (**373–987 ms/10 cyc**,
i.e. *comparable to or larger than the whole sweep*). So the **prototype's whole-run ZCPS is slower**
than baseline (0.62–0.92× at nmu=6). **This is a prototype artifact, not the native result:** the
prototype exists only to measure the sweep-kernel prize in isolation. A native angle-innermost `ir`
carries no transpose, so it would realize the 1.5–1.9× sweep speedup directly.

## Verdict — prize CONFIRMED (≫ the 25% build threshold); native I2 justified, but costly and wavefront-only

The coalescing prize is real and large (**1.5–1.9× on the wavefront sweep kernel**), so per the plan's
decision rule the **native angle-innermost, wavefront-specialized path is worth designing.** What that
entails (from the exploration, not trivial):
- Store `ir`/`coarse_ir` angle-innermost and give them an **`ir`-specific transposed CC-exchange + AMR
  path** (the generic bvals/inflow/restriction/prolongation/load-balance all hard-assume angle =
  `extent_int(1)`), or transpose only at the (small) ghost-pack boundary each cycle.
- **Reorder `get_moments`/`ComputeJ`** to an angle-warp reduction (they currently coalesce over `i` and
  would otherwise be pessimized).
- It **helps only the wavefront** — the diagonal/diagonal_compact (warp = cells, angle = team) is
  pessimized by angle-innermost and cannot use it (occupancy wall). So native I2 couples to "wavefront is
  the production sweep."

**Recommendation:** the prize warrants pursuing native I2 as a dedicated wavefront-specialized effort
(a larger project than I1). The `wavefront_coalesced` prototype should **not** be used in production (its
transpose makes it a net slowdown) — it stays as the measurement vehicle and the `UpdateCellSC<ANG_INNER>`
seam is already in place for the native version. Lower-effort alternative to bank first: **I3** (hoist
angle-only invariants), which is cheap, portable, and helps both sweeps.
