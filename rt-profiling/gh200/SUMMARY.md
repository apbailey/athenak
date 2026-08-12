# GH200 — hydro-vs-RT sweep study

*TACC Vista, partition `gh`, GH200 120GB (~95.6 GiB usable). Run 2026-08-11 (jobs 903230
phase-1; 903247/8/9 sweeps). Branch `rt-sc`, `build_gh200` (`Kokkos_ARCH_HOPPER90=ON`,
Kokkos 4.7.02). Setup in `DEVICE_INFO.md`; cross-architecture analysis in `../ARCH_SCALING.md`.*

## Mesh design (phase 1)

**N★ = 224³** (11.2 M zones, 16³ blocks, 168 rays): peak **73.6 GiB ≈ 77 % fill**; 256³ needs
93.5 GiB and **OOMs**. Block ladder: divisors of 224 ≥ 16 → **{16, 28, 32, 56, 112, 224}**
→ nmb {2744, 512, 343, 64, 8, 1}. This is the finest ladder of the three devices — six rungs,
including two adjacent points (28/32) that bracket the sweep crossover.

## Hydro baseline (radiation off)

| B | 16 | 28 | 32 | 56 | 112 | 224 |
|---|--:|--:|--:|--:|--:|--:|
| nmb | 2744 | 512 | 343 | 64 | 8 | 1 |
| ZCPS | 1.91e8 | 3.32e8 | 3.84e8 | 6.12e8 | **9.80e8** | 7.90e8 |

**2.5× the A100 hydro peak**, peaking one rung below the single-block mesh — the same shape as
A100 and B200.

## Radiation on — ZCPS by sweep (168 rays)

| B | wavefront | diagonal | **diagonal_compact** | best | slowdown (best) |
|--:|--:|--:|--:|:--|--:|
| 16 | 3.15e7 | 3.46e7 | **3.71e7** | dc | 5.2× |
| 28 | 3.96e7 | 4.02e7 | **4.86e7** | dc | 6.8× |
| 32 | 4.07e7 | 4.35e7 | **4.43e7** | dc | 8.6× |
| 56 | **4.32e7** | 4.11e7 | 4.16e7 | wf (+4 %) | 14.2× |
| 112 | **4.27e7** | 2.08e7 | 2.52e7 | wf | 22.9× |
| 224 | **3.71e7** | 3.17e7 | 3.39e7 | wf | 21.3× |

- Peak RT-on throughput **4.86e7 ZCPS** at B=28 with `diagonal_compact`.
- **`diagonal_compact` ≥ `diagonal` at 34 of 36 points.** The two exceptions are
  (B=32, 8 rays) at 0.994× and (B=112, 24 rays) at 0.951× — the first is inside run-to-run
  noise, the second is not obviously so and is the only measured point on any device where
  compaction appears to cost something. Everywhere else the gain is +1 % to +23 %.
- Crossover: **dc wins nmb ≥ 343, wavefront wins nmb ≤ 64** — but the B=56 (nmb=64) margin is
  only 4 %, so on the nmb axis GH200 agrees with the other two devices to within the ladder's
  resolution.
- GH200 posts the **highest gap in the entire study** — 21.9 at B=112, 168 rays — and is the one
  device where radiation scales slightly *worse* than hydro at large blocks (2.11× vs 2.48× from
  A100). At B=16 it scales better (2.04× vs 1.67×), like the others.

## Register pressure

`FormalSolutionDiagonal` 94 regs on sm90 — identical to sm80, no Hopper penalty (the jump to 122
appears only on sm_100). Wavefront 54–60, Jacobi 62, `SweepUpdateGS` 47–66. No spills.

## Relative cost

Ray cost `k = gap/rays`: **2.47 %** at B=16 → **13.1 %** at B=112, the widest span of the three
devices. Sits between A100 (2.99 %) and B200 (2.25 %) at B=16.

## Thermal caveat on the dc-vs-wf comparison

The three sweep jobs ran back-to-back and the GPU warmed through the batch: at (B=16, 168 rays)
the wavefront job logged **39 °C / 450 W** and the diagonal_compact job **61 °C / 423 W**, at the
same 1980 MHz. Clocks were not pinned (admin), so a few percent of any dc-vs-wf difference on
GH200 may be thermal rather than algorithmic. This is the most likely explanation for the two
sub-1.0 points above. Hydro-baseline reproducibility across the three jobs was nonetheless tight
(median 0.3 %, max 0.4 %).

## Files

`results_phase1.csv`, `results_{wavefront,diagonal,diagonal_compact}.csv`,
`kernel_resources.csv`, `memfill.png`, `zcps_vs_block_*.png`, `sweep_crossover.png`,
`slowdown_vs_rays.png`, `DEVICE_INFO.md`.
