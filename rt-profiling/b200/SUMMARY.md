# B200 (GB200) — hydro-vs-RT sweep study

*TACC Vista, partition `gb`, 1 GPU of a 4-GPU GB200 node. Run 2026-08-11 (jobs 903411 phase-1;
903471/2/3 sweeps). Branch `rt-sc`, `build_b200` (`Kokkos_ARCH_BLACKWELL100=ON`, Kokkos 4.7.02).
Setup details in `DEVICE_INFO.md`; cross-architecture analysis in `../ARCH_SCALING.md`.*

## Mesh design (phase 1)

**N★ = 256³** (16.8 M zones, 16³ blocks, 168 rays): peak **109.7 GiB ≈ 59 % of the 185 GiB
device**. The next rung on the 16-divisible ladder, 320³, needs 182 GiB and **OOMs**. B200 is
therefore the one device in the study running well below its memory ceiling — a ladder artifact,
not a solver limit. Block ladder: divisors of 256 ≥ 16 → **{16, 32, 64, 128, 256}**
→ nmb {4096, 512, 64, 8, 1}.

## Hydro baseline (radiation off)

| B | 16 | 32 | 64 | 128 | 256 |
|---|--:|--:|--:|--:|--:|
| nmb | 4096 | 512 | 64 | 8 | 1 |
| ZCPS | 2.20e8 | 4.64e8 | 8.51e8 | **1.30e9** | 9.93e8 |

Same shape as A100/GH200 — peaks one rung below the single-block mesh. **3.3× the A100 hydro
peak.**

## Radiation on — ZCPS by sweep (168 rays, and the slowdown)

| B | wavefront | diagonal | **diagonal_compact** | best | slowdown (best) |
|--:|--:|--:|--:|:--|--:|
| 16 | 4.30e7 | 4.38e7 | **4.73e7** | dc | 4.7× |
| 32 | 6.44e7 | 7.12e7 | **8.61e7** | dc | 5.4× |
| 64 | 7.24e7 | 7.11e7 | **7.33e7** | dc (+1 %) | 11.6× |
| 128 | **7.07e7** | 5.15e7 | 5.81e7 | wf | 18.3× |
| 256 | **6.75e7** | 3.83e7 | 4.53e7 | wf | 14.7× |

- **Peak RT-on throughput: 8.61e7 ZCPS** at B=32 with `diagonal_compact` — the best RT
  configuration measured on any device in this study.
- **`diagonal_compact` ≥ `diagonal` at all 30 points** (ratio 1.015–1.51), reproducing the A100
  A/B verdict. Largest gains where the baseline diagonal collapses: **+51 %** at B=128/8 rays,
  **+18–34 %** across the whole B=256 column.
- Crossover: **dc wins nmb ≥ 64, wavefront wins nmb ≤ 8** — the same team-count boundary as the
  other two devices (`league = nmb × nang_tot`).
- The wavefront's biggest margin anywhere in the study is here: **2.80×** over dc at B=256,
  48 rays. Consistent with the diagonal's register jump on Blackwell (below).

## Register pressure — the Blackwell effect

`FormalSolutionDiagonal` compiles to **122 registers on sm_100**, up from **94** on both sm80 and
sm90; `FormalSolutionWavefront` rises more modestly (62 → 72). No kernel spills. This deepens the
diagonal's occupancy handicap exactly where it already hurts (few teams, big blocks), and raises
the expected value of ledger **I3** (hoisting angle-only invariants out of `UpdateCellSC`) on
this architecture.

## Relative cost

Ray cost `k = gap/rays` (see `../ARCH_SCALING.md` §1 for the definition): **2.18 %** at B=16,
rising to **10.3 %** at B=128 — the lowest k of the three devices at every block size. Radiation
scales **2.67×** from A100 at B=16 (hydro: 1.93×) and **3.52×** at the largest block
(hydro: 3.28×), i.e. **radiation scales at least as well as hydro on Blackwell.**

## Data caveat — GPU telemetry columns are node-averaged, ignore them

`sm_util_mean` 20.7 %, `dcgm_sm_occ_mean` 0.094, `sm_clock_mhz` 640, `power_w` 260 are **not
plausible** for a GPU delivering 1.3e9 ZCPS, and each is ≈¼ of a sensible value
(20.7×4 ≈ 83 % SM-util, 0.094×4 ≈ 0.37 occupancy — matching A100/GH200). The node has 4 GPUs and
the job uses one: the sampler is averaging all four. **Fix:** restrict the `nvidia-smi dmon` /
`dcgmi dmon` invocation in `run_sweep.py` to `$CUDA_VISIBLE_DEVICES`. ZCPS and
`kernel_resources.csv` are unaffected.

## Files

`results_phase1.csv`, `results_{wavefront,diagonal,diagonal_compact}.csv`,
`kernel_resources.csv`, `memfill.png`, `zcps_vs_block_*.png`, `sweep_crossover.png`,
`slowdown_vs_rays.png`, `DEVICE_INFO.md`.
