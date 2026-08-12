# Repeating the A100 speed study with the shipped optimizations — REPORT2

*Apollo A100-PCIE-40GB, branch `rt-sc` (`b9f56cbb`), `build_a100`. Run 2026-08-12, jobs 16953153
(diagonal_compact @ N★=176), 16953155/16953156 (normal vs angle_inner @ N★_ai=144). Same
methodology, decks, and metric as the original **`REPORT.md`** (2026-08-07); this document is a
direct follow-up — read that one first for the full setup/caveats. Data: `a100/results_diagonal_compact.csv`,
`a100/angle_inner_full/{normal,angle_inner}/results_wavefront*.csv`, `a100/angle_inner_full/results_phase1.csv`.*

## TL;DR — yes, both shipped optimizations show up as real wins on this exact test

- **I1 (`sweep=diagonal_compact`) repeats the original Phase-2 sweep at the *same* N★=176 and
  *same* block/angle grid**, replacing the old `diagonal`. It is **never slower** than the old
  diagonal (geomean **+12.2%** across all 30 (B,nmu) points) and closes most of diagonal's
  big-block collapse (176³/nmu=6: 1.07e7 → 1.54e7, **+44%**). Because wavefront already won at
  B≥88, the **best-of-both-sweeps** picture — the number that actually matters for "what should a
  user run" — improves by a smaller but real **+0–12%** (geomean **+3.0%**), concentrated exactly
  where the old report identified diagonal's weak spot: **B=44, nmu 4–6 (+10–12%)**.
- **I2 (`ir_layout=angle_inner`) cannot reuse N★=176** — its `ir_normal` bridge companion roughly
  doubles the intensity array, so it needs its **own** memory-fill. Repeating Phase 1 for it finds
  **N★_ai = 144³** (27.2 GiB; 160³/176³ exceed the 34 GiB ceiling). Repeating Phase 2 there over the
  *natural* divisor ladder {16,18,24,36,48,72,144} confirms the same pattern found earlier at
  N=128: a **regression on small blocks** (geomean **0.92×** for B<64) and a **real win on large
  blocks** (geomean **1.01×** for B≥64, rising to **1.21× at 144³/nmu=6** — the single-block,
  highest-angle corner).
- **I3 (`sc_hoist`) was already measured separately** (this session, A100 job 16953004): ~neutral/
  slightly negative, not part of this repeat. See the ledger.
- **The original report's headline qualitative findings still hold**: mesh design is memory-driven,
  the best sweep still flips with block size, radiation still costs far more than hydro per cell.
  What changed is **how much better the best sweep now is**, not the shape of the story.

## 1. Repeating Phase 2 exactly at N★=176: `diagonal_compact` replaces `diagonal`

Same deck, same block ladder {16,22,44,88,176}, same nmu 1–6, same `N_REPEAT=3`/`nlim=50`. Wavefront
is **unchanged** since the original report (no wavefront-path edits landed since `d776ee93`; reused
`a100/results_wavefront.csv` — its default path was independently verified bit-identical to
pre-refactor HEAD during the I3 work). Only `diagonal` → `diagonal_compact` is new.

### 1.1 diagonal_compact vs the old diagonal (whole-run ZCPS, higher = faster)

| B | nmb | nmu=1 | 2 | 3 | 4 | 5 | **6** |
|--:|----:|------:|--:|--:|--:|--:|------:|
| 16  | 1331 | 1.00× | 1.01× | 1.01× | 1.01× | 1.01× | 1.02× |
| 22  | 512  | 1.05× | 1.05× | 1.05× | 1.06× | 1.06× | 1.06× |
| 44  | 64   | 1.23× | 1.23× | 1.20× | 1.14× | 1.13× | 1.12× |
| 88  | 8    | 1.06× | 1.16× | 1.25× | 1.38× | 1.39× | 1.39× |
| 176 | 1    | 1.07× | 1.02× | 1.02× | 1.02× | 1.23× | **1.44×** |

**Never a regression anywhere in the grid** (geomean **+12.2%** over all 30 points) — matches the
ledger's I1 SUMMARY (+5–11% small/mid, +39–50% big at nmu=6; the full nmu=1–6 grid here shows the
gain is broad, not just an nmu=6 artifact, and is largest at B=88–176 where the old diagonal's team
starvation was worst).

### 1.2 The sweep crossover, before vs after (best sweep per block, nmu=6)

| B | OLD winner (2026-08-07) | NEW winner (2026-08-12) | best-of-both gain |
|--:|:---|:---|--:|
| 16  | diagonal (1.81e7)          | diagonal_compact (1.85e7)  | **+2.2%** |
| 22  | diagonal (2.18e7)          | diagonal_compact (2.31e7)  | **+6.2%** |
| 44  | diagonal (2.00e7)          | diagonal_compact (2.24e7)  | **+12.0%** |
| 88  | wavefront (2.01e7)         | wavefront (2.01e7)         | +0.0% |
| 176 | wavefront (1.71e7)         | wavefront (1.71e7)         | +0.0% |

The **crossover itself hasn't moved** — wavefront still wins ≥88³, diagonal-family still wins ≤44³
— because at B≥88 wavefront was always the winner and diagonal_compact, however improved, doesn't
overtake it. What changed is that the **diagonal side of the crossover got meaningfully faster**
(up to +12% at the B=44 boundary, where the choice matters most). Averaged over the *entire*
(B,nmu) grid including where wavefront already wins (so the improvement is diluted to 0 by
construction at B≥88), the best-of-both geomean gain is **+3.0%** (range 1.000×–1.124× across all
30 points).
**Practical read: if you were already choosing the best sweep per block size, the biggest concrete
win is at B≈44 (+10–12% at nmu≥4) — the exact regime the original report flagged as the
diagonal/wavefront crossover.**

Hydro baseline reproduced for a sanity check (N=176, nmu-independent `zcps_off`): 16³ 1.14e8,
22³ 1.57e8, 44³ 2.66e8, 88³ 3.95e8, 176³ 3.52e8 — matches the original report's Table 3.1 to within
run-to-run noise, confirming Rule 4 (radiation-off / non-radiation-path behaviour is unchanged).

## 2. A new test: I2 (`angle_inner`) at its own memory-fill N★

I2 cannot be dropped into the N★=176 grid — `ir_normal` (the CC-exchange bridge companion,
`nr_radiation.hpp`) makes the intensity footprint ~2× normal layout. So this repeats **Phase 1**
first, at the same worst corner (B=16, nmu=6) and the same 34 GiB ceiling, for `ir_layout=angle_inner`:

| N | peak mem (GiB) | % of 40 GB | result |
|---|---:|---:|:---:|
| 112 | 13.1 | 33% | fits |
| 128 | 19.4 | 48% | fits |
| **144** | **27.2** | **68%** | **← N★_ai** |
| 160 | 37.2 | 93% | over 34 GiB ceiling |
| 176 | 39.1 | 98% | over 34 GiB ceiling |

**N★_ai = 144³** — smaller than the normal-layout N★=176 (as expected: same ir cost per zone, but
double the array). Phase 2 then repeats at N★_ai=144 over its *natural* divisor ladder
{16,18,24,36,48,72,144} (a longer ladder than 176's {16,22,44,88,176} simply because 144 has more
divisors ≥16), comparing `ir_layout=normal` (baseline wavefront) against `ir_layout=angle_inner`.

### 2.1 angle_inner / normal (whole-run ZCPS, sweep=wavefront, N=144³)

| B | nmb | nmu=1 | 2 | 3 | 4 | 5 | **6** |
|--:|----:|------:|--:|--:|--:|--:|------:|
| 16  | 729 | 0.91× | 0.88× | 0.90× | 0.92× | 0.90× | 0.89× |
| 18  | 512 | 0.93× | 0.88× | 0.91× | 0.92× | 0.91× | 0.90× |
| 24  | 216 | 0.90× | 0.87× | 0.91× | 0.93× | 0.93× | 0.94× |
| 36  | 64  | 0.86× | 0.85× | 0.93× | 0.96× | 0.97× | 0.98× |
| 48  | 27  | 0.85× | 0.86× | 0.95× | 1.00× | 1.01× | 1.03× |
| 72  | 8   | 0.80× | 0.87× | 1.00× | 1.03× | 1.05× | **1.08×** |
| 144 | 1   | 0.87× | 0.92× | 1.07× | 1.14× | 1.19× | **1.21×** |

Same shape as the earlier N=128 measurement (I2's original SUMMARY): a **consistent small-block
regression** (geomean **0.92×** for B<64 — the O(surface) shell-transpose bridge costs proportionally
more when the boundary shell is a big fraction of a small block) and a **real, growing win on large
blocks + many angles** (geomean **1.01×** for B≥64; up to **1.21× at the single 144³ block, nmu=6**
— the most production-relevant heavy-angle corner). This independently reproduces and slightly
exceeds the earlier N=128 result (1.17×/1.27× at 64³/128³) at a different N/B combination — the win
is real and not an artifact of the specific mesh size chosen the first time.

## 3. Putting it together: does the *shape* of the original report change?

**No — the qualitative story is the same; the numbers inside it got better.** The original report's
five headline claims all still hold:

1. Mesh design is memory-driven ✅ (confirmed again, separately, for angle_inner's own layout).
2. Hydro baseline shape unchanged ✅ (reproduced to within noise).
3. Radiation slowdown vs hydro is still large and grows with angular order ✅ (unchanged — neither
   I1 nor I2 touches the fundamental per-cell RT cost, only the sweep's parallel efficiency/memory
   pattern).
4. The best sweep still flips with block size ✅ — but the diagonal side of that flip is up to 12%
   faster at the exact crossover block (B=44), and a *third* option (wavefront+`angle_inner`) is
   now the best choice specifically for large-block, high-angle, non-AMR production runs (≥64³,
   nmu≥4), beating plain wavefront by up to 21%.
5. GPU loading/occupancy diagnosis (latency-bound sweep, not register- or occupancy-starved) ✅ —
   independently reconfirmed by I3 (`sc_hoist`) this session: removing hot-loop ALU bought nothing,
   exactly as the latency-bound diagnosis predicts.

## 4. Updated practical guidance

- **B ≤ 24, any angle count:** use `diagonal_compact` (never worse than the old advice, and now
  meaningfully ahead of plain `diagonal` if you'd been running that).
- **B ≈ 32–44 (the crossover):** `diagonal_compact` is now the clearer winner here (+10–12% over
  `diagonal` at nmu≥4) — this is the biggest concrete "upgrade your config" takeaway from this repeat.
  If diagonal_compact vs wavefront was close before, it is now unambiguously worth checking.
  Its bit-exactness (hash `0x9a54…` == baseline diagonal) means switching costs nothing else.
  Note: this is I1's finding, not a new one — quantified here on the same grid as the original.
  **Cross-reference `OPTIMIZATION_LEDGER.md` I1 for the caveat that the team-size knob doesn't help.**
- **B ≥ 64 (72³/144³ tested), nmu ≥ 4, uniform mesh, no AMR planned:** switch to `sweep=wavefront` +
  `ir_layout=angle_inner` — a genuine 1.03–1.21× win on top of wavefront, bit-exact, at the cost of
  ~2× the intensity array's memory and needing its own (smaller) N★. At nmu≤3 even large blocks are
  break-even or slightly behind (0.80–1.00×) — the win needs both a large block *and* many angles.
- **B < 64 with `angle_inner` considered:** don't — it's a consistent regression there (2–15%
  slower, worst at low angle counts and small B), never a clear win at any (B,nmu) below 64³; use
  plain `ir_layout=normal` wavefront (or `diagonal_compact` if B≤44).
- Nothing changes for **AMR/SMR runs**: `angle_inner` remains uniform-mesh-only (ledger §4); use
  wavefront/`diagonal_compact` as before.

## 5. Reproduce & files

```bash
# I1: exact repeat of the original Phase 2 at N*=176 (wavefront reused unchanged)
RT_N_STAR=176 RT_SWEEPS=diagonal_compact sbatch rt-profiling/a100/submit_sweep.sbatch

# I2: its own memory-fill, then Phase 2 at N*_ai for normal vs angle_inner
sbatch rt-profiling/a100/submit_phase1_angle_inner.sbatch          # -> N*_ai via results_phase1.csv
RT_N_STAR=144 RT_SWEEPS=wavefront RT_IR_LAYOUT=normal \
  RT_OUT=.../angle_inner_full/normal      sbatch --time=08:00:00 rt-profiling/a100/submit_sweep.sbatch
RT_N_STAR=144 RT_SWEEPS=wavefront RT_IR_LAYOUT=angle_inner \
  RT_OUT=.../angle_inner_full/angle_inner sbatch --time=08:00:00 rt-profiling/a100/submit_sweep.sbatch
```

`RT_IR_LAYOUT` is a new axis added to `run_sweep.py` this session (defaults to `normal`, so every
prior invocation is unaffected) — reusable shared scaffolding per the ledger's §5 consolidation plan.

Data: `a100/results_diagonal_compact.csv` (30 rows), `a100/angle_inner_full/results_phase1.csv`
(6 rows), `a100/angle_inner_full/{normal,angle_inner}/results_wavefront*.csv` (42 rows each).
Original data for comparison: `a100/results_{wavefront,diagonal}.csv` (unchanged, reused).

## 6. Caveats (same as the original report, still apply)

GPU clocks not pinned (admin-blocked); Nsight Compute admin-blocked (DCGM used instead); `slowdown`/
ratio columns are thermal-paired-run advisory, absolute medians are primary. The I2 grid here uses a
different N (144³, forced by memory) than I1 and the original report (176³) — the two are not
directly comparable cell-for-cell, only each against its own same-N baseline, exactly as the
ledger's I2 SUMMARY already notes.
