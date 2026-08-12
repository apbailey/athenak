# Tiled diagonal sweep (KBA) — design

*Status: **implemented, bit-identical, and MEASURED — it wins.** `sweep=tiled` +
`<nr_radiation>/tile_size` landed 2026-08-11; performance measured the same day on Vista GH200
(job 904039) and B200 (904040). Ledger row I7. Companion to `OPTIMIZATION_LEDGER.md` (which this
amends — see §8) and `ARCH_SCALING.md`. Analysis: `tile_study.py`, data `{gh200,b200}/tile/`.*

## Result (2026-08-11)

**Best sweep at every large-block point measured on both devices**, at 168 rays:

| device | B | nmb | wavefront | best tiled | **vs wavefront** |
|---|--:|--:|--:|--:|--:|
| GH200 | 56 | 64 | 4.33e7 | 5.46e7 (t=28) | **1.26×** |
| GH200 | 112 | 8 | 4.30e7 | 6.18e7 (t=16) | **1.44×** |
| GH200 | 224 | 1 | 3.74e7 | 6.08e7 (t=16) | **1.63×** |
| B200 | 64 | 64 | 7.24e7 | 1.03e8 (t=32) | **1.43×** |
| B200 | 128 | 8 | 7.07e7 | 1.10e8 (t=32) | **1.56×** |
| B200 | 256 | 1 | 6.76e7 | 1.06e8 (t=32) | **1.57×** |

**Against the best configuration previously known anywhere on the ladder** — the number that
matters for production:

| device | rays | previous best | new best | gain |
|---|--:|---|---|--:|
| GH200 | 48 | 1.22e8 (wavefront, B=112) | **1.70e8** (tiled t=16, B=112) | **1.40×** |
| GH200 | 168 | 4.86e7 (diag_compact, B=28) | **6.18e7** (tiled t=16, B=112) | **1.27×** |
| B200 | 48 | 2.04e8 (diag_compact, B=32) | **2.85e8** (tiled t=32, B=128) | **1.40×** |
| B200 | 168 | 8.61e7 (diag_compact, B=32) | **1.10e8** (tiled t=32, B=128) | **1.28×** |

Four things the data says:

1. **The gain grows with block size** (1.26→1.44→1.63 on GH200) — the predicted mechanism, since
   large blocks are where the diagonal starved and where there was most to recover.
2. **The optimal tile is ~16³ (GH200) / ~32³ (B200)** — i.e. roughly the block size at which the
   diagonal already wins on that device, exactly as §5 predicted, with no tuning needed. B200's
   larger optimum is consistent with its needing more work per team to fill.
3. **8³ tiles are worse than the wavefront everywhere** (0.63–0.85×) — the tile-face re-read and
   tile-plane count predicted in §7 as the lower bound is real, and it sits between 8 and 14.
4. **The new optimum moves to a LARGE block** (B=112 / B=128 instead of B=28 / B=32). That
   compounds with everything else: hydro is ~3× faster there, and per `LAG_STUDY.md` a thin
   problem at nmb=8 needs ~5 iterations instead of ~21 at nmb=512. Tiling is what makes the
   large-block regime affordable for radiation — the regime hydro and the iteration count
   already wanted.

**GPU determinism gate PASSED on both devices**: wavefront, diagonal_compact and tiled all hash
`0x9a543797d2295dea` under real GPU parallelism, not just the serial CPU check.

### Open anomaly

At `tile_size=0` (one tile per block, which *should* be `diagonal_compact` via a different code
path) the tiled kernel is **32 % slower** than `diagonal_compact` at GH200 B=224 and **26 %
slower** at B200 B=256 — while being *faster* at GH200 B=112. Run-to-run spread is ≤0.9 %, so it
is real, and it is not register pressure (tiled 88 regs vs diagonal 94). Unexplained. No
practical consequence — `tile_size=0` is never the setting to use — but `diagonal_compact`
should stay the recommendation when not tiling, and this is worth understanding before the
tiled sweep becomes a default.

## 0. What has landed

| piece | where |
|---|---|
| `BuildPlaneIndex()` — the hyperplane rule, extracted and parameterized on dims | `sc/formal_solution.cpp` (file-local) |
| `SC::BuildWavefrontIndex()` — now a thin caller of it (behaviour unchanged) | same |
| `SC::BuildTileIndex()` — the two nested maps (cells-in-tile, tiles) | same |
| `SC::FormalSolutionTiled()` — one launch per tile-plane | same |
| `sweep=tiled`, `tile_size` (0 = one tile per block), divisibility guard | `nr_radiation.cpp`, `nr_radiation.hpp` |
| `sc_determinism_tiled.athinput` at `tile_size=8` (**real** tiling), wired into the CPU+GPU suites | `tst/inputs/`, `tst/test_suite/sc/` |

**Correctness — all PASS, all bit-identical.** The `sc_sweep_determinism` hash is
`0x9a543797d2295dea` for wavefront, diagonal, diagonal_compact **and tiled**, and for tiled at
`tile_size` = 0 (degenerate), 16, 8 and 4 on a 32³ block — i.e. up to 8 tiles per side, 22
tile-planes. So the upwind-corner argument of §3 holds in practice, not just on paper. The
1D + Jacobi-ALI path is covered too: `sc_atmosphere` (Davis Eq. 30) PASSES with `sweep=tiled` at
`tile_size` 0 and 32. The divisibility guard fires with the list of valid divisors.

**Not yet done:** step 3 — the performance sweep. No throughput claim is made here yet.

## 1. The problem this solves

`sweep=diagonal` issues **one** kernel launch with `league_size = nmb × nang_tot` teams; each
team owns one (meshblock, angle) pair and marches the cell-hyperplanes `h = 0…hmax` with a
`team_barrier()` between them. That is optimal when there are many meshblocks and fatal when
there are few:

| B (at N★=176³) | nmb | teams | outcome |
|--:|--:|--:|:--|
| 16 | 1331 | 223,608 | best sweep |
| 44 | 64 | 10,752 | best sweep |
| 88 | 8 | 1,344 | loses to wavefront 1.6× |
| 176 | 1 | **168** | loses to wavefront, was 58 % of it before compaction |

168 teams cannot fill any of A100/GH200/B200, and each team must chew a ≤15,000-cell hyperplane
with ~128 threads. Meanwhile the wavefront has ample parallelism at large blocks but pays
`3B−2 = 526` launches per sweep and streams a whole 176² plane's working set.

**The idea:** partition the meshblock into tiles and run a wavefront *over tiles*, one kernel
launch per tile-hyperplane, with each team sweeping one tile internally exactly the way today's
diagonal sweeps a small meshblock. A big meshblock is made to behave like a bag of small ones.

This is the Koch–Baker–Alcouffe (KBA) sweep, standard in parallel discrete-ordinates transport.

## 2. Why this is *not* the I1 data race

The ledger's I1 row closes "tile a plane across the league" as a data race. That verdict is
correct **for what it describes** — splitting a single *cell*-plane across teams inside one
launch, where the footpoints on planes `h−1…h−3` are only `team_barrier`-synchronized within a
team, so there is no ordering between teams.

This design is a different algorithm that happens to share the word "tile":

| | I1's tiling (racy) | this (safe) |
|---|---|---|
| what is partitioned | one cell-hyperplane | the meshblock volume |
| cross-partition sync | none available (needs cross-team barrier) | **the kernel boundary** |
| launches | 1 | one per tile-plane |

Because each launch is a global barrier, every tile's upwind neighbours are provably complete
before it starts. The ledger row should be narrowed accordingly (§8).

## 3. Correctness argument (bit-identical)

The footpoint stencil of `UpdateCellSC` reaches at most one cell in each direction, so a cell in
tile `(ta,tb,tc)` reads only cells in that tile or in its **upwind corner set**:

```
(ta−1,tb,tc) (ta,tb−1,tc) (ta,tb,tc−1) (ta−1,tb−1,tc)
(ta−1,tb,tc−1) (ta,tb−1,tc−1) (ta−1,tb−1,tc−1)
```

Every member has at least one index decremented, so all of them lie on tile-planes with
`ta+tb+tc` strictly smaller than the current one — i.e. in earlier launches, hence complete.
Within a tile the existing `h`-ordering is unchanged.

Therefore each cell reads exactly the same fully-updated upwind values as today, in the same
arithmetic order, so the result is **bit-identical** to `wavefront` / `diagonal` /
`diagonal_compact`. Gate on the determinism hash (`0x9a543797d2295dea` on A100).

*Caveat:* with `use_ali=true` the `lamstr` `atomic_add` accumulation order changes with the
thread mapping, so ALI runs will differ in the last bits (benign, and true of any remapping).
The determinism test runs `ops=0, eps=1 ⇒ use_ali=false`, so the gate itself is unaffected.

## 4. Index algebra

### 4.1 Tiles

Tile edge `(tx1,tx2,tx3)`, from `<nr_radiation>/tile_size` (scalar or per-axis).
**Require exact division** — `nx1 % tx1 == 0`, etc. — validated at construction with a fatal
error listing the valid divisors. Partial tiles would need up to 8 distinct local plane maps for
no real gain; meshblock dims are chosen from divisor ladders anyway.

```
T1 = nx1/tx1,  T2 = nx2/tx2,  T3 = nx3/tx3
HmaxT = T1 + T2 + T3 − 3                     // number of tile-planes − 1
```

Tile coordinates `(ta,tb,tc)` are measured **from the octant's upwind corner**, exactly as `li`
is for cells, so the same tile-plane index `H = ta+tb+tc` works for all 8 octants and the octant
sign only flips the mapping to absolute cell indices:

```
sx > 0:  i_lo = is + ta*tx1;        i_hi = i_lo + tx1 − 1
sx < 0:  i_hi = ie − ta*tx1;        i_lo = i_hi − tx1 + 1
```

and likewise for `(tb,sy,j)`, `(tc,sz,k)`.

### 4.2 Two precomputed maps, both built once

Both are pure functions of the meshblock interior dims — identical for every meshblock and every
octant — so they are built lazily once, exactly like the existing `wf_cell_`:

1. **Tile-plane map** `tp_cell_` / `tp_start_` — for each tile-plane `H`, the packed tile indices
   `(tc*T2 + tb)*T1 + ta` on it. Built by the same triple loop as `BuildWavefrontIndex`, with
   `(T1,T2,T3)` in place of `(nx1,nx2,nx3)`. `tp_start_` is needed on the **host** (it drives the
   launch loop).
2. **Local cell-plane map** `tile_cell_` / `tile_start_dev_` — the compact per-plane cell list for
   a *tile* of dims `(tx1,tx2,tx3)`. This is literally the existing `BuildWavefrontIndex` called
   with the tile dims; refactor it to take dims as arguments and fill caller-supplied arrays.
   `tile_start_dev_` must be on the **device** (the `h`-loop runs inside the kernel).

Tiles per plane peaks at `H = 3(T−1)/2`; for a cubic tiling
`N(H) = C(H+2,2) − 3·C(H−T+2,2) + 3·C(H−2T+2,2)` (terms with `n<2` vanish).
For `T=11`: `N(15) = 136 − 45 = 91`.

### 4.3 Launch structure

```cpp
for (int H = 0; H <= HmaxT; ++H) {                    // host loop: kernel boundary = tile barrier
  int tlo = tp_start_[H], ntile = tp_start_[H+1] - tlo;
  if (ntile <= 0) continue;
  int nang_teams = nang_tot / Na;                     // Na = angles per team (§5)
  Kokkos::TeamPolicy<> policy(DevExeSpace(), ntile * nmb * nang_teams, Kokkos::AUTO);
  Kokkos::parallel_for("sc_sweep_tiled", policy, KOKKOS_LAMBDA(const TeamMember_t &tmember) {
    // decode: tile OUTERMOST so same-tile teams are co-resident and share chi/srad in L2
    int lid = tmember.league_rank();
    int t    = lid / (nmb * nang_teams);
    int rem  = lid - t*(nmb * nang_teams);
    int m    = rem / nang_teams;
    int ablk = rem - m*nang_teams;                    // which block of Na angles

    int lin  = tp_cell_(tlo + t);                     // packed tile index
    int tc   = lin / (T1*T2);
    int r    = lin - tc*(T1*T2);
    int tb   = r / T1;
    int ta   = r - tb*T1;

    // ... per-angle setup (mux/muy/muz, sx/sy/sz) as today ...
    // absolute tile origin from (ta,tb,tc) and the octant signs, per §4.1

    for (int h = 0; h <= (tx1+tx2+tx3-3); ++h) {      // inner wavefront, IDENTICAL to today
      int lo = tile_start_dev_(h), cnt = tile_start_dev_(h+1) - lo;
      Kokkos::parallel_for(Kokkos::TeamThreadRange(tmember, cnt*Na), [&](const int idx) {
        int c = idx / Na, ai = idx - c*Na;            // cell on plane, angle within the block
        int angg = ablk*Na + ai;
        // decode li1,li2,li3 from tile_cell_(lo+c); add the tile origin; call UpdateCellSC
      });
      tmember.team_barrier();
    }
  });
}
```

**`tile = block, Na = 1` reduces exactly to today's `diagonal_compact`** — one tile-plane, one
launch, `league = nmb·nang_tot`, the same inner loop. So the new kernel *subsumes* the old one
and the existing determinism test validates the whole family at the degenerate setting before
any tiling is switched on. That is the cheapest possible correctness scaffold, and it is the
reason to implement it as a generalisation rather than a new sweep.

## 5. The two knobs

| knob | meaning | trades |
|---|---|---|
| `tile_size` | tile edge in cells | parallelism & locality ↑ vs launches & face re-reads ↑ as it shrinks |
| `Na` | angles per team | `chi`/`srad` reuse ↑ vs parallelism ↓ as it grows |

`Na` is the same quantity as in the array-layout experiment: with `Na>1` a thread handles several
angles of one cell, so the 9+9 `chi`/`srad` stencil is loaded once and reused `Na` times from
L1/registers. Under the **current** `ir` layout the `ir` reads for those `Na` angles are strided
by `n1·n2·n3`, so `Na>1` is only clearly worthwhile once `ir` is angle-major — which is why the
layout experiment should report first. Ship `Na=1` and leave the knob wired but pinned.

**Tile-size guidance:** pick the largest divisor of `nx` that is ≤ ~32 — the block size at which
the diagonal already wins. That is the whole thesis of the design, stated as a rule.

## 6. Expected effect (hypothesis, not measurement)

At N★=176³, nmu=6 (168 rays), 16³ tiles where divisibility allows:

| B | nmb | today: diagonal teams | tiled: teams at widest plane | launches (tiled) | launches (wavefront) |
|--:|--:|--:|--:|--:|--:|
| 16 | 1331 | 223,608 | 223,608 *(1 tile — no-op)* | 1 | 46 |
| 44 | 64 | 10,752 | 129,024 (12 tiles, 11³) | 10 | 130 |
| 88 | 8 | 1,344 | 122,304 (91 tiles, 8³) | 31 | 262 |
| 176 | 1 | **168** | **15,288** (91 tiles) | **31** | 526 |

Three mechanisms move together at large blocks: **91× the teams**, **17× fewer launches than the
wavefront**, and a tile-sized working set that is far more cache-resident than a 176² plane —
the last mattering most, since the kernel is latency-bound with the memory interface only ~35 %
busy.

Note the top row: at B=16 the tiling is a **no-op by construction**, which is the correct
behaviour — that is where the diagonal already wins.

## 7. Risks and limits

- **Thin end-planes.** `H=0` and `H=HmaxT` hold one tile each, so those launches have
  `nmb·nang_teams` teams — as starved as today's diagonal, for 2 of 31 launches. Bounded, but it
  caps the achievable gain; worth measuring the per-plane time distribution with nsys.
- **Face re-reads.** A tile re-reads its neighbours' faces from L2 rather than L1. Surface/volume
  is `6/tx` (0.375 at 16³). Tiles below ~8³ will lose more to this than they gain in parallelism.
- **Exact division required** (§4.1) — constrains `tile_size` to divisors of the block dims.
- **Not applicable to `ali_mode=gauss_seidel`.** `SweepUpdateGS` needs the *global* center-out
  completion-shell ordering to decide when a cell has received its last octant; tiles break that
  ordering. The tiled sweep covers the formal solution used by LTE and Jacobi-ALI only. GS keeps
  the host-plane wavefront.
- **More index math per thread** (tile decode + local plane decode). Registers are the currency
  here — the diagonal already sits at 94 regs (122 on sm_100), so watch
  `kernel_resources.py` after implementing; if it pushes occupancy down, ledger I3 (hoisting the
  angle-only invariants) becomes a prerequisite rather than a nice-to-have.

## 8. Ledger amendment

I1's parenthetical currently reads as closing all tiling. It should be narrowed to intra-plane
tiling, and this design added as **I7** (see the updated `OPTIMIZATION_LEDGER.md`).

## 9. Implementation order

1. ~~Refactor `BuildWavefrontIndex` to take `(nx1,nx2,nx3)` and fill caller-supplied arrays.~~ ✅
2. ~~Add the tiled kernel with `tile_size = block`, `Na = 1`; confirm it is **bit-identical**.~~ ✅
   (and verified bit-identical with real tiling at 16/8/4 as well — see §0)
3. ~~Turn on tiling; sweep `tile_size` × `B` at nmu=3 and 6 on clean whole-run ZCPS.~~ ✅ — see
   the Result section above. Won on both GH200 and B200.
4. **← NEXT.** Revisit `Na` (angle blocking) **after** the array-layout experiment reports; and
   resolve the `tile_size=0` anomaly. Consider promoting `tiled` to the default for nmb ≤ 64.
