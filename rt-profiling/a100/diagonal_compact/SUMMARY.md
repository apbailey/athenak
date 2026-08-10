# `sweep=diagonal_compact` A/B — result (A100)

*Apollo A100-PCIE-40GB, branch `rt-sc` (`35c4e65e`), `build_a100` rebuilt with the diagonal_compact
changes. Run 2026-08-10 via `a100/submit_diagonal_compact_ab.sbatch`. Data: `results_both.csv`,
`launch_overhead/results_nsys_{diagonal,diagonal_compact}.csv`, `team_size_sweep.csv`.*

## What was tested (ledger I1, re-scoped)

The literal I1 (tile a plane across teams to add league breadth) is a **data race** — the 1-cell
upwind footpoint reach means a plane's footpoints on planes h−1..h−3 are only `team_barrier`-synchronized;
splitting a plane across teams needs a cross-team barrier (= the retired Lever 1). So the valid,
bit-identical improvements keep each whole plane in one team:
- **(A) compact-plane iteration** — reuse the wavefront's `wf_cell_` list so the 3D diagonal's
  `TeamThreadRange` iterates only real cells per plane, instead of `nx1·nx2` candidate slots with
  off-plane early-returns (~⅔ waste at 176³).
- **(B) `diag_team_size` knob** — explicit `TeamPolicy` team size vs `Kokkos::AUTO`.

## Correctness (GPU, all PASS)

`sc_sweep_determinism` (nrepeat=64): wavefront / diagonal / **diagonal_compact** all hash
`0x9a543797d2295dea` — **diagonal_compact is bit-for-bit identical to baseline diagonal** (and
wavefront). `sc_atmosphere` (Davis Eq. 30) PASSED with `sweep=diagonal_compact`.

## Throughput — clean whole-run ZCPS (the golden rule), nmu=6 (168 rays), N★=176³

| block B | nmb | diagonal | **diagonal_compact** | wavefront | compact/diagonal | best sweep |
|--:|--:|--:|--:|--:|--:|:--|
| 16 | 1331 | 1.80e7 | **1.89e7** | 1.58e7 | **+4.7 %** | diagonal_compact |
| 22 | 512 | 2.16e7 | **2.31e7** | 1.80e7 | **+6.8 %** | diagonal_compact |
| 44 | 64 | 1.98e7 | **2.19e7** | 2.01e7 | **+10.9 %** | diagonal_compact |
| 88 | 8 | 9.06e6 | **1.26e7** | 2.04e7 | **+39.2 %** | wavefront |
| 176 | 1 | 1.00e7 | **1.50e7** | 1.73e7 | **+49.8 %** | wavefront |

- **diagonal_compact ≥ baseline diagonal at every point measured** (all blocks, all nmu 1–6) — a strict
  Pareto improvement — with the gain growing sharply on big blocks (where the ~⅔ wasted iterations were
  *not* hidden by other teams, since there are only 8 or 1 × nang teams).
- It becomes the **best sweep for B ≤ 44** (beats both diagonal and wavefront), moving the wavefront
  crossover from ≈32³ (baseline diagonal) to ≈64³.
- On the single **176³** block it **closes the gap to the wavefront from 58 % → 87 %** of wavefront ZCPS
  (baseline diagonal 1.00e7 → compact 1.50e7 vs wavefront 1.73e7). The **wavefront still wins B ≥ 88**
  (its flat per-plane parallelism has no team-count ceiling) — as predicted, the gap narrows, it doesn't
  close. The gain is largest at high nmu (e.g. 176³: +8.5 % at nmu=3 → +49.8 % at nmu=6), because the
  baseline diagonal falls off a high-nmu "crater" on big blocks that the compaction largely avoids.

## nsys per-cycle sweep-kernel active time (fence-free), nmu=6

Direct confirmation the compaction shrinks the sweep kernel itself (`sc_sweep_diag`, one launch/cycle;
active-ms summed over 10 cycles):

| block B | diagonal active (ms) | diagonal_compact active (ms) | kernel speedup |
|--:|--:|--:|--:|
| 16 | 1317 | 1185 | 1.11× (−10 %) |
| 88 | 5576 | 3866 | 1.44× (−31 %) |
| 176 | 5104 | 3284 | **1.55× (−36 %)** |

Monotonic with block size (more off-plane waste removed at bigger blocks), corroborating the ZCPS gains.
*(The `gap_fraction` column is meaningless for the diagonal — it is a single launch/cycle, not the
wavefront's per-plane chain — so `active_ms` is the relevant metric here.)*

## Team-size knob (B) — no help; AUTO is optimal

`diag_team_size` sweep at 176³ nmu=6 (ZCPS): **AUTO=1.50e7, 128=1.50e7, 256=1.14e7, 512=1.29e7**.
`Kokkos::AUTO` (≈128) is optimal; forcing larger teams **hurts** — with register-limited occupancy
(94 regs → occ ≈0.25) bigger teams reduce resident teams/SM without adding useful parallelism.
**Recommendation: keep `diag_team_size=0` (AUTO)**; the knob stays as a documented tuning hook. (A future
I3 — hoisting angle-only invariants to cut registers — could raise the occupancy ceiling and change this.)

## Verdict

**diagonal_compact is a clear, bit-identical win** — a strict Pareto improvement over baseline diagonal
(+5–11 % small/mid blocks, **+39–50 % big blocks**), the best sweep for B ≤ 44, and it nearly catches the
wavefront on a single 176³ block. Improvement (A) delivered it entirely; the team-size knob (B) does not
help. Because it is bit-identical and never slower, **consider promoting it to be the default `diagonal`**
(or the recommended diagonal). The wavefront remains the winner for the largest blocks (B ≥ 88), so the
practical guidance is: **diagonal_compact for many small/mid blocks, wavefront for few big blocks.**
