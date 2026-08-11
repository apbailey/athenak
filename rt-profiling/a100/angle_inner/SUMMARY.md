# Native I2 — `ir_layout=angle_inner` real whole-run speedup (A100)

*Apollo A100-PCIE-40GB, branch `rt-sc` (`bbfcc1f6`), `build_a100`. Run 2026-08-11 via
`a100/submit_angle_inner_ab.sbatch`. Mesh N=128³ (the angle-inner path keeps `ir` + a normal-layout
companion `ir_normal` = 2×`ir`, so N=128 fits). Data: `results_angle_inner.csv`.*

## What this is

The **native** realization of I2: `ir` stored ANGLE-INNERMOST `(m,k,j,i,angg)` so the wavefront sweep +
moments run coalesced with **no per-sweep transpose** — the real speedup the transpose prototype
(`wavefront_coalesced`, whole-run 0.62–0.92×) could only *measure*. Opt-in `<nr_radiation>/ir_layout=angle_inner`
(default `normal` byte-identical); requires sweep=wavefront, 3D, uniform mesh (FATAL-guarded).

**Isolation (the key constraint):** the generic CC boundary-exchange, AMR, and hydro/MHD/Z4c code is
**byte-unchanged**. The per-cycle exchange is bridged SC-side by a normal-layout companion `ir_normal`
transposed around the UNCHANGED `PackAndSendCC`/`RecvAndUnpackCC`/`RadiationBCs` — and only over the
**boundary shell** (O(surface), not the full-array transpose that sank the prototype). SMR/AMR guarded off.

## Correctness (GPU, bit-exact vs normal-layout wavefront)

- reordered sweep + ComputeJ: `jmean` hash `0x0a5a28a1ddc9319c` **== normal** (MATCH);
- exchange bridge: refilled-ghost hash `0xed93742693d8b175` **== normal** (MATCH, `test_exchange` round-trip).

## Whole-run ZCPS speedup — `angle_inner` / `normal` (sweep=wavefront, N=128³)

| block B | nmb | **nmu=6 (168 rays)** | nmu=3 (48 rays) |
|--:|--:|--:|--:|
| 16 | 512 | 0.95× | 0.97× |
| 32 | 64 | 1.03× | 1.02× |
| 64 | 8 | **1.17×** | 1.09× |
| 128 | 1 | **1.27×** | 1.16× |

**Real whole-run speedup on large blocks (1.17× at 64³, 1.27× at 128³, nmu=6)** — the coalescing prize
flows through natively and the shell-transpose bridge is cheap enough to net positive. Contrast the
transpose *prototype*, which was a net slowdown (0.62–0.92×) everywhere: **the shell-only bridge is what
converts the measured sweep-kernel prize into a genuine whole-run gain.**

**Why less than the sweep-kernel prize (1.94× at 128³):** the whole run also includes `ComputeJ`, moments,
bvals, the shell transpose, and hydro, which dilute the sweep gain. **Why small blocks don't win:** at
16³ the boundary shell is ~78 % of the block, so the shell-transpose bridge costs ≈ a full transpose and
eats the modest sweep gain; and `ComputeJ`/moments were kept as a simple index-flip (warp over cells,
serial angle → *uncoalesced* under angle-inner), a per-cell pessimization that hurts most at small blocks.

## Guidance & follow-ups

- **Use `ir_layout=angle_inner` for large-meshblock wavefront runs** (B ≥ 64³): a real 1.2–1.3× radiation
  speedup, bit-exact, standard codebase untouched.
- **Recover the small/mid-block dilution** (next opt): reorder `ComputeJ`/`CalculateMoments` to an
  angle-warp reduction (they currently pessimize under angle-inner), and/or tighten the shell for small
  blocks. That would lift the whole-run gain toward the sweep-kernel 1.5–1.9×.
- **Memory:** the `ir_normal` companion doubles the `ir` footprint (the exchange bridge's cost of keeping
  the shared code untouched). A future SC-private angle-inner pack/unpack would remove it.
- **AMR:** unsupported (uniform mesh only) — a later increment needs the shared mesh restrict/prolong path
  to be bridged too.
