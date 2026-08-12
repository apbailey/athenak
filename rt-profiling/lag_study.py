#!/usr/bin/env python3
"""Boundary-lag study: how many SC iterations does a domain decomposition cost?

The SC sweep is meshblock-local; light crosses a block boundary only via the ghost exchange at
the top of each iteration, so information advances at most one meshblock per iteration. Every
throughput number in rt-profiling was taken at iter_max=1, which hides that cost entirely.

This runs the `sc_lag` probe (src/pgen/unit_tests/sc_lag.cpp) over a 2D grid of
  meshblock size B   (at FIXED mesh size, so only the decomposition changes)
  x  opacity chi     (domain optical depth tau = chi * L, L = 1)
and records the iteration count to reach a fixed iter_tol.

Hypothesis: niter ~ min(block hops a ray must cross, the optical-depth horizon in blocks).
Thin medium -> the decomposition penalty is real and grows with block count; thick medium ->
light is absorbed before crossing many blocks, so it saturates.

Iteration count is a property of the algorithm, not the hardware, so this runs on a laptop
CPU build and the answer transfers to any GPU.

Usage:
  python3 lag_study.py --athena <path> [--outdir DIR] [--mesh 64] [--nmu 3]
  python3 lag_study.py --analyze-only [--outdir DIR]
"""
import argparse
import csv
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

RESULT_RE = re.compile(r"SC_LAG RESULT (.*)")


def parse_result(stdout):
    for line in stdout.splitlines():
        m = RESULT_RE.search(line)
        if not m:
            continue
        out = {}
        for tok in m.group(1).split():
            k, _, v = tok.partition("=")
            out[k] = v
        return out
    return None


def write_input(path, template, mesh, block, chi, nmu, iter_max, tol):
    s = open(template).read()
    # <mesh> block sizes
    s = re.sub(r"(<mesh>.*?)nx1    = \d+", lambda m: m.group(1) + f"nx1    = {mesh}", s,
               flags=re.S)
    s = re.sub(r"(<mesh>.*?)nx2    = \d+", lambda m: m.group(1) + f"nx2    = {mesh}", s,
               flags=re.S)
    s = re.sub(r"(<mesh>.*?)nx3    = \d+", lambda m: m.group(1) + f"nx3    = {mesh}", s,
               flags=re.S)
    # <meshblock>
    s = re.sub(r"(<meshblock>\s*\n)nx1 = \d+\nnx2 = \d+\nnx3 = \d+",
               lambda m: m.group(1) + f"nx1 = {block}\nnx2 = {block}\nnx3 = {block}", s)
    s = re.sub(r"^nmu          = \d+", f"nmu          = {nmu}", s, flags=re.M)
    s = re.sub(r"^iter_max     = \d+", f"iter_max     = {iter_max}", s, flags=re.M)
    s = re.sub(r"^iter_tol     = \S+", f"iter_tol     = {tol}", s, flags=re.M)
    s = re.sub(r"^chi    = \S+", f"chi    = {chi}", s, flags=re.M)
    open(path, "w").write(s)


def run_grid(args):
    template = os.path.join(REPO, "tst", "inputs", "sc_lag.athinput")
    rundir = os.path.join(args.outdir, "runs")
    os.makedirs(rundir, exist_ok=True)
    rows = []
    blocks = [b for b in args.blocks if args.mesh % b == 0]
    total = len(blocks) * len(args.chis)
    n = 0
    for chi in args.chis:
        for B in blocks:
            n += 1
            tag = f"m{args.mesh}_B{B}_chi{chi:g}"
            inp = os.path.join(rundir, f"sc_lag_{tag}.athinput")
            write_input(inp, template, args.mesh, B, chi, args.nmu, args.iter_max, args.tol)
            print(f"[{n}/{total}] mesh={args.mesh} B={B} chi={chi:g} ...",
                  end=" ", flush=True)
            try:
                p = subprocess.run([args.athena, "-i", inp, "-d", rundir],
                                   capture_output=True, text=True, timeout=args.timeout)
            except subprocess.TimeoutExpired:
                print("TIMEOUT")
                continue
            r = parse_result(p.stdout)
            if r is None:
                print("NO RESULT")
                sys.stderr.write(p.stdout[-2000:] + p.stderr[-2000:])
                continue
            rows.append(r)
            print(f"nblk_side={r['nblk_side']} nmb={r['nmb']} niter={r['niter']} "
                  f"conv={r['converged']} jbar={float(r['jbar']):.6e}")

    csv_path = os.path.join(args.outdir, "lag_results.csv")
    if rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {csv_path}")
    return rows


def analyze(rows, outdir):
    if not rows:
        print("no rows")
        return
    for r in rows:
        for k in ("nblk_side", "nmb", "niter", "converged", "B", "nx1"):
            r[k] = int(r[k])
        for k in ("chi", "tau_dom", "tau_blk", "jbar", "res"):
            r[k] = float(r[k])

    chis = sorted({r["chi"] for r in rows})
    sides = sorted({r["nblk_side"] for r in rows})

    print("\n## Iterations to converge (rows = meshblocks per side, cols = domain optical depth)\n")
    hdr = "| blocks/side | nmb | " + " | ".join(f"tau={c:g}" for c in chis) + " |"
    print(hdr)
    print("|" + "---|" * (2 + len(chis)))
    for s in sides:
        nmb = next(r["nmb"] for r in rows if r["nblk_side"] == s)
        cells = []
        for c in chis:
            m = [r for r in rows if r["nblk_side"] == s and r["chi"] == c]
            cells.append(f"{m[0]['niter']}" + ("" if m[0]["converged"] else "*") if m else "-")
        print(f"| {s} | {nmb} | " + " | ".join(cells) + " |")
    print("\n`*` = hit iter_max without converging.")

    # Cost model: total work ~ niter x (per-sweep cost). Report the iteration penalty
    # relative to the single-block reference at the same optical depth.
    print("\n## Iteration penalty relative to a single meshblock\n")
    print("| blocks/side | " + " | ".join(f"tau={c:g}" for c in chis) + " |")
    print("|" + "---|" * (1 + len(chis)))
    for s in sides:
        cells = []
        for c in chis:
            ref = [r for r in rows if r["nblk_side"] == 1 and r["chi"] == c]
            m = [r for r in rows if r["nblk_side"] == s and r["chi"] == c]
            cells.append(f"{m[0]['niter']/ref[0]['niter']:.1f}x" if ref and m else "-")
        print(f"| {s} | " + " | ".join(cells) + " |")

    # Decomposition invariance: the converged answer must not depend on how the domain was cut.
    print("\n## Decomposition invariance of the converged solution (max spread in mean J)\n")
    print("| tau | mean J | max rel spread across decompositions |")
    print("|---|---|---|")
    for c in chis:
        js = [r["jbar"] for r in rows if r["chi"] == c and r["converged"]]
        if len(js) < 2:
            continue
        spread = (max(js) - min(js)) / (sum(js) / len(js))
        flag = "OK" if spread < 1e-4 else "<-- CHECK"
        print(f"| {c:g} | {sum(js)/len(js):.6e} | {spread:.2e} {flag} |")

    make_figure(rows, chis, sides, outdir)


def combine_with_throughput(outdir):
    """The deliverable: total radiation cost = niter x per-sweep cost.

    Per-sweep throughput ZCPS_rad comes from the cross-architecture study (arch_scaling.py);
    niter comes from the law measured here. Both regimes are reported because the answer
    genuinely depends on the optical depth per meshblock:
      thin  (tau_blk << 1): niter = 3*nblk_side - 1   (light must cross every block)
      thick (tau_blk >~ 3): niter ~ 4, flat           (absorbed before it can cross)
    """
    sys.path.insert(0, HERE)
    try:
        import arch_scaling as A
    except Exception as e:
        print(f"(no throughput combine: {e})")
        return
    D = A.load_all()
    NSTAR = {"a100": 176, "gh200": 224, "b200": 256}

    print("\n## Total radiation cost = iterations x per-sweep cost  (nmu=6, at each device's N*)\n")
    print("Relative cost per zone-cycle, normalised to the cheapest column entry. "
          "`niter` from the law measured above; `ZCPS_rad` measured in ARCH_SCALING.md.\n")
    for dev in A.DEVICES:
        N = NSTAR[dev]
        blocks = [b for b in A.blocks(D, dev) if N % b == 0]
        if not blocks:
            continue
        recs = []
        for B in blocks:
            b = A.best(D, dev, 6, B)
            if not b:
                continue
            n = N // B
            recs.append(dict(B=B, n=n, rad=b["rad"],
                             thin=(3*n - 1) / b["rad"],
                             thick=4.0 / b["rad"]))
        if not recs:
            continue
        tmin = min(r["thin"] for r in recs)
        kmin = min(r["thick"] for r in recs)
        print(f"### {A.LABEL[dev]} (N* = {N}³)\n")
        print("| B | blocks/side | ZCPS_rad | niter thin | **cost thin** | niter thick | **cost thick** |")
        print("|--:|--:|--:|--:|--:|--:|--:|")
        for r in recs:
            print(f"| {r['B']} | {r['n']} | {r['rad']:.3g} | {3*r['n']-1} | "
                  f"**{r['thin']/tmin:.1f}x** | 4 | **{r['thick']/kmin:.2f}x** |")
        bt = min(recs, key=lambda r: r["thin"])
        bk = min(recs, key=lambda r: r["thick"])
        print(f"\nBest: **B={bt['B']}** when optically thin, **B={bk['B']}** when thick.\n")


def make_figure(rows, chis, sides, outdir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        import cmasher  # noqa: F401
    except Exception as e:
        print(f"(no figure: {e})")
        return

    grid = np.full((len(sides), len(chis)), np.nan)
    for r in rows:
        grid[sides.index(r["nblk_side"]), chis.index(r["chi"])] = r["niter"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))

    ax = axes[0]
    im = ax.imshow(grid, cmap="cmr.ember_r", aspect="auto",
                   norm=matplotlib.colors.LogNorm(vmin=np.nanmin(grid), vmax=np.nanmax(grid)))
    ax.set_xticks(range(len(chis)), [f"{c:g}" for c in chis])
    ax.set_yticks(range(len(sides)), [str(s) for s in sides])
    ax.set_xlabel(r"domain optical depth  $\tau = \chi L$")
    ax.set_ylabel("meshblocks per side")
    ax.set_title("Iterations to converge — bright = cheap")
    for a in range(len(sides)):
        for b in range(len(chis)):
            if np.isfinite(grid[a, b]):
                rgba = plt.get_cmap("cmr.ember_r")(
                    matplotlib.colors.LogNorm(vmin=np.nanmin(grid),
                                              vmax=np.nanmax(grid))(grid[a, b]))
                lum = 0.2126*rgba[0] + 0.7152*rgba[1] + 0.0722*rgba[2]
                ax.text(b, a, f"{int(grid[a,b])}", ha="center", va="center", fontsize=9,
                        color="black" if lum > 0.55 else "white")
    ax.set_xticks(np.arange(-0.5, len(chis), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(sides), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)
    fig.colorbar(im, ax=ax, label="iterations")

    ax = axes[1]
    # Okabe-Ito, CVD-checked (see arch_scaling.py --check-palette)
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9"]
    for n, c in enumerate(chis):
        ys = [grid[sides.index(s), n] for s in sides]
        ax.plot(sides, ys, "o-", color=colors[n % len(colors)], label=rf"$\tau$ = {c:g}")
    ax.plot(sides, sides, "k--", lw=1, alpha=0.5, label="linear in blocks/side")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("meshblocks per side")
    ax.set_ylabel("iterations to converge")
    ax.set_title("The decomposition penalty, vs optical depth")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    out = os.path.join(outdir, "lag_iterations.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"\nfigure -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--athena", default=os.path.join(REPO, "build_lag", "src", "athena"))
    ap.add_argument("--outdir", default=os.path.join(HERE, "lag"))
    ap.add_argument("--mesh", type=int, default=64)
    ap.add_argument("--blocks", type=int, nargs="+", default=[64, 32, 16, 8])
    ap.add_argument("--chis", type=float, nargs="+", default=[0.1, 1.0, 10.0, 100.0])
    ap.add_argument("--nmu", type=int, default=3)
    ap.add_argument("--iter-max", type=int, default=400)
    ap.add_argument("--tol", default="1.0e-6")
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    if args.analyze_only:
        path = os.path.join(args.outdir, "lag_results.csv")
        rows = list(csv.DictReader(open(path)))
    else:
        rows = run_grid(args)
    analyze(rows, args.outdir)
    combine_with_throughput(args.outdir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
