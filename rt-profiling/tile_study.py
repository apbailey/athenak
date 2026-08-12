#!/usr/bin/env python3
"""Tiled (KBA) sweep performance — analysis of ledger I7 step 3.

Reads {device}/tile/results_both.csv (produced by run_sweep.py with RT_SWEEPS including
`tiled` and RT_TILE_SIZES set) and answers three questions:

  1. Does tiling beat the wavefront in the large-block regime, where the diagonal collapses?
  2. What is the optimal tile size, and does it match the design's prediction (~the block size
     at which the diagonal already wins, i.e. 16-32)?
  3. Does the new best beat the best configuration previously known ANYWHERE on the ladder
     (from ARCH_SCALING.md)? That is the number that matters for production.

Metric convention is ARCH_SCALING.md's: ZCPS_rad = 1/(1/ZCPS_on - 1/ZCPS_hydro) is the
radiation-only throughput with hydro time removed, and both terms come from the same run.

Usage: python3 tile_study.py [--outdir DIR]
"""
import argparse
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEVICES = ["gh200", "b200"]
LABEL = {"a100": "A100", "gh200": "GH200-96GB", "b200": "GB200 (B200)"}
# Okabe-Ito, CVD-checked (arch_scaling.py --check-palette)
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9", "#E69F00"]


def rad_only(hydro, on):
    return 1.0 / (1.0 / on - 1.0 / hydro) if on < hydro else float("nan")


def load(dev):
    path = os.path.join(HERE, dev, "tile", "results_both.csv")
    if not os.path.exists(path):
        return []
    out = []
    for r in csv.DictReader(open(path)):
        if not r.get("zcps_on_med"):
            continue
        hydro, on = float(r["zcps_off_med"]), float(r["zcps_on_med"])
        out.append(dict(B=int(r["B"]), nmb=int(r["nmb"]), nmu=int(r["nmu"]),
                        sweep=r["sweep"], tile=int(r["tile_size"] or 0),
                        hydro=hydro, on=on, rad=rad_only(hydro, on),
                        lo=float(r["zcps_on_min"]), hi=float(r["zcps_on_max"])))
    return out


def label_of(r):
    return f"tiled/t{r['tile']}" if r["sweep"] == "tiled" else r["sweep"]


def report(dev, rows):
    if not rows:
        print(f"\n## {LABEL[dev]} — no data\n")
        return
    Bs = sorted({r["B"] for r in rows})
    nmus = sorted({r["nmu"] for r in rows})
    print(f"\n## {LABEL[dev]}\n")
    for nmu in nmus:
        print(f"### {nmu*(nmu+1)//2*8} rays (nmu={nmu}) — ZCPS with radiation on\n")
        tiles = sorted({r["tile"] for r in rows if r["sweep"] == "tiled"})
        hdr = "| B | nmb | wavefront | diag_compact | " + \
              " | ".join(f"tiled t={t}" for t in tiles) + " | best | vs wavefront |"
        print(hdr)
        print("|" + "---|" * (5 + len(tiles)))
        for B in Bs:
            sel = [r for r in rows if r["B"] == B and r["nmu"] == nmu]
            if not sel:
                continue
            nmb = sel[0]["nmb"]
            get = lambda **kw: next((r for r in sel if all(r[k] == v for k, v in kw.items())), None)
            wf = get(sweep="wavefront")
            dc = get(sweep="diagonal_compact")
            cells = []
            for t in tiles:
                x = get(sweep="tiled", tile=t)
                cells.append(f"{x['on']:.3g}" if x else "-")
            best = max(sel, key=lambda r: r["on"])
            spd = f"**{best['on']/wf['on']:.2f}x**" if wf else "-"
            print(f"| {B} | {nmb} | {wf['on']:.3g} | {dc['on']:.3g} | " + " | ".join(cells) +
                  f" | **{label_of(best)}** | {spd} |")
        print()


def headline(dev, rows):
    """New best vs the best previously known anywhere on this device's block ladder."""
    try:
        sys.path.insert(0, HERE)
        import arch_scaling as A
    except Exception:
        return
    D = A.load_all()
    out = []
    for nmu in sorted({r["nmu"] for r in rows}):
        prev = []
        for k in D:
            if k[0] != dev or k[1] != nmu:
                continue
            b = A.best(D, dev, nmu, k[2])
            if b:
                prev.append((b["on"], b["B"], b["sweep"]))
        if not prev:
            continue
        pon, pB, psw = max(prev)
        new = max((r for r in rows), key=lambda r: r["on"] if r["nmu"] == nmu else -1)
        new = max((r for r in rows if r["nmu"] == nmu), key=lambda r: r["on"])
        out.append((nmu, pon, pB, psw, new))
    if out:
        print(f"\n### {LABEL[dev]} — best config, before vs after\n")
        print("| rays | previous best (whole ladder) | new best | gain |")
        print("|---|---|---|---|")
        for nmu, pon, pB, psw, new in out:
            print(f"| {nmu*(nmu+1)//2*8} | {pon:.3g} ({psw}, B={pB}) | "
                  f"**{new['on']:.3g}** ({label_of(new)}, B={new['B']}) | "
                  f"**{new['on']/pon:.2f}x** |")


def make_figure(data, outdir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as e:
        print(f"(no figure: {e})")
        return
    devs = [d for d in DEVICES if data.get(d)]
    if not devs:
        return
    fig, axes = plt.subplots(1, len(devs), figsize=(6.2 * len(devs), 4.6), squeeze=False)
    for ax, dev in zip(axes[0], devs):
        rows = data[dev]
        nmu = max(r["nmu"] for r in rows)
        Bs = sorted({r["B"] for r in rows})
        for n, B in enumerate(Bs):
            sel = sorted([r for r in rows if r["B"] == B and r["nmu"] == nmu
                          and r["sweep"] == "tiled" and r["tile"] > 0], key=lambda r: r["tile"])
            wf = next((r for r in rows if r["B"] == B and r["nmu"] == nmu
                       and r["sweep"] == "wavefront"), None)
            if not sel or not wf:
                continue
            ax.plot([r["tile"] for r in sel], [r["on"] / wf["on"] for r in sel],
                    "o-", color=COLORS[n % len(COLORS)], label=f"B={B} (nmb={sel[0]['nmb']})")
            ax.axhline(1.0, color="0.6", lw=1, ls="--")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("tile size (cells per side)")
        ax.set_title(f"{LABEL[dev]} — {nmu*(nmu+1)//2*8} rays")
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, fontsize=8)
    axes[0][0].set_ylabel("tiled ZCPS / wavefront ZCPS")
    fig.suptitle("Tiled (KBA) sweep vs the wavefront in the large-block regime "
                 "(dashed = wavefront parity)", y=1.00)
    fig.tight_layout()
    out = os.path.join(outdir, "tile_speedup.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nfigure -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.join(HERE, "tile"))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    data = {d: load(d) for d in DEVICES}
    print("# Tiled (KBA) sweep — performance (ledger I7 step 3)")
    for d in DEVICES:
        report(d, data[d])
    print("\n## Headline: does it beat the previously known best?")
    for d in DEVICES:
        if data[d]:
            headline(d, data[d])
    make_figure(data, args.outdir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
