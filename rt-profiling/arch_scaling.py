#!/usr/bin/env python3
"""Cross-architecture analysis of the hydro-vs-RT cost gap (A100 / GH200 / B200).

Companion to analyze.py, which summarises ONE device dir. This one reads several and answers
the cross-device question: how does the cost of the SC radiation sweep, relative to hydro,
depend on the GPU architecture?

Derived metrics (all from the clean whole-run ZCPS pair, per the ledger's golden rule --
never the fenced per-kernel rad_cells_per_s):

    ZCPS_hydro   = zcps_off_med                       radiation off
    ZCPS_on      = zcps_on_med                        radiation on
    ZCPS_rad     = 1 / (1/ZCPS_on - 1/ZCPS_hydro)     RADIATION-ONLY throughput; subtracts the
                                                      hydro time from the same run
    gap G        = ZCPS_hydro / ZCPS_rad              cost of the radiation work for one
                                                      zone-cycle, in hydro-zone-cycle units
    ray cost k   = G / rays                           the same, per ray -- the quantity that
                                                      turns out to be nearly config-invariant

Identity worth knowing: slowdown = ZCPS_hydro/ZCPS_on = 1 + G exactly, so G is just the
slowdown with the hydro baseline removed. G is the honest "gap" because it is comparable
across devices with different hydro speeds.

IMPORTANT (why matched configs): G must be formed from ZCPS_hydro and ZCPS_rad measured in
the SAME run (same block size, same ray count). Pairing one device's best-RT config against
its best-hydro config -- which sit at different block sizes -- inflates or deflates G by the
hydro block-size curve and produces a spurious architecture trend.

Usage: python3 arch_scaling.py [outdir]      (default: rt-profiling/arch)
"""
import csv
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "arch")

DEVICES = ["a100", "gh200", "b200"]
LABEL = {"a100": "A100-PCIE-40GB", "gh200": "GH200-96GB", "b200": "GB200 (B200)"}
# Okabe-Ito blue / vermillion / bluish-green. Chosen over the seaborn-deep default because that
# one FAILS dichromat separation: its orange-green pair sits at OKLab dE 5.2 (protan) / 6.6
# (deutan), under the dE>=8 floor. These three clear it in every simulation -- run
# `python3 arch_scaling.py --check-palette` to re-derive both numbers.
COLOR = {"a100": "#0072B2", "gh200": "#D55E00", "b200": "#009E73"}
SWEEPS = ["wavefront", "diagonal", "diagonal_compact"]
RAYS = {1: 8, 2: 24, 3: 48, 4: 80, 5: 120, 6: 168}

# Heatmap ramps. Sequential = perceptually uniform, monotonic in OKLab lightness (verified by
# --check-palette), oriented so BRIGHT ALWAYS MEANS GOOD: cmr.ember for throughput (bright =
# fast), reversed for cost (bright = cheap). Diverging = cmr.fusion, whose midpoint is exactly
# neutral (chroma 0.00) so "no difference" reads as white; warm = wavefront wins, cool =
# diagonal_compact wins.
CMAP_SEQ = "cmr.ember"
CMAP_DIV = "cmr.fusion"

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import numpy as np
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False

try:
    import cmasher  # noqa: F401  -- registers the cmr.* colormaps
    HAVE_CMR = True
except Exception:
    HAVE_CMR = False


def seq_cmap(reverse=False):
    if HAVE_CMR:
        return CMAP_SEQ + ("_r" if reverse else "")
    return "inferno_r" if reverse else "inferno"


def div_cmap():
    return CMAP_DIV if HAVE_CMR else "RdBu"


NMB = {}   # (device, B) -> number of meshblocks, for axis labels


def load_all():
    """-> {(device, nmu, B): {sweep: (zcps_hydro, zcps_on)}}"""
    D = defaultdict(dict)

    def add(dev, sweep, r):
        if r.get("phase") != "2" or not r.get("zcps_on_med"):
            return
        try:
            key = (dev, int(r["nmu"]), int(r["B"]))
        except (KeyError, ValueError):
            return
        D[key][sweep] = (float(r["zcps_off_med"]), float(r["zcps_on_med"]))
        if r.get("nmb"):
            NMB[(dev, int(r["B"]))] = int(r["nmb"])

    for dev in DEVICES:
        for sweep in SWEEPS:
            path = os.path.join(HERE, dev, f"results_{sweep}.csv")
            if os.path.exists(path):
                with open(path) as f:
                    for r in csv.DictReader(f):
                        add(dev, sweep, r)
        # A100 never got diagonal_compact in its phase-2 grid; it was measured later in the
        # dedicated A/B run. Same device and deck, a different job -- folded in so all three
        # devices have all three sweeps, and flagged in the report.
        ab = os.path.join(HERE, dev, "diagonal_compact", "results_both.csv")
        if os.path.exists(ab):
            with open(ab) as f:
                for r in csv.DictReader(f):
                    if r.get("sweep") == "diagonal_compact":
                        add(dev, "diagonal_compact", r)
    return D


def rad_only(hydro, on):
    """Radiation-only throughput: strip the hydro time out of the RT-on run."""
    return 1.0 / (1.0 / on - 1.0 / hydro) if on < hydro else float("nan")


def best(D, dev, nmu, B, by="on"):
    """Best sweep at this config. by='on' -> fastest whole run; by='rad' -> same ordering
    (hydro is common), kept explicit so the intent is readable at the call site."""
    v = D.get((dev, nmu, B))
    if not v:
        return None
    sweep = max(v, key=lambda s: v[s][1])
    hydro, on = v[sweep]
    return dict(sweep=sweep, hydro=hydro, on=on, rad=rad_only(hydro, on),
                gap=hydro / rad_only(hydro, on), B=B, nmu=nmu, dev=dev)


def blocks(D, dev):
    return sorted({k[2] for k in D if k[0] == dev})


def hydro_baseline_spread(D):
    """The same hydro baseline is measured once per sweep job. Its spread is a free
    run-to-run error bar on every derived number."""
    out = {}
    for dev in DEVICES:
        spreads = []
        for k, v in D.items():
            if k[0] != dev or len(v) < 2:
                continue
            offs = [x[0] for x in v.values()]
            spreads.append((max(offs) - min(offs)) / (sum(offs) / len(offs)))
        if spreads:
            spreads.sort()
            out[dev] = (len(spreads), spreads[len(spreads) // 2], spreads[-1])
    return out


def check_palette():
    """Recompute the colour claims instead of asserting them: OKLab dE between every pair of
    the categorical device colours, under normal vision and simulated protan/deutan/tritan
    (Vienot 1999), plus OKLab-lightness monotonicity of the sequential ramp and chroma of the
    diverging midpoint. Floors: dE >= 15 normal, >= 8 dichromat."""
    def srgb2lin(c):
        return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

    def lin2oklab(rgb):
        M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                       [0.2119034982, 0.6806995451, 0.1073969566],
                       [0.0883024619, 0.2817188376, 0.6299787005]])
        M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                       [1.9779984951, -2.4285922050, 0.4505937099],
                       [0.0259040371, 0.7827717662, -0.8086757660]])
        return np.cbrt(np.clip(rgb @ M1.T, 0, None)) @ M2.T

    RGB2LMS = np.array([[17.8824, 43.5161, 4.11935],
                        [3.45565, 27.1554, 3.86714],
                        [0.0299566, 0.184309, 1.46709]])
    LMS2RGB = np.linalg.inv(RGB2LMS)
    SIM = {"protan": np.array([[0, 2.02344, -2.52581], [0, 1, 0], [0, 0, 1]]),
           "deutan": np.array([[1, 0, 0], [0.494207, 0, 1.24827], [0, 0, 1]]),
           "tritan": np.array([[1, 0, 0], [0, 1, 0], [-0.395913, 0.801109, 0]])}

    hexes = [COLOR[d] for d in DEVICES]
    lin = srgb2lin(np.array([mcolors.to_rgb(h) for h in hexes]))
    print("categorical palette:", ", ".join(f"{d}={COLOR[d]}" for d in DEVICES))
    ok = True
    for kind in ["normal", "protan", "deutan", "tritan"]:
        rgb = lin if kind == "normal" else np.clip((lin @ RGB2LMS.T) @ SIM[kind].T @ LMS2RGB.T, 0, 1)
        lab = lin2oklab(rgb)
        pairs = [(i, j) for i in range(len(hexes)) for j in range(i + 1, len(hexes))]
        des = [np.linalg.norm(lab[i] - lab[j]) * 100 for i, j in pairs]
        floor = 15.0 if kind == "normal" else 8.0
        good = min(des) >= floor
        ok &= good
        print(f"  {kind:>7}: dE " + " ".join(f"{d:5.1f}" for d in des)
              + f"   min {min(des):5.1f} (floor {floor:.0f})  {'PASS' if good else 'FAIL'}")

    for name, rev in ((seq_cmap(False), False), (seq_cmap(True), True)):
        cm = plt.get_cmap(name)
        L = lin2oklab(srgb2lin(np.array([mcolors.to_rgb(cm(x)) for x in np.linspace(0, 1, 32)])))[:, 0]
        d = np.diff(L)
        mono = (d > 0).all() or (d < 0).all()
        print(f"  sequential {name:>14}: L {L[0]:.2f}->{L[-1]:.2f}  "
              f"{'monotonic' if mono else 'NON-MONOTONIC'}  {'PASS' if mono else 'FAIL'}")
        ok &= mono
    cm = plt.get_cmap(div_cmap())
    mid = lin2oklab(srgb2lin(np.array([mcolors.to_rgb(cm(0.5))])))[0]
    chroma = float(np.hypot(mid[1], mid[2]) * 100)
    print(f"  diverging  {div_cmap():>14}: midpoint L {mid[0]:.2f} chroma {chroma:.2f}  "
          f"{'PASS' if chroma < 2 else 'FAIL'} (neutral midpoint)")
    return 0 if ok and chroma < 2 else 1


def _cell_text_color(rgba):
    """Black or white annotation, whichever has more contrast against the cell."""
    r, g, b = rgba[:3]
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "black" if lum > 0.55 else "white"


def heatmap_panels(D, quantity, outfile, title, cbar_label, fmt,
                   cmap, norm_kind="log", vlim=None):
    """One figure, one panel per device, x = rays, y = meshblock size.

    quantity(dev, nmu, B) -> float or None. Every cell is annotated with its value, so the
    figure doubles as the table (identity never rests on colour alone), and the colour scale is
    SHARED across the three panels so cross-device differences are visible as colour."""
    grids, ylabels = {}, {}
    vals = []
    for dev in DEVICES:
        Bs = blocks(D, dev)
        g = np.full((len(Bs), len(RAYS)), np.nan)
        for r, B in enumerate(Bs):
            for c, nmu in enumerate(sorted(RAYS)):
                v = quantity(dev, nmu, B)
                if v is not None and np.isfinite(v):
                    g[r, c] = v
                    vals.append(v)
        grids[dev] = g
        ylabels[dev] = Bs

    if vlim is not None:
        vmin, vmax = vlim
    elif norm_kind == "div":
        m = max(abs(min(vals)), abs(max(vals)))
        vmin, vmax = -m, m
    else:
        vmin, vmax = min(vals), max(vals)
    if norm_kind == "log":
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    nrows = max(len(ylabels[d]) for d in DEVICES)
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 1.05 * nrows + 2.0))
    im = None
    for ax, dev in zip(axes, DEVICES):
        g = grids[dev]
        im = ax.imshow(g, cmap=cmap, norm=norm, aspect="auto")
        ax.set_xticks(range(len(RAYS)), [RAYS[n] for n in sorted(RAYS)])
        ax.set_yticks(range(len(ylabels[dev])),
                      [f"{B}  (n={NMB.get((dev, B), '?')})" for B in ylabels[dev]], fontsize=8)
        ax.set_xlabel("rays")
        ax.set_title(LABEL[dev], fontsize=10)
        for r in range(g.shape[0]):
            for c in range(g.shape[1]):
                if np.isfinite(g[r, c]):
                    ax.text(c, r, fmt(g[r, c]), ha="center", va="center", fontsize=7.5,
                            color=_cell_text_color(plt.get_cmap(cmap)(norm(g[r, c]))))
        ax.set_xticks(np.arange(-0.5, len(RAYS), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(ylabels[dev]), 1), minor=True)
        # 2px surface gap between cells, per the mark spec
        ax.grid(which="minor", color="white", linewidth=1.5)
        ax.tick_params(which="minor", length=0)
    axes[0].set_ylabel("meshblock size B  (n = meshblocks)")
    fig.suptitle(title, y=0.99)
    fig.tight_layout(rect=(0, 0, 0.93, 1))
    cax = fig.add_axes([0.945, 0.13, 0.014, 0.72])
    cb = fig.colorbar(im, cax=cax, extend="both" if norm_kind == "div" else "neither")
    cb.set_label(cbar_label, fontsize=9)
    if norm_kind == "div":
        # Label the diverging bar in ratio units -- log2 is the right scale to *compute* on
        # and the wrong one to read off a legend.
        ticks = [t for t in (0.5, 0.7, 0.85, 1.0, 1.2, 1.4, 2.0)
                 if vmin <= np.log2(t) <= vmax]
        cb.set_ticks([np.log2(t) for t in ticks])
        cb.set_ticklabels([f"{t:g}x" for t in ticks])
    fig.savefig(os.path.join(OUTDIR, outfile), dpi=140, bbox_inches="tight")
    plt.close(fig)


def md_table(rows, headers):
    w = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    sep = "|" + "|".join("-" * (x + 2) for x in w) + "|"
    head = "| " + " | ".join(str(h).ljust(w[i]) for i, h in enumerate(headers)) + " |"
    body = ["| " + " | ".join(str(r[i]).ljust(w[i]) for i in range(len(headers))) + " |"
            for r in rows]
    return "\n".join([head, sep] + body)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    if "--check-palette" in sys.argv:
        return check_palette()
    D = load_all()
    if not D:
        print("no phase-2 data found", file=sys.stderr)
        return 1

    print("# Hydro-vs-RT gap across architectures\n")

    print("## Hydro-baseline reproducibility (same config, different sweep jobs)\n")
    rows = [(dev, n, f"{med*100:.1f}%", f"{mx*100:.1f}%")
            for dev, (n, med, mx) in hydro_baseline_spread(D).items()]
    print(md_table(rows, ["device", "n configs", "median spread", "max spread"]), "\n")

    # --- the common block size, B=16, is the only one all three ladders share -------------
    print("## Gap at B=16 (the one block size common to all three ladders)\n")
    rows = []
    for nmu in sorted(RAYS):
        row = [RAYS[nmu]]
        for dev in DEVICES:
            b = best(D, dev, nmu, 16)
            row += [f"{b['gap']:.2f}", f"{b['gap']/RAYS[nmu]*100:.2f}%"] if b else ["-", "-"]
        rows.append(row)
    hdr = ["rays"] + [f"{d} {x}" for d in DEVICES for x in ("gap", "k")]
    print(md_table(rows, hdr), "\n")

    # --- ray cost k vs block size ---------------------------------------------------------
    print("## Relative ray cost k = gap/rays (nmu=6) vs block size\n")
    for dev in DEVICES:
        rows = []
        for B in blocks(D, dev):
            b = best(D, dev, 6, B)
            if b:
                rows.append((B, f"{b['hydro']:.3g}", f"{b['rad']:.3g}",
                             f"{b['gap']:.1f}", f"{b['gap']/168*100:.2f}%", b["sweep"]))
        print(f"### {LABEL[dev]}\n")
        print(md_table(rows, ["B", "ZCPS hydro", "ZCPS rad-only", "gap", "k", "best sweep"]), "\n")

    # --- generational scaling -------------------------------------------------------------
    print("## Generational scaling (A100 = 1.0), matched configs\n")
    BIGB = {"a100": 88, "gh200": 112, "b200": 128}   # each device's best-hydro block
    for nmu in (3, 6):
        for name, pick in (("B=16 (common)", lambda d: 16),
                           ("best-hydro block", lambda d: BIGB[d])):
            rows = []
            ref = None
            for dev in DEVICES:
                b = best(D, dev, nmu, pick(dev))
                if not b:
                    continue
                if ref is None:
                    ref = b
                rows.append((dev, b["B"], f"{b['hydro']:.3g}", f"{b['hydro']/ref['hydro']:.2f}x",
                             f"{b['rad']:.3g}", f"{b['rad']/ref['rad']:.2f}x",
                             f"{b['gap']:.1f}", b["sweep"]))
            print(f"### {RAYS[nmu]} rays, {name}\n")
            print(md_table(rows, ["device", "B", "hydro", "vs A100", "rad-only",
                                  "vs A100", "gap", "sweep"]), "\n")

    if HAVE_MPL:
        make_figures(D, BIGB)
        print(f"figures -> {OUTDIR}")
    return 0


def make_figures(D, BIGB):
    # --- heatmaps over the full (block size x rays) plane --------------------------------
    def q_zcps(dev, nmu, B):
        b = best(D, dev, nmu, B)
        return b["on"] if b else None

    def q_radonly(dev, nmu, B):
        b = best(D, dev, nmu, B)
        return b["rad"] if b else None

    def q_sweepup(dev, nmu, B):
        v = D.get((dev, nmu, B), {})
        if "diagonal_compact" not in v or "wavefront" not in v:
            return None
        return np.log2(v["diagonal_compact"][1] / v["wavefront"][1])

    def q_relcost(dev, nmu, B):
        b = best(D, dev, nmu, B)
        return b["gap"] / RAYS[nmu] * 100 if b else None

    def q_slowdown(dev, nmu, B):
        b = best(D, dev, nmu, B)
        return b["hydro"] / b["on"] if b else None

    heatmap_panels(D, q_zcps, "heat_zcps.png",
                   "RT-on throughput (best sweep at each point) — bright = fast",
                   "ZCPS with radiation on", lambda v: f"{v/1e6:.0f}M",
                   seq_cmap(), norm_kind="log")
    heatmap_panels(D, q_radonly, "heat_zcps_radonly.png",
                   "Radiation-only throughput (hydro time removed) — bright = fast",
                   "ZCPS$_{rad}$", lambda v: f"{v/1e6:.0f}M",
                   seq_cmap(), norm_kind="log")
    # Clipped at +/-1.2 (0.44x-2.3x): the wavefront's blowouts on single-block meshes reach
    # 0.18x, and letting them set the limits washes out the 1.0-1.35x band that the sweep
    # choice actually turns on. The 5 clipped cells still carry their printed value.
    heatmap_panels(D, q_sweepup, "heat_sweep_speedup.png",
                   "Sweep speedup: diagonal_compact vs wavefront  "
                   "(cool = compact wins, warm = wavefront wins)",
                   "diagonal_compact / wavefront", lambda v: f"{2**v:.2f}x",
                   div_cmap(), norm_kind="div", vlim=(-1.2, 1.2))
    heatmap_panels(D, q_slowdown, "heat_slowdown.png",
                   "Whole-run slowdown from enabling radiation (hydro / RT-on) — bright = cheap",
                   "ZCPS$_{hydro}$ / ZCPS$_{on}$", lambda v: f"{v:.1f}x",
                   seq_cmap(reverse=True), norm_kind="log")
    heatmap_panels(D, q_relcost, "heat_relcost.png",
                   "Relative cost of one ray-cell update, k = gap/rays — bright = cheap",
                   "k  (% of a hydro cell update)", lambda v: f"{v:.1f}",
                   seq_cmap(reverse=True), norm_kind="lin")

    # Fig 1: gap vs rays, two panels (small blocks / big blocks)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (title, pick) in zip(axes, [("B = 16 (many small blocks)", lambda d: 16),
                                        ("largest block on each device", lambda d: BIGB[d])]):
        for dev in DEVICES:
            xs, ys = [], []
            for nmu in sorted(RAYS):
                b = best(D, dev, nmu, pick(dev))
                if b:
                    xs.append(RAYS[nmu])
                    ys.append(b["gap"])
            ax.plot(xs, ys, "o-", color=COLOR[dev], label=LABEL[dev])
        ax.set_xlabel("rays (3D, Bruls type-A)")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("gap  G = ZCPS$_{hydro}$ / ZCPS$_{rad}$")
    axes[0].legend(frameon=False)
    fig.suptitle("Radiation cost relative to hydro is set by ray count, not architecture",
                 y=1.00)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "gap_vs_rays.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)

    # Fig 2: relative ray cost k vs block size -- the money plot
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for dev in DEVICES:
        xs, ys = [], []
        for B in blocks(D, dev):
            b = best(D, dev, 6, B)
            if b:
                xs.append(B)
                ys.append(b["gap"] / 168 * 100)
        ax.plot(xs, ys, "o-", color=COLOR[dev], label=LABEL[dev])
    ax.set_xscale("log", base=2)
    ax.set_xlabel("meshblock size B (cells per side)")
    ax.set_ylabel("k = cost of one ray-cell update\n(% of a hydro cell update)")
    ax.set_title("Block size moves the ray cost ~4x; architecture ~1.4x  (168 rays)")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "raycost_vs_block.png"), dpi=140)
    plt.close(fig)

    # Fig 3: generational scaling, hydro vs radiation

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (title, pick) in zip(axes, [("B = 16 (many small blocks)", lambda d: 16),
                                        ("largest block on each device", lambda d: BIGB[d])]):
        hyd, rad = [], []
        for dev in DEVICES:
            b = best(D, dev, 6, pick(dev))
            hyd.append(b["hydro"])
            rad.append(b["rad"])
        hyd = np.array(hyd) / hyd[0]
        rad = np.array(rad) / rad[0]
        x = np.arange(len(DEVICES))
        ax.bar(x - 0.19, hyd, 0.38, label="hydro", color="#8C8C8C")
        ax.bar(x + 0.19, rad, 0.38, label="radiation only", color="#C44E52")
        for i, (h, r) in enumerate(zip(hyd, rad)):
            ax.text(i - 0.19, h + 0.05, f"{h:.2f}x", ha="center", fontsize=8)
            ax.text(i + 0.19, r + 0.05, f"{r:.2f}x", ha="center", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([LABEL[d] for d in DEVICES], fontsize=8)
        ax.set_title(title)
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("throughput relative to A100")
    axes[0].legend(frameon=False)
    fig.suptitle("Radiation scales across generations at least as well as hydro (168 rays)",
                 y=1.00)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "generational_scaling.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
