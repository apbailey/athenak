#!/usr/bin/env python3
"""Schematics for TILING_REPORT.md — how the tiled (KBA) sweep works.

Four figures, all 2D projections of what is really a 3D construction (the 3D rule is the same
with a third index added; 2D is what a reader can actually follow):

  sch_wavefront.png  the hyperplane rule itself: which cells can be updated together, and why
  sch_twolevel.png   the two-level construction: cells within a tile, tiles within a block
  sch_dependency.png the correctness argument: a tile's upwind corner set lies on earlier planes
  sch_launch.png     what each sweep actually launches, and the parallelism it exposes

Usage: python3 tile_schematics.py [outdir]
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "tile")
os.makedirs(OUT, exist_ok=True)

# Okabe-Ito (CVD-checked, see arch_scaling.py --check-palette) + a light sequential ramp for
# hyperplane index, which is ordinal data and so wants a ramp, not categorical hues.
BLUE, VERM, GREEN, PURPLE = "#0072B2", "#D55E00", "#009E73", "#CC79A7"
GREY, INK = "#BBBBBB", "#222222"


def plane_cmap(n):
    return [plt.get_cmap("cividis")(i / max(1, n - 1)) for i in range(n)]


def cell_grid(ax, nx, ny, colors=None, lw=0.8, ec="white"):
    for j in range(ny):
        for i in range(nx):
            c = colors[j][i] if colors else "none"
            ax.add_patch(Rectangle((i, j), 1, 1, facecolor=c, edgecolor=ec, lw=lw))
    ax.set_xlim(-0.4, nx + 0.4)
    ax.set_ylim(-0.4, ny + 0.4)
    ax.set_aspect("equal")
    ax.axis("off")


# ---------------------------------------------------------------- 1. the hyperplane rule
def fig_wavefront():
    nx = ny = 8
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))

    ax = axes[0]
    cmap = plane_cmap(nx + ny - 1)
    colors = [[cmap[i + j] for i in range(nx)] for j in range(ny)]
    cell_grid(ax, nx, ny, colors)
    for j in range(ny):
        for i in range(nx):
            ax.text(i + 0.5, j + 0.5, str(i + j), ha="center", va="center",
                    fontsize=7, color="white" if (i + j) < 9 else "black")
    ax.set_title("Hyperplane index $h = l_1 + l_2$\n(distance from the upwind corner)",
                 fontsize=10)
    ax.annotate("upwind\ncorner", xy=(0.5, 0.5), xytext=(-0.2, -1.6), fontsize=8,
                ha="center", color=VERM,
                arrowprops=dict(arrowstyle="->", color=VERM, lw=1.5))

    ax = axes[1]
    hl = 6
    colors = [["#F0F0F0" for _ in range(nx)] for _ in range(ny)]
    for j in range(ny):
        for i in range(nx):
            if i + j == hl:
                colors[j][i] = GREEN
            elif i + j < hl:
                colors[j][i] = "#D8D8D8"
    cell_grid(ax, nx, ny, colors)
    # the footpoint stencil of one cell on the active plane
    ci, cj = 3, 3
    for di, dj in ((-1, 0), (0, -1), (-1, -1)):
        ax.add_patch(Rectangle((ci + di, cj + dj), 1, 1, facecolor="none",
                               edgecolor=VERM, lw=2.2, zorder=5))
        ax.add_patch(FancyArrowPatch((ci + di + 0.5, cj + dj + 0.5), (ci + 0.5, cj + 0.5),
                                     arrowstyle="->", color=VERM, lw=1.4,
                                     mutation_scale=11, zorder=6))
    ax.add_patch(Rectangle((ci, cj), 1, 1, facecolor="none", edgecolor=INK, lw=2.2, zorder=5))
    ax.set_title("Plane $h=6$ (green) updates in parallel\nits footpoints (red) all lie on "
                 "$h-1$, $h-2$ — already done", fontsize=10)
    ax.text(0.5, -1.5, "done ($h<6$)", fontsize=8, color="0.35")
    ax.text(5.2, -1.5, "not yet ($h>6$)", fontsize=8, color="0.55")

    fig.suptitle("The rule the whole solver rests on: within a hyperplane, cells are causally "
                 "independent", y=0.99, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "sch_wavefront.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- 2. two-level construction
def fig_twolevel():
    T, tx = 4, 4          # 4x4 tiles of 4x4 cells => a 16x16 block
    n = T * tx
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))

    ax = axes[0]
    cmap = plane_cmap(2 * n - 1)
    colors = [[cmap[i + j] for i in range(n)] for j in range(n)]
    cell_grid(ax, n, n, colors, lw=0.25)
    ax.set_title(f"(a) diagonal / wavefront\none level: {2*n-1} cell-planes over the whole block",
                 fontsize=10)

    ax = axes[1]
    cmapT = plane_cmap(2 * T - 1)
    colors = [[cmapT[(i // tx) + (j // tx)] for i in range(n)] for j in range(n)]
    cell_grid(ax, n, n, colors, lw=0.25)
    for k in range(T + 1):
        ax.axhline(k * tx, color="white", lw=2.5)
        ax.axvline(k * tx, color="white", lw=2.5)
    for tb in range(T):
        for ta in range(T):
            ax.text(ta * tx + tx / 2, tb * tx + tx / 2, str(ta + tb), ha="center", va="center",
                    fontsize=11, color="white" if (ta + tb) < 4 else "black", weight="bold")
    ax.set_title(f"(b) tiled — outer level\n{2*T-1} TILE-planes, "
                 f"one kernel launch each", fontsize=10)

    ax = axes[2]
    colors = [["#F0F0F0" for _ in range(n)] for _ in range(n)]
    for tb in range(T):
        for ta in range(T):
            if ta + tb == 3:
                cm = plane_cmap(2 * tx - 1)
                for j in range(tx):
                    for i in range(tx):
                        colors[tb * tx + j][ta * tx + i] = cm[i + j]
            elif ta + tb < 3:
                for j in range(tx):
                    for i in range(tx):
                        colors[tb * tx + j][ta * tx + i] = "#D8D8D8"
    cell_grid(ax, n, n, colors, lw=0.25)
    for k in range(T + 1):
        ax.axhline(k * tx, color="white", lw=2.5)
        ax.axvline(k * tx, color="white", lw=2.5)
    ax.set_title("(c) tiled — inner level\nwithin each tile of tile-plane 3, the SAME\n"
                 "hyperplane rule, one team per tile", fontsize=10)

    fig.suptitle("The tiled sweep applies one rule at two nested levels — that is the whole idea",
                 y=1.00, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "sch_twolevel.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- 3. correctness
def fig_dependency():
    T = 4
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ta, tb = 2, 2
    colors = [["#F5F5F5" for _ in range(T)] for _ in range(T)]
    for j in range(T):
        for i in range(T):
            if i + j < ta + tb:
                colors[j][i] = "#DCDCDC"
    colors[tb][ta] = GREEN
    for di, dj in ((-1, 0), (0, -1), (-1, -1)):
        colors[tb + dj][ta + di] = VERM
    cell_grid(ax, T, T, colors, lw=1.2, ec="white")
    for j in range(T):
        for i in range(T):
            ax.text(i + 0.5, j + 0.5, f"H={i+j}", ha="center", va="center", fontsize=9,
                    color="white" if (i, j) == (ta, tb) or colors[j][i] == VERM else "0.35")
    ax.set_title("A tile reads only its upwind corner set (red).\nEvery member has a strictly "
                 "smaller H, so it finished\nin an EARLIER LAUNCH — the kernel boundary is the "
                 "barrier.", fontsize=10)
    ax.text(T / 2, -0.95, "green: the tile being swept (H=4)   grey: already done (H<4)",
            ha="center", fontsize=8, color="0.4")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "sch_dependency.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- 4. what gets launched
def fig_launch():
    """Concrete numbers: a single 224^3 meshblock at 168 rays (GH200 N*), the worst case."""
    B, nmb, nang = 224, 1, 168
    tile = 16
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))

    names = ["wavefront", "diagonal_compact", f"tiled (t={tile})"]
    launches = [3 * B - 2, 1, 3 * (B // tile) - 2]
    T = B // tile
    peak_tiles = sum(1 for a in range(T) for b in range(T) for c in range(T)
                     if a + b + c == 3 * (T - 1) // 2)
    # Express all three in the SAME unit -- resident threads -- or the comparison is meaningless:
    # the wavefront has no teams, and a team is ~128 threads, so plotting its threads against the
    # others' teams would overstate it by two orders of magnitude. TEAM_THREADS is what
    # Kokkos::AUTO picks in practice on these devices (see section 6 of TILING_REPORT.md -- the
    # fact that AUTO can pick differently per kernel is exactly the tile_size=0 anomaly).
    TEAM_THREADS = 128
    teams = [None, nmb * nang, peak_tiles * nmb * nang]
    parallel = [B * B * nang // 2, teams[1] * TEAM_THREADS, teams[2] * TEAM_THREADS]
    plabel = ["threads\n(plane cells x rays)",
              f"threads\n({teams[1]:,} teams x {TEAM_THREADS})",
              f"threads\n({teams[2]:,} teams x {TEAM_THREADS})"]

    ax = axes[0]
    bars = ax.bar(names, launches, color=[BLUE, VERM, GREEN], width=0.6)
    ax.set_yscale("log")
    ax.set_ylabel("kernel launches per sweep")
    for b, v in zip(bars, launches):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.15, str(v), ha="center", fontsize=10)
    ax.set_title("Launches", fontsize=10)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    bars = ax.bar(names, parallel, color=[BLUE, VERM, GREEN], width=0.6)
    ax.set_yscale("log")
    ax.set_ylabel(f"concurrent threads (teams x {TEAM_THREADS})")
    for b, v, lb in zip(bars, parallel, plabel):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.2, f"{v:,}\n{lb}", ha="center", fontsize=8)
    ax.set_title("Parallelism exposed", fontsize=10)
    ax.grid(alpha=0.3, axis="y")
    ax.set_ylim(top=max(parallel) * 12)

    fig.suptitle(f"Single {B}³ meshblock, {nang} rays — the configuration where the diagonal "
                 f"starves\n(tiling exposes {parallel[2]//parallel[1]}× the diagonal's threads, "
                 f"at {launches[0]//launches[2]}× fewer launches than the wavefront)",
                 y=1.02, fontsize=10.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "sch_launch.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  single {B}^3 block, {nang} rays: wavefront {launches[0]} launches / "
          f"diagonal {parallel[1]} teams / tiled {launches[2]} launches, {parallel[2]} teams")


if __name__ == "__main__":
    fig_wavefront()
    fig_twolevel()
    fig_dependency()
    fig_launch()
    print(f"schematics -> {OUT}")
