#!/usr/bin/env python3
"""Summarize + plot the controlled hydro-vs-RT sweep (companion to run_sweep.py).

Reads, from a results dir (default rt-profiling/a100):
  results_phase1.csv     -- the memory-fill curve (mesh design)
  results_wavefront.csv  -- Phase-2 sweep, wavefront
  results_diagonal.csv   -- Phase-2 sweep, diagonal   (either sweep file is optional)
  kernel_resources.csv   -- static register / occupancy sidecar (optional)

Prints Markdown tables (paste into REPORT.md) and, if matplotlib is available, writes PNGs:
  memfill.png            -- peak GiB vs mesh N (with the fill ceiling)
  zcps_vs_block.png      -- ZCPS with/without RT vs meshblock size, one line per ray count
  slowdown_vs_rays.png   -- RT slowdown (off/on) vs number of rays
  sweep_crossover.png    -- wavefront vs diagonal ZCPS vs block size (fixed nmu)

Usage: python3 analyze.py [results_dir]
"""
import os
import sys
import csv
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "a100")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False


def load(name):
    path = os.path.join(OUT, name)
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def g(unit=""):
    return lambda v: ("-" if v is None else f"{v:.3g}{unit}")


# ---- memory-fill (mesh design) ---------------------------------------------------------------
def report_memfill(rows):
    if not rows:
        return None
    print("\n## Phase 1 — memory-fill (mesh design)\n")
    print("| N | zones | peak mem (GiB) | %fill | fits |")
    print("|---|------:|---------------:|------:|:----:|")
    fitting = []
    for r in sorted(rows, key=lambda r: int(r["N"])):
        oom = r.get("oom") in ("True", "1", "true")
        peak = fnum(r["peak_mem_gib"])
        pct = fnum(r["pct_fill"])
        fits = (not oom)
        if fits:
            fitting.append(int(r["N"]))
        print(f"| {r['N']} | {int(r['zones']):,} | {g()(peak)} | {g('%')(pct)} | "
              f"{'yes' if fits else 'OOM'} |")
    n_star = max(fitting) if fitting else None
    print(f"\n**N★ (largest mesh that fills memory) = {n_star}**")
    if HAVE_MPL and any(fnum(r["peak_mem_gib"]) for r in rows):
        xs = [int(r["N"]) for r in rows]
        ys = [fnum(r["peak_mem_gib"]) for r in rows]
        plt.figure(figsize=(6, 4))
        plt.plot(xs, ys, "o-", label="peak device memory")
        if n_star:
            plt.axvline(n_star, ls="--", color="green", label=f"N★={n_star}")
        plt.xlabel("mesh N (cells/side, 16³ blocks)"); plt.ylabel("peak memory (GiB)")
        plt.title("Phase 1 — memory fill"); plt.legend(); plt.grid(alpha=0.3)
        plt.tight_layout(); plt.savefig(os.path.join(OUT, "memfill.png"), dpi=130); plt.close()
        print("wrote memfill.png")
    return n_star


# ---- Phase-2 sweep ---------------------------------------------------------------------------
def report_sweep(rows, sweep):
    if not rows:
        return
    print(f"\n## Phase 2 — {sweep}: ZCPS with/without RT (mesh {rows[0]['N']}³)\n")
    # sanity: OFF ZCPS is nmu-independent -> constant per B
    offs = defaultdict(set)
    for r in rows:
        offs[r["B"]].add(round(fnum(r["zcps_off_med"]) or 0, -4))
    warn = [b for b, s in offs.items() if len(s) > 1]
    if warn:
        print(f"> note: hydro-OFF ZCPS varied across nmu at B={warn} (thermal noise; expected small)\n")
    print("| B (block) | nmb | nmu | rays | ZCPS off | ZCPS on | slowdown | t_rad/t_hyd | "
          "rad cells/s | SM-util% |")
    print("|----------:|----:|----:|-----:|---------:|--------:|---------:|------------:|"
          "-----------:|---------:|")
    for r in sorted(rows, key=lambda r: (int(r["B"]), int(r["nmu"]))):
        print(f"| {r['B']} | {r['nmb']} | {r['nmu']} | {r['rays_total']} | "
              f"{g()(fnum(r['zcps_off_med']))} | {g()(fnum(r['zcps_on_med']))} | "
              f"{g('x')(fnum(r['slowdown_med']))} | {g('x')(fnum(r['rad_over_hydro']))} | "
              f"{g()(fnum(r['rad_cells_per_s']))} | {g()(fnum(r['sm_util_mean']))} |")


def plot_zcps_vs_block(sweeps):
    if not HAVE_MPL:
        return
    for sweep, rows in sweeps.items():
        if not rows:
            continue
        by_nmu = defaultdict(list)
        for r in rows:
            by_nmu[int(r["nmu"])].append(r)
        plt.figure(figsize=(6.5, 4.5))
        for nmu in sorted(by_nmu):
            rs = sorted(by_nmu[nmu], key=lambda r: int(r["B"]))
            xs = [int(r["B"]) for r in rs]
            ys = [fnum(r["zcps_on_med"]) for r in rs]
            plt.plot(xs, ys, "o-", label=f"RT on, {rs[0]['rays_total']} rays")
        # hydro-off (nmu-independent): take from nmu=min
        rs0 = sorted(by_nmu[min(by_nmu)], key=lambda r: int(r["B"]))
        plt.plot([int(r["B"]) for r in rs0], [fnum(r["zcps_off_med"]) for r in rs0],
                 "k--", lw=2, label="hydro only (off)")
        plt.xlabel("meshblock size B (cells/side)"); plt.ylabel("ZCPS")
        plt.title(f"ZCPS vs block size — {sweep}"); plt.legend(fontsize=8); plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT, f"zcps_vs_block_{sweep}.png"), dpi=130); plt.close()
        print(f"wrote zcps_vs_block_{sweep}.png")


def plot_slowdown_vs_rays(sweeps):
    if not HAVE_MPL:
        return
    plt.figure(figsize=(6.5, 4.5))
    for sweep, rows in sweeps.items():
        if not rows:
            continue
        # pick a representative mid block size present in the data
        Bs = sorted(set(int(r["B"]) for r in rows))
        Bmid = Bs[len(Bs) // 2]
        rs = sorted([r for r in rows if int(r["B"]) == Bmid], key=lambda r: int(r["rays_total"]))
        xs = [int(r["rays_total"]) for r in rs]
        ys = [fnum(r["slowdown_med"]) for r in rs]
        plt.plot(xs, ys, "o-", label=f"{sweep} (B={Bmid})")
    plt.xlabel("number of rays (3D total)"); plt.ylabel("slowdown = ZCPS off / on")
    plt.title("RT slowdown vs angular resolution"); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "slowdown_vs_rays.png"), dpi=130); plt.close()
    print("wrote slowdown_vs_rays.png")


def plot_crossover(sweeps):
    if not HAVE_MPL or len([s for s in sweeps.values() if s]) < 2:
        return
    # fixed nmu (median available) -> ZCPS_on vs B for each sweep
    nmus = set()
    for rows in sweeps.values():
        nmus |= set(int(r["nmu"]) for r in rows)
    nmu = sorted(nmus)[len(nmus) // 2] if nmus else 3
    plt.figure(figsize=(6.5, 4.5))
    for sweep, rows in sweeps.items():
        rs = sorted([r for r in rows if int(r["nmu"]) == nmu], key=lambda r: int(r["B"]))
        if rs:
            plt.plot([int(r["B"]) for r in rs], [fnum(r["zcps_on_med"]) for r in rs],
                     "o-", label=sweep)
    plt.xlabel("meshblock size B"); plt.ylabel(f"ZCPS (RT on, nmu={nmu})")
    plt.title("Sweep crossover: wavefront vs diagonal"); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "sweep_crossover.png"), dpi=130); plt.close()
    print("wrote sweep_crossover.png")


def report_kernels(rows):
    if not rows:
        return
    print("\n## Kernel resources (static — register pressure & theoretical occupancy)\n")
    print("| kind | method/kernel | regs | smem (B) | spills | occ@128 | occ@256 |")
    print("|------|---------------|-----:|---------:|:------:|--------:|--------:|")
    for r in rows:
        if r.get("kind") == "other":
            continue
        print(f"| {r['kind']} | {r['kernel']} | {r['regs']} | {r['shared_bytes']} | "
              f"{r['spills']} | {r['occ_bs128']} | {r['occ_bs256']} |")
    print("\n_Achieved occupancy needs Nsight Compute counters (admin-blocked on Apollo)._")


def main():
    print(f"# Analysis of {OUT}\n")
    report_memfill(load("results_phase1.csv"))
    sweeps = {"wavefront": load("results_wavefront.csv"),
              "diagonal": load("results_diagonal.csv")}
    # also accept a combined file
    both = load("results_both.csv")
    if both:
        sweeps["wavefront"] = [r for r in both if r["sweep"] == "wavefront"] or sweeps["wavefront"]
        sweeps["diagonal"] = [r for r in both if r["sweep"] == "diagonal"] or sweeps["diagonal"]
    for sweep, rows in sweeps.items():
        report_sweep(rows, sweep)
    plot_zcps_vs_block(sweeps)
    plot_slowdown_vs_rays(sweeps)
    plot_crossover(sweeps)
    report_kernels(load("kernel_resources.csv"))


if __name__ == "__main__":
    main()
