#!/usr/bin/env python3
"""Summarise + plot rad_cost results. Usage: analyze_rad_cost.py <results.csv> [<results.csv> ...]

Each results.csv (one per device, written by run_rad_cost.py) has a 'device' column, so pass the
A100 and H200 CSVs together to get cross-device tables/plots. Emits text tables to stdout and PNG
plots (saturation curve, angular scaling, sweep-mode + device comparison) next to the first CSV.
"""
import csv
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(paths):
    rows = []
    for p in paths:
        for r in csv.DictReader(open(p)):
            for k in ("B", "nmb", "zones", "nmu", "nang", "niter", "zcps_off", "zcps_on",
                      "slowdown", "t_rad_ms", "t_hydro_ms", "t_sweep_ms", "rad_over_hydro",
                      "sweep_over_hydro", "rad_cells_per_s", "hydro_cells_per_s", "gcaups"):
                try:
                    r[k] = float(r[k]) if r.get(k) not in (None, "", "None") else None
                except ValueError:
                    r[k] = None
            rows.append(r)
    return rows


def tbl(rows, cols, keyf=None):
    if keyf:
        rows = sorted(rows, key=keyf)
    w = {c: max(len(c), *(len(fmt(r.get(c))) for r in rows)) for c in cols}
    print("  ".join(c.ljust(w[c]) for c in cols))
    for r in rows:
        print("  ".join(fmt(r.get(c)).ljust(w[c]) for c in cols))


def fmt(v):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.3g}" if abs(v) < 1e4 else f"{v:.3e}"
    return str(v)


def main(paths):
    rows = load(paths)
    devs = sorted({r["device"] for r in rows})
    outdir = os.path.dirname(os.path.abspath(paths[0]))
    COLS = ["device", "suite", "B", "nmb", "zones", "nmu", "nang", "sweep",
            "zcps_off", "zcps_on", "slowdown", "rad_over_hydro", "sweep_over_hydro",
            "rad_cells_per_s", "gcaups"]

    for suite in ("sat", "bsize", "ang", "val"):
        s = [r for r in rows if r["suite"] == suite]
        if not s:
            continue
        print(f"\n===== suite {suite} =====")
        tbl(s, COLS, keyf=lambda r: (r["device"], r["sweep"], r["nmb"], r["B"], r["nmu"]))

    # ---- plots ----
    # (1) saturation: radiation completed-cell throughput vs total zones
    sat = [r for r in rows if r["suite"] == "sat"]
    if sat:
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        for dev in devs:
            for sw in ("wavefront", "diagonal"):
                d = sorted([r for r in sat if r["device"] == dev and r["sweep"] == sw],
                           key=lambda r: r["zones"])
                if not d:
                    continue
                ax[0].plot([r["zones"] for r in d], [r["rad_cells_per_s"] for r in d],
                           "o-", label=f"{dev} {sw}")
                ax[1].plot([r["zones"] for r in d], [r["slowdown"] for r in d], "o-",
                           label=f"{dev} {sw}")
        ax[0].set_xscale("log"); ax[0].set_yscale("log")
        ax[0].set_xlabel("total zones"); ax[0].set_ylabel("radiation completed cells/s")
        ax[0].set_title("Saturation: radiation throughput"); ax[0].grid(alpha=.3); ax[0].legend(fontsize=8)
        ax[1].set_xscale("log")
        ax[1].set_xlabel("total zones"); ax[1].set_ylabel("ZCPS slowdown (off/on)")
        ax[1].set_title("Radiation slowdown vs load"); ax[1].grid(alpha=.3); ax[1].legend(fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(outdir, "rc_saturation.png"), dpi=140)
        plt.close(fig)

    # (2) angular: t_rad/t_hydro and slowdown vs nmu
    ang = [r for r in rows if r["suite"] == "ang"]
    if ang:
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        for dev in devs:
            for sw in ("wavefront", "diagonal"):
                d = sorted([r for r in ang if r["device"] == dev and r["sweep"] == sw],
                           key=lambda r: r["nmu"])
                if not d:
                    continue
                ax[0].plot([r["nmu"] for r in d], [r["rad_over_hydro"] for r in d], "o-",
                           label=f"{dev} {sw}")
                ax[1].plot([r["nmu"] for r in d], [r["slowdown"] for r in d], "o-",
                           label=f"{dev} {sw}")
        for a, t, yl in ((ax[0], "radiation/hydro kernel time", "t(rad)/t(hydro)"),
                         (ax[1], "ZCPS slowdown", "off/on")):
            a.set_xlabel("nmu"); a.set_ylabel(yl); a.set_title(t); a.grid(alpha=.3); a.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(outdir, "rc_angular.png"), dpi=140)
        plt.close(fig)

    print(f"\nplots -> {outdir}/rc_saturation.png, rc_angular.png")


if __name__ == "__main__":
    main(sys.argv[1:] or ["results.csv"])
