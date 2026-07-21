"""
analyze.py -- wavefront vs diagonal: which VET sweep is faster, and by how much.

Reads one performance_driver.py run directory (e.g. benchmark/compare_sweep/<device>/) and writes
its verdict to analysis.txt there:
    manifest.json              configs + provenance (device/commit/host)
    raw/<label>/perf.kernels    per-kernel cumulative time; sum of vet_sweep* = sweep time
    raw/<label>/perf.iteration  cum_niter, cells, nblocks, nang -- the work done

For each config it derives the SC sweep throughput
    gcaups = cum_niter * cells * nblocks * nang / sweep_seconds / 1e9   (giga cell-angle updates/s)
then compares configs that solved the SAME problem (same cells x nblocks x nang) and reports which
is fastest and by how much. Grouping by the MEASURED problem size keeps a method x size sweep honest
-- one verdict per size, never a big run judged against a small one.

    cd tst && python benchmark/compare_sweep/analyze.py benchmark/compare_sweep/<device>   # e.g. cpu/

Caveat: sweep time is the sum of the fenced vet_sweep* kernel times (device only). It misses the host
plane-loop overhead of the wavefront traversal, which biases this comparison in wavefront's favor at
small problem sizes (~12% at 32^3, ~1% at 128^3). Treat a narrow small-size wavefront win with
suspicion -- it may not survive a host-inclusive timer.
"""

import json
import os
import sys


# ---- load a performance_driver.py results directory ------------------------------------------
def read_probe(path):
    """Parse a wide perf.<probe> file ('# cycle time <cols>' header + rows) into row dicts."""
    if not os.path.exists(path):
        return []
    rows, header = [], None
    with open(path) as f:
        for line in f:
            tok = line.split()
            if not tok:
                continue
            if tok[0] == "#":
                header = tok[1:]
            elif header is not None:
                rows.append(dict(zip(header, tok)))
    return rows


def snap_cycles(rows):
    """Sorted distinct cycle numbers present (probe files are cumulative snapshots, one set per cycle)."""
    return sorted({int(float(r["cycle"])) for r in rows if r.get("cycle") not in (None, "-")})


def at_cycle(rows, c):
    return [r for r in rows if r.get("cycle") not in (None, "-") and int(float(r["cycle"])) == c]


def load(run_dir):
    with open(os.path.join(run_dir, "manifest.json")) as f:
        man = json.load(f)
    configs = {}
    for label in man["configs"]:
        d = os.path.join(run_dir, "raw", label)                            # per-config raw files
        configs[label] = {
            "params": man["configs"][label],                              # incl. nr_radiation/sweep
            "kernels": read_probe(os.path.join(d, "perf.kernels")),        # all snapshots (cumulative)
            "iteration": read_probe(os.path.join(d, "perf.iteration")),
        }
    return man, configs


# ---- derive per-config sweep throughput ------------------------------------------------------
def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _sum_sweep(kern_rows):
    """Sum of the fenced vet_sweep* kernel times in one snapshot."""
    return sum(_f(r.get("total_ms")) or 0.0 for r in kern_rows
               if r.get("kernel", "").startswith("vet_sweep"))


def metrics(cfg):
    """(sweep_ms, work|None, gcaups|None) for one config -- MAIN-LOOP ONLY.

    Probes are cumulative, so the cycle-0 snapshot already carries any sweeps the pgen ran at setup
    (vet_uniform does one FormalSolution for its I==b check). Subtracting the first (baseline)
    snapshot from the last isolates the sweeps driven through the main loop -- the controlled effort
    we mean to time. One snapshot -> nothing to subtract.
    """
    kc = snap_cycles(cfg["kernels"])
    if not kc:
        return 0.0, None, None
    sms = _sum_sweep(at_cycle(cfg["kernels"], kc[-1]))
    if len(kc) >= 2:
        sms -= _sum_sweep(at_cycle(cfg["kernels"], kc[0]))

    work = gcaups = None
    ic = snap_cycles(cfg["iteration"])
    if ic and sms > 0:
        last = at_cycle(cfg["iteration"], ic[-1])[0]
        base = at_cycle(cfg["iteration"], ic[0])[0] if len(ic) >= 2 else {}
        niter = (_f(last.get("cum_niter")) or 0.0) - (_f(base.get("cum_niter")) or 0.0)
        cells, nblocks, nang = (_f(last.get(k)) for k in ("cells", "nblocks", "nang"))
        if niter > 0 and None not in (cells, nblocks, nang):
            work = {"cum_niter": niter, "cells": cells, "nblocks": nblocks, "nang": nang}
            gcaups = niter * cells * nblocks * nang / (sms / 1e3) / 1e9
    return sms, work, gcaups


# ---- compare + report ------------------------------------------------------------------------
_METHOD_ORDER = ["wavefront", "diagonal", "jacobi"]


def _method_cols(seen):
    """Ordered method (column) list: known methods in canonical order, then any extras."""
    return [m for m in _METHOD_ORDER if m in seen] + sorted(m for m in seen if m not in _METHOD_ORDER)


def _blockdim(params, cells):
    """Block dimensions for display: 'n^3' for a cube, else 'axbxc'; cube-root of cells as fallback."""
    try:
        d = [int(params[f"meshblock/nx{i}"]) for i in (1, 2, 3)]
        return f"{d[0]}^3" if d[0] == d[1] == d[2] else f"{d[0]}x{d[1]}x{d[2]}"
    except (KeyError, TypeError, ValueError):
        n = round(cells ** (1.0 / 3.0))
        return f"{n}^3" if n ** 3 == cells else str(cells)


def report(man, configs):
    """One flat table, one row per measured problem size: each method's gcaups + winner + ratio."""
    # Rows are ordered by first appearance in the manifest, so they follow the config's own axis
    # structure (the spec emits one sweep axis at a time) instead of re-sorting by problem size.
    order_of = {label: i for i, label in enumerate(man["configs"])}
    groups, first_idx, methods_seen, skipped = {}, {}, set(), []
    for label, cfg in configs.items():
        sms, work, g = metrics(cfg)
        if sms <= 0 or work is None or g is None:
            skipped.append(label)
            continue
        method = cfg["params"].get("nr_radiation/sweep", label)
        methods_seen.add(method)
        key = (int(work["cells"]), int(work["nblocks"]), int(work["nang"]))
        groups.setdefault(key, {})[method] = (g, sms, int(work["cum_niter"]), cfg["params"])
        first_idx[key] = min(first_idx.get(key, len(order_of)), order_of.get(label, len(order_of)))
    if not groups:
        return "no config had sweep timing + iteration data to compare"

    cols = _method_cols(methods_seen)
    prov = man.get("provenance", {})
    tag = f"commit {prov.get('commit', '?')}" + (f" on {prov['host']}" if prov.get("host") else "")

    out = [f"wavefront vs diagonal   {man.get('name', '?')}   ({tag})",
           "gcaups = cum_niter*cells*nblocks*nang / sweep_s  (higher = faster); one row per problem size",
           "",
           f"  {'block':>8}  {'nblocks':>7}  {'nmu':>3}"
           + "".join(f"  {c:>10}" for c in cols) + f"  {'winner':>9}  {'xfaster':>7}"]

    for cells, nb, nang in sorted(groups, key=lambda k: first_idx[k]):
        grp = groups[(cells, nb, nang)]
        params = next(iter(grp.values()))[3]
        nmu = params.get("nr_radiation/nmu", "?")
        row = f"  {_blockdim(params, cells):>8}  {nb:>7}  {str(nmu):>3}"
        for c in cols:
            row += f"  {grp[c][0]:>10.4g}" if c in grp else f"  {'-':>10}"
        gv = {m: grp[m][0] for m in grp}
        winner = max(gv, key=gv.get)
        if len(gv) > 1:
            row += f"  {winner:>9}  {gv[winner] / min(gv.values()):>6.2f}x"
        else:
            row += f"  {winner:>9}  {'-':>7}"
        out.append(row)

    if [k for k, grp in groups.items() if len({grp[m][2] for m in grp}) > 1]:
        out += ["", "  !! iteration counts differ between methods at some sizes -- the two traversals "
                "should converge",
                "     identically; 'xfaster' there blends per-sweep speed with convergence."]
    if skipped:
        out.append(f"\n  (skipped, no sweep+iteration data: {', '.join(skipped)})")
    out += ["", "note: sweep time is device-only (fenced vet_sweep* kernels); it hides the host "
            "plane-loop overhead of",
            "the wavefront traversal, biasing toward wavefront at small sizes (~12% at 32^3, ~1% at "
            "128^3). A",
            "narrow small-size wavefront win may not survive host timing."]
    return "\n".join(out)


def main():
    if len(sys.argv) != 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0 if len(sys.argv) == 2 else 1)
    run_dir = sys.argv[1]
    if not os.path.isdir(run_dir):
        sys.exit(f"not a run directory: {run_dir}")
    man, configs = load(run_dir)
    text = report(man, configs)
    print(text)
    out = os.path.join(run_dir, "analysis.txt")
    with open(out, "w") as f:
        f.write(text + "\n")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
