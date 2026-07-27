"""
analyze.py -- jacobi vs wavefront vs diagonal SC sweep: kernel throughput AND full wall-time.

Reads one performance_driver.py run directory (benchmark/jacobi/<device>/) and writes two verdicts to
analysis.txt:
    manifest.json               configs + provenance (device/commit/host)
    raw/<label>/perf.kernels     per-kernel cumulative time; sum of vet_sweep* = sweep-kernel time
    raw/<label>/perf.iteration   cum_niter, cells, nblocks, nang -- the work done
    raw/<label>/run.log          the run's stdout; "cpu time used = X" is the full driver wall time

For each config it forms the SC work = cum_niter * cells * nblocks * nang and two throughputs:
    kernel gcaups = work / sweep_kernel_seconds / 1e9    (pure vet_sweep* kernel cost, device-only)
    wall   gcaups = work / cpu_time_used     / 1e9        (full run: sweeps + exchange + ComputeJ +
                                                            residual + launches + host<->dev syncs)
then compares the three modes at each MEASURED problem size, one verdict per size. The gap between the
two metrics is the per-iteration overhead each mode carries -- the point of the jacobi comparison:
jacobi's flat par_for should win kernel throughput, but does overhead erode it in wall time?

    cd tst && python benchmark/jacobi/analyze.py benchmark/jacobi/<device>       # e.g. p100/

Caveat: kernel time is device-only (fenced vet_sweep*); it hides host plane-loop overhead of the
wavefront traversal (small-size bias toward wavefront). The WALL metric is host-inclusive, so a
wavefront kernel win that doesn't survive in the wall table is exactly that overhead showing up.
"""

import json
import os
import re
import sys


def read_probe(path):
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


def read_walltime(path):
    """Parse 'cpu time used = X' (the driver-loop wall time) from a saved run.log; None if absent."""
    if not os.path.exists(path):
        return None
    m = None
    with open(path) as f:
        for line in f:
            hit = re.search(r"cpu time used\s*=\s*([0-9.eE+-]+)", line)
            if hit:
                m = float(hit.group(1))          # last one wins (there is only one)
    return m


def snap_cycles(rows):
    return sorted({int(float(r["cycle"])) for r in rows if r.get("cycle") not in (None, "-")})


def at_cycle(rows, c):
    return [r for r in rows if r.get("cycle") not in (None, "-") and int(float(r["cycle"])) == c]


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _sum_sweep(kern_rows):
    return sum(_f(r.get("total_ms")) or 0.0 for r in kern_rows
               if r.get("kernel", "").startswith("vet_sweep"))


def load(run_dir):
    with open(os.path.join(run_dir, "manifest.json")) as f:
        man = json.load(f)
    configs = {}
    for label in man["configs"]:
        d = os.path.join(run_dir, "raw", label)
        configs[label] = {
            "params": man["configs"][label],
            "kernels": read_probe(os.path.join(d, "perf.kernels")),
            "iteration": read_probe(os.path.join(d, "perf.iteration")),
            "wall_s": read_walltime(os.path.join(d, "run.log")),
        }
    return man, configs


def metrics(cfg):
    """(work|None, kernel_gcaups|None, wall_gcaups|None) for one config, main-loop only.

    Probes are cumulative: subtract the cycle-0 (post-pgen) snapshot from the last to isolate the
    sweeps driven through the main loop. Kernel time subtracts likewise; wall time ("cpu time used")
    is already the driver-loop wall clock (setup excluded), so it is used as-is.
    """
    kc = snap_cycles(cfg["kernels"])
    if not kc:
        return None, None, None
    sms = _sum_sweep(at_cycle(cfg["kernels"], kc[-1]))
    if len(kc) >= 2:
        sms -= _sum_sweep(at_cycle(cfg["kernels"], kc[0]))

    ic = snap_cycles(cfg["iteration"])
    if not ic:
        return None, None, None
    last = at_cycle(cfg["iteration"], ic[-1])[0]
    base = at_cycle(cfg["iteration"], ic[0])[0] if len(ic) >= 2 else {}
    niter = (_f(last.get("cum_niter")) or 0.0) - (_f(base.get("cum_niter")) or 0.0)
    cells, nblocks, nang = (_f(last.get(k)) for k in ("cells", "nblocks", "nang"))
    if niter <= 0 or None in (cells, nblocks, nang):
        return None, None, None
    work = {"cum_niter": niter, "cells": cells, "nblocks": nblocks, "nang": nang}
    caups = niter * cells * nblocks * nang
    kern_g = caups / (sms / 1e3) / 1e9 if sms > 0 else None
    wall_s = cfg["wall_s"]
    wall_g = caups / wall_s / 1e9 if wall_s and wall_s > 0 else None
    return work, kern_g, wall_g


_METHOD_ORDER = ["wavefront", "diagonal", "jacobi"]


def _method_cols(seen):
    return [m for m in _METHOD_ORDER if m in seen] + sorted(m for m in seen if m not in _METHOD_ORDER)


def _blockdim(params, cells):
    try:
        d = [int(params[f"meshblock/nx{i}"]) for i in (1, 2, 3)]
        return f"{d[0]}^3" if d[0] == d[1] == d[2] else f"{d[0]}x{d[1]}x{d[2]}"
    except (KeyError, TypeError, ValueError):
        n = round(cells ** (1.0 / 3.0))
        return f"{n}^3" if n ** 3 == cells else str(cells)


def _table(title, groups, first_idx, cols):
    """One gcaups table (higher = faster): a row per problem size, a column per mode, winner+ratio."""
    out = [title,
           f"  {'block':>8}  {'nblocks':>7}  {'nmu':>3}"
           + "".join(f"  {c:>10}" for c in cols) + f"  {'winner':>9}  {'xfaster':>7}"]
    for key in sorted(groups, key=lambda k: first_idx[k]):
        cells, nb, nang = key
        grp = groups[key]
        params = next(iter(grp.values()))[1]
        nmu = params.get("nr_radiation/nmu", "?")
        row = f"  {_blockdim(params, cells):>8}  {nb:>7}  {str(nmu):>3}"
        gv = {}
        for c in cols:
            g = grp[c][0] if c in grp else None
            row += f"  {g:>10.4g}" if g is not None else f"  {'-':>10}"
            if g is not None:
                gv[c] = g
        if len(gv) > 1:
            winner = max(gv, key=gv.get)
            row += f"  {winner:>9}  {gv[winner] / min(gv.values()):>6.2f}x"
        elif gv:
            row += f"  {next(iter(gv)):>9}  {'-':>7}"
        else:
            row += f"  {'-':>9}  {'-':>7}"
        out.append(row)
    return out


def report(man, configs):
    order_of = {label: i for i, label in enumerate(man["configs"])}
    kern_groups, wall_groups, first_idx, seen, skipped = {}, {}, {}, set(), []
    for label, cfg in configs.items():
        work, kg, wg = metrics(cfg)
        if work is None:
            skipped.append(label)
            continue
        method = cfg["params"].get("nr_radiation/sweep", label)
        seen.add(method)
        key = (int(work["cells"]), int(work["nblocks"]), int(work["nang"]))
        if kg is not None:
            kern_groups.setdefault(key, {})[method] = (kg, cfg["params"])
        if wg is not None:
            wall_groups.setdefault(key, {})[method] = (wg, cfg["params"])
        first_idx[key] = min(first_idx.get(key, len(order_of)), order_of.get(label, len(order_of)))
    if not kern_groups and not wall_groups:
        return "no config had sweep timing + iteration data to compare"

    cols = _method_cols(seen)
    prov = man.get("provenance", {})
    tag = f"commit {prov.get('commit', '?')}" + (f" on {prov['host']}" if prov.get("host") else "")

    out = [f"jacobi vs wavefront vs diagonal   {man.get('name', '?')}   ({tag})",
           "gcaups = cum_niter*cells*nblocks*nang / seconds  (higher = faster)", ""]
    out += _table("KERNEL throughput  (fenced vet_sweep* time -- pure sweep kernel, device-only):",
                  kern_groups, first_idx, cols)
    out += ["", ""]
    out += _table("WALL throughput  (full run 'cpu time used' -- sweeps + all per-iteration overhead):",
                  wall_groups, first_idx, cols)

    if not any(cfg["wall_s"] for cfg in configs.values()):
        out += ["", "  !! no wall times found (run.log missing 'cpu time used' -- old driver?); "
                "WALL table empty."]
    if skipped:
        out.append(f"\n  (skipped, no sweep+iteration data: {', '.join(skipped)})")
    out += ["", "notes:",
            "  - KERNEL is device-only (fenced vet_sweep*); it hides host plane-loop overhead of the",
            "    wavefront traversal (small-size bias toward wavefront). WALL is host-inclusive.",
            "  - kernel_gcaps >> wall_gcaps means large per-iteration overhead (launches/exchange/",
            "    ComputeJ/residual) around a cheap sweep -- watch this for jacobi's flat par_for."]
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
