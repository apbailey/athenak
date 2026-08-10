#!/usr/bin/env python3
"""Measure the wavefront SC-sweep's inter-plane launch-bubble prize (nsys, fence-free).

WHY: the earlier rt-profiling study reported the wavefront's radiation throughput dropping
~34% from 16^3 to 176^3 meshblocks. That number comes from the perf-probe run, which fences
after EVERY kernel (src/utils/perf.cpp KernelsProbe::OnEnd) -- 526 fully-exposed fences/cycle
at 176^3 vs 46 at 16^3 -- and excludes radiation bvals. The CLEAN whole-run radiation cost
(t_on - t_off) is U-shaped, not monotonic. So before building a fused single-launch sweep
("Lever 1"), this measures the ONLY thing such a fusion can recover: the wall-time spent in
inter-plane launch/relaunch/ramp GAPS during a normal (unfenced, async) run.

metric, per steady-state cycle, for the sweep kernel:
    active       = sum of kernel durations
    span         = last_kernel_end - first_kernel_start
    gap_fraction = (span - active) / span      # recoverable-by-fusion upper bound

DECISION (see rt-profiling/REPORT.md / plan): gap_fraction > ~10% -> a fused sweep is worth
building (Kokkos Graph first, cooperative-groups only if needed); < ~5% -> retire Lever 1 and
pivot to the latency-bound per-cell gather / diagonal team-tuning.

The deck is the SAME LTE recipe as run_rad_cost (eps=1, ops=0 => use_ali=false, iter_max=
itermin=1 => one formal solution/cycle) but with NO <output file_type=perf> block, so par_for
stays async (no fences) -- production-like. No solver code is changed or rebuilt.

Env:
  ATHENAK_BUILD    dir containing 'athena'      (default build/src)
  ATHENAK_LAUNCHER e.g. 'srun -n 1'             (default none)
  ATHENAK_DEVICE   output-dir tag               (default cpu)
  RT_OUT           output root                  (default <here>/../<device>)
  RT_NSYS_NLIM     cycles per run               (default 10)
  RT_NSYS_CONFIGS  "B:N:nmu,..." override       (default the 176^3/88^3/16^3 nmu=6 set + 176^3 nmu=3)
  RT_NSYS_MATCH    sweep-kernel name regex      (default FormalSolutionWavefront|sc_sweep3d)
  RT_SMOKE=1       skip nsys; unit-check the parser on synthetic data (CPU-safe, no GPU)

Run (Apollo, after a CUDA build on the login node):
  cd $ATHENAK && ATHENAK_BUILD=build_a100/src ATHENAK_LAUNCHER='srun -n 1' ATHENAK_DEVICE=a100 \
  python3 rt-profiling/launch_overhead/measure_nsys.py
"""
import os
import re
import sys
import csv
import shutil
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
# import the LTE deck templates (single source of truth) from the rad_cost harness
RAD_COST = os.path.normpath(os.path.join(HERE, "..", "..", "tst", "benchmark", "rad_cost"))
sys.path.insert(0, RAD_COST)
import run_rad_cost as rc  # noqa: E402  (DECK, RAD_BLOCK -- NOT PERF_BLOCK: fence-free)

BUILD = os.environ.get("ATHENAK_BUILD", "build/src")
LAUNCHER = os.environ.get("ATHENAK_LAUNCHER", "").split()
DEVICE = os.environ.get("ATHENAK_DEVICE", "cpu")
BIN = os.path.abspath(os.path.join(BUILD, "athena"))
OUT = os.environ.get("RT_OUT", os.path.normpath(os.path.join(HERE, "..", DEVICE)))
LO_OUT = os.path.join(OUT, "launch_overhead")
NLIM = int(os.environ.get("RT_NSYS_NLIM", "10"))
SMOKE = os.environ.get("RT_SMOKE", "") == "1"
# default: fence-free sweep-kernel symbol. Without the Kokkos NVTX connector the par_for label
# "sc_sweep3d" is absent; the demangled enclosing method "FormalSolutionWavefront" is present
# (see handover/apollo-access.md). Match either.
MATCH = re.compile(os.environ.get("RT_NSYS_MATCH", r"FormalSolutionWavefront|sc_sweep3d"))


def rays_total(nmu):
    return 8 * nmu * (nmu + 1) // 2       # 3D Bruls type-A: {8,24,48,80,120,168}


def parse_configs():
    """List of dict(B,N,nmu). Mesh fixed at N*=176^3; sweep the block ladder + one angle check."""
    env = os.environ.get("RT_NSYS_CONFIGS")
    if env:
        out = []
        for tok in env.split(","):
            b, n, nmu = tok.split(":")
            out.append(dict(B=int(b), N=int(n), nmu=int(nmu)))
        return out
    return [
        dict(B=176, N=176, nmu=6),   # single block, 526 planes/cycle -- the extreme (primary)
        dict(B=88,  N=176, nmu=6),   # nmb=8,  262 planes/cycle -- trend
        dict(B=16,  N=176, nmu=6),   # nmb=1331, 46 planes/cycle -- many-block anchor (gap~0)
        dict(B=176, N=176, nmu=3),   # angle check: gap_fraction should be ~angle-independent
    ]


# ---------------------------------------------------------------------------
# nsys invocation (fence-free deck; profile CUDA timeline; export per-launch trace to CSV)
# ---------------------------------------------------------------------------
def make_deck(cfg):
    d = os.path.join(LO_OUT, f"B{cfg['B']}_N{cfg['N']}_nmu{cfg['nmu']}")
    os.makedirs(d, exist_ok=True)
    # DECK + RAD_BLOCK only -- deliberately NO PERF_BLOCK, so no per-kernel Kokkos::fence().
    text = rc.DECK.format(base=os.path.join(d, "lo"), N=cfg["N"], B=cfg["B"], nlim=NLIM)
    text += rc.RAD_BLOCK.format(nmu=cfg["nmu"], sweep="wavefront")
    deck = os.path.join(d, "deck.athinput")
    with open(deck, "w") as f:
        f.write(text)
    return d, deck


def nsys_profile(d, deck, tag):
    rep = os.path.join(d, f"nsys_{tag}")
    cmd = LAUNCHER + ["nsys", "profile", "--trace=cuda", "--force-overwrite=true",
                      "--sample=none", "-o", rep, BIN, "-i", deck]
    print("  ", " ".join(cmd))
    subprocess.run(cmd, cwd=d, check=False)
    return rep + ".nsys-rep"


def nsys_trace_csv(rep, d, tag):
    """Export the CUDA GPU trace (per-launch Start/Duration/Name) to CSV; return its path.

    Report alias differs across nsys versions: cuda_gpu_trace (>=2023) vs gputrace (older)."""
    base = os.path.join(d, f"trace_{tag}")
    for report in ("cuda_gpu_trace", "gputrace"):
        cmd = ["nsys", "stats", "--report", report, "--format", "csv",
               "--force-export=true", "-o", base, rep]
        print("  ", " ".join(cmd))
        r = subprocess.run(cmd, cwd=d, capture_output=True, text=True, check=False)
        # nsys writes <base>_<report>.csv; some versions honor -o verbatim, others append.
        for cand in (f"{base}_{report}.csv", f"{base}.csv"):
            if os.path.exists(cand):
                return cand
        # fall back: it may have streamed CSV to stdout
        if r.stdout and "Start" in r.stdout:
            cand = f"{base}_{report}.csv"
            with open(cand, "w") as f:
                f.write(r.stdout)
            return cand
    return None


# ---------------------------------------------------------------------------
# trace parsing + gap-fraction (pure; unit-tested under RT_SMOKE=1)
# ---------------------------------------------------------------------------
def read_trace(path):
    """Return list of (start_ns, dur_ns, name) for rows matching the sweep kernel."""
    rows = []
    with open(path, newline="") as f:
        rd = csv.reader(f)
        header = None
        for r in rd:
            if not r:
                continue
            if header is None:
                # find the real header row (it contains a 'Start' and 'Name' column)
                if any(c.strip().startswith("Start") for c in r) and \
                   any(c.strip() == "Name" for c in r):
                    header = [c.strip() for c in r]
                    ci_start = next(i for i, c in enumerate(header) if c.startswith("Start"))
                    ci_dur = next(i for i, c in enumerate(header) if c.startswith("Duration"))
                    ci_name = header.index("Name")
                continue
            if len(r) <= max(ci_start, ci_dur, ci_name):
                continue
            name = r[ci_name].strip()
            if not MATCH.search(name):
                continue
            try:
                rows.append((float(r[ci_start]), float(r[ci_dur]), name))
            except ValueError:
                continue
    rows.sort(key=lambda t: t[0])
    return rows


def split_cycles(rows, gap_factor=20.0):
    """Split sorted sweep launches into per-cycle bursts. A cycle boundary is an inter-launch
    gap far larger than the median (the hydro time-integrator ran between sweeps)."""
    if len(rows) < 2:
        return [rows] if rows else []
    gaps = [rows[i + 1][0] - (rows[i][0] + rows[i][1]) for i in range(len(rows) - 1)]
    sg = sorted(g for g in gaps if g >= 0)
    med = sg[len(sg) // 2] if sg else 0.0
    thresh = max(med * gap_factor, med + 1.0)
    bursts, cur = [], [rows[0]]
    for i, g in enumerate(gaps):
        if g > thresh:
            bursts.append(cur)
            cur = []
        cur.append(rows[i + 1])
    bursts.append(cur)
    return bursts


def cycle_metrics(burst):
    """active/span/gap_fraction (ns) for one cycle's sweep launches."""
    if not burst:
        return None
    active = sum(d for _, d, _ in burst)
    span = (burst[-1][0] + burst[-1][1]) - burst[0][0]
    gap = span - active
    n = len(burst)
    return dict(n_launches=n, active_ns=active, span_ns=span, gap_ns=gap,
                gap_fraction=(gap / span if span > 0 else 0.0),
                mean_interplane_gap_us=(gap / (n - 1) / 1e3 if n > 1 else 0.0))


def analyze_trace(path, expected_planes):
    """Pick a steady-state cycle (median-span burst near the middle) and return its metrics."""
    rows = read_trace(path)
    if not rows:
        return None, "no matching sweep-kernel rows in trace"
    bursts = split_cycles(rows)
    # prefer bursts whose size == expected_planes (a clean full cycle); else the largest bursts.
    full = [b for b in bursts if len(b) == expected_planes]
    pool = full if full else sorted(bursts, key=len, reverse=True)[:max(1, len(bursts) // 2)]
    if not pool:
        return None, "no usable cycle bursts"
    # steady state: skip the first burst (JIT/warmup); take the median-span one from the pool
    if len(pool) > 1:
        pool = pool[1:] if full else pool
    pool_sorted = sorted(pool, key=lambda b: cycle_metrics(b)["span_ns"])
    chosen = pool_sorted[len(pool_sorted) // 2]
    m = cycle_metrics(chosen)
    note = f"{len(bursts)} bursts; sizes~{sorted(set(len(b) for b in bursts))[:6]}; " \
           f"expected {expected_planes}/cycle"
    return m, note


# ---------------------------------------------------------------------------
def blank_row(cfg):
    B, N, nmu = cfg["B"], cfg["N"], cfg["nmu"]
    nmb = (N // B) ** 3
    return dict(device=DEVICE, B=B, N=N, nmb=nmb, zones=N ** 3, nmu=nmu,
                rays_total=rays_total(nmu), n_planes_expected=3 * B - 2,
                nlim=NLIM, cycle_launches="", active_ms="", span_ms="", gap_ms="",
                gap_fraction="", mean_interplane_gap_us="", note="")


def run_config(cfg):
    row = blank_row(cfg)
    d, deck = make_deck(cfg)
    tag = f"B{cfg['B']}_N{cfg['N']}_nmu{cfg['nmu']}"
    if shutil.which("nsys") is None or not os.path.exists(BIN):
        row["note"] = f"nsys or bin missing (deck written: {deck})"
        print(f"  [skip {tag}: nsys/bin unavailable -- deck written for inspection]")
        return row
    rep = nsys_profile(d, deck, tag)
    if not os.path.exists(rep):
        row["note"] = "nsys produced no .nsys-rep"
        return row
    csvp = nsys_trace_csv(rep, d, tag)
    if not csvp:
        row["note"] = "no cuda_gpu_trace CSV export"
        return row
    m, note = analyze_trace(csvp, 3 * cfg["B"] - 2)
    row["note"] = note
    if m:
        row.update(cycle_launches=m["n_launches"],
                   active_ms=round(m["active_ns"] / 1e6, 4),
                   span_ms=round(m["span_ns"] / 1e6, 4),
                   gap_ms=round(m["gap_ns"] / 1e6, 4),
                   gap_fraction=round(m["gap_fraction"], 5),
                   mean_interplane_gap_us=round(m["mean_interplane_gap_us"], 4))
    return row


def write_csv(rows):
    os.makedirs(LO_OUT, exist_ok=True)
    p = os.path.join(LO_OUT, "results_nsys.csv")
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", p, f"({len(rows)} rows)")


# ---------------------------------------------------------------------------
def smoke():
    """CPU unit-check of the parser+metric on synthetic data -- no GPU, no nsys."""
    print("[RT_SMOKE] unit-checking gap-fraction parser on synthetic traces")
    assert rays_total(6) == 168 and rays_total(3) == 48 and rays_total(1) == 8

    # 3 cells (0..2) per axis toy: build one cycle of 5 launches, 100us active each,
    # 25us gap between -> span = 5*100 + 4*25 = 600us; gap = 100us; frac = 1/6.
    burst = []
    t = 0.0
    for i in range(5):
        burst.append((t, 100e3, "FormalSolutionWavefront"))   # ns
        t += 100e3 + 25e3
    m = cycle_metrics(burst)
    assert abs(m["active_ns"] - 500e3) < 1, m
    assert abs(m["span_ns"] - 600e3) < 1, m
    assert abs(m["gap_fraction"] - (100e3 / 600e3)) < 1e-9, m
    assert abs(m["mean_interplane_gap_us"] - 25.0) < 1e-6, m

    # two cycles separated by a big gap (hydro) -> split_cycles finds 2 bursts of 5
    rows = []
    t = 0.0
    for _c in range(2):
        for _i in range(5):
            rows.append((t, 100e3, "sc_sweep3d"))
            t += 100e3 + 25e3
        t += 10e6   # 10 ms hydro gap
    bursts = split_cycles(rows)
    assert len(bursts) == 2 and all(len(b) == 5 for b in bursts), [len(b) for b in bursts]

    # write a synthetic CSV and round-trip read_trace + analyze_trace (expected 5 planes/cycle)
    sample = os.path.join(LO_OUT, "smoke_trace.csv")
    os.makedirs(LO_OUT, exist_ok=True)
    with open(sample, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Start (ns)", "Duration (ns)", "Name"])
        for s, dd, nm in rows:
            w.writerow([f"{s:.1f}", f"{dd:.1f}", nm])
        w.writerow(["0.0", "5000.0", "some_hydro_kernel"])   # must be filtered out
    got = read_trace(sample)
    assert len(got) == 10 and all(MATCH.search(n) for _, _, n in got), len(got)
    m2, note = analyze_trace(sample, expected_planes=5)
    assert m2 and abs(m2["gap_fraction"] - (100e3 / 600e3)) < 1e-9, (m2, note)
    print("[RT_SMOKE] OK -- parser, cycle split, and gap-fraction all correct")


def main():
    if SMOKE:
        smoke()
        return
    os.makedirs(LO_OUT, exist_ok=True)
    print(f"device={DEVICE} bin={BIN} launcher={LAUNCHER or '(none)'} nlim={NLIM}")
    print(f"out={LO_OUT}  match=/{MATCH.pattern}/")
    rows = [run_config(c) for c in parse_configs()]
    write_csv(rows)
    print("\n B    nmb   nmu rays  planes  launches  active_ms  span_ms  gap_ms  gap_fraction")
    for r in rows:
        print(f" {r['B']:<4} {r['nmb']:<5} {r['nmu']:<3} {r['rays_total']:<5} "
              f"{r['n_planes_expected']:<7} {str(r['cycle_launches']):<9} "
              f"{str(r['active_ms']):<10} {str(r['span_ms']):<8} {str(r['gap_ms']):<7} "
              f"{r['gap_fraction']}")


if __name__ == "__main__":
    main()
