"""
performance_driver.py -- run a perf experiment: sweep a deck over configs with measurement on.

The binary writes per-probe time-series files when a deck has an <output file_type=perf> block
(src/utils/perf.hpp). This driver is pure orchestration: for each config it copies the deck, injects
the block, points job/basename at a results subdir, and runs the binary (which writes the raw files
there). It never reads or reformats measurements -- analyze.py does all of that from the directory.

Two front-ends, one runner:
    performance_driver.py <config.py>
        Run a committed Python spec. The spec is a module defining:
            deck    : str                     base athinput (required)
            configs : {label: {param: value}} the sweep -- Python builds it, so covariation and
                                              multi-axis sweeps are just loops (required)
            probes  : list[str]               default: kernels, memory, comms, iteration
            cadence : str                     "final" (default) | "every N" | "dt X"
            filter  : str                     kernel-name glob (e.g. "vet_sweep*")
    performance_driver.py <deck.athinput> [--kernels --memory --comms --iter] [--every N | --dt X]
                   [--filter GLOB] [block/name=v1,v2,... ...] [-o results/<name>]
        Quick one-off: builds a one-config experiment (a comma-valued override still gives a small
        sweep with auto labels). For anything complex, write a spec.

Add --rerun-with-nsys (either front-end) to rerun each config a second time under a LEAN
`nsys profile --stats=true -t cuda,nvtx --sample=none --cpuctxsw=none` (GPU kernels only -- CPU
sampling/OSRT is dropped because its report finalization hangs under mpirun+UCX on Ocelote). Saves
raw/<label>/<label>.nsys-rep (binary, for the Nsight Systems GUI), <label>.nsys.txt (the readable
kernel/API/mem summary --stats=true prints), and <label>.kern.csv (untruncated per-kernel table from
`nsys stats --format csv` on the finished report -- full symbol names the --stats table clips). Trace
only -- no perf probes, so the clean throughput run is unperturbed; a `timeout` guards against a hung
profile stalling the sweep. Needs nsys on PATH; composes with the launcher (mpirun/srun). ncu is the
same pattern but is admin-blocked on Ocelote.

A benchmark is a folder: benchmark/<name>/ with config.py, analyze.py, and one <device>/ subdir per
GPU. Each <device>/ holds that device's submit.sbatch plus its run outputs -- manifest.json +
analysis.txt at top, per-config deck.athinput + raw perf.<probe> files under raw/<label>/. A spec run
takes its name from the folder; output goes to <bench>/<device>/ (a CLI one-off lands in central
benchmark/results/<name>/<device>/). <device> (p100/v100/a100/cpu, auto from nvidia-smi or
$ATHENAK_DEVICE) keeps hardware apart. analyze.py reads one run dir. Run from tst/. Cluster env:
ATHENAK_BUILD, ATHENAK_LAUNCHER, ATHENAK_DEVICE.
"""

import itertools
import json
import os
import platform
import re
import runpy
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import test_suite.testutils as testutils  # noqa: E402

PROBES = ("kernels", "memory", "comms", "iteration")
RESULTS = "benchmark/results"
NSYS_TIMEOUT = 900   # seconds; guard so a hung `nsys` profile skips its config instead of stalling


def device_tag():
    """Hardware tag for this run: $ATHENAK_DEVICE override, else auto-detect the GPU from nvidia-smi
    (p100/v100/a100/...), else 'cpu'. Used to partition results by hardware (<bench>/<device>/)."""
    tag = os.environ.get("ATHENAK_DEVICE", "").strip()
    if tag:
        return tag
    try:
        name = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                       text=True, stderr=subprocess.DEVNULL).splitlines()[0].lower()
    except Exception:
        return "cpu"
    for known in ("a100", "v100", "p100", "h100", "a40", "l40", "rtx"):
        if known in name:
            return known
    return "_".join(name.split()) or "gpu"


def _arch():
    """Kokkos GPU arch from the build's CMakeCache (e.g. PASCAL60), '' if not found."""
    cache = os.path.join(os.path.dirname(testutils.ATHENAK_BUILD.rstrip("/")), "CMakeCache.txt")
    gpu = ("PASCAL", "VOLTA", "AMPERE", "HOPPER", "TURING", "MAXWELL", "ADA")
    try:
        with open(cache) as f:
            on = [ln.split(":", 1)[0].replace("Kokkos_ARCH_", "") for ln in f
                  if ln.startswith("Kokkos_ARCH_") and ":BOOL=ON" in ln]
        return next((a for a in on if any(g in a for g in gpu)), "")
    except Exception:
        return ""


def provenance():
    """Which device / commit / host / launcher produced this experiment -- recorded in the manifest."""
    def _git(a):
        try:
            return subprocess.check_output(["git"] + a, text=True,
                                           stderr=subprocess.DEVNULL).strip()
        except Exception:
            return ""
    return {
        "device": device_tag(),
        "arch": _arch(),
        "commit": _git(["rev-parse", "--short", "HEAD"]),
        "dirty": "1" if _git(["status", "--porcelain"]) else "0",
        "host": platform.node(),
        "launcher": os.environ.get("ATHENAK_LAUNCHER", ""),
        "build_dir": testutils.ATHENAK_BUILD,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def cadence_line(cadence):
    """Translate a spec cadence -> the perf <output> block's trigger line. 'final' never fires
    during the run (dt huge), so only the end-of-run Finalize writes -> one snapshot."""
    parts = (cadence or "final").split()
    if parts[0] == "every":
        return f"dcycle = {parts[1]}"
    if parts[0] == "dt":
        return f"dt = {parts[1]}"
    return "dt = 1e30"                                    # "final"


def _perf_block(probes, cadence, kfilter):
    lines = ["<output77>", "file_type = perf"]
    if set(probes) != set(PROBES):
        lines.append("perf_probe = " + ",".join(probes))
    if kfilter:
        lines.append("kernels = " + kfilter)
    lines.append(cadence_line(cadence))
    return "\n".join(lines) + "\n"


def _safe(label):
    # labels become subdir names AND appear in the deck path; '=' (and other chars) there would be
    # mis-parsed by athena's ModifyFromCmdline (any argv with '/' and '=' looks like an override).
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(label)).strip("_") or "run"


def run_experiment(name, deck, configs, probes, cadence, kfilter, bench_dir, rerun_nsys=False):
    """Run each labeled config. The run dir holds manifest.json + analysis.txt (the summary) at top and
    per-config outputs in raw/<label>/. For a spec that run dir is <bench>/<device>/ -- beside
    <bench>/<device>/submit.sbatch, so a device's submit script and its results sit together; a CLI
    one-off lands in central results/<name>/<device>/. <device> keeps p100/v100/a100 apart."""
    deck = os.path.abspath(deck)
    prov = provenance()                                   # includes the device tag (drives the path)
    configs = {_safe(k): v for k, v in configs.items()}   # keep labels path/CLI-safe
    # run dir holds manifest.json + analysis.txt (the summary) at top; per-config raw files go in raw/
    rundir = (os.path.join(bench_dir, prov["device"]) if bench_dir
              else os.path.join(RESULTS, name, prov["device"]))
    rawdir = os.path.join(rundir, "raw")
    shutil.rmtree(rawdir, ignore_errors=True)
    os.makedirs(rawdir)
    for stale in ("manifest.json", "analysis.txt"):       # clear the prior run's summary (not submit.sbatch)
        try:
            os.remove(os.path.join(rundir, stale))
        except OSError:
            pass
    with open(deck) as f:
        base_text = f.read()
    block = _perf_block(probes, cadence, kfilter)

    failed = []
    for i, (label, params) in enumerate(configs.items()):
        sub = os.path.join(rawdir, label)
        os.makedirs(sub)
        deckpath = os.path.join(sub, "deck.athinput")
        with open(deckpath, "w") as f:
            f.write(base_text + "\n" + block)
        basename = os.path.abspath(os.path.join(sub, "perf"))
        overrides = [f"{k}={v}" for k, v in params.items()] + [f"job/basename={basename}"]
        print(f"[{i + 1}/{len(configs)}] {label}  "
              + "  ".join(f"{k}={v}" for k, v in params.items()))
        try:
            # binary writes perf.* here; also save its stdout (holds "cpu time used = ..." wall time)
            out = testutils.run_capture(os.path.abspath(deckpath), overrides)
            with open(os.path.join(sub, "run.log"), "w") as lf:
                lf.write(out)
        except Exception:
            # one bad config (e.g. a self-checking test pgen that aborts) shouldn't lose the others
            print(f"  !! '{label}' failed to run -- skipping (rerun: athena -i {deckpath})")
            failed.append(label)
            continue
        if rerun_nsys:
            # Trace pass: rerun under nsys on the BASE deck (no perf block -> the clean throughput run
            # above is untouched). LEAN trace -- `-t cuda,nvtx --sample=none --cpuctxsw=none`: keeps the
            # per-kernel GPU times (the point) but drops the OSRT/CPU-backtrace sampling whose report
            # finalization HANGS under mpirun+UCX on Ocelote (verified 2026-07-21). --stats=true prints
            # the summary tables to stdout -> saved as the readable <label>.nsys.txt next to the binary
            # <label>.nsys-rep (GUI). `timeout` guards against any future hang stalling the sweep. The
            # profiler goes between the launcher and ./athena: mpirun -np 1 nsys profile ... ./athena.
            nsys_out = os.path.join(sub, label)
            launcher = (["timeout", "-k", "30", str(NSYS_TIMEOUT)] + list(testutils.DEFAULT_LAUNCHER)
                        + ["nsys", "profile", "--force-overwrite=true", "--stats=true",
                           "-t", "cuda,nvtx", "--sample=none", "--cpuctxsw=none", "-o", nsys_out])
            print(f"        nsys -> {label}.nsys-rep + {label}.nsys.txt + {label}.kern.csv")
            ok = False
            try:
                out = testutils.run_capture(deck, overrides, launcher=launcher)
                with open(nsys_out + ".nsys.txt", "w") as f:
                    f.write(out)
                ok = True
            except Exception:
                print(f"  !! nsys rerun failed/timed out for '{label}' (nsys on PATH? check the run log)")
            # Untruncated per-kernel CSV via `nsys stats` on the FINISHED .nsys-rep (no mpirun -> no
            # hang; the --stats table above clips long Kokkos symbols, csv doesn't). The report name
            # drifts across nsys versions (newer `cuda_gpu_kern_sum`, older `gpukernsum`) and nsys
            # EXITS 0 even on an unknown name -- so try both and keep the first that yields a real CSV
            # (a comma line, not just the "Processing ..." progress). Non-fatal if none works.
            if ok and os.path.exists(nsys_out + ".nsys-rep"):
                for report in ("cuda_gpu_kern_sum", "gpukernsum"):
                    try:
                        kc = subprocess.run(
                            ["nsys", "stats", "--report", report, "--format", "csv",
                             nsys_out + ".nsys-rep"], capture_output=True, text=True, timeout=120)
                    except Exception:
                        break
                    csv_text = "\n".join(ln for ln in kc.stdout.splitlines()
                                         if not ln.lstrip().startswith("Processing")).strip()
                    if "," in csv_text:                       # a real table, not just progress/error
                        with open(nsys_out + ".kern.csv", "w") as f:
                            f.write(csv_text + "\n")
                        break

    manifest = {"name": name, "device": prov["device"], "deck": deck, "probes": list(probes),
                "cadence": cadence, "filter": kfilter, "rerun_nsys": rerun_nsys,
                "provenance": prov, "configs": configs}
    with open(os.path.join(rundir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    ok = len(configs) - len(failed)
    print(f"-> {rundir}/  ({ok}/{len(configs)} configs ran, probes: {', '.join(probes)})"
          + (f"   FAILED: {', '.join(failed)}" if failed else ""))


def spec_from_file(path):
    ns = runpy.run_path(path)                            # runs the spec, returns its globals
    if "deck" not in ns or "configs" not in ns:
        print(f"spec '{path}' must define `deck` and `configs`")
        sys.exit(1)
    bench_dir = os.path.dirname(os.path.abspath(path))    # a benchmark is a folder: <bench>/config.py
    return {
        "name": os.path.basename(bench_dir),              # experiment name = the folder (not "config")
        "deck": ns["deck"],
        "configs": {str(k): {kk: vv for kk, vv in v.items()} for k, v in ns["configs"].items()},
        "probes": list(ns.get("probes", PROBES)),
        "cadence": ns.get("cadence", "final"),
        "kfilter": ns.get("filter", ""),
        "bench_dir": bench_dir,                            # results -> <bench>/<device>/results/
    }


def spec_from_cli(argv):
    deck = argv[0]
    flagmap = {"--kernels": "kernels", "--memory": "memory",
               "--comms": "comms", "--iter": "iteration"}
    probes, cadence, kfilter, overrides, name = [], "final", "", [], None
    i = 1
    while i < len(argv):
        a = argv[i]
        if a in flagmap:
            probes.append(flagmap[a])
        elif a == "--every":
            i += 1; cadence = f"every {argv[i]}"
        elif a == "--dt":
            i += 1; cadence = f"dt {argv[i]}"
        elif a == "--filter":
            i += 1; kfilter = argv[i]
        elif a in ("-o", "--out", "--name"):
            i += 1; name = os.path.basename(argv[i])
        elif "=" in a:
            overrides.append(a)
        else:
            print(f"unknown arg '{a}'")
            sys.exit(1)
        i += 1
    # comma-expand overrides -> configs, auto-labelled by the swept params
    axes = [(k, v.split(",")) for k, v in (o.split("=", 1) for o in overrides)]
    swept = [k for k, vs in axes if len(vs) > 1]
    configs = {}
    for combo in itertools.product(*[vs for _, vs in axes]) if axes else [()]:
        params = dict(zip([k for k, _ in axes], combo))
        label = "_".join(str(params[k]) for k in swept) or "run"   # values only -> path/CLI-safe
        configs[label] = params
    name = name or os.path.splitext(os.path.basename(deck))[0]
    return {
        "name": name,
        "deck": deck, "configs": configs,
        "probes": probes or list(PROBES), "cadence": cadence, "kfilter": kfilter,
        "bench_dir": None,                                # CLI one-off: central results/<name>/<device>/
    }


def main():
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0 if argv else 1)
    rerun_nsys = "--rerun-with-nsys" in argv               # global flag; works with a spec or a one-off
    argv = [a for a in argv if a != "--rerun-with-nsys"]
    spec = spec_from_file(argv[0]) if argv[0].endswith(".py") else spec_from_cli(argv)
    run_experiment(spec["name"], spec["deck"], spec["configs"], spec["probes"], spec["cadence"],
                   spec["kfilter"], spec["bench_dir"], rerun_nsys=rerun_nsys)


if __name__ == "__main__":
    main()
