#!/usr/bin/env python3
"""Static per-kernel register/shared-mem/spill usage + theoretical occupancy (no admin needed).

Secondary sidecar to run_sweep.py.  The *achieved* (runtime) occupancy and warp-execution
efficiency need Nsight Compute hardware counters, which are admin-blocked on Apollo
(NVreg_RestrictProfilingToAdminUsers=1).  This script gets the *static* subset that needs neither
admin nor a rebuild:

  * `cuobjdump --dump-resource-usage <athena>` reads registers/thread, shared-mem, and stack
    (spill) per compiled kernel straight from the existing binary.
  * theoretical occupancy is then computed from regs + shared-mem vs the A100 (SM80) limits, for a
    bracket of launch block sizes (Kokkos picks the block size at launch; the wavefront sweep is a
    flat par_for, the diagonal a TeamPolicy with Kokkos::AUTO).

Kernel symbols are the Kokkos launch wrappers templated on the functor (lambda) type; the enclosing
C++ method name is embedded in the *demangled* name (that is why the sweep is grep-able as
"FormalSolutionWavefront").  We demangle with c++filt and tag kernels by enclosing method:
  SC sweep : FormalSolutionWavefront | FormalSolutionDiagonal | FormalSolutionJacobi | SweepUpdateGS
  hydro    : RKUpdate (h_update) | CalculateFluxes (hflux_*)

Off a GPU/CUDA box (no cuobjdump) it prints the command it would run and exits 0, so it is
verifiable before submitting to Apollo.

Env: ATHENAK_BUILD (dir with 'athena'; default build/src), RT_OUT (output dir; default <here>/<dev>),
     ATHENAK_DEVICE (tag; default cpu), KR_ARCH (sm80|sm90; default sm80 = A100).
"""
import os
import re
import csv
import math
import shutil
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = os.environ.get("ATHENAK_BUILD", "build/src")
BIN = os.path.abspath(os.path.join(BUILD, "athena"))
DEVICE = os.environ.get("ATHENAK_DEVICE", "cpu")
OUT = os.environ.get("RT_OUT", os.path.join(HERE, DEVICE))
ARCH = os.environ.get("KR_ARCH", "sm80")

# tag patterns (substrings expected in the demangled kernel name)
SWEEP_PATTERNS = ["FormalSolutionWavefront", "FormalSolutionDiagonal",
                  "FormalSolutionJacobi", "SweepUpdateGS"]
HYDRO_PATTERNS = ["RKUpdate", "CalculateFluxes"]

# GPU limits per SM (occupancy model)
LIMITS = {
    # regs_per_sm, max_warps, max_blocks, smem_per_sm(bytes), reg_alloc_unit, warp_alloc_gran
    "sm80": dict(regs=65536, warps=64, blocks=32, smem=164 * 1024, reg_unit=256, warp_gran=4),
    "sm90": dict(regs=65536, warps=64, blocks=32, smem=228 * 1024, reg_unit=256, warp_gran=4),
}
BLOCK_SIZES = (128, 256, 512)


def occupancy(regs_per_thread, smem_per_block, threads_per_block, arch="sm80"):
    """Theoretical occupancy fraction for a kernel launched with the given block config."""
    L = LIMITS[arch]
    if regs_per_thread <= 0:
        regs_per_thread = 1
    warps_per_block = math.ceil(threads_per_block / 32)
    # register limit -> warps, then blocks
    regs_per_warp = math.ceil(regs_per_thread * 32 / L["reg_unit"]) * L["reg_unit"]
    warps_by_reg = (L["regs"] // regs_per_warp)
    warps_by_reg -= warps_by_reg % L["warp_gran"]
    warps_by_reg = min(warps_by_reg, L["warps"])
    blocks_by_reg = warps_by_reg // warps_per_block if warps_per_block else 0
    # shared-mem limit
    blocks_by_smem = (L["smem"] // smem_per_block) if smem_per_block > 0 else L["blocks"]
    # thread + hard block limits
    blocks_by_threads = L["warps"] // warps_per_block if warps_per_block else 0
    active_blocks = min(blocks_by_reg, blocks_by_smem, L["blocks"], blocks_by_threads)
    active_warps = active_blocks * warps_per_block
    return active_warps / L["warps"]


def parse_resource_usage(text):
    """Yield (mangled_symbol, regs, shared_bytes, stack_bytes) from cuobjdump output."""
    # blocks start at "Function <sym>:" and carry REG:/SHARED:/STACK: tokens
    parts = re.split(r"\n\s*Function\s+", "\n" + text)
    for blk in parts[1:]:
        head, _, rest = blk.partition(":")
        sym = head.strip()
        body = rest
        reg = _int(re.search(r"\bREG:(\d+)", body))
        shared = _int(re.search(r"\bSHARED:(\d+)", body))
        stack = _int(re.search(r"\bSTACK:(\d+)", body))
        if reg is not None:
            yield sym, reg, shared or 0, stack or 0


def _int(m):
    return int(m.group(1)) if m else None


def demangle(symbols):
    if not symbols or not shutil.which("c++filt"):
        return {s: s for s in symbols}
    try:
        r = subprocess.run(["c++filt"], input="\n".join(symbols), capture_output=True,
                           text=True, timeout=60, check=False)
        out = r.stdout.splitlines()
        return {s: (out[i] if i < len(out) else s) for i, s in enumerate(symbols)}
    except Exception:
        return {s: s for s in symbols}


def classify(name):
    for p in SWEEP_PATTERNS:
        if p in name:
            return "sweep", p
    for p in HYDRO_PATTERNS:
        if p in name:
            return "hydro", p
    return "other", ""


def main():
    os.makedirs(OUT, exist_ok=True)
    cmd = ["cuobjdump", "--dump-resource-usage", BIN]
    if not shutil.which("cuobjdump"):
        print("cuobjdump not on PATH (need the CUDA toolkit: `module load cuda`).")
        print("Would run:  " + " ".join(cmd))
        print("On Apollo:  module load cuda && python3 rt-profiling/kernel_resources.py")
        return
    if not os.path.exists(BIN):
        print(f"binary not found: {BIN} (set ATHENAK_BUILD)"); return
    r = subprocess.run(cmd, capture_output=True, text=True, check=False)
    raw = (r.stdout or "") + (r.stderr or "")
    entries = list(parse_resource_usage(raw))
    names = demangle([s for s, *_ in entries])
    rows = []
    for sym, reg, shared, stack in entries:
        dem = names.get(sym, sym)
        kind, method = classify(dem)
        row = dict(device=DEVICE, arch=ARCH, kind=kind, method=method, regs=reg,
                   shared_bytes=shared, stack_bytes=stack, spills=("yes" if stack > 0 else "no"))
        for bs in BLOCK_SIZES:
            row[f"occ_bs{bs}"] = round(occupancy(reg, shared, bs, ARCH), 3)
        # a short readable label: enclosing method if tagged, else a trimmed demangled name
        row["kernel"] = method or (dem[:80] + ("..." if len(dem) > 80 else ""))
        rows.append(row)
    # sweep + hydro first, then by registers desc
    rows.sort(key=lambda x: (x["kind"] == "other", -x["regs"]))
    path = os.path.join(OUT, "kernel_resources.csv")
    if rows:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("wrote", path, f"({len(rows)} kernels)")
        print(f"\n{'kind':6} {'method/kernel':32} {'regs':>4} {'smem':>6} {'spill':>5} "
              f"{'occ128':>6} {'occ256':>6}")
        for x in rows:
            if x["kind"] == "other":
                continue
            print(f"{x['kind']:6} {str(x['kernel'])[:32]:32} {x['regs']:>4} "
                  f"{x['shared_bytes']:>6} {x['spills']:>5} {x['occ_bs128']:>6} {x['occ_bs256']:>6}")
        print("\nNote: theoretical (static-sense) occupancy; achieved occupancy needs ncu "
              "(admin-blocked on Apollo).")
    else:
        print("no kernels parsed from cuobjdump output (unexpected)")


if __name__ == "__main__":
    main()
