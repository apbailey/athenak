#!/usr/bin/env python3
"""White-box GPU profiling for the rt-sc LTE short-characteristics sweep.

Companion to run_rad_cost.py. That script is a BLACK-BOX throughput study (wall-clock
ZCPS + device-fenced kernel times); it establishes the structural story (radiation is
dependency-limited, pinned near a flat ~T/3 ceiling while hydro climbs) but records NO
white-box GPU counters. This script captures exactly those missing counters for ONE
representative *saturated* sweep, to answer the questions the throughput study cannot:

  * Roofline / SpeedOfLight  -> is the saturated sweep MEMORY-bound or COMPUTE-bound?
      (Predicted memory-bound: one 3D cell-update is a ~20-element bilinear footpoint
       gather of FP64 across 4 cache-line streams vs ~1 exp + a store; sc_interp.hpp.)
  * Achieved occupancy       -> confirm/refute the occupancy story (register-limited?).
  * Warp execution efficiency-> tests the ~1/3 over-issue hypothesis: the wavefront
      launches an nx1*nx2 (=B^2) grid per plane and early-returns off-plane slots
      (formal_solution.cpp), so 3B*B^2 slots are issued for B^3 real cells => ~1/3 of
      lanes should be active if that waste dominates. This is the # (i) lever's target.
  * Launch/sync gap timeline -> the ~3B serial plane-chain (nsys): how much wall-time is
      inter-launch gap (structural cost (i)) vs kernel body.

It runs the SAME LTE deck as run_rad_cost (eps=1, ops=0 => use_ali=false, iter_max=
itermin=1 => one formal solution per cycle, no in-meshblock iteration; decoupled
linear-wave background so the sweep does its full value-independent work) at a saturated
geometry, under Nsight Compute (ncu) and Nsight Systems (nsys).

Filtering Kokkos kernels by their par_for LABEL (e.g. "sc_sweep3d") requires the Kokkos
NVTX connector so labels become NVTX ranges that ncu/nsys can include; the sbatch sets
KOKKOS_TOOLS_LIBS to it. Without the connector, set PROF_FILTER=window to fall back to a
launch-index window (-s/-c) with no label filter (the sweep is then the highest-count
Kokkos parallel_for in the report).

If ncu/nsys are absent (e.g. a CPU dev box), this PRINTS the exact commands and writes the
generated deck, then exits 0 -- so the harness is fully verifiable off-GPU before submit.

Env:
  ATHENAK_BUILD   dir containing 'athena' (default build/src)
  ATHENAK_LAUNCHER e.g. 'srun -n 1' (default none)
  ATHENAK_DEVICE  tag for the output dir (default cpu)
  PROFILE_OUT     output dir (default <here>/<device>/profile)
Knobs (all optional; defaults reproduce the report's saturated wavefront point at a
tractable size):
  PROF_B      meshblock size            (default 32)
  PROF_NMB    meshblock count           (default 64;  set 216 for the exact 192^3 headline)
  PROF_NMU    polar angles              (default 3 => 48 angles, the headline)
  PROF_SWEEP  wavefront|diagonal|jacobi (default wavefront)
  PROF_NLIM   cycles                    (default 2)
  PROF_LABEL  Kokkos label to profile   (default derived: sc_sweep{ndim}d / gs_sweepa..)
  PROF_FILTER nvtx|window               (default nvtx; window => -s/-c only, no label match)
  PROF_SKIP   ncu launch-skip           (default centers a window on the LARGEST planes)
  PROF_COUNT  ncu launch-count          (default 24)
  PROF_NCU_SET ncu section set          (default full)

Run (Apollo, after building a CUDA build on the login node):
  cd $ATHENAK/tst && \
  ATHENAK_BUILD=../build_a100/src ATHENAK_LAUNCHER='srun -n 1' ATHENAK_DEVICE=a100 \
  python3 benchmark/rad_cost/profile_sweep.py
"""
import os
import sys
import shutil
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_rad_cost as rc  # single source of truth for the LTE deck (DECK, RAD_BLOCK)

BUILD = os.environ.get("ATHENAK_BUILD", "build/src")
LAUNCHER = os.environ.get("ATHENAK_LAUNCHER", "").split()
DEVICE = os.environ.get("ATHENAK_DEVICE", "cpu")
BIN = os.path.abspath(os.path.join(BUILD, "athena"))
OUT = os.environ.get("PROFILE_OUT", os.path.join(HERE, DEVICE, "profile"))

B = int(os.environ.get("PROF_B", "32"))
NMB = int(os.environ.get("PROF_NMB", "64"))
NMU = int(os.environ.get("PROF_NMU", "3"))
SWEEP = os.environ.get("PROF_SWEEP", "wavefront")
NLIM = int(os.environ.get("PROF_NLIM", "2"))
FILTER = os.environ.get("PROF_FILTER", "nvtx")
NCU_SET = os.environ.get("PROF_NCU_SET", "full")
COUNT = int(os.environ.get("PROF_COUNT", "24"))


def cube_root_int(n):
    r = round(n ** (1.0 / 3.0))
    if r ** 3 != n:
        raise SystemExit(f"PROF_NMB={n} is not a perfect cube (need (N/B)^3 blocks)")
    return r


def default_label(sweep, ndim):
    if sweep == "jacobi":
        return "sc_sweep_jacobi"
    if sweep == "diagonal":
        return "sc_sweep_diag"
    return f"sc_sweep{ndim}d"          # wavefront: one kernel launch per plane


def make_deck():
    side = cube_root_int(NMB)
    N = B * side                       # cubic mesh N^3, N/B = side blocks per dim
    ndim = 3                           # the benchmark problem is always 3D
    os.makedirs(OUT, exist_ok=True)
    base = os.path.join(OUT, "prof")
    text = rc.DECK.format(base=base, N=N, B=B, nlim=NLIM)
    text += rc.RAD_BLOCK.format(nmu=NMU, sweep=SWEEP)
    deck = os.path.join(OUT, "deck.athinput")
    with open(deck, "w") as f:
        f.write(text)
    # The largest hyperplanes sit near h ~ 1.5*B (the middle of the 3B-2 plane chain);
    # center a COUNT-wide window there so the profiled launches are the most saturated,
    # most representative ones. All within cycle 0 (plane count per cycle = 3B-2), so no
    # cross-cycle ambiguity. Override with PROF_SKIP.
    skip = int(os.environ.get("PROF_SKIP", str(max(0, int(1.5 * B) - COUNT // 2))))
    label = os.environ.get("PROF_LABEL", default_label(SWEEP, ndim))
    n_planes = 3 * B - 2
    return deck, N, ndim, skip, label, n_planes


def ncu_cmd(deck, skip, label, tag):
    out = os.path.join(OUT, f"ncu_{tag}")
    cmd = ["ncu", "--set", NCU_SET, "-f", "-o", out,
           "-s", str(skip), "-c", str(COUNT)]
    if FILTER == "nvtx":
        # requires the Kokkos NVTX connector (KOKKOS_TOOLS_LIBS) so the par_for label
        # "<label>" is emitted as an NVTX range ncu can include.
        cmd += ["--nvtx", "--nvtx-include", f"{label}/"]
    # FILTER == window: no label match; -s/-c select a bounded launch window.
    return LAUNCHER + cmd + [BIN, "-i", deck]


def nsys_cmd(deck, tag):
    out = os.path.join(OUT, f"nsys_{tag}")
    cmd = ["nsys", "profile", "-t", "cuda,nvtx", "-o", out,
           "-f", "true", "--stats", "true"]
    return LAUNCHER + cmd + [BIN, "-i", deck]


def run_or_print(name, cmd, tool):
    print(f"\n# --- {name} ---")
    print(" ", " ".join(cmd))
    if shutil.which(tool) is None:
        print(f"  [{tool} not on PATH -> command printed only, not executed]")
        return None
    r = subprocess.run(cmd, cwd=OUT, check=False)
    print(f"  [{tool} exit {r.returncode}]")
    return r.returncode


def main():
    deck, N, ndim, skip, label, n_planes = make_deck()
    tag = f"{SWEEP}_B{B}_nmb{NMB}_nmu{NMU}"
    print(f"device={DEVICE} bin={BIN}")
    print(f"deck: {deck}")
    print(f"geometry: mesh {N}^3, {B}^3 blocks x {NMB} = {B**3 * NMB} zones, "
          f"nmu={NMU}, sweep={SWEEP}")
    print(f"sweep has {n_planes} plane-launches/cycle; profiling label='{label}' "
          f"filter={FILTER} window: skip={skip} count={COUNT}")
    if not os.path.exists(BIN):
        print(f"  [note: {BIN} missing -- build a CUDA athena on the login node first]")

    run_or_print("Nsight Compute (roofline / occupancy / warp-efficiency)",
                 ncu_cmd(deck, skip, label, tag), "ncu")
    run_or_print("Nsight Systems (per-plane launch-gap timeline)",
                 nsys_cmd(deck, tag), "nsys")

    print("\n# Extract after the run (Apollo):")
    print(f"#   ncu -i {os.path.join(OUT, 'ncu_' + tag)}.ncu-rep --page details "
          "--section SpeedOfLight  # memory- vs compute-bound + achieved BW/FLOP %")
    print(f"#   ncu -i {os.path.join(OUT, 'ncu_' + tag)}.ncu-rep --page raw "
          "| grep -E 'achieved_occupancy|warp_execution_efficiency'")
    print(f"#   nsys stats --report gpukernsum,gpumemtimesum "
          f"{os.path.join(OUT, 'nsys_' + tag)}.nsys-rep  # kernel time vs launch gaps")


if __name__ == "__main__":
    main()
