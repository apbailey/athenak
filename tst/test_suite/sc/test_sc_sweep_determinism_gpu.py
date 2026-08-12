"""
SC sweep determinism / GPU-race regression.

Generic pgen `sc_sweep_determinism` freezes a gradient intensity field and runs the
configured sweep `nrepeat` times from the SAME input. The pgen prints a one-line verdict
("... determinism PASS ..." or "... FAIL (race) ...") and exits 0 either way -- it does NOT
std::exit on failure, which under Kokkos+MPI would skip Kokkos::finalize() and make mpirun
abort the job (poisoning the GPU for the next test). This wrapper judges pass/fail by grepping
that line, so a failing sweep is non-destructive and the suite continues.

The unordered single-buffer jacobi sweep races (reads an upwind cell a sibling thread
overwrites) -> non-deterministic on GPU -> FAIL; the double-buffered jacobi and the ordered
wavefront/diagonal sweeps are reproducible -> PASS. CPU-serial passes all (no concurrency).
"""

# Modules
import re
import pytest
import test_suite.testutils as testutils


def _run_capture(inputfile, extra=None):
    cmd = list(testutils.DEFAULT_LAUNCHER) + ["./athena", "-i", inputfile] + (extra or [])
    rc, out, err = testutils.run_command_capture(cmd)
    return rc, (out or "") + (err or "")


@pytest.mark.parametrize(
    "deck",
    [
        "sc_determinism_jacobi.athinput",
        "sc_determinism_wavefront.athinput",
        "sc_determinism_diagonal.athinput",
        "sc_determinism_diagonal_compact.athinput",
        "sc_determinism_wavefront_coalesced.athinput",
        "sc_determinism_angle_inner.athinput",
        "sc_determinism_hoist_wavefront.athinput",
        "sc_determinism_tiled.athinput",
        "sc_determinism_tiled_angle_inner.athinput",
    ],
)
def test_sc_sweep_determinism(deck):
    rc, out = _run_capture(f"inputs/{deck}")
    assert rc == 0, f"athena exited non-zero (crash?):\n{out[-2000:]}"
    assert "determinism PASS" in out, f"sweep is non-deterministic (race):\n{out[-2000:]}"


def _pass_hash(out):
    """Extract the interior XOR-hash from a determinism PASS line, or None."""
    m = re.search(r"determinism PASS.*hash (0x[0-9a-fA-F]+)", out)
    return m.group(1) if m else None


def test_sc_diagonal_compact_matches_diagonal():
    """diagonal_compact must reproduce baseline diagonal BIT-FOR-BIT: same interior cells, iterated
    in a different order, and UpdateCellSC is a pure function of strictly-lower planes (use_ali=false).
    The interior XOR-hash of both sweeps must be identical -- the key correctness gate for I1."""
    rc_d, out_d = _run_capture("inputs/sc_determinism_diagonal.athinput")
    rc_c, out_c = _run_capture("inputs/sc_determinism_diagonal_compact.athinput")
    assert rc_d == 0 and rc_c == 0, f"athena crashed:\n{out_d[-1000:]}\n{out_c[-1000:]}"
    h_d, h_c = _pass_hash(out_d), _pass_hash(out_c)
    assert h_d is not None and h_c is not None, (
        f"could not parse PASS hash (diagonal={h_d}, compact={h_c})")
    assert h_d == h_c, f"diagonal_compact hash {h_c} != diagonal {h_d} -- NOT bit-identical"


def test_sc_wavefront_coalesced_matches_wavefront():
    """wavefront_coalesced (I2: angle-innermost transposed scratch + angle-warp sweep) must reproduce
    baseline wavefront BIT-FOR-BIT: the transpose is a pure copy and UpdateCellSC<true> runs the same
    FP ops (use_ali=false). The interior XOR-hash of both sweeps must be identical."""
    rc_w, out_w = _run_capture("inputs/sc_determinism_wavefront.athinput")
    rc_c, out_c = _run_capture("inputs/sc_determinism_wavefront_coalesced.athinput")
    assert rc_w == 0 and rc_c == 0, f"athena crashed:\n{out_w[-1000:]}\n{out_c[-1000:]}"
    h_w, h_c = _pass_hash(out_w), _pass_hash(out_c)
    assert h_w is not None and h_c is not None, (
        f"could not parse PASS hash (wavefront={h_w}, coalesced={h_c})")
    assert h_w == h_c, f"wavefront_coalesced hash {h_c} != wavefront {h_w} -- NOT bit-identical"


def test_sc_hoist_matches_baseline():
    """sc_hoist=true (ledger I3: per-ray precomputed interpolation invariants read from sc_inv_)
    must reproduce the baseline wavefront (sc_hoist=false) BIT-FOR-BIT. The table is built by the
    same ComputeSCAngleInv() the recompute path uses, on the same dx/mu, so GatherSolveSC runs the
    identical FP ops (use_ali=false). The interior XOR-hash of both must match -- the I3 gate."""
    rc_b, out_b = _run_capture("inputs/sc_determinism_wavefront.athinput")
    rc_h, out_h = _run_capture("inputs/sc_determinism_hoist_wavefront.athinput")
    assert rc_b == 0 and rc_h == 0, f"athena crashed:\n{out_b[-1000:]}\n{out_h[-1000:]}"
    h_b, h_h = _pass_hash(out_b), _pass_hash(out_h)
    assert h_b is not None and h_h is not None, (
        f"could not parse PASS hash (baseline={h_b}, hoist={h_h})")
    assert h_b == h_h, f"sc_hoist hash {h_h} != baseline wavefront {h_b} -- NOT bit-identical"


def _grep(out, pat):
    m = re.search(pat, out)
    return m.group(1) if m else None


def test_sc_angle_inner_matches_normal():
    """Native I2 (ir_layout=angle_inner): the reordered sweep+ComputeJ and the CC-exchange bridge
    must reproduce the normal-layout wavefront BIT-FOR-BIT. (1) jmean hash of angle_inner ==
    wavefront (reordered kernels); (2) boundary-exchange refill hash (problem/test_exchange=true)
    of angle_inner == wavefront (the SyncIrNormal bridge around the UNCHANGED exchange)."""
    rc_w, out_w = _run_capture("inputs/sc_determinism_wavefront.athinput")
    rc_a, out_a = _run_capture("inputs/sc_determinism_angle_inner.athinput")
    assert rc_w == 0 and rc_a == 0, f"athena crashed:\n{out_w[-1000:]}\n{out_a[-1000:]}"
    jw, ja = _grep(out_w, r"jmean (0x[0-9a-fA-F]+)"), _grep(out_a, r"jmean (0x[0-9a-fA-F]+)")
    assert jw and ja and jw == ja, f"angle_inner jmean {ja} != wavefront {jw} (reordered kernels)"

    rc_we, out_we = _run_capture("inputs/sc_determinism_wavefront.athinput",
                                 ["problem/test_exchange=true"])
    rc_ae, out_ae = _run_capture("inputs/sc_determinism_angle_inner.athinput",
                                 ["problem/test_exchange=true"])
    assert rc_we == 0 and rc_ae == 0, f"athena crashed:\n{out_we[-1000:]}\n{out_ae[-1000:]}"
    ew = _grep(out_we, r"exchange_test.*hash (0x[0-9a-fA-F]+)")
    ea = _grep(out_ae, r"exchange_test.*hash (0x[0-9a-fA-F]+)")
    assert ew and ea and ew == ea, f"angle_inner exchange {ea} != wavefront {ew} (bridge)"
