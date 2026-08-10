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


def _run_capture(inputfile):
    cmd = list(testutils.DEFAULT_LAUNCHER) + ["./athena", "-i", inputfile]
    rc, out, err = testutils.run_command_capture(cmd)
    return rc, (out or "") + (err or "")


@pytest.mark.parametrize(
    "deck",
    [
        "sc_determinism_jacobi.athinput",
        "sc_determinism_wavefront.athinput",
        "sc_determinism_diagonal.athinput",
        "sc_determinism_diagonal_compact.athinput",
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
