"""
VET sweep determinism / GPU-race regression.

Generic pgen `vet_sweep_determinism` freezes a gradient intensity field and runs the
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
import pytest
import test_suite.testutils as testutils


def _run_capture(inputfile):
    cmd = list(testutils.DEFAULT_LAUNCHER) + ["./athena", "-i", inputfile]
    rc, out, err = testutils.run_command_capture(cmd)
    return rc, (out or "") + (err or "")


@pytest.mark.parametrize(
    "deck",
    [
        "vet_determinism_jacobi.athinput",
        "vet_determinism_wavefront.athinput",
        "vet_determinism_diagonal.athinput",
    ],
)
def test_vet_sweep_determinism(deck):
    rc, out = _run_capture(f"inputs/{deck}")
    assert rc == 0, f"athena exited non-zero (crash?):\n{out[-2000:]}"
    assert "determinism PASS" in out, f"sweep is non-deterministic (race):\n{out[-2000:]}"
