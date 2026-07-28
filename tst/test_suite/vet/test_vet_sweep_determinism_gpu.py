"""
VET sweep determinism / GPU-race regression.

Generic pgen `vet_sweep_determinism` freezes a gradient intensity field and runs the
configured sweep `nrepeat` times from the SAME input, requiring bitwise-identical output
(order-independent XOR hash) and std::exit(EXIT_FAILURE) on any mismatch. The unordered
single-buffer jacobi sweep would race (read an upwind cell a sibling thread overwrites) ->
non-deterministic on GPU; the double-buffered jacobi and the ordered wavefront/diagonal
sweeps are reproducible. The three decks share the pgen (differ only in nr_radiation/sweep).
On CPU-serial all pass trivially (no concurrency); on GPU the jacobi deck is the real guard.
"""

# Modules
import pytest
import test_suite.testutils as testutils


@pytest.mark.parametrize(
    "deck",
    [
        "vet_determinism_jacobi.athinput",
        "vet_determinism_wavefront.athinput",
        "vet_determinism_diagonal.athinput",
    ],
)
def test_vet_sweep_determinism(deck):
    testutils.run(f"inputs/{deck}")
