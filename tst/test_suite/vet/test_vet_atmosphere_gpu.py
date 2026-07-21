"""
VET coupling test (level 3): internally-heated atmosphere -> radiative equilibrium.

Gray, internally-heated slab (const opacity) relaxed via Jacobi-ALI to radiative
equilibrium; the pgen checks the converged J/temperature against the exact nmu=1
two-stream solution and std::exit(EXIT_FAILURE)s on non-convergence or error. Exercises
the source update + iteration + fluid coupling.

Three variants share the pgen:
  vet_atmosphere               single block, direct ALI loop, analytic check
  vet_atmosphere_2mb           split across 2 MeshBlocks (inter-block ghost exchange)
  vet_atmosphere_solvetransfer driven through the Driver/task list (production path)
"""

# Modules
import pytest
import test_suite.testutils as testutils


@pytest.mark.parametrize(
    "deck",
    [
        "vet_atmosphere.athinput",
        "vet_atmosphere_2mb.athinput",
        "vet_atmosphere_solvetransfer.athinput",
    ],
)
def test_vet_atmosphere(deck):
    testutils.run(f"inputs/{deck}")
