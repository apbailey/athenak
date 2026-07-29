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


# Davis+2012 Fig 4: deep exponential NLTE scattering atmosphere. Across four decades of the
# photon destruction probability eps, the source function thermalizes to the analytic surface
# sqrt(eps) law and J -> B[1 - exp(-sqrt(3 eps) tau)/(1+sqrt(eps))] (Eq. 30). The pgen requires
# the ALI to converge (residual <= ali_tol=1e-5) AND the mean intensity to match the analytic
# solution to within 5% (rms|J-Jan|/B < tol=5e-2), std::exit(EXIT_FAILURE) otherwise.
@pytest.mark.parametrize("eps", ["1.0e-4", "1.0e-6", "1.0e-8", "1.0e-10"])
def test_vet_atmosphere_thermalization(eps):
    testutils.run(
        "inputs/vet_atmosphere_exp.athinput",
        flags=[
            f"nr_radiation/eps={eps}",
            "nr_radiation/ali_tol=1.0e-5",
            "problem/require_analytic=true",
            "problem/tol=5.0e-2",
        ],
    )
