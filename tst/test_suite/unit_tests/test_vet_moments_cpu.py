"""
VET unit test (level 1): angular-moment identities of the Carlson quadrature.

The vet_moments pgen computes J, H, K from a known intensity and checks them against
the exact quadrature identities (J=1, H=(1/3,0,0), K_ii=1/3, K_ij=0), calling
std::exit(EXIT_FAILURE) if any exceeds machine-precision tolerance. testutils.run()
raises on that nonzero exit, so the assertion lives entirely in the pgen.
"""

# Modules
import test_suite.testutils as testutils


def test_vet_moments():
    testutils.run("inputs/vet_moments.athinput")
