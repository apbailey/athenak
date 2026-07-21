"""
VET transport test (level 2): two crossing pencil beams, mirror-symmetry invariant.

Two collimated beams are injected from the lower boundary of a 2D vacuum box (up-right and
up-left, offset to x = -/+ x0) and cross in an X. Because the two ordinates differ only in the
sign of mu_x (equal weights), the beams are exact mirror images, so the resulting E_r field
must be symmetric under x -> -x -- short characteristics commutes with reflection, so the
asymmetry is machine-zero. The pgen checks this and std::exit(EXIT_FAILURE)s if the x-mirror
asymmetry exceeds tol (1e-12).

This gives the (otherwise closed-form-free) spreading pencil beam real teeth via a symmetry
invariant, complementing the sphere convergence ladder on the transport operator.
"""

# Modules
import test_suite.testutils as testutils


def test_vet_twobeam():
    testutils.run("inputs/vet_twobeam.athinput")
