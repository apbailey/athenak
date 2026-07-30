"""
SC transport test (level 2): 1D pure-absorption attenuation.

Upwind beam through a uniform absorbing slab; the pgen compares the swept intensity
to the exact I(x) = I_bc * exp(-tau), tau = chi * dist / |mu|, and std::exit(EXIT_FAILURE)s
if the max relative error exceeds tolerance. Isolates the formal-solution operator with
a fixed source.
"""

# Modules
import test_suite.testutils as testutils


def test_sc_attenuation():
    testutils.run("inputs/sc_attenuation.athinput")
