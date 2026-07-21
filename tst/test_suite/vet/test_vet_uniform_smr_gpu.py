"""
VET framework test (level 6): SMR restrict/prolong of the radiation field.

2D uniform S=b, I=b with one static-refinement region. Beyond the constant-source
identity, the pgen restricts -> fills-coarse -> prolongs the intensity `ir` and source
`bb` across the refinement boundary and re-checks I==b, S==b to machine precision,
std::exit(EXIT_FAILURE)ing on error. Exercises the multilevel transfer operators.
"""

# Modules
import test_suite.testutils as testutils


def test_vet_uniform_smr():
    testutils.run("inputs/vet_uniform_smr.athinput")
