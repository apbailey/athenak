"""
SC unit test (level 1): short-characteristics constant-source identity.

Uniform chi, S=b, I=b everywhere: one FormalSolution must leave I==b to machine
precision (exercises the SC edtau+sum(a)=1 and bilinear partition-of-unity identities).
The pgen std::exit(EXIT_FAILURE)s if the max relative error exceeds tolerance.
"""

# Modules
import test_suite.testutils as testutils


def test_sc_uniform():
    testutils.run("inputs/sc_uniform.athinput")
