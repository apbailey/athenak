"""
SC coupling test: the physical-boundary face flux of the differential form (Davis Eq. 28).

A uniform absorbing medium with a NON-uniform emission fills the domain; no fluid. The
divergence telescopes to the boundary, so the heating must integrate to minus the net
outward flux, and the pgen recomputes that flux on the host ray by ray, independently of
the device kernel. Columns of sc_qrad_bdry-errs.dat:
Nx1 Nx2 Nx3 Ncycle rel-diff residual SumQdV Sum|Q|dV bdry-flux naive-flux sum-w-mu niter.

The emission is non-uniform deliberately. A uniform source in a uniform medium is already
in equilibrium -- J = brad, Q = 0, H = 0 -- and every assertion below would pass on
nothing.

Each boundary flag says something different, and each is checked:
  reflect   nothing crosses a mirror, so the boundary flux is identically zero and the
            heating must integrate to zero. This is the sharpest case: exact, and with no
            analytic input at all. The test also requires Sum |Q| dV > 0, so that the
            zero is a cancellation and not an empty problem.
  vacuum    nothing enters; the flux is what escapes, so the gas cools on balance.
  inflow    in minus out, with an isotropic incident intensity on every face.
  outflow   the ghost copies the last active cell, so the face flux is the interior H.

For inflow the naive alternative -- a full-sphere quadrature of the ghost averaged with
the interior -- is also required to DIFFER materially from the rule. With an isotropic
incident field the ghost's first moment vanishes and the naive flux loses the incoming
radiation entirely, so if the two ever agreed, the test would have stopped discriminating.

sum-w-mu is checked too: the rule leans on Sum_k w_k mu_k = 0, and this branch dropped the
startup quadrature check.

Parametrized over sweep_kernel, so every assertion is also a kernel-identity check.
"""

import pytest
import test_suite.testutils as testutils

_KERNELS = [("wavefront", 0), ("tiled", 8), ("plane", 0)]
_BCS = ["inflow", "vacuum", "outflow", "reflect"]
# machine-precision gates; measured 2026-09-28 on CPU
_REL_TOL = 1.0e-11       # Sum Q dV vs the independent host-side boundary flux
_WMU_TOL = 1.0e-12       # |Sum w mu| per axis
_REFLECT_TOL = 1.0e-11   # |Sum Q dV| / Sum |Q| dV with reflecting walls
_NAIVE_MIN = 1.0e-2      # inflow: the naive flux must be materially wrong


def _bc_args(bc, ndim):
    a = [f"mesh/ix1_bc={bc}", f"mesh/ox1_bc={bc}"]
    if ndim >= 2:
        a += [f"mesh/ix2_bc={bc}", f"mesh/ox2_bc={bc}"]
    if ndim >= 3:
        a += [f"mesh/ix3_bc={bc}", f"mesh/ox3_bc={bc}"]
    return a


def _dim_args(ndim):
    if ndim == 1:
        return ["mesh/nx2=1", "mesh/nx3=1", "meshblock/nx1=8", "meshblock/nx2=1",
                "meshblock/nx3=1", "mesh/ix2_bc=periodic", "mesh/ox2_bc=periodic",
                "mesh/ix3_bc=periodic", "mesh/ox3_bc=periodic"]
    if ndim == 3:
        # meshblock dims must stay divisible by tile_size for the tiled kernel
        return ["mesh/nx3=8", "mesh/x3min=0.0", "mesh/x3max=1.0",
                "meshblock/nx1=8", "meshblock/nx2=8", "meshblock/nx3=8"]
    return []


def _row(fname="sc_qrad_bdry-errs.dat"):
    data = testutils.athena_read.error_dat(fname)
    assert len(data) == 1, f"expected one row in {fname}, got {len(data)}"
    return data[0]


@pytest.mark.parametrize("bc", _BCS)
@pytest.mark.parametrize("kernel,tile", _KERNELS)
def test_sc_qrad_bdry(bc, kernel, tile):
    input_file = "inputs/sc_qrad_bdry.athinput"
    testutils.cleanup()
    try:
        args = _bc_args(bc, 2) + [f"nr_radiation/sweep_kernel={kernel}",
                                  f"nr_radiation/tile_size={tile}"]
        assert testutils.run(input_file, args), f"run failed: {bc} {kernel}"
        row = _row()
        assert row[10] < _WMU_TOL, \
            f"{bc} {kernel}: quadrature identity broken, max |Sum w mu| = {row[10]:g}"
        assert row[4] < _REL_TOL, \
            f"{bc} {kernel}: heating does not match the independent boundary flux, " \
            f"rel diff {row[4]:g} (Sum Q dV {row[6]:g} vs flux {row[8]:g})"
        assert row[7] > 0.0, f"{bc} {kernel}: no heating anywhere; test is vacuous"

        if bc == "reflect":
            # nothing crosses a mirror: the boundary flux, and so the total, is zero
            assert row[5] < _REFLECT_TOL, \
                f"reflect {kernel}: heating does not integrate to zero, " \
                f"|Sum Q dV|/Sum |Q| dV = {row[5]:g}"
        if bc == "vacuum":
            assert row[6] < 0.0, \
                f"vacuum {kernel}: expected net cooling as radiation escapes, " \
                f"got Sum Q dV = {row[6]:g}"
        if bc == "inflow":
            # the naive ghost-average flux must be materially wrong, or this stops being
            # a test of the rule
            scale = max(abs(row[8]), abs(row[9]))
            assert abs(row[9] - row[8]) > _NAIVE_MIN*scale, \
                f"inflow {kernel}: the naive flux ({row[9]:g}) agrees with the " \
                f"half-range rule ({row[8]:g}); the test no longer discriminates"
    finally:
        testutils.cleanup()


@pytest.mark.parametrize("ndim", [1, 3])
@pytest.mark.parametrize("kernel,tile", _KERNELS)
def test_sc_qrad_bdry_dims(ndim, kernel, tile):
    """1D and 3D. In 3D the corner cells touch three physical faces at once, which is the
    case a single-face argument does not cover."""
    input_file = "inputs/sc_qrad_bdry.athinput"
    testutils.cleanup()
    try:
        args = (_dim_args(ndim) + _bc_args("inflow", ndim)
                + [f"nr_radiation/sweep_kernel={kernel}",
                   f"nr_radiation/tile_size={tile}"])
        assert testutils.run(input_file, args), f"run failed: {ndim}D {kernel}"
        row = _row()
        assert row[10] < _WMU_TOL, \
            f"{ndim}D {kernel}: quadrature identity broken, {row[10]:g}"
        assert row[4] < _REL_TOL, \
            f"{ndim}D {kernel}: rel diff {row[4]:g} (Sum Q dV {row[6]:g} vs " \
            f"flux {row[8]:g})"
    finally:
        testutils.cleanup()
