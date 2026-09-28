"""
SC coupling test: closed-box conservation of the differential form (Davis Eq. 28).

A periodic, fluid-coupled box with a smooth density and pressure perturbation, so the
opacity and the emission both vary and the radiation field carries a real flux. With no
boundary, the face-flux divergence telescopes to nothing and the heating must integrate
to zero over the domain. Columns of sc_qrad_conserve-errs.dat:
Nx1 Nx2 Nx3 Ncycle residual SumQdV Sum|Q|dV face-vs-mom face-vs-mean niter, where
residual is |Sum Q dV| / Sum |Q| dV.

What makes this a unit test rather than a convergence study is that the identity is
algebraic: the cancellation is between face values shared by neighbouring cells, so it
does not care whether the solve converged or whether the angular grid resolves anything.
Hence the iter_max=1 and nmu=1 cases, which must pass just as tightly as the default.

Also asserted:
  * the integral form does NOT conserve -- the contrast that motivates the differential
    form at all. Its residual is ~2e-3 here, ten orders above divh.
  * the face flux agrees with the mean of the cell-centred moments on interior faces
    (column 7). Those are the same quantity by two different routes; BuildHFlux averages
    intensities and then takes the moment, the moments array does the reverse. They agree
    to round-off in the interior and only there, which is why the pgen restricts the
    comparison to faces with two interior neighbours.
  * the face flux agrees with a host-side mean of the intensities on EVERY face, block
    edges included (column 8). On a periodic mesh every face is interior, so every one
    must take the mean. This is the check that catches a face being misclassified as a
    physical boundary -- the boundary rule conserves too, and agrees with the moments
    away from the block edge, so nothing else here notices.

Parametrized over sweep_kernel: the three kernels are the same sweep, so each assertion
is also a kernel-identity check, and the residuals are required to match exactly.
"""

import pytest
import test_suite.testutils as testutils

_KERNELS = [("wavefront", 0), ("tiled", 8), ("plane", 0)]
# machine-precision gates; measured 2026-09-28 on CPU: residual 1.4e-16 (default),
# 3.5e-17 (iter_max=1), 2.3e-16 (nmu=1), 4.0e-16 (3D); face-vs-moments 5.8e-16
_RESID_TOL = 1.0e-13
_FACE_TOL = 1.0e-12
_INTEGRAL_MIN = 1.0e-5   # the integral form must be decisively worse, not marginally


def _cols(fname="sc_qrad_conserve-errs.dat"):
    data = testutils.athena_read.error_dat(fname)
    assert len(data) == 1, f"expected one row in {fname}, got {len(data)}"
    return data[0]


@pytest.mark.parametrize("kernel,tile", _KERNELS)
def test_sc_qrad_conserve(kernel, tile):
    input_file = "inputs/sc_qrad_conserve.athinput"
    base = [f"nr_radiation/sweep_kernel={kernel}", f"nr_radiation/tile_size={tile}"]
    testutils.cleanup()
    try:
        # the identity is algebraic, so it must hold for a converged solve, for a single
        # sweep, and for the coarsest possible angular grid, all to the same tolerance
        for extra, label in [([], "default"),
                             (["nr_radiation/iter_max=1"], "iter_max=1"),
                             (["nr_radiation/nmu=1"], "nmu=1")]:
            testutils.cleanup()
            assert testutils.run(input_file, base + extra), \
                f"run failed: {kernel} {label}"
            row = _cols()
            assert row[4] < _RESID_TOL, \
                f"{kernel} {label}: heating does not integrate to zero, " \
                f"residual {row[4]:g} >= {_RESID_TOL:g}"
            assert row[7] < _FACE_TOL, \
                f"{kernel} {label}: face flux disagrees with the cell-centred moments " \
                f"on interior faces, {row[7]:g} >= {_FACE_TOL:g}"
            assert row[8] < _FACE_TOL, \
                f"{kernel} {label}: face flux is not the mean on every face (a face is " \
                f"likely misclassified as a boundary), {row[8]:g} >= {_FACE_TOL:g}"

        # the contrast: Eq. 27 has no such identity
        testutils.cleanup()
        assert testutils.run(input_file, base + ["nr_radiation/qrad_form=integral"]), \
            f"run failed: {kernel} integral"
        assert _cols()[4] > _INTEGRAL_MIN, \
            f"{kernel}: the integral form unexpectedly conserves; the comparison is no " \
            f"longer meaningful"
    finally:
        testutils.cleanup()


@pytest.mark.parametrize("kernel,tile", _KERNELS)
def test_sc_qrad_conserve_decomposition(kernel, tile):
    """The answer must not depend on how the domain is cut into meshblocks.

    This is the check that catches a face being classified wrongly. mb_bcs gives a
    domain-edge block the mesh flag rather than block, so on a periodic mesh an edge face
    reports periodic even though the tree wraps and the ghost is a real neighbour's
    interior. Treating that as a physical boundary applies the boundary rule to an
    interior face -- which still conserves (the boundary rule telescopes too) and still
    matches the moments on faces away from the block edge, so neither of the other checks
    notices. It does change the answer, and it changes it differently depending on where
    the block edges fall, so comparing decompositions finds it.
    """
    input_file = "inputs/sc_qrad_conserve.athinput"
    base = [f"nr_radiation/sweep_kernel={kernel}", f"nr_radiation/tile_size={tile}"]
    testutils.cleanup()
    try:
        heat = {}
        for nx1, nx2 in [(32, 16), (16, 16), (16, 8), (8, 8)]:
            testutils.cleanup()
            assert testutils.run(input_file, base + [f"meshblock/nx1={nx1}",
                                                     f"meshblock/nx2={nx2}"]), \
                f"run failed: {kernel} meshblock {nx1}x{nx2}"
            row = _cols()
            heat[(nx1, nx2)] = row[6]
            assert row[4] < _RESID_TOL, \
                f"{kernel} meshblock {nx1}x{nx2}: residual {row[4]:g}"
        ref = heat[(32, 16)]                       # one block covers the whole mesh
        for key, val in heat.items():
            assert abs(val - ref) <= 1.0e-12*ref, \
                f"{kernel}: total heating depends on the decomposition: " \
                f"{key} gives {val:.12g}, one block gives {ref:.12g}"
    finally:
        testutils.cleanup()


@pytest.mark.parametrize("kernel,tile", _KERNELS)
def test_sc_qrad_hybrid_regimes(kernel, tile):
    """The Davis sec. 4 switch, and the price it charges.

    hybrid takes Eq. 27 where chi*dx <= 1 and Eq. 28 above, so it should reduce to divh
    when every cell is thick and to integral when every cell is thin. In between it is
    NOT conservative, and cannot be: telescoping needs a face to be debited by one cell
    and credited by its neighbour, and at the seam between a differential cell and an
    integral one that does not happen. The transition case is asserted to fail
    conservation on purpose -- if it ever starts passing, the scheme has changed and the
    reason should be understood rather than welcomed.
    """
    input_file = "inputs/sc_qrad_conserve.athinput"
    base = [f"nr_radiation/sweep_kernel={kernel}", f"nr_radiation/tile_size={tile}"]

    def run_resid(kappa, form, guard=1.0e-2):
        # guard is the pgen's crash-guard, which only catches NaN and blow-up; the
        # transition case is legitimately non-conservative and needs it relaxed
        testutils.cleanup()
        assert testutils.run(input_file, base + [f"nr_radiation/kappa_a={kappa}",
                                                 f"nr_radiation/qrad_form={form}",
                                                 f"problem/tol={guard}"]), \
            f"run failed: {kernel} kappa_a={kappa} {form}"
        return _cols()[4]

    try:
        # thick everywhere (chi*dx = 6.25): every cell takes Eq. 28
        assert run_resid(200.0, "hybrid") < _RESID_TOL, \
            f"{kernel}: hybrid does not reduce to divh when every cell is thick"
        # thin everywhere (chi*dx = 0.0125): every cell takes Eq. 27, so it matches it
        thin_h = run_resid(0.4, "hybrid")
        thin_i = run_resid(0.4, "integral")
        assert abs(thin_h - thin_i) <= 1.0e-12*max(thin_h, thin_i), \
            f"{kernel}: hybrid does not reduce to integral when every cell is thin " \
            f"({thin_h:g} vs {thin_i:g})"
        # the transition (chi*dx = 1.25): mixed cells, so conservation is lost by design
        assert run_resid(40.0, "hybrid", guard=1.0) > _RESID_TOL, \
            f"{kernel}: hybrid unexpectedly conserves across a thick/thin transition; " \
            f"mixing the two forms per zone should break the telescoping"
    finally:
        testutils.cleanup()


@pytest.mark.parametrize("kernel,tile", _KERNELS)
def test_sc_qrad_conserve_3d(kernel, tile):
    input_file = "inputs/sc_qrad_conserve.athinput"
    testutils.cleanup()
    try:
        assert testutils.run(input_file, [
            "mesh/nx3=16", "mesh/x3min=0.0", "mesh/x3max=1.0",
            "meshblock/nx1=8", "meshblock/nx2=8", "meshblock/nx3=8",
            f"nr_radiation/sweep_kernel={kernel}", f"nr_radiation/tile_size={tile}",
        ]), f"run failed: {kernel} 3D"
        row = _cols()
        assert row[4] < _RESID_TOL, \
            f"{kernel} 3D: residual {row[4]:g} >= {_RESID_TOL:g}"
        assert row[7] < _FACE_TOL, \
            f"{kernel} 3D: face vs moments {row[7]:g} >= {_FACE_TOL:g}"
        assert row[8] < _FACE_TOL, \
            f"{kernel} 3D: face vs mean {row[8]:g} >= {_FACE_TOL:g}"
    finally:
        testutils.cleanup()
