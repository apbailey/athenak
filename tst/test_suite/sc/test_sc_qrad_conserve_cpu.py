"""
SC coupling test: closed-box conservation of the differential form (Davis Eq. 28).

A periodic, fluid-coupled box with a smooth density and pressure perturbation, so the
opacity and the emission both vary and the radiation field carries a real flux. With no
boundary, the face-flux divergence telescopes to nothing and the heating must integrate
to zero over the domain. Columns of sc_qrad_conserve-errs.dat:
Nx1 Nx2 Nx3 Ncycle residual SumQdV Sum|Q|dV face-vs-mom niter, where residual is
|Sum Q dV| / Sum |Q| dV.

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
    finally:
        testutils.cleanup()
