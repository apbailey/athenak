"""
SC coupling test: conservation of the differential form ACROSS A COARSE-FINE INTERFACE.

Same closed, periodic, fluid-coupled box as test_sc_qrad_conserve, but with a STATIC
refined patch, so a level jump is guaranteed to be present when the residual is measured.
That guarantee is the whole point of the file. The adaptive deck cannot provide it: it
refines every block at every level, so by the time the error file is written the mesh is
uniform again and contains no coarse-fine face at all -- which is exactly how
"coarse-fine conservation needs no flux correction" came to be believed on 2026-09-29.
Every case here therefore asserts maxlev > minlev as well as the residual.

Columns of sc_qrad_conserve-errs.dat:
Nx1 Nx2 Nx3 Ncycle residual SumQdV Sum|Q|dV face-vs-mom face-vs-mean niter nmb minlev
maxlev.

Note that face-vs-mean (column 8) is NOT asserted on a refined mesh. It recomputes the
face flux from each block's own ir plus ghosts, so at a corrected coarse face it
legitimately disagrees with the stored value -- by the size of the correction itself,
which is the O(dx^2) mismatch being removed. It is meaningful only on a uniform mesh, and
the uniform control below still checks it.

Parametrized over sweep_kernel: the three kernels are the same sweep, so each assertion is
also a kernel-identity check.
"""

import pytest
import test_suite.testutils as testutils

_INPUT = "inputs/sc_qrad_conserve_smr.athinput"
_KERNELS = [("wavefront", 0), ("tiled", 8), ("plane", 0)]

# machine-precision gates; measured 2026-10-01 on CPU with the correction on:
# 1.54e-16 (2D one jump), 3.30e-16 (2D nested), 9.47e-16 (3D), 1.57e-16 (uniform control)
_RESID_TOL = 1.0e-13
_FACE_TOL = 1.0e-12
# before the coarse-fine flux correction existed the same decks gave 5.43e-04 /
# 4.48e-04 / 4.75e-05; see logs/validation/qrad_cf_conservation_2026-10-01/REPORT.md
# Sec. 7

_3D = ["mesh/nx3=32", "mesh/x3min=0.0", "mesh/x3max=1.0", "meshblock/nx3=8"]
_NESTED = ["refined_region2/level=2"]


def _cols(fname="sc_qrad_conserve-errs.dat"):
    data = testutils.athena_read.error_dat(fname)
    assert len(data) == 1, f"expected one row in {fname}, got {len(data)}"
    return data[0]


def _run(kernel, tile, extra, label):
    testutils.cleanup()
    assert testutils.run(_INPUT, [f"nr_radiation/sweep_kernel={kernel}",
                                  f"nr_radiation/tile_size={tile}"] + extra), \
        f"run failed: {kernel} {label}"
    return _cols()


@pytest.mark.parametrize("extra,label", [([], "2D, one level jump"),
                                         (_NESTED, "2D, nested L1+L2"),
                                         (_3D, "3D, one level jump")])
@pytest.mark.parametrize("kernel,tile", _KERNELS)
def test_sc_qrad_conserve_smr(kernel, tile, extra, label):
    """The heating must integrate to zero with a coarse-fine face in the mesh.

    It does not without the flux correction. The coarse block's ghost is RestrictCC of
    the fine data, a volume average over TWO layers normal to the face, while the fine
    block uses the layer that touches the face; the two sides therefore put different
    numbers on the same face and the divergence stops telescoping. The correction sends
    the fine side's area-weighted mean across and the coarse side uses it instead.
    """
    try:
        row = _run(kernel, tile, extra, label)
        assert row[12] > row[11], \
            f"{kernel} {label}: no level jump survived (minlev {int(row[11])}, maxlev " \
            f"{int(row[12])}, {int(row[10])} blocks) -- the test is not measuring " \
            f"anything, fix the deck rather than the tolerance"
        assert row[4] < _RESID_TOL, \
            f"{kernel} {label}: heating does not integrate to zero across the " \
            f"coarse-fine face, residual {row[4]:g} >= {_RESID_TOL:g}"
    finally:
        testutils.cleanup()


@pytest.mark.parametrize("kernel,tile", _KERNELS)
def test_sc_qrad_conserve_smr_uniform_is_untouched(kernel, tile):
    """With refinement off the correction must not exist, let alone act.

    pbval_hflx is only constructed on a multilevel mesh, so this case must reproduce the
    plain uniform-grid answer exactly. face-vs-mean (column 8) is the sharp one: it is
    identically zero only where every face still carries the plain mean of the two cells
    either side, so it is non-zero the moment a correction touches a face. If it is zero
    here and the residual is at round-off, nothing ran.
    """
    off = ["mesh_refinement/refinement=none"]
    try:
        row = _run(kernel, tile, off, "uniform, refinement off")
        assert row[11] == row[12] == 0, \
            f"{kernel}: refinement=none produced levels {int(row[11])}..{int(row[12])}"
        assert row[4] < _RESID_TOL, f"{kernel}: uniform residual {row[4]:g}"
        assert row[7] < _FACE_TOL, \
            f"{kernel}: uniform face-vs-moments {row[7]:g} >= {_FACE_TOL:g}"
        assert row[8] < _FACE_TOL, \
            f"{kernel}: uniform face-vs-mean {row[8]:g} >= {_FACE_TOL:g} -- the " \
            f"coarse-fine correction has touched a face on a mesh that has none"
    finally:
        testutils.cleanup()
