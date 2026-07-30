"""
SC transport test (level 2): homogeneous emitting/absorbing sphere convergence ladder.

Runs the sc_sphere problem generator (a uniform emitting/absorbing sphere in vacuum) over a
ladder of resolutions, in both 2D and 3D. One FormalSolution sweep is the exact formal solution
for the fixed source on a single block; the pgen compares J(r) to the analytic angular quadrature
of I = b[1-exp(-tau)] over interior cells (using the SAME octant-indexed quadrature as
CalculateMoments, so the angle integration cancels and this isolates transport + interpolation
error) and appends the relative L2 (RMS-L1 column) and L-infty errors to sc_sphere-errs.dat.

Checks that the error CONVERGES with resolution, not just that a single run is under a threshold.
The SC bilinear sweep is ~1st order here (limited by the chi discontinuity at the staircased
surface), so we require the finest-grid error under a floor AND an observed order above ~1. In 2D
the angular grid drops mu_z (|mu| < 1), so the problem is an infinite cylinder and the analytic
accounts for the sweep transporting the full 3D ray path (see the pgen).
"""

# Modules
import math
import pytest
import test_suite.testutils as testutils

_res = [32, 64, 128]  # resolutions to test


def _args(res, ndim):
    nx3 = res if ndim == 3 else 1
    return [
        f"mesh/nx1={res}", f"mesh/nx2={res}", f"mesh/nx3={nx3}",
        f"meshblock/nx1={res}", f"meshblock/nx2={res}", f"meshblock/nx3={nx3}",
    ]


@pytest.mark.parametrize("ndim", [2, 3])
def test_sc_sphere(ndim):
    input_file = "inputs/sc_sphere.athinput"
    testutils.cleanup()  # drop any stale *-errs.dat so appends start clean
    try:
        for res in _res:
            assert testutils.run(input_file, _args(res, ndim)), \
                f"run failed at {ndim}D nx={res}"
        data = testutils.athena_read.error_dat("sc_sphere-errs.dat")
        l2 = [row[4] for row in data]  # RMS-L1 column = relative L2 error

        # error must decrease monotonically as the grid is refined
        assert all(l2[i + 1] < l2[i] for i in range(len(l2) - 1)), \
            f"{ndim}D L2 error not monotonically decreasing: {l2}"

        # finest-grid error below an absolute floor
        assert l2[-1] < 4.0e-3, \
            f"{ndim}D L2 at nx={_res[-1]} too large: {l2[-1]:g} (L2 ladder {l2})"

        # observed order of convergence over the ladder must be ~1 or better
        rate = math.log(l2[0] / l2[-1]) / math.log(_res[-1] / _res[0])
        assert rate > 0.8, \
            f"{ndim}D insufficient convergence: observed order {rate:.2f} (L2 {l2})"
    finally:
        testutils.cleanup()
