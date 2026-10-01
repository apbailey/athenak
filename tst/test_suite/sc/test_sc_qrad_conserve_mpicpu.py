"""
SC coupling test: the differential form under MPI.

The same closed-box identity as test_sc_qrad_conserve, but distributed. Two things are
checked that a single rank cannot show:

  * the heating still integrates to zero. The cancellation is between face values shared
    by neighbouring meshblocks, and across a rank boundary those two blocks reach each
    other only through the exchange -- so this is really a test that the extra exchange
    SolveTransfer performs after the iteration loop does its job. Without it the two sides
    of a shared face disagree by one sweep's update and the residual rises to the order of
    the iteration residual.
  * the total heating is independent of the rank count. Cutting the domain differently
    must not change the answer.

Columns as in test_sc_qrad_conserve: residual is 4, Sum |Q| dV is 6.
"""

import pytest
import test_suite.testutils as testutils

_KERNELS = [("wavefront", 0), ("tiled", 8), ("plane", 0)]
_RESID_TOL = 1.0e-13
_RANKS = [1, 2, 4]


def _cols(fname="sc_qrad_conserve-errs.dat"):
    data = testutils.athena_read.error_dat(fname)
    assert len(data) == 1, f"expected one row in {fname}, got {len(data)}"
    return data[0]


@pytest.mark.parametrize("kernel,tile", _KERNELS)
def test_sc_qrad_conserve_mpi(kernel, tile):
    input_file = "inputs/sc_qrad_conserve.athinput"
    # four meshblocks, so 1, 2 and 4 ranks all divide the work differently
    base = ["meshblock/nx1=16", "meshblock/nx2=8",
            f"nr_radiation/sweep_kernel={kernel}", f"nr_radiation/tile_size={tile}"]
    testutils.cleanup()
    try:
        heat = {}
        for nranks in _RANKS:
            testutils.cleanup()
            assert testutils.mpi_run(input_file, base, threads=nranks), \
                f"run failed: {kernel} on {nranks} ranks"
            row = _cols()
            assert row[4] < _RESID_TOL, \
                f"{kernel} on {nranks} ranks: heating does not integrate to zero, " \
                f"residual {row[4]:g} >= {_RESID_TOL:g}"
            heat[nranks] = row[6]
        ref = heat[1]
        for n, val in heat.items():
            assert abs(val - ref) <= 1.0e-12*ref, \
                f"{kernel}: total heating depends on the rank count: {n} ranks give " \
                f"{val:.12g}, 1 rank gives {ref:.12g}"
    finally:
        testutils.cleanup()


@pytest.mark.parametrize("kernel,tile", _KERNELS)
def test_sc_qrad_conserve_smr_mpi(kernel, tile):
    """Coarse-fine flux correction across a RANK boundary.

    The correction is the one piece of this scheme whose data genuinely has to travel:
    the coarse block cannot reconstruct the fine side's face flux from its own ghosts,
    because RestrictCC already averaged away the layer that touches the face. On one rank
    PackAndSendFluxCC copies straight into the receive buffer and the MPI path is never
    taken at all, so this is the only case that exercises InitFluxRecv, the MPI_Isend and
    the MPI_Test loop. It is also where a wrong nvar would surface -- as a truncation
    error or as silent corruption.
    """
    input_file = "inputs/sc_qrad_conserve_smr.athinput"
    base = [f"nr_radiation/sweep_kernel={kernel}", f"nr_radiation/tile_size={tile}"]
    testutils.cleanup()
    try:
        heat = {}
        for nranks in _RANKS:
            testutils.cleanup()
            assert testutils.mpi_run(input_file, base, threads=nranks), \
                f"run failed: {kernel} SMR on {nranks} ranks"
            row = _cols()
            assert row[12] > row[11], \
                f"{kernel} on {nranks} ranks: no level jump survived (levels " \
                f"{int(row[11])}..{int(row[12])}) -- the test is measuring nothing"
            assert row[4] < _RESID_TOL, \
                f"{kernel} SMR on {nranks} ranks: heating does not integrate to zero " \
                f"across the coarse-fine face, residual {row[4]:g} >= {_RESID_TOL:g}"
            heat[nranks] = row[6]
        ref = heat[1]
        for n, val in heat.items():
            assert abs(val - ref) <= 1.0e-12*ref, \
                f"{kernel} SMR: total heating depends on the rank count: {n} ranks " \
                f"give {val:.12g}, 1 rank gives {ref:.12g}"
    finally:
        testutils.cleanup()
