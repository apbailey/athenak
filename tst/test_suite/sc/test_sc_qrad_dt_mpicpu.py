"""
SC coupling test: the Q_rad energy-depletion timestep limit under MPI.

The limit is a Kokkos::Min over the rank's own cells followed by an MPI_Allreduce inside
LimitDtByQrad, and it lowers Mesh::dt directly rather than reporting a dtnew that
Mesh::NewTimeStep would reduce for it. That is a *new* collective on a path that
previously had none, so the thing worth checking is the one a single rank cannot show:
every rank must come away with the same dt, and therefore the same cycle count and the
same final state, however the domain is cut.

If the Allreduce were missing or misplaced, ranks would take different steps and the run
would either diverge between rank counts or deadlock.

Columns of sc_qrad_dt-errs.dat: Nx1 Nx2 Nx3 Ncycle min_eint dt max_absQ cfl_qrad nbad.
"""

import os
import pytest
import test_suite.testutils as testutils

_INPUT = "inputs/sc_qrad_dt.athinput"
_ERRS = "sc_qrad_dt-errs.dat"
_LOG = "SCQradDt.log"
_RANKS = [1, 2, 4]


def _row():
    data = testutils.athena_read.error_dat(_ERRS)
    assert len(data) == 1, f"expected one row in {_ERRS}, got {len(data)}"
    return data[0]


def _clean():
    testutils.cleanup()
    if os.path.exists(_LOG):
        os.remove(_LOG)


def test_sc_qrad_dt_mpi():
    # four meshblocks, so 1, 2 and 4 ranks all divide the work differently
    base = ["meshblock/nx1=16", "meshblock/nx2=16", "nr_radiation/cfl_qrad=0.25"]
    _clean()
    try:
        results = {}
        for nranks in _RANKS:
            _clean()
            assert testutils.mpi_run(_INPUT, base, threads=nranks), \
                f"run failed on {nranks} ranks"
            row = _row()
            assert row[8] == 0 and row[4] > 0.0, \
                f"{nranks} ranks: min e_int = {row[4]:g}, {int(row[8])} bad cells"
            results[nranks] = (int(row[3]), row[5])

        ref_ncycle, ref_dt = results[_RANKS[0]]
        for nranks, (ncycle, dt) in results.items():
            assert ncycle == ref_ncycle, \
                f"{nranks} ranks took {ncycle} cycles, {_RANKS[0]} rank(s) took " \
                f"{ref_ncycle}: the timestep limit is not rank-invariant"
            assert dt == pytest.approx(ref_dt, rel=1.0e-12), \
                f"{nranks} ranks ended at dt = {dt:g} vs {ref_dt:g}; the MPI_Allreduce " \
                f"in LimitDtByQrad is not giving every rank the same bound"
    finally:
        _clean()
