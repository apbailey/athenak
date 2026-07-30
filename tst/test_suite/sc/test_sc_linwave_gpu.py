"""
SC coupled-dynamics test (level 4): radiatively-damped acoustic wave (Davis+2012 sec 5.4).

A small-amplitude acoustic wave in a radiating gas is damped and phase-shifted by radiative
heating/cooling. The pgen evolves the coupled rad-hydro system for one period, solves the
analytic cubic dispersion relation (phase speed omega_r and damping omega_i, both / (k a)),
Fourier-fits the measured density mode, and writes both plus the L1 error of the full state vs
the analytic damped wave to SCLinWave-davis54.dat.

This test checks TWO things over a resolution ladder (N = 256, 512, 1024):
  physics      the fitted phase speed and damping match the analytic dispersion (at finest N)
  convergence  the L1 rms error decreases at ~1st order (operator-split coupling is first order
               in the CFL-limited regime N>=256, Davis+2012 sec 6)

It is the only test exercising the two-way rad-hydro coupling (affect_fluid=true) as a
time-dependent dynamics problem, so it guards the operator-split energy source term end to end.

SCLinWave-davis54.dat columns (see sc_linwave.cpp::SCLinwaveErrors):
  Bo  tau  nx1  nx2  omega_r  omega_i  omega_r_fit  omega_i_fit  rms  d_err  e_err  m1_err
   0    1    2    3      4        5          6            7         8     9      10     11
"""

# Modules
import math
import test_suite.testutils as testutils

_RES = [256, 512, 1024]            # CFL-limited 1st-order regime (Davis sec 5.4)
# GPU-only: cap the meshblock so large N decomposes into multiple blocks. The hydro flux kernel
# requests per-team shared-memory scratch = 2*nvars*(nx1+2*nghost)*8 B, which must fit the GPU's
# ~48 KB/block limit; a single block with nx1 > ~600 overruns it and Kokkos throws (aborts). Block
# decomposition is physics-identical (single rank, periodic), so the convergence ladder is unchanged.
_MB = 256
_OMEGA_R, _OMEGA_I = 4, 5          # analytic phase speed / damping, omega/(k a)
_OMEGA_R_FIT, _OMEGA_I_FIT = 6, 7  # Fourier-fitted from the evolved wave
_RMS = 8                           # L1 rms error of the full state vs the analytic wave


def test_sc_linwave_1d():
    testutils.cleanup()  # clear any stale SCLinWave-davis54.dat
    try:
        for n in _RES:
            assert testutils.run("inputs/sc_linwave_1d.athinput",
                                 [f"mesh/nx1={n}", f"meshblock/nx1={min(n, _MB)}"]), \
                f"run failed at N={n}"
        data = testutils.athena_read.error_dat("SCLinWave-davis54.dat")
        rms = [row[_RMS] for row in data]

        # --- physics: fitted dispersion matches analytic at the finest resolution ---
        fine = data[-1]
        assert abs(fine[_OMEGA_R_FIT] - fine[_OMEGA_R]) < 1.0e-3, \
            f"phase speed off: fitted {fine[_OMEGA_R_FIT]:g} vs analytic {fine[_OMEGA_R]:g}"
        assert abs(fine[_OMEGA_I_FIT] - fine[_OMEGA_I]) < 1.0e-3, \
            f"damping off: fitted {fine[_OMEGA_I_FIT]:g} vs analytic {fine[_OMEGA_I]:g}"

        # --- convergence: L1 rms error decreases at ~1st order ---
        assert all(rms[i + 1] < rms[i] for i in range(len(rms) - 1)), \
            f"L1 rms not monotonically decreasing: {rms}"
        order = math.log(rms[0] / rms[-1]) / math.log(_RES[-1] / _RES[0])
        assert order > 0.6, f"insufficient convergence: observed order {order:.2f} (rms {rms})"
    finally:
        testutils.cleanup()
