"""
SC coupling test: does the timestep bound the gas-energy update that Q_rad drives?

Cold uniform gas lit from one face. kappa_a and p0 are small enough that the
radiation-relaxation limit of sc/newdt.cpp,

    nu_rad = 4(gamma-1) T^3 (sigma_a/rho) crat prat / (1 + 3(chi dx/pi)^2),

is inert by roughly twenty orders of magnitude, while Q is set by the incident beam and
is not small. Columns of sc_qrad_dt-errs.dat:
Nx1 Nx2 Nx3 Ncycle min_eint dt max_absQ cfl_qrad nbad.

Two distinct failures are exercised, and it is worth being precise about which is which,
because they are often conflated:

  * divh: Q = -crat prat div(H) has no relation to sigma_a at all, so nu_rad -> 0 while
    |Q| stays large. This is the failure the cfl_qrad limit was added for.
  * integral: Q = crat prat sigma_a (J - brad) IS proportional to sigma_a, and in the
    thin COOLING limit (J << brad) 1/nu_rad = (1/4) e_int/|Q| identically, so nu_rad
    bounds it for free. That identity does not hold in the thin HEATING limit (J >> brad)
    reached here, where nu_rad is evaluated at the cold pre-heating temperature and badly
    underestimates the rate. Measured: the integral form runs away just as hard, to
    max|Q| ~ 1e11, and the same cfl_qrad limit rescues it.

The second point is why cfl_qrad defaults to 0 for the integral form but is not claimed
to be unnecessary there -- see theory/timestep-constraints.md Sec. 4.1. The default is 0
to keep every existing integral run bit-identical, not because the form is safe.

nbad (column 8) counts cells with e_int not strictly positive, which is also true for
NaN; fmin would silently swallow a NaN and report a healthy minimum, so assertions use
the event log and nbad rather than min_eint alone.
"""

import os
import pytest
import test_suite.testutils as testutils

_INPUT = "inputs/sc_qrad_dt.athinput"
_ERRS = "sc_qrad_dt-errs.dat"
_LOG = "SCQradDt.log"

# measured 2026-09-29 on CPU: guarded dt 6.9e-10 vs a hydro CFL of 7.3, so the limit is
# binding by ~10 orders. A generous gate still excludes "passed because nothing happened".
_DT_LIMITED_MAX = 1.0e-4
# unguarded max|Q| reaches ~1e38 (divh) and ~1e11 (integral) against a true value of ~1e3
_BLOWUP_MIN = 1.0e6


def _row():
    data = testutils.athena_read.error_dat(_ERRS)
    assert len(data) == 1, f"expected one row in {_ERRS}, got {len(data)}"
    return data[0]


def _events():
    """Summed event counters from the log, or all-zero if nothing was ever written.

    eventlog.cpp only writes a row on cycles where some counter is non-zero, so a clean
    run leaves a header-only file (or none at all, if no cycle triggered it).
    """
    totals = {"eos_dfloor": 0, "eos_efloor": 0, "eos_tfloor": 0, "qrad_neg": 0}
    if not os.path.exists(_LOG):
        return totals
    with open(_LOG) as f:
        lines = [ln for ln in f if not ln.startswith("#")]
    for ln in lines:
        c = ln.split()
        if len(c) < 9:
            continue
        totals["eos_dfloor"] += int(c[1])
        totals["eos_efloor"] += int(c[2])
        totals["eos_tfloor"] += int(c[3])
        totals["qrad_neg"] += int(c[8])
    return totals


def _clean():
    testutils.cleanup()
    if os.path.exists(_LOG):
        os.remove(_LOG)


def _run(flags, label, input_file=_INPUT):
    _clean()
    assert testutils.run(input_file, flags), f"run failed: {label}"
    return _row(), _events()


@pytest.mark.parametrize("form", ["divh", "hybrid"])
def test_sc_qrad_dt_unguarded_blows_up(form):
    """Without the limit the differential form drives the internal energy negative.

    Asserted deliberately as a FAILURE of positivity: if this ever starts passing, either
    the deck has stopped being stiff or something else has begun bounding the update, and
    the guarded case below would then be passing for the wrong reason.
    """
    try:
        row, ev = _run([f"nr_radiation/qrad_form={form}", "nr_radiation/cfl_qrad=0.0"],
                       f"{form} unguarded")
        assert ev["eos_efloor"] > 0, \
            f"{form}: no energy floor fired without the cfl_qrad limit; the deck is no " \
            f"longer stiff enough for this test to mean anything"
        assert row[6] > _BLOWUP_MIN, \
            f"{form}: max|Q| = {row[6]:g} did not run away as expected " \
            f"(> {_BLOWUP_MIN:g})"
    finally:
        _clean()


@pytest.mark.parametrize("form", ["divh", "hybrid"])
def test_sc_qrad_dt_limited(form):
    """With the default cfl_qrad the same run stays positive, and is genuinely limited."""
    try:
        row, ev = _run([f"nr_radiation/qrad_form={form}", "nr_radiation/cfl_qrad=0.25"],
                       f"{form} limited")
        assert ev["eos_efloor"] == 0 and ev["eos_dfloor"] == 0, \
            f"{form}: floors fired despite the cfl_qrad limit: {ev}"
        assert ev["qrad_neg"] == 0, \
            f"{form}: {ev['qrad_neg']} cells would have been driven non-positive by " \
            f"Q_rad; the one-cycle margin in the limit was not enough"
        assert row[8] == 0, f"{form}: {int(row[8])} cells with non-positive or NaN e_int"
        assert row[4] > 0.0, f"{form}: min e_int = {row[4]:g} is not positive"
        assert row[5] < _DT_LIMITED_MAX, \
            f"{form}: dt = {row[5]:g} was never limited, so this run cannot be " \
            f"distinguished from one that simply never became stiff"
    finally:
        _clean()


def test_sc_qrad_dt_integral_is_not_protected_when_heating():
    """The integral form is NOT safe here either, and the same limit rescues it.

    nu_rad bounds Eq. 27 only in the thin cooling limit. This deck is the thin heating
    limit, where nu_rad is evaluated at the cold pre-heating temperature. Recorded as a
    test so the claim in theory/timestep-constraints.md Sec. 4.1 stays honest.
    """
    try:
        row0, ev0 = _run(["nr_radiation/qrad_form=integral", "nr_radiation/cfl_qrad=0.0"],
                         "integral unguarded")
        assert ev0["eos_efloor"] > 0 and row0[6] > _BLOWUP_MIN, \
            f"the integral form no longer runs away here (max|Q| = {row0[6]:g}, " \
            f"floors {ev0['eos_efloor']}); Sec. 4.1 of the theory note needs revisiting"

        row1, ev1 = _run(["nr_radiation/qrad_form=integral",
                          "nr_radiation/cfl_qrad=0.25"], "integral limited")
        assert ev1["eos_efloor"] == 0, \
            f"cfl_qrad did not rescue the integral form: {ev1}"
        assert row1[6] < _BLOWUP_MIN, \
            f"cfl_qrad did not stop the integral runaway: max|Q| = {row1[6]:g}"
    finally:
        _clean()


@pytest.mark.parametrize("form,expected", [("divh", 0.25), ("hybrid", 0.25),
                                           ("integral", 0.0)])
def test_sc_qrad_dt_default_depends_on_form(form, expected):
    """cfl_qrad defaults to 0.25 for the differential forms and 0 for the integral one.

    AthenaK only accepts command-line overrides of parameters already present in the
    deck, so the default path is reached by stripping the line rather than by an
    override. Tests the behaviour instead of trusting the comment in nr_radiation.cpp.
    """
    stripped = "sc_qrad_dt_nodefault.athinput"
    with open(_INPUT) as f:
        lines = [ln for ln in f if not ln.startswith("cfl_qrad")]
    with open(stripped, "w") as f:
        f.writelines(lines)
    try:
        row, _ = _run([f"nr_radiation/qrad_form={form}"], f"{form} default",
                      input_file=stripped)
        assert row[7] == pytest.approx(expected), \
            f"{form}: cfl_qrad defaulted to {row[7]:g}, expected {expected:g}"
    finally:
        _clean()
        if os.path.exists(stripped):
            os.remove(stripped)


def _run_capture(flags, label, input_file=_INPUT):
    """Run, and return (row, events, the stdout this run appended to the test log)."""
    _clean()
    log = testutils.LOG_FILE_PATH
    start = os.path.getsize(log) if os.path.exists(log) else 0
    assert testutils.run(input_file, flags), f"run failed: {label}"
    with open(log, errors="replace") as f:
        f.seek(start)
        out = f.read()
    return _row(), _events(), out


def test_sc_qrad_dt_warns_only_when_out_of_regime():
    """The limit warns when it is in charge by >=100x for >=10 cycles, and not otherwise.

    The distinction matters and is the point of the test. Measured on this deck: a
    slowdown of ~5-12x is ordinary accuracy control, ~20-120x is the band where the run
    is slow but correct while the unguarded run is quietly wrong, and a sustained factor
    beyond ~100 means the explicit operator split is simply out of regime -- a different
    integrator, not a smaller dt. Only the last case should nag.

    p0 sets which regime: e_int/|Q| scales with p0 while the hydro CFL goes as p0^-1/2,
    so the slowdown scales as p0^-3/2. p0=1e-6 gives ~1e10x, p0=1.0 gives ~12x.
    """
    marker = "Q_rad depletion limit"
    try:
        # deep in the stiff regime: must warn, exactly once, however long the run
        _, _, out = _run_capture(["problem/p0=1e-6", "time/nlim=30"], "out of regime")
        assert out.count(marker) == 1, \
            f"expected exactly one warning when the limit dominates by ~1e10x, got " \
            f"{out.count(marker)}"
        assert "Boltzmann" in out and "implicit" in out, \
            "the warning should name the cause and the remedy, not just the symptom"

        # limiting, but by ~12x: correct, affordable, and none of the user's business
        _, _, out = _run_capture(["problem/p0=1.0", "time/nlim=30"], "in regime")
        assert marker not in out, \
            "warned on a run the limit merely slows by ~12x; that is ordinary accuracy " \
            "control and nagging about it would train people to ignore the warning"
    finally:
        _clean()


def test_sc_qrad_dt_amr():
    """The limit still holds when blocks are refined and redistributed.

    qrad is not packed with the blocks across a remesh, so a limit evaluated from a stale
    qrad would read another block's heating. Evaluating it inside SolveTransfer, where Q
    has just been rebuilt, is what makes this pass; a limit placed in NewTimeStep would
    not.
    """
    try:
        row, ev = _run(["mesh_refinement/refinement=adaptive",
                        "mesh_refinement/num_levels=2",
                        "nr_radiation/cfl_qrad=0.25"], "divh AMR")
        assert ev["eos_efloor"] == 0 and ev["qrad_neg"] == 0, \
            f"AMR: the limit did not hold across a remesh: {ev}"
        assert row[8] == 0 and row[4] > 0.0, \
            f"AMR: min e_int = {row[4]:g}, {int(row[8])} bad cells"
    finally:
        _clean()
