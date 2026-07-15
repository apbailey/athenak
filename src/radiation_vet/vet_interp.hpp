#ifndef RADIATION_VET_VET_INTERP_HPP_
#define RADIATION_VET_VET_INTERP_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file vet_interp.hpp
//! \brief Short-characteristics interpolation coefficients for the Eq. 20 formal-solution
//! update (Davis, Stone & Jiang 2012), following Kunasz & Auer (1988, JQSRT 39, 67) for
//! the quadratic source-function interpolation, with the Bezier-type opacity
//! interpolation and slope-limited source control point of Auer (2003, ASP 288, 3) /
//! Hayek et al. (2010, A&A 517, 49) that keep I >= 0 for positive S. Both routines are
//! direct, validated ports of Athena-C's radiation/utils_rad.c (interp_quad_chi(),
//! interp_quad_source_slope_lim()) which implement exactly this scheme; ported here as
//! KOKKOS_INLINE_FUNCTION so they can be called from device sweep kernels.

#include <cmath>

#include "athena.hpp"

namespace radiation_vet {

//----------------------------------------------------------------------------------------
//! \fn Real InterpQuadChi()
//! \brief 2nd-order (parabolic) representation of the opacity chi at the upwind or
//! downwind footpoint of a ray, with Bezier-type slope limiting to avoid overshoots
//! (Auer 2003). chi0,chi1,chi2 are the opacities at the far, near, and center points
//! of the 3-point stencil along the ray (chi1 = local cell).

KOKKOS_INLINE_FUNCTION
Real InterpQuadChi(Real chi0, Real chi1, Real chi2) {
  Real chic = chi1 - 0.25*(chi2 - chi0);
  if ((chi0-chic)*(chi1-chic) <= 0.0) {
    return 0.4166666666666667*chi0 + 0.6666666666666667*chi1 - 0.0833333333333333*chi2;
  } else {
    return 0.3333333333333333*chi0 + 0.6666666666666667*chi1;
  }
}

//----------------------------------------------------------------------------------------
//! \fn void InterpQuadSourceSlopeLim()
//! \brief Eq. 20 formal-solution weights (edtau, a0, a1, a2) from quadratic
//! interpolation of the source function S along the ray, with Bezier slope limiting
//! (Kunasz & Auer 1988; Auer 2003). dtaum, dtaup are the upwind/downwind optical-depth
//! increments; S0, S1, S2 are the source function at the upwind footpoint, the local
//! cell, and the downwind footpoint. On return:
//!   I_cell = a0*S0 + a1*S1 + a2*S2 + edtau*I_upwind        (Eq. 20)

KOKKOS_INLINE_FUNCTION
void InterpQuadSourceSlopeLim(Real dtaum, Real dtaup, Real S0, Real S1, Real S2,
                              Real *edtau, Real *a0, Real *a1, Real *a2) {
  const Real taumin = 1.0e-6;
  if (dtaum < taumin) {
    // expand to 2nd order in dtaum to avoid catastrophic cancellation at small tau
    (*edtau) = 1.0 - dtaum*(1.0 - 0.5*dtaum);
    Real taurat = dtaup / dtaum;
    Real Sc = S1 - 0.5/(1.0+taurat) * (taurat*(S1-S0) + (S2-S1)/taurat);
    if ((S0-Sc)*(S1-Sc) <= 0.0) {
      (*a0) = dtaum * (taurat/(1.0+taurat) - 0.5*dtaum);
      (*a1) = dtaum * 0.5 * (1.0+taurat) / taurat;
      (*a2) = -dtaum * 0.5 / (taurat*(1.0+taurat));
    } else {
      (*a0) = 0.6666666666666667 * dtaum;
      (*a1) = 0.3333333333333333 * dtaum;
      (*a2) = 0.0;
    }
  } else {
    Real dtaus  = dtaum + dtaup;
    Real dtaus1 = 1.0 / dtaus;
    Real dtaup1 = 1.0 / dtaup;
    Real dtaum1 = 1.0 / dtaum;
    Real dtausp1 = dtaus1 * dtaup1;
    Real dtaum2 = dtaum * dtaum;

    (*edtau) = exp(-dtaum);
    Real c0 = 1.0 - (*edtau);
    Real c1 = dtaum - c0;
    Real c2 = dtaum2 - 2.0*c1;

    Real Sc = S1 - 0.5*(dtaup*(S1-S0)*dtaus1 + dtaum2*(S2-S1)*dtausp1);
    if ((S0-Sc)*(S1-Sc) <= 0.0) {
      (*a0) = c0 + (c2 - (dtaus+dtaum)*c1) * dtaum1 * dtaus1;
      (*a1) = (dtaus*c1 - c2) * dtaum1 * dtaup1;
      (*a2) = (c2 - dtaum*c1) * dtausp1;
    } else {
      (*a1) = c2 / dtaum2;
      (*a0) = c0 - (*a1);
      (*a2) = 0.0;
    }
  }
}

}  // namespace radiation_vet

#endif  // RADIATION_VET_VET_INTERP_HPP_
