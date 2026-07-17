#ifndef NR_RADIATION_VET_VET_INTERP_HPP_
#define NR_RADIATION_VET_VET_INTERP_HPP_
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

namespace nr_radiation {

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

//----------------------------------------------------------------------------------------
//! \fn Real UpdateCellSC()
//! \brief Per-cell short-characteristics intensity update (Davis, Stone & Jiang 2012
//! Eq. 20) for all dimensionalities (1D/2D/3D). Encapsulates upwind/downwind footpoint
//! interpolation (linear in 2D, bilinear in 3D) and calls InterpQuadChi +
//! InterpQuadSourceSlopeLim. Returns the updated intensity for cell (m,angg,k,j,i).
//!
//! Parameters:
//!   chi_       — opacity (nmb,nx3,nx2,nx1)
//!   bb_        — source S as 5D (nmb,1,nx3,nx2,nx1) for CC boundary exchange
//!   ir_        — intensity array (nmb,nang_tot,nx3,nx2,nx1)
//!   m, angg    — meshblock and flattened angle index
//!   i, j, k    — cell indices in the active domain
//!   sx,sy,sz   — sign(mu_x/y/z), encoding sweep direction (+1 or -1; sy=sz=0 if unused)
//!   mux,muy,muz— direction cosines (muy/muz unused in lower dims)
//!   dx1,dx2,dx3— cell widths
//!   ndim       — 1, 2 or 3
//!   ks, js     — start-of-active-zone indices for k, j
//!   a1_out     — if non-null, returns Ψ⁰ (local-source weight) for Λ* accumulation

KOKKOS_INLINE_FUNCTION
Real UpdateCellSC(
    const DvceArray4D<Real> &chi_, const DvceArray5D<Real> &bb_,
    const DvceArray5D<Real> &ir_, int m, int angg,
    int i, int j, int k,
    int sx, int sy, int sz,
    Real mux, Real muy, Real muz,
    Real dx1, Real dx2, Real dx3,
    int ndim, int ks, int js,
    Real *a1_out) {
  int im = i - sx, ip = i + sx;
  Real chi1 = chi_(m,k,j,i);
  Real S1   = bb_(m,0,k,j,i);

  Real S0, S2, chi0, chi2, imu0, dtaum, dtaup;

  if (ndim == 1) {
    chi0 = chi_(m,ks,js,im);
    chi2 = chi_(m,ks,js,ip);
    S0   = bb_(m,0,ks,js,im);
    S2   = bb_(m,0,ks,js,ip);
    imu0 = ir_(m,angg,ks,js,im);
    dtaum = InterpQuadChi(chi0,chi1,chi2) * dx1 / fabs(mux);
    dtaup = InterpQuadChi(chi2,chi1,chi0) * dx1 / fabs(mux);
  } else if (ndim == 2) {
    int jm = j - sy, jp = j + sy;
    Real am = fabs((dx2*mux) / (dx1*muy));
    if (am <= 1.0) {
      Real am1 = 1.0 - am;
      S0    = am*bb_(m,0,ks,jm,im)  + am1*bb_(m,0,ks,jm,i);
      S2    = am*bb_(m,0,ks,jp,ip)  + am1*bb_(m,0,ks,jp,i);
      chi0  = am*chi_(m,ks,jm,im) + am1*chi_(m,ks,jm,i);
      chi2  = am*chi_(m,ks,jp,ip) + am1*chi_(m,ks,jp,i);
      imu0  = am*ir_(m,angg,ks,jm,im) + am1*ir_(m,angg,ks,jm,i);
      dtaum = InterpQuadChi(chi0,chi1,chi2) * dx2/fabs(muy);
      dtaup = InterpQuadChi(chi2,chi1,chi0) * dx2/fabs(muy);
    } else {
      Real bm  = 1.0/am;
      Real bm1 = 1.0 - bm;
      S0    = bm*bb_(m,0,ks,jm,im)  + bm1*bb_(m,0,ks,j,im);
      S2    = bm*bb_(m,0,ks,jp,ip)  + bm1*bb_(m,0,ks,j,ip);
      chi0  = bm*chi_(m,ks,jm,im) + bm1*chi_(m,ks,j,im);
      chi2  = bm*chi_(m,ks,jp,ip) + bm1*chi_(m,ks,j,ip);
      imu0  = bm*ir_(m,angg,ks,jm,im) + bm1*ir_(m,angg,ks,j,im);
      dtaum = InterpQuadChi(chi0,chi1,chi2) * dx1/fabs(mux);
      dtaup = InterpQuadChi(chi2,chi1,chi0) * dx1/fabs(mux);
    }
  } else {
    int jm = j - sy, jp = j + sy;
    int km = k - sz, kp = k + sz;

    Real lx = dx1/fabs(mux), ly = dx2/fabs(muy), lz = dx3/fabs(muz);
    Real lmin = fmin(fmin(lx,ly),lz);

    if (lmin == lx) {
      Real am_r = lmin/ly, bm = lmin/lz;
      Real c0 = (1.0-am_r)*(1.0-bm), c1_ = (1.0-am_r)*bm;
      Real c2_ = am_r*bm, c3 = am_r*(1.0-bm);
      S0    = c0*bb_(m,0,k ,j ,im) + c1_*bb_(m,0,km,j ,im) +
              c2_*bb_(m,0,km,jm,im) + c3*bb_(m,0,k ,jm,im);
      chi0  = c0*chi_(m,k ,j ,im) + c1_*chi_(m,km,j ,im) +
              c2_*chi_(m,km,jm,im) + c3*chi_(m,k ,jm,im);
      imu0  = c0*ir_(m,angg,k ,j ,im) + c1_*ir_(m,angg,km,j ,im) +
              c2_*ir_(m,angg,km,jm,im) + c3*ir_(m,angg,k ,jm,im);
      S2    = c0*bb_(m,0,k ,j ,ip) + c1_*bb_(m,0,kp,j ,ip) +
              c2_*bb_(m,0,kp,jp,ip) + c3*bb_(m,0,k ,jp,ip);
      chi2  = c0*chi_(m,k ,j ,ip) + c1_*chi_(m,kp,j ,ip) +
              c2_*chi_(m,kp,jp,ip) + c3*chi_(m,k ,jp,ip);
      dtaum = InterpQuadChi(chi0,chi1,chi2) * dx1/fabs(mux);
      dtaup = InterpQuadChi(chi2,chi1,chi0) * dx1/fabs(mux);
    } else if (lmin == ly) {
      Real am_r = lmin/lx, bm = lmin/lz;
      Real c0 = (1.0-am_r)*(1.0-bm), c1_ = (1.0-am_r)*bm;
      Real c2_ = am_r*bm, c3 = am_r*(1.0-bm);
      S0    = c0*bb_(m,0,k ,jm,i ) + c1_*bb_(m,0,km,jm,i ) +
              c2_*bb_(m,0,km,jm,im) + c3*bb_(m,0,k ,jm,im);
      chi0  = c0*chi_(m,k ,jm,i ) + c1_*chi_(m,km,jm,i ) +
              c2_*chi_(m,km,jm,im) + c3*chi_(m,k ,jm,im);
      imu0  = c0*ir_(m,angg,k ,jm,i ) + c1_*ir_(m,angg,km,jm,i ) +
              c2_*ir_(m,angg,km,jm,im) + c3*ir_(m,angg,k ,jm,im);
      S2    = c0*bb_(m,0,k ,jp,i ) + c1_*bb_(m,0,kp,jp,i ) +
              c2_*bb_(m,0,kp,jp,ip) + c3*bb_(m,0,k ,jp,ip);
      chi2  = c0*chi_(m,k ,jp,i ) + c1_*chi_(m,kp,jp,i ) +
              c2_*chi_(m,kp,jp,ip) + c3*chi_(m,k ,jp,ip);
      dtaum = InterpQuadChi(chi0,chi1,chi2) * dx2/fabs(muy);
      dtaup = InterpQuadChi(chi2,chi1,chi0) * dx2/fabs(muy);
    } else {
      Real am_r = lmin/lx, bm = lmin/ly;
      Real c0 = (1.0-am_r)*(1.0-bm), c1_ = (1.0-am_r)*bm;
      Real c2_ = am_r*bm, c3 = am_r*(1.0-bm);
      S0    = c0*bb_(m,0,km,j ,i ) + c1_*bb_(m,0,km,jm,i ) +
              c2_*bb_(m,0,km,jm,im) + c3*bb_(m,0,km,j ,im);
      chi0  = c0*chi_(m,km,j ,i ) + c1_*chi_(m,km,jm,i ) +
              c2_*chi_(m,km,jm,im) + c3*chi_(m,km,j ,im);
      imu0  = c0*ir_(m,angg,km,j ,i ) + c1_*ir_(m,angg,km,jm,i ) +
              c2_*ir_(m,angg,km,jm,im) + c3*ir_(m,angg,km,j ,im);
      S2    = c0*bb_(m,0,kp,j ,i ) + c1_*bb_(m,0,kp,jp,i ) +
              c2_*bb_(m,0,kp,jp,ip) + c3*bb_(m,0,kp,j ,ip);
      chi2  = c0*chi_(m,kp,j ,i ) + c1_*chi_(m,kp,jp,i ) +
              c2_*chi_(m,kp,jp,ip) + c3*chi_(m,kp,j ,ip);
      dtaum = InterpQuadChi(chi0,chi1,chi2) * dx3/fabs(muz);
      dtaup = InterpQuadChi(chi2,chi1,chi0) * dx3/fabs(muz);
    }
  }

  Real edtau, a0, a1, a2;
  InterpQuadSourceSlopeLim(dtaum, dtaup, S0, S1, S2, &edtau, &a0, &a1, &a2);
  if (a1_out != nullptr) { *a1_out = a1; }
  return a0*S0 + a1*S1 + a2*S2 + edtau*imu0;
}

}  // namespace nr_radiation

#endif  // NR_RADIATION_VET_VET_INTERP_HPP_
