#ifndef NR_RADIATION_SC_SC_INTERP_HPP_
#define NR_RADIATION_SC_SC_INTERP_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file sc_interp.hpp
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
//!   srad_        — source S as 5D (nmb,1,nx3,nx2,nx1) for CC boundary exchange
//!   ir_        — intensity array (nmb,nang_tot,nx3,nx2,nx1)
//!   m, angg    — meshblock and flattened angle index
//!   i, j, k    — cell indices in the active domain
//!   sx,sy,sz   — sign(mu_x/y/z), encoding sweep direction (+1 or -1; sy=sz=0 if unused)
//!   mux,muy,muz— direction cosines (muy/muz unused in lower dims)
//!   dx1,dx2,dx3— cell widths
//!   ndim       — 1, 2 or 3
//!   ks, js     — start-of-active-zone indices for k, j
//!   a1_out     — if non-null, returns Ψ⁰ (local-source weight) for Λ* accumulation
//!   a0_out, edtau_out — if non-null, also return a0 and e^{-Δτ} (Eq. 20). Together with
//!     a1_out these give the local-scatter coupling coefficient w·(a0+e^{-Δτ}·a1) used by the
//!     Gauss-Seidel "psiint" correction (Olson & Kunasz 1987; TF95; SweepUpdateGS in
//!     formal_solution.cpp) — the sensitivity of a downstream cell's intensity on this ray to a
//!     change in *this* cell's S, evaluated here as a zeroth-order (same-cell) stand-in for the
//!     downstream cell's own coefficients (exact only for a locally-uniform medium; see
//!     SweepUpdateGS's doc comment for the approximation this implies).

//----------------------------------------------------------------------------------------
//! \fn Real IrGet()
//! \brief Intensity read that hides the `ir` storage layout. ANG_INNER=false is the default
//! layout `ir(m,angg,k,j,i)`; ANG_INNER=true is the angle-innermost scratch `ir(m,k,j,i,angg)`
//! used by sweep=wavefront_coalesced (I2 gather-coalescing prototype). Same value either way, so
//! callers stay bit-identical; only the memory access pattern differs.

template <bool ANG_INNER>
KOKKOS_INLINE_FUNCTION
Real IrGet(const DvceArray5D<Real> &ir_, int m, int angg, int k, int j, int i) {
  if constexpr (ANG_INNER) {
    return ir_(m, k, j, i, angg);
  } else {
    return ir_(m, angg, k, j, i);
  }
}

//----------------------------------------------------------------------------------------
//! \struct SCRayInv
//! \brief Per-(octant,angle) short-characteristics interpolation invariants: the
//! sweep-direction signs, the dominant-axis selector, the upwind/downwind interpolation
//! weights, and the dominant-direction (dx, |mu|) that set the optical-depth path length.
//! These depend only on the ray direction and the (uniform) cell size dx — NOT on the cell
//! (k,j,i) — so they can be hoisted out of the per-cell loop (optimization I3): precomputed
//! once per ray into a table (sweep=wavefront) or once per team (sweep=diagonal). Computed by
//! ComputeSCAngleInv(); consumed by GatherSolveSC(). This is a pure reorganisation —
//! GatherSolveSC reproduces UpdateCellSC's arithmetic bit-for-bit (see the pdx/pamu note).

struct SCRayInv {
  int sx, sy, sz;        // sign(mu_x/y/z); +/-1 (sy=sz=0 if the dim is unused)
  int axis;              // 3D: 0/1/2 = x/y/z-dominant; 2D: 0 = |am|<=1, 1 = |am|>1; 1D: 0
  Real c0, c1, c2, c3;   // interp weights (3D: 4 bilinear; 2D: c0,c1 linear; 1D: unused)
  Real pdx, pamu;        // dominant dx and |mu|: dtau = InterpQuadChi(...) * pdx / pamu
};

//----------------------------------------------------------------------------------------
//! \fn SCRayInv ComputeSCAngleInv()
//! \brief Compute the per-ray SC interpolation invariants (I3 hoist target). Extracted
//! verbatim from UpdateCellSC's angle-only arithmetic; results are bit-identical. Note the
//! path length is carried as the pair (pdx = dominant dx, pamu = |mu_dom|), NOT a single
//! quotient, because the caller applies it as `InterpQuadChi(...) * pdx / pamu` — the same
//! left-associative (x*dx)/|mu| order as the original, so no rounding change.

KOKKOS_INLINE_FUNCTION
SCRayInv ComputeSCAngleInv(Real mux, Real muy, Real muz,
                           Real dx1, Real dx2, Real dx3, int ndim,
                           int sx, int sy, int sz) {
  SCRayInv inv;
  inv.sx = sx; inv.sy = sy; inv.sz = sz;
  inv.axis = 0;
  inv.c0 = 0.0; inv.c1 = 0.0; inv.c2 = 0.0; inv.c3 = 0.0;
  if (ndim == 1) {
    inv.pdx = dx1; inv.pamu = fabs(mux);
  } else if (ndim == 2) {
    Real am = fabs((dx2*mux) / (dx1*muy));
    if (am <= 1.0) {
      inv.axis = 0;
      inv.c0 = am; inv.c1 = 1.0 - am;
      inv.pdx = dx2; inv.pamu = fabs(muy);
    } else {
      inv.axis = 1;
      Real bm = 1.0/am;
      inv.c0 = bm; inv.c1 = 1.0 - bm;
      inv.pdx = dx1; inv.pamu = fabs(mux);
    }
  } else {
    Real lx = dx1/fabs(mux), ly = dx2/fabs(muy), lz = dx3/fabs(muz);
    Real lmin = fmin(fmin(lx,ly),lz);
    if (lmin == lx) {
      inv.axis = 0;
      Real am_r = lmin/ly, bm = lmin/lz;
      inv.c0 = (1.0-am_r)*(1.0-bm); inv.c1 = (1.0-am_r)*bm;
      inv.c2 = am_r*bm;             inv.c3 = am_r*(1.0-bm);
      inv.pdx = dx1; inv.pamu = fabs(mux);
    } else if (lmin == ly) {
      inv.axis = 1;
      Real am_r = lmin/lx, bm = lmin/lz;
      inv.c0 = (1.0-am_r)*(1.0-bm); inv.c1 = (1.0-am_r)*bm;
      inv.c2 = am_r*bm;             inv.c3 = am_r*(1.0-bm);
      inv.pdx = dx2; inv.pamu = fabs(muy);
    } else {
      inv.axis = 2;
      Real am_r = lmin/lx, bm = lmin/ly;
      inv.c0 = (1.0-am_r)*(1.0-bm); inv.c1 = (1.0-am_r)*bm;
      inv.c2 = am_r*bm;             inv.c3 = am_r*(1.0-bm);
      inv.pdx = dx3; inv.pamu = fabs(muz);
    }
  }
  return inv;
}

//----------------------------------------------------------------------------------------
//! \fn Real GatherSolveSC()
//! \brief The per-cell short-characteristics gather + Eq. 20 solve, given the precomputed
//! per-ray invariants `inv` (from ComputeSCAngleInv). This is UpdateCellSC's body with the
//! angle-only arithmetic factored out; bit-identical results. Called directly by the hoisted
//! I3 sweep variants (with `inv` read from a table or hoisted to team scope), and via the thin
//! UpdateCellSC wrapper by every other call site.

template <bool ANG_INNER = false>
KOKKOS_INLINE_FUNCTION
Real GatherSolveSC(
    const DvceArray4D<Real> &chi_, const DvceArray5D<Real> &srad_,
    const DvceArray5D<Real> &ir_, int m, int angg,
    int i, int j, int k, const SCRayInv &inv,
    int ndim, int ks, int js,
    Real *a1_out, Real *a0_out = nullptr, Real *edtau_out = nullptr) {
  int sx = inv.sx, sy = inv.sy, sz = inv.sz;
  int im = i - sx, ip = i + sx;
  Real chi1 = chi_(m,k,j,i);
  Real S1   = srad_(m,0,k,j,i);
  Real c0 = inv.c0, c1_ = inv.c1, c2_ = inv.c2, c3 = inv.c3;

  Real S0, S2, chi0, chi2, imu0, dtaum, dtaup;

  if (ndim == 1) {
    chi0 = chi_(m,ks,js,im);
    chi2 = chi_(m,ks,js,ip);
    S0   = srad_(m,0,ks,js,im);
    S2   = srad_(m,0,ks,js,ip);
    imu0 = IrGet<ANG_INNER>(ir_,m,angg,ks,js,im);
  } else if (ndim == 2) {
    int jm = j - sy, jp = j + sy;
    if (inv.axis == 0) {          // |am| <= 1
      S0    = c0*srad_(m,0,ks,jm,im)  + c1_*srad_(m,0,ks,jm,i);
      S2    = c0*srad_(m,0,ks,jp,ip)  + c1_*srad_(m,0,ks,jp,i);
      chi0  = c0*chi_(m,ks,jm,im) + c1_*chi_(m,ks,jm,i);
      chi2  = c0*chi_(m,ks,jp,ip) + c1_*chi_(m,ks,jp,i);
      imu0  = c0*IrGet<ANG_INNER>(ir_,m,angg,ks,jm,im) + c1_*IrGet<ANG_INNER>(ir_,m,angg,ks,jm,i);
    } else {                      // |am| > 1
      S0    = c0*srad_(m,0,ks,jm,im)  + c1_*srad_(m,0,ks,j,im);
      S2    = c0*srad_(m,0,ks,jp,ip)  + c1_*srad_(m,0,ks,j,ip);
      chi0  = c0*chi_(m,ks,jm,im) + c1_*chi_(m,ks,j,im);
      chi2  = c0*chi_(m,ks,jp,ip) + c1_*chi_(m,ks,j,ip);
      imu0  = c0*IrGet<ANG_INNER>(ir_,m,angg,ks,jm,im) + c1_*IrGet<ANG_INNER>(ir_,m,angg,ks,j,im);
    }
  } else {
    int jm = j - sy, jp = j + sy;
    int km = k - sz, kp = k + sz;
    if (inv.axis == 0) {          // x-dominant
      S0    = c0*srad_(m,0,k ,j ,im) + c1_*srad_(m,0,km,j ,im) +
              c2_*srad_(m,0,km,jm,im) + c3*srad_(m,0,k ,jm,im);
      chi0  = c0*chi_(m,k ,j ,im) + c1_*chi_(m,km,j ,im) +
              c2_*chi_(m,km,jm,im) + c3*chi_(m,k ,jm,im);
      imu0  = c0*IrGet<ANG_INNER>(ir_,m,angg,k ,j ,im) + c1_*IrGet<ANG_INNER>(ir_,m,angg,km,j ,im) +
              c2_*IrGet<ANG_INNER>(ir_,m,angg,km,jm,im) + c3*IrGet<ANG_INNER>(ir_,m,angg,k ,jm,im);
      S2    = c0*srad_(m,0,k ,j ,ip) + c1_*srad_(m,0,kp,j ,ip) +
              c2_*srad_(m,0,kp,jp,ip) + c3*srad_(m,0,k ,jp,ip);
      chi2  = c0*chi_(m,k ,j ,ip) + c1_*chi_(m,kp,j ,ip) +
              c2_*chi_(m,kp,jp,ip) + c3*chi_(m,k ,jp,ip);
    } else if (inv.axis == 1) {   // y-dominant
      S0    = c0*srad_(m,0,k ,jm,i ) + c1_*srad_(m,0,km,jm,i ) +
              c2_*srad_(m,0,km,jm,im) + c3*srad_(m,0,k ,jm,im);
      chi0  = c0*chi_(m,k ,jm,i ) + c1_*chi_(m,km,jm,i ) +
              c2_*chi_(m,km,jm,im) + c3*chi_(m,k ,jm,im);
      imu0  = c0*IrGet<ANG_INNER>(ir_,m,angg,k ,jm,i ) + c1_*IrGet<ANG_INNER>(ir_,m,angg,km,jm,i ) +
              c2_*IrGet<ANG_INNER>(ir_,m,angg,km,jm,im) + c3*IrGet<ANG_INNER>(ir_,m,angg,k ,jm,im);
      S2    = c0*srad_(m,0,k ,jp,i ) + c1_*srad_(m,0,kp,jp,i ) +
              c2_*srad_(m,0,kp,jp,ip) + c3*srad_(m,0,k ,jp,ip);
      chi2  = c0*chi_(m,k ,jp,i ) + c1_*chi_(m,kp,jp,i ) +
              c2_*chi_(m,kp,jp,ip) + c3*chi_(m,k ,jp,ip);
    } else {                      // z-dominant
      S0    = c0*srad_(m,0,km,j ,i ) + c1_*srad_(m,0,km,jm,i ) +
              c2_*srad_(m,0,km,jm,im) + c3*srad_(m,0,km,j ,im);
      chi0  = c0*chi_(m,km,j ,i ) + c1_*chi_(m,km,jm,i ) +
              c2_*chi_(m,km,jm,im) + c3*chi_(m,km,j ,im);
      imu0  = c0*IrGet<ANG_INNER>(ir_,m,angg,km,j ,i ) + c1_*IrGet<ANG_INNER>(ir_,m,angg,km,jm,i ) +
              c2_*IrGet<ANG_INNER>(ir_,m,angg,km,jm,im) + c3*IrGet<ANG_INNER>(ir_,m,angg,km,j ,im);
      S2    = c0*srad_(m,0,kp,j ,i ) + c1_*srad_(m,0,kp,jp,i ) +
              c2_*srad_(m,0,kp,jp,ip) + c3*srad_(m,0,kp,j ,ip);
      chi2  = c0*chi_(m,kp,j ,i ) + c1_*chi_(m,kp,jp,i ) +
              c2_*chi_(m,kp,jp,ip) + c3*chi_(m,kp,j ,ip);
    }
  }

  // Path-length applied as (x*pdx)/pamu — the exact left-associative order of the original
  // `InterpQuadChi(...) * dx_dom / fabs(mu_dom)`, so dtau is bit-identical.
  dtaum = InterpQuadChi(chi0,chi1,chi2) * inv.pdx / inv.pamu;
  dtaup = InterpQuadChi(chi2,chi1,chi0) * inv.pdx / inv.pamu;

  Real edtau, a0, a1, a2;
  InterpQuadSourceSlopeLim(dtaum, dtaup, S0, S1, S2, &edtau, &a0, &a1, &a2);
  if (a1_out != nullptr) { *a1_out = a1; }
  if (a0_out != nullptr) { *a0_out = a0; }
  if (edtau_out != nullptr) { *edtau_out = edtau; }
  return a0*S0 + a1*S1 + a2*S2 + edtau*imu0;
}

//----------------------------------------------------------------------------------------
//! \fn Real UpdateCellSC()
//! \brief Thin wrapper: compute the per-ray invariants then do the per-cell gather+solve.
//! Signature unchanged from the original monolithic version, so every existing call site is
//! bit-identical. The hoisted I3 sweep variants bypass this and call GatherSolveSC directly
//! with a precomputed `inv`.

template <bool ANG_INNER = false>
KOKKOS_INLINE_FUNCTION
Real UpdateCellSC(
    const DvceArray4D<Real> &chi_, const DvceArray5D<Real> &srad_,
    const DvceArray5D<Real> &ir_, int m, int angg,
    int i, int j, int k,
    int sx, int sy, int sz,
    Real mux, Real muy, Real muz,
    Real dx1, Real dx2, Real dx3,
    int ndim, int ks, int js,
    Real *a1_out, Real *a0_out = nullptr, Real *edtau_out = nullptr) {
  SCRayInv inv = ComputeSCAngleInv(mux, muy, muz, dx1, dx2, dx3, ndim, sx, sy, sz);
  return GatherSolveSC<ANG_INNER>(chi_, srad_, ir_, m, angg, i, j, k, inv,
                                  ndim, ks, js, a1_out, a0_out, edtau_out);
}

}  // namespace nr_radiation

#endif  // NR_RADIATION_SC_SC_INTERP_HPP_
