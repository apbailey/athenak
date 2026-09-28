//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file get_moments.cpp
//! \brief Radiation moment quadrature (Davis 2012 Eq. 17-19): J every sweep for the
//! iteration, H and K once per solve; and the coupling
//! Q = crat * prat * sigma_a * (J - brad) (Davis Eq. 27 absorption form).
//! Moments array ordering (Athena++ convention):
//!   moments: 0:J, 1-3:H_1,H_2,H_3, 4:K_11, 5:K_22, 6:K_33, 7:K_12, 8:K_13, 9:K_23
//! -- Athena++ rad_mom order (IER, IFR1-3, IPR...), with the symmetric six K components

#include <algorithm>
#include <cmath>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "nr_radiation/nr_radiation.hpp"

namespace nr_radiation {

//----------------------------------------------------------------------------------------
//! \fn void SC::ComputeJ
//! \brief Mean intensity only, J = Sum w I (Eq. 17), into moments slot 0. The cheap one
//! iteration runs after every sweep (residual) and MeshRefinement after a remesh.

void SC::ComputeJ() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  int nang = pang->nang;
  int nang_tot_ = nang_tot;
  auto &wmu = pang->wmu;
  auto ir_ = ir;
  auto mom_ = moments;

  par_for("sc_compute_j", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real J = 0.0;
    for (int angg = 0; angg < nang_tot_; ++angg) {
      int a = angg % nang;
      J += wmu.d_view(a) * ir_(m, angg, k, j, i);
    }
    mom_(m, 0, k, j, i) = J;
  });
}

//----------------------------------------------------------------------------------------
//! \fn void SC::ComputeHK
//! \brief The first and second moments H_i, K_ij into moments slots 1-9: one par_for
//! over cells with an inner serial loop over all nang_tot angles, nine accumulators in
//! registers. Slots in Athena++ order: H1(0), H2(1), H3(2), K11(3), K22(4), K33(5),
//! K12(6), K13(7), K23(8). Run once per solve, over the active cells PLUS one ghost
//! layer: the face-centred radiative flux is 0.5*(H(i-1) + H(i)), so the faces bounding
//! the block read H one cell outside it. Ghost H cannot be communicated -- moments has no
//! bvals object -- so it is computed here from the ghost ir the exchange supplies.
//! Not gated on a consumer: slot 0 shares
//! the array and a restart rebuilds it from the restored ir before the outputs exist, so
//! the array cannot be reallocated later and there is nothing to defer. Consumers today
//! are the outputs; a radiation force or an Eddington-tensor closure would be next. J
//! itself is slot 0 (ComputeJ, per sweep).

void SC::ComputeHK() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  // one ghost layer beyond the active range in every dimension the problem has: the face
  // flux 0.5*(H(i-1) + H(i)) at i = is and i = ie+1 reads H there. moments is allocated
  // over ncells* (ghosts included), so the wider range is in bounds.
  int is = indcs.is - 1, ie = indcs.ie + 1;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  if (pmy_pack->pmesh->multi_d) { js -= 1; je += 1; }
  if (pmy_pack->pmesh->three_d) { ks -= 1; ke += 1; }
  int nmb1 = pmy_pack->nmb_thispack - 1;
  int nang = pang->nang;
  int nang_tot_ = nang_tot;
  auto &mu = pang->mu;
  auto &wmu = pang->wmu;
  auto ir_ = ir;
  auto mom_ = moments;

  par_for("sc_compute_hk", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real H1 = 0.0, H2 = 0.0, H3 = 0.0;
    Real K11 = 0.0, K22 = 0.0, K33 = 0.0;
    Real K12 = 0.0, K13 = 0.0, K23 = 0.0;

    for (int angg = 0; angg < nang_tot_; ++angg) {
      int oct = angg / nang;
      int a = angg - oct * nang;
      Real w = wmu.d_view(a);
      Real nx = mu.d_view(oct, a, 0);
      Real ny = mu.d_view(oct, a, 1);
      Real nz = mu.d_view(oct, a, 2);
      Real wI = w * ir_(m,angg,k,j,i);

      H1  += nx * wI;
      H2  += ny * wI;
      H3  += nz * wI;
      K11 += nx * nx * wI;
      K22 += ny * ny * wI;
      K33 += nz * nz * wI;
      K12 += nx * ny * wI;
      K13 += nx * nz * wI;
      K23 += ny * nz * wI;
    }

    mom_(m, 1, k, j, i) = H1;
    mom_(m, 2, k, j, i) = H2;
    mom_(m, 3, k, j, i) = H3;
    mom_(m, 4, k, j, i) = K11;
    mom_(m, 5, k, j, i) = K22;
    mom_(m, 6, k, j, i) = K33;
    mom_(m, 7, k, j, i) = K12;
    mom_(m, 8, k, j, i) = K13;
    mom_(m, 9, k, j, i) = K23;
  });
}

//----------------------------------------------------------------------------------------
//! \fn void SC::FillHflx
//! \brief Face-centred first moment: x1f(i) = 0.5*(H1(i-1) + H1(i)), and likewise for
//! x2f/x3f in the dimensions the problem has.
//!
//! Differencing this over a cell reproduces the centred difference of the cell-centred H
//! exactly, so the face form costs nothing in accuracy -- and it telescopes: the face
//! between two cells enters each with opposite sign and the same value, so a sum of
//! Q_dif over cells collapses to the boundary faces. That is only true if neighbouring
//! meshblocks compute the SAME number for a shared face, which needs two things: ghost H
//! (ComputeHK covers one layer) taken from ghost ir that is current (the post-sweep
//! exchange), and the operand order below fixed as 0.5*(left + right) on every block, so
//! the floating-point sum is identical from either side.

void SC::FillHflx() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  auto mom_ = moments;
  auto x1f_ = hflx.x1f;

  // slot 1 = H1; faces is..ie+1 bound the active cells
  par_for("sc_hflx_x1", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie+1,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    x1f_(m,0,k,j,i) = 0.5*(mom_(m,1,k,j,i-1) + mom_(m,1,k,j,i));
  });

  if (pmy_pack->pmesh->multi_d) {
    auto x2f_ = hflx.x2f;
    par_for("sc_hflx_x2", DevExeSpace(), 0, nmb1, ks, ke, js, je+1, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      x2f_(m,0,k,j,i) = 0.5*(mom_(m,2,k,j-1,i) + mom_(m,2,k,j,i));
    });
  }
  if (pmy_pack->pmesh->three_d) {
    auto x3f_ = hflx.x3f;
    par_for("sc_hflx_x3", DevExeSpace(), 0, nmb1, ks, ke+1, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      x3f_(m,0,k,j,i) = 0.5*(mom_(m,3,k-1,j,i) + mom_(m,3,k,j,i));
    });
  }
}

//----------------------------------------------------------------------------------------
//! \fn void SC::BoundaryHflxAtOpenFaces
//! \brief Replace the face flux on inflow/vacuum physical faces with the per-ray upwind
//! value, each ray contributing from whichever side it arrives on.
//!
//! FillHflx averages the two cells sharing a face, which is a cell-centre-to-face
//! interpolation and second order WHERE THE FIELD IS SMOOTH. Across an open boundary it
//! is not: an entering ray is i_in outside and grows inside as it picks up emission, so
//! intensity has a kink exactly at the face and interpolating across it converges to the
//! wrong value.
//!
//! No construction built from the full moment H can fix that, because the outward flux
//! at a surface is a HALF-moment. In a thick surface cell every ray tends to the local
//! source function, entering and leaving alike, so H = S * sum(w*mu) = 0 by its own
//! symmetry -- zero precisely where the outward flux is largest. The hemisphere asymmetry
//! that carries the flux has already been summed away. Hence the angular loop here: it is
//! the only form that can represent the quantity, not a stylistic choice.
//!
//! Only inflow and vacuum need this. outflow and diode copy the interior into the ghost
//! for every ray, reflect mirrors it, and block/periodic ghosts are real neighbour data.
//! In all of those the ghost is a complete physical field and the average is right. A
//! user BC is trusted to have filled every ray (see sc_hooks.hpp).
//!
//! sum(w*mu*i_in) over entering rays is a per-face constant and could be hoisted to
//! setup; it is left in the loop because it is zero for vacuum and the loop runs once per
//! solve over a boundary layer.

void SC::BoundaryHflxAtOpenFaces() {
  auto &pm = pmy_pack->pmesh;
  if (pm->strictly_periodic) return;

  auto &indcs = pm->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  int nang = pang->nang;
  int nang_tot_ = nang_tot;
  auto &mu = pang->mu;
  auto &wmu = pang->wmu;
  auto &mb_bcs = pmy_pack->pmb->mb_bcs;
  auto ir_ = ir;
  auto i_in_ = pbval_ir->i_in;

  auto x1f_ = hflx.x1f;
  par_for("sc_hflx_bdry_x1", DevExeSpace(), 0, nmb1, ks, ke, js, je,
  KOKKOS_LAMBDA(int m, int k, int j) {
    for (int side = 0; side < 2; ++side) {
      const int face = (side == 0) ? BoundaryFace::inner_x1 : BoundaryFace::outer_x1;
      const auto flag = mb_bcs.d_view(m, face);
      if (flag != BoundaryFlag::inflow && flag != BoundaryFlag::vacuum) continue;
      // inner: the face is at is and a ray ENTERS if it points +x. outer: face ie+1, -x.
      const int icell = (side == 0) ? is : ie;
      const int iface = (side == 0) ? is : ie + 1;
      const Real sgn = (side == 0) ? 1.0 : -1.0;
      Real flx = 0.0;
      for (int angg = 0; angg < nang_tot_; ++angg) {
        const int oct = angg / nang;
        const int a = angg - oct * nang;
        const Real nx = mu.d_view(oct, a, 0);
        const Real intensity = (sgn * nx > 0.0)
            ? ((flag == BoundaryFlag::vacuum) ? 0.0 : i_in_.d_view(angg, face))
            : ir_(m, angg, k, j, icell);
        flx += wmu.d_view(a) * nx * intensity;
      }
      x1f_(m, 0, k, j, iface) = flx;
    }
  });

  if (pm->multi_d) {
    auto x2f_ = hflx.x2f;
    par_for("sc_hflx_bdry_x2", DevExeSpace(), 0, nmb1, ks, ke, is, ie,
    KOKKOS_LAMBDA(int m, int k, int i) {
      for (int side = 0; side < 2; ++side) {
        const int face = (side == 0) ? BoundaryFace::inner_x2 : BoundaryFace::outer_x2;
        const auto flag = mb_bcs.d_view(m, face);
        if (flag != BoundaryFlag::inflow && flag != BoundaryFlag::vacuum) continue;
        const int jcell = (side == 0) ? js : je;
        const int jface = (side == 0) ? js : je + 1;
        const Real sgn = (side == 0) ? 1.0 : -1.0;
        Real flx = 0.0;
        for (int angg = 0; angg < nang_tot_; ++angg) {
          const int oct = angg / nang;
          const int a = angg - oct * nang;
          const Real ny = mu.d_view(oct, a, 1);
          const Real intensity = (sgn * ny > 0.0)
              ? ((flag == BoundaryFlag::vacuum) ? 0.0 : i_in_.d_view(angg, face))
              : ir_(m, angg, k, jcell, i);
          flx += wmu.d_view(a) * ny * intensity;
        }
        x2f_(m, 0, k, jface, i) = flx;
      }
    });
  }

  if (pm->three_d) {
    auto x3f_ = hflx.x3f;
    par_for("sc_hflx_bdry_x3", DevExeSpace(), 0, nmb1, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int j, int i) {
      for (int side = 0; side < 2; ++side) {
        const int face = (side == 0) ? BoundaryFace::inner_x3 : BoundaryFace::outer_x3;
        const auto flag = mb_bcs.d_view(m, face);
        if (flag != BoundaryFlag::inflow && flag != BoundaryFlag::vacuum) continue;
        const int kcell = (side == 0) ? ks : ke;
        const int kface = (side == 0) ? ks : ke + 1;
        const Real sgn = (side == 0) ? 1.0 : -1.0;
        Real flx = 0.0;
        for (int angg = 0; angg < nang_tot_; ++angg) {
          const int oct = angg / nang;
          const int a = angg - oct * nang;
          const Real nz = mu.d_view(oct, a, 2);
          const Real intensity = (sgn * nz > 0.0)
              ? ((flag == BoundaryFlag::vacuum) ? 0.0 : i_in_.d_view(angg, face))
              : ir_(m, angg, kcell, j, i);
          flx += wmu.d_view(a) * nz * intensity;
        }
        x3f_(m, 0, kface, j, i) = flx;
      }
    });
  }
}

//----------------------------------------------------------------------------------------
//! \fn void SC::ComputeQrad
//! \brief Gas-radiation coupling, Davis Eq. 27 in its absorption form:
//!   Q = crat * prat * sigma_a * (J - brad)
//!
//! Eq. 27 gives the two as an identity, 4 pi chi_tot (J - S) = 4 pi sigma_a (J - B):
//! with S = eps B + (1 - eps) J and eps = sigma_a/chi_tot, the scattering contribution
//! cancels exactly. So scattering is in the transfer equation and in srad, and is
//! absent from Q -- not because the reference omits it, but because it cancels.
//!
//! The ABSORPTION coefficient, then, not chi. The two arrays are equal today, so this
//! reads as a free choice; it is not. Once sigma_s exists chi = sigma_a + sigma_s, and
//! chi here would overstate heating and cooling by the scattering fraction, with no
//! crash and nothing obviously wrong in the output. Of the two valid forms this one is
//! also the better conditioned: when scattering dominates, eps -> 0 and S -> J, so
//! chi_tot (J - S) becomes a small difference of large numbers, while sigma_a (J - B)
//! does not. newdt.cpp already draws the same distinction, using sigma_a for the
//! coupling and chi only in the diffusion denominator.
//!
//! Both forms lose precision in the opposite limit -- optically thick and near
//! equilibrium, where J -> B is a small difference of large numbers multiplied by a large
//! sigma_a. Davis Eq. 28 is the remedy:
//!
//!   Q_int =  C sigma_a (J - brad)          accurate thin, cancels thick
//!   Q_dif = -C div(H)                      accurate thick, noisy thin
//!
//! identical in exact arithmetic (Eq. 28 is the zeroth moment of the transfer equation),
//! and this blends them on tau = chi * min(dx over the dimensions the run has). Davis
//! switches hard at tau = 1; a smooth blend over one decade follows Hubeny & Lanz's
//! practice in the stellar-atmosphere literature instead, and is better behaved at the
//! interface, where a hard switch puts the largest possible jump.
//!
//! The two are NOT interchangeable in their conservation properties. Q_dif is built from
//! face-centred H (FillHflx) and telescopes: summed over cells it collapses to the
//! boundary faces exactly, whatever H's accuracy. Q_int is a pointwise source with no
//! face representation, conservative only to the accuracy of the transfer solve. So the
//! scheme is exactly conservative where w = 1 -- the deep thick region, where drift would
//! otherwise accumulate -- and not elsewhere. In the transition band w varies from cell
//! to cell, which breaks the telescoping by C*F*A*(w_L - w_R) per face; that residual is
//! inherent to blending and is measured, not assumed away.
//!
//! The divergence is a centred difference of face values, so it is blind to odd-even
//! structure in H. With few rays (nmu small) ray effects can put exactly that there.

void SC::ComputeQrad() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  auto sigma_a_ = sigma_a;
  auto chi_ = chi;
  auto brad_ = brad;
  auto mom_ = moments;
  auto qrad_ = qrad;
  auto x1f_ = hflx.x1f;
  auto x2f_ = hflx.x2f;
  auto x3f_ = hflx.x3f;
  auto &mbsize = pmy_pack->pmb->mb_size;
  const Real crat_prat = crat * prat;
  const bool multi_d = pmy_pack->pmesh->multi_d;
  const bool three_d = pmy_pack->pmesh->three_d;
  const Real tau_lo = kTauLo, tau_hi = kTauHi;
  const Real inv_decade = 1.0/(std::log10(kTauHi) - std::log10(kTauLo));
  const Real log_tau_lo = std::log10(kTauLo);

  par_for("sc_compute_qrad", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    const Real qint = crat_prat * sigma_a_(m,0,k,j,i)
                      * (mom_(m,0,k,j,i) - brad_(m,k,j,i));

    // dx over the dimensions the run HAS: mb_size.dx2 with nx2 == 1 is the whole x2
    // extent, not a cell width, so including it would make tau meaningless in 1D/2D.
    Real dxmin = mbsize.d_view(m).dx1;
    if (multi_d) dxmin = fmin(dxmin, mbsize.d_view(m).dx2);
    if (three_d) dxmin = fmin(dxmin, mbsize.d_view(m).dx3);
    const Real tau = chi_(m,k,j,i) * dxmin;

    // Branch rather than compute both and weight: tau is exactly 0 in vacuum cells (the
    // sphere and beam problems), where log10(0) is -inf and a NaN in qdif would poison a
    // cell whose weight is 0 anyway. The first branch also keeps today's bits exactly.
    if (tau <= tau_lo) {
      qrad_(m,k,j,i) = qint;
    } else {
      Real divh = (x1f_(m,0,k,j,i+1) - x1f_(m,0,k,j,i)) / mbsize.d_view(m).dx1;
      if (multi_d) {
        divh += (x2f_(m,0,k,j+1,i) - x2f_(m,0,k,j,i)) / mbsize.d_view(m).dx2;
      }
      if (three_d) {
        divh += (x3f_(m,0,k+1,j,i) - x3f_(m,0,k,j,i)) / mbsize.d_view(m).dx3;
      }
      const Real qdif = -crat_prat * divh;
      if (tau >= tau_hi) {
        qrad_(m,k,j,i) = qdif;
      } else {
        const Real t = (log10(tau) - log_tau_lo) * inv_decade;   // in [0,1]
        const Real w = t*t*(3.0 - 2.0*t);                        // smoothstep
        qrad_(m,k,j,i) = (1.0 - w)*qint + w*qdif;
      }
    }
  });
}

}  // namespace nr_radiation
