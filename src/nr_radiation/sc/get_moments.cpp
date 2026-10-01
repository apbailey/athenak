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
#include <cstdlib>
#include <iostream>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "nr_radiation/nr_radiation.hpp"

namespace nr_radiation {

namespace {
//----------------------------------------------------------------------------------------
//! \fn Real FaceIntensity
//! \brief The intensity of ONE ray at a cell face, given its values in the two cells
//! either side (lo = the cell at lower index, hi = the cell at higher index) and its
//! direction cosine along the face normal axis. This is the single place the face rule
//! lives.
//!
//! Interior face: both cells are real, and a linear reconstruction gives the mean. Summed
//! over rays that is (H_lo + H_hi)/2, the mean of the cell-centred moments.
//!
//! Physical-boundary face: one of the two cells is a ghost, and a ghost is only half
//! valid. A boundary condition can fix the intensity ENTERING the domain and no more --
//! the transfer equation is hyperbolic along rays, so data may be imposed on inflow
//! characteristics only -- yet RadiationBCs writes every ray into the ghost regardless of
//! direction, because the sweep never reads a downwind ghost and so never notices. So the
//! outbound half of a boundary ghost is not a physical intensity, and averaging would mix
//! it in. Take each ray from the cell it arrived from instead: that is the value actually
//! crossing the face, and it is well defined for every ray.
//!
//! "Arrived from" is just upwind along the axis, the same sense the sweep uses for
//! its own
//! footpoints, and it needs no knowledge of which end of the domain this face is: a ray
//! with mu > 0 travels towards higher index and so comes from lo, whichever side happens
//! to be the ghost. At an inner face lo is the ghost and mu > 0 is inbound; at an outer
//! face hi is the ghost and mu < 0 is inbound. One expression covers both.
//!
//! The boundary conditions then reduce correctly without special cases. outflow and diode
//! copy the last active cell, so both halves read the same numbers and the flux is the
//! interior H. vacuum zeroes the ghost, leaving only what escapes. reflect writes each
//! ghost ray from its mirror image, and mirroring flips mu at equal weight, so the two
//! halves cancel and no flux crosses -- which is what a mirror means.

KOKKOS_INLINE_FUNCTION
Real FaceIntensity(Real i_lo, Real i_hi, Real mu_n, bool boundary) {
  if (boundary) return (mu_n > 0.0) ? i_lo : i_hi;
  return 0.5*(i_lo + i_hi);
}

//----------------------------------------------------------------------------------------
//! \fn bool IsPhysicalFace
//! \brief True when nothing lies across this face of the meshblock. mb_bcs holds block
//! for a face with a neighbour in the tree, but a block at the DOMAIN EDGE is given the
//! mesh flag itself (meshblock.cpp) -- and that includes periodic, where the tree does
//! wrap and the ghost really is a neighbour's interior. Periodic must therefore be
//! excluded explicitly: counting it as physical would apply the boundary rule to what is
//! an interior face, quietly changing the scheme on every periodic domain edge.

KOKKOS_INLINE_FUNCTION
bool IsPhysicalFace(BoundaryFlag f) {
  return (f != BoundaryFlag::block && f != BoundaryFlag::periodic);
}
}  // namespace


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
//! K12(6), K13(7), K23(8). Run once per solve. Not gated on a consumer: slot 0 shares
//! the array and a restart rebuilds it from the restored ir before the outputs exist, so
//! the array cannot be reallocated later and there is nothing to defer. Consumers today
//! are the outputs; a radiation force or an Eddington-tensor closure would be next. J
//! itself is slot 0 (ComputeJ, per sweep).

void SC::ComputeHK() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
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
//! \fn void SC::ComputeQrad
//! \brief Gas-radiation coupling: dispatch on <nr_radiation>/qrad_form. The branch is
//! resolved on the host, as with sweep_kernel, so no kernel carries it.

void SC::ComputeQrad() {
  switch (qrad_form) {
    case QradForm::integral:
      ComputeQradIntegral();
      break;
    case QradForm::divh:
    case QradForm::hybrid:
      BuildHFlux();
      CorrectHFluxCoarseFine();
      ComputeQradDivH();
      break;
  }
}

//----------------------------------------------------------------------------------------
//! \fn void SC::ComputeQradIntegral
//! \brief Davis Eq. 27 in its absorption form:
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
//! equilibrium, where J -> brad and the difference is small while sigma_a is large.
//! That is what Davis Eq. 28 is for; it is qrad_form = divh (ComputeQradDivH), with
//! hybrid switching between the two on chi*dx as the reference does.

void SC::ComputeQradIntegral() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  auto sigma_a_ = sigma_a;
  auto brad_ = brad;
  auto mom_ = moments;
  auto qrad_ = qrad;
  const Real crat_prat = crat * prat;

  par_for("sc_compute_qrad", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    qrad_(m,k,j,i) = crat_prat * sigma_a_(m,0,k,j,i)
                     * (mom_(m,0,k,j,i) - brad_(m,k,j,i));
  });
}

//----------------------------------------------------------------------------------------
//! \fn void SC::BuildHFlux
//! \brief The radiative flux normal to each cell face, H.n, into hflx. x1f(i) is the
//! face at i-1/2, so the range is [is, ie+1]; likewise x2f, x3f.
//!
//! On an interior face the two cells either side are both real, and the flux is the
//! moment of a linear reconstruction of the intensity there:
//!
//!   Hhat_{i-1/2} = Sum_k w_k mu_k * (I_k(i-1) + I_k(i))/2
//!
//! which is identically (H_{i-1} + H_i)/2, the mean of the cell-centred moments. It is
//! written per ray rather than as that mean for two reasons. It never reads a moment in a
//! ghost zone, so moments stays an interior-only, output-only array and ComputeHK is
//! untouched. And it is the form that survives at a physical boundary, where the mean
//! does not: a boundary condition fixes only the inbound half of the sphere, so the
//! outbound half of a boundary ghost cell is not a physical intensity and a full-sphere
//! quadrature of it is meaningless. Boundary faces are therefore taken per ray from
//! whichever side of the face that ray arrived from, which is the same expression with
//! the average replaced by a choice. (The two routes agree on interior faces to
//! round-off, not bitwise: this one averages then weights, the other weights then
//! averages.)
//!
//! Faces normal to an inactive dimension are not built. In 2D that also keeps the flux
//! clear of mu_z, which carries no sign there (angular_grid.cpp), so the x3 moment is not
//! a physical flux in 2D.

void SC::BuildHFlux() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int &ng = indcs.ng;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  int nang = pang->nang;
  int nang_tot_ = nang_tot;
  const bool multi_d = pmy_pack->pmesh->multi_d;
  const bool three_d = pmy_pack->pmesh->three_d;

  // allocated on first use, so the integral form never pays for it
  int nc1 = indcs.nx1 + 2*ng;
  int nc2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng) : 1;
  int nc3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng) : 1;
  if (hflx.x1f.extent_int(0) != (nmb1+1) || hflx.x1f.extent_int(4) != nc1) {
    Kokkos::realloc(hflx.x1f, (nmb1+1), 1, nc3, nc2, nc1);
    Kokkos::realloc(hflx.x2f, (nmb1+1), 1, nc3, nc2, nc1);
    Kokkos::realloc(hflx.x3f, (nmb1+1), 1, nc3, nc2, nc1);
    Kokkos::deep_copy(hflx.x1f, 0.0);
    Kokkos::deep_copy(hflx.x2f, 0.0);
    Kokkos::deep_copy(hflx.x3f, 0.0);
  }

  auto &mu = pang->mu;
  auto &wmu = pang->wmu;
  auto ir_ = ir;
  auto f1 = hflx.x1f;
  auto f2 = hflx.x2f;
  auto f3 = hflx.x3f;
  auto &mb_bcs = pmy_pack->pmb->mb_bcs;   // see IsPhysicalFace

  par_for("sc_hflx_x1", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie+1,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    bool blo = IsPhysicalFace(mb_bcs.d_view(m, BoundaryFace::inner_x1));
    bool bhi = IsPhysicalFace(mb_bcs.d_view(m, BoundaryFace::outer_x1));
    bool bdry = (i == is && blo) || (i == ie+1 && bhi);
    Real h = 0.0;
    for (int angg = 0; angg < nang_tot_; ++angg) {
      int oct = angg / nang;
      int a = angg - oct * nang;
      Real mu_n = mu.d_view(oct, a, 0);
      h += wmu.d_view(a) * mu_n
           * FaceIntensity(ir_(m, angg, k, j, i-1), ir_(m, angg, k, j, i), mu_n, bdry);
    }
    f1(m, 0, k, j, i) = h;
  });
  if (multi_d) {
    par_for("sc_hflx_x2", DevExeSpace(), 0, nmb1, ks, ke, js, je+1, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      bool blo = IsPhysicalFace(mb_bcs.d_view(m, BoundaryFace::inner_x2));
      bool bhi = IsPhysicalFace(mb_bcs.d_view(m, BoundaryFace::outer_x2));
      bool bdry = (j == js && blo) || (j == je+1 && bhi);
      Real h = 0.0;
      for (int angg = 0; angg < nang_tot_; ++angg) {
        int oct = angg / nang;
        int a = angg - oct * nang;
        Real mu_n = mu.d_view(oct, a, 1);
        h += wmu.d_view(a) * mu_n
             * FaceIntensity(ir_(m, angg, k, j-1, i), ir_(m, angg, k, j, i), mu_n, bdry);
      }
      f2(m, 0, k, j, i) = h;
    });
  }
  if (three_d) {
    par_for("sc_hflx_x3", DevExeSpace(), 0, nmb1, ks, ke+1, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      bool blo = IsPhysicalFace(mb_bcs.d_view(m, BoundaryFace::inner_x3));
      bool bhi = IsPhysicalFace(mb_bcs.d_view(m, BoundaryFace::outer_x3));
      bool bdry = (k == ks && blo) || (k == ke+1 && bhi);
      Real h = 0.0;
      for (int angg = 0; angg < nang_tot_; ++angg) {
        int oct = angg / nang;
        int a = angg - oct * nang;
        Real mu_n = mu.d_view(oct, a, 2);
        h += wmu.d_view(a) * mu_n
             * FaceIntensity(ir_(m, angg, k-1, j, i), ir_(m, angg, k, j, i), mu_n, bdry);
      }
      f3(m, 0, k, j, i) = h;
    });
  }
}

//----------------------------------------------------------------------------------------
//! \fn void SC::CorrectHFluxCoarseFine
//! \brief Replace the coarse side's face flux with the area-weighted mean of the fine
//! face fluxes over the same face, so that both sides of a coarse-fine interface use ONE
//! number for it. No-op unless a differential form is in use on a multilevel mesh, in
//! which case pbval_hflx is nullptr and this returns immediately.
//!
//! WHY IT IS NEEDED. Eq. 28 telescopes only because each interior face enters two cells
//! with opposite signs -- the SAME number, twice. At a coarse-fine face the two sides
//! compute different numbers, because they are built from different data:
//!
//!   coarse side   0.5*(H_C + H_G), and its ghost H_G is RestrictCC of the fine block,
//!                 a plain volume average over 2x2 (2D) or 2x2x2 (3D) cells -- including
//!                 TWO layers normal to the face, so H_G = 0.5*(L1 + L2).
//!   fine side     0.5*(ghost + L1), and its ghost is ProlongCC of the coarse cell, so
//!                 the children adjacent to the face have transverse mean H_C + dvar1.
//!
//! The gap is 0.5*[dvar1 + 0.5*(L1 - L2)], identically zero for a field linear in x
//! (dvar1 = 0.5*a*h against 0.5*(L1-L2) = -0.5*a*h) and O(dx^2) otherwise. Measured
//! residual |Sum Q dV|/Sum|Q|dV = 5.4e-04 (2D, one level jump), 4.5e-04 (2D, nested),
//! 4.7e-05 (3D), against 1.6e-16 on the same decks with refinement off, and converging at
//! second order -- a truncation-level flux mismatch, not round-off.
//!
//! The coarse side cannot repair this alone: restriction has already destroyed the layer
//! adjacent to the face. One number per coarse face must travel fine -> coarse, which is
//! exactly what PackAndSendFluxCC does -- it packs the fine faces at a FIXED normal index
//! (fi = 2*il - cis, i.e. is or ie+1 since cis == is) with only a transverse average,
//! 0.5* in 2D and 0.25* in 3D. hflx is per-unit-area on a uniform Cartesian face and
//! ComputeQradDivH divides by dx, so that plain mean is the correct conservative
//! restriction and no area weight is applied anywhere.
//!
//! Conservation alone does not single out this convention -- overwriting every FINE face
//! with the coarse value telescopes too -- but it is the one that keeps the resolution
//! that was paid for at the interface, and it is what hydro, MHD and the GR radiation
//! module already do. Note this buys CONSERVATION, not accuracy: the agreed value is
//! still built from a prolongated ghost on one side, so the interface flux remains
//! coarse-grid limited. The O(dx^2) error does not leave the solution, it stops being a
//! conservation error.
//!
//! NOT a DSJ12/JSD12 algorithm: the papers are silent on refinement and Athena-C's
//! rad_to_hydro has no coarse-fine awareness. This is Berger & Colella (1989) as already
//! implemented in src/bvals/flux_correct_cc.cpp, applied to hflx unchanged. Recorded as a
//! deliberate deviation in logs/impl-athenak.md.
//!
//! The receives were posted at the top of SolveTransfer; see the comment there for why.

void SC::CorrectHFluxCoarseFine() {
  if (pbval_hflx == nullptr) return;
  (void)pbval_hflx->PackAndSendFluxCC(hflx);
  // RecvAndUnpackFluxCC returns incomplete while an MPI_Irecv is outstanding. Spinning
  // here rather than driving a task list is not a lost overlap: the chain is linear, and
  // "before_timeintegrator" holds nothing but this solve. The traffic is nvar = 1 over
  // coarse-fine faces only, once per cycle, against sc_bvals moving nang_tot over every
  // face iter_max + 1 times -- four orders of magnitude smaller.
  while (pbval_hflx->RecvAndUnpackFluxCC(hflx) == TaskStatus::incomplete) {}
  (void)pbval_hflx->ClearFluxSend();
  (void)pbval_hflx->ClearFluxRecv();
}

//----------------------------------------------------------------------------------------
//! \fn void SC::ComputeQradDivH
//! \brief Davis Eq. 28, the differential form, as the divergence of the face flux:
//!   Q = -crat * prat * div H
//! No 4 pi: the weights sum to one, so this carries the same crat*prat as Eq. 27.
//!
//! Conservative by construction. Each interior face enters two cells with opposite signs,
//! so summing Q over the domain telescopes to the flux through the domain boundary alone.
//! That holds whatever the state of the solve -- an unconverged iteration or a coarse
//! angular grid moves energy to the wrong place, but cannot create or destroy it.
//!
//! Under qrad_form = hybrid the cell instead takes Eq. 27 where it is the better
//! conditioned of the two, chi*dx <= 1 (Davis sec. 4). The criterion is per cell and
//! scalar, on the smallest active cell width: a per-direction test would let one cell mix
//! an Eq. 27 contribution along x with an Eq. 28 contribution along y, leaving its x
//! faces no longer telescoping against the neighbour's. Whole cells keep the bookkeeping
//! intact between differential cells, though the blend still breaks conservation wherever
//! the two forms meet -- that is inherent to switching, and is why divh, not hybrid, is
//! the conservative setting.

void SC::ComputeQradDivH() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  auto f1 = hflx.x1f;
  auto f2 = hflx.x2f;
  auto f3 = hflx.x3f;
  auto chi_ = chi;
  auto sigma_a_ = sigma_a;
  auto brad_ = brad;
  auto mom_ = moments;
  auto qrad_ = qrad;
  auto &mbsize = pmy_pack->pmb->mb_size;
  const Real crat_prat = crat * prat;
  const bool multi_d = pmy_pack->pmesh->multi_d;
  const bool three_d = pmy_pack->pmesh->three_d;
  const bool blend = (qrad_form == QradForm::hybrid);

  par_for("sc_compute_qrad_divh", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real dx1 = mbsize.d_view(m).dx1;
    if (blend) {
      Real dxm = dx1;
      if (multi_d) dxm = fmin(dxm, mbsize.d_view(m).dx2);
      if (three_d) dxm = fmin(dxm, mbsize.d_view(m).dx3);
      if (chi_(m,k,j,i)*dxm <= 1.0) {
        qrad_(m,k,j,i) = crat_prat * sigma_a_(m,0,k,j,i)
                         * (mom_(m,0,k,j,i) - brad_(m,k,j,i));
        return;
      }
    }
    Real divh = (f1(m,0,k,j,i+1) - f1(m,0,k,j,i)) / dx1;
    if (multi_d) {
      divh += (f2(m,0,k,j+1,i) - f2(m,0,k,j,i)) / mbsize.d_view(m).dx2;
    }
    if (three_d) {
      divh += (f3(m,0,k+1,j,i) - f3(m,0,k,j,i)) / mbsize.d_view(m).dx3;
    }
    qrad_(m,k,j,i) = -crat_prat * divh;
  });
}

}  // namespace nr_radiation
