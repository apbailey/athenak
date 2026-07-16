//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file formal_solution.cpp
//! \brief Short-characteristics formal solution (Davis, Stone & Jiang 2012 Eq. 20)
//! with three dispatch modes selectable via the input parameter nr_radiation/sweep:
//!
//!   "wavefront"  — host loop over hyperplane index h, flat par_for per plane (default)
//!   "diagonal"   — TeamPolicy with league=(nmb*nang_tot), device loop over h with
//!                  team_barrier between planes
//!   "jacobi"     — single par_for over all (m,angg,k,j,i); no sweep ordering enforced,
//!                  effectively Gauss-Seidel in GPU thread order (requires more outer
//!                  iterations to converge, but each iteration is fully parallel)
//!
//! All three modes call the shared UpdateCellSC() device function defined in
//! vet_interp.hpp, which encapsulates the per-cell geometry logic, transverse
//! interpolation, opacity interpolation and source-function quadrature weights.

#include <algorithm>
#include <cmath>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "nr_radiation/nr_radiation.hpp"
#include "nr_radiation/vet/vet_interp.hpp"

namespace nr_radiation {

//----------------------------------------------------------------------------------------
//! \fn void VET::FormalSolution
//! \brief Dispatcher — delegates to the implementation selected by sweep_method.

void VET::FormalSolution() {
  if (sweep_method == "diagonal") {
    FormalSolutionDiagonal();
  } else if (sweep_method == "jacobi") {
    FormalSolutionJacobi();
  } else {
    FormalSolutionWavefront();
  }
}

//----------------------------------------------------------------------------------------
//! \fn void VET::FormalSolutionWavefront
//! \brief Hyperplane-swept formal solution.  Host loop over hyperplane index h;
//! device par_for flattens (MeshBlock x direction x transverse-plane cell).

void VET::FormalSolutionWavefront() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  int ndim = pang->ndim;
  int nang = pang->nang;
  int nangt1 = nang_tot - 1;
  auto &mu = pang->mu;
  auto &mbsize = pmy_pack->pmb->mb_size;
  auto ir_ = ir;
  auto chi_ = chi;
  auto bb_ = bb;

  if (ndim == 1) {
    for (int h = 0; h < nx1; ++h) {
      par_for("vet_sweep1d", DevExeSpace(), 0, nmb1, 0, nangt1,
      KOKKOS_LAMBDA(int m, int angg) {
        int oct = angg / nang;
        int a = angg - oct*nang;
        Real mux = mu.d_view(oct,a,0);
        int sx = (mux > 0.0) ? 1 : -1;
        int i  = (sx > 0) ? (is + h) : (ie - h);
        Real dx1v = mbsize.d_view(m).dx1;
        ir_(m,angg,ks,js,i) = UpdateCellSC(chi_,bb_,ir_,m,angg,
                                            i,js,ks, sx,0,0,
                                            mux,0.0,0.0,
                                            dx1v,0.0,0.0, ndim,ks,js);
      });
    }
  } else if (ndim == 2) {
    int hmax = nx1 + nx2 - 2;
    for (int h = 0; h <= hmax; ++h) {
      par_for("vet_sweep2d", DevExeSpace(), 0, nmb1, 0, nangt1, 0, (nx1-1),
      KOKKOS_LAMBDA(int m, int angg, int li1) {
        int li2 = h - li1;
        if (li2 < 0 || li2 > nx2-1) return;
        int oct = angg / nang;
        int a = angg - oct*nang;
        Real mux = mu.d_view(oct,a,0);
        Real muy = mu.d_view(oct,a,1);
        int sx = (mux > 0.0) ? 1 : -1;
        int sy = (muy > 0.0) ? 1 : -1;
        int i = (sx > 0) ? (is + li1) : (ie - li1);
        int j = (sy > 0) ? (js + li2) : (je - li2);
        Real dx1v = mbsize.d_view(m).dx1;
        Real dx2v = mbsize.d_view(m).dx2;
        ir_(m,angg,ks,j,i) = UpdateCellSC(chi_,bb_,ir_,m,angg,
                                           i,j,ks, sx,sy,0,
                                           mux,muy,0.0,
                                           dx1v,dx2v,0.0, ndim,ks,js);
      });
    }
  } else {
    int hmax = nx1 + nx2 + nx3 - 3;
    for (int h = 0; h <= hmax; ++h) {
      par_for("vet_sweep3d", DevExeSpace(), 0, nmb1, 0, nangt1, 0,(nx1-1), 0,(nx2-1),
      KOKKOS_LAMBDA(int m, int angg, int li1, int li2) {
        int li3 = h - li1 - li2;
        if (li3 < 0 || li3 > nx3-1) return;
        int oct = angg / nang;
        int a = angg - oct*nang;
        Real mux = mu.d_view(oct,a,0);
        Real muy = mu.d_view(oct,a,1);
        Real muz = mu.d_view(oct,a,2);
        int sx = (mux > 0.0) ? 1 : -1;
        int sy = (muy > 0.0) ? 1 : -1;
        int sz = (muz > 0.0) ? 1 : -1;
        int i = (sx > 0) ? (is + li1) : (ie - li1);
        int j = (sy > 0) ? (js + li2) : (je - li2);
        int k = (sz > 0) ? (ks + li3) : (ke - li3);
        Real dx1v = mbsize.d_view(m).dx1;
        Real dx2v = mbsize.d_view(m).dx2;
        Real dx3v = mbsize.d_view(m).dx3;
        ir_(m,angg,k,j,i) = UpdateCellSC(chi_,bb_,ir_,m,angg,
                                          i,j,k, sx,sy,sz,
                                          mux,muy,muz,
                                          dx1v,dx2v,dx3v, ndim,ks,js);
      });
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn void VET::FormalSolutionDiagonal
//! \brief TeamPolicy sweep: one team per (MeshBlock, angle) pair, device loop over
//! hyperplane planes h with team_barrier between planes, TeamThreadRange over cells on
//! each plane. Same algorithmic sweep order as wavefront (so bit-identical results).

void VET::FormalSolutionDiagonal() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  int nmb = pmy_pack->nmb_thispack;
  int ndim = pang->ndim;
  int nang = pang->nang;
  int nang_tot_ = nang_tot;
  auto &mu = pang->mu;
  auto &mbsize = pmy_pack->pmb->mb_size;
  auto ir_ = ir;
  auto chi_ = chi;
  auto bb_ = bb;

  int league_size = nmb * nang_tot_;
  int hmax;
  if (ndim == 1) hmax = nx1 - 1;
  else if (ndim == 2) hmax = nx1 + nx2 - 2;
  else hmax = nx1 + nx2 + nx3 - 3;

  Kokkos::TeamPolicy<> policy(DevExeSpace(), league_size, Kokkos::AUTO);
  Kokkos::parallel_for("vet_sweep_diag", policy,
  KOKKOS_LAMBDA(const TeamMember_t &tmember) {
    int league_id = tmember.league_rank();
    int m = league_id / nang_tot_;
    int angg = league_id - m * nang_tot_;
    int oct = angg / nang;
    int a = angg - oct * nang;
    Real mux = mu.d_view(oct,a,0);
    Real muy = (ndim >= 2) ? mu.d_view(oct,a,1) : 0.0;
    Real muz = (ndim == 3) ? mu.d_view(oct,a,2) : 0.0;
    int sx = (mux > 0.0) ? 1 : -1;
    int sy = (ndim >= 2) ? ((muy > 0.0) ? 1 : -1) : 0;
    int sz = (ndim == 3) ? ((muz > 0.0) ? 1 : -1) : 0;
    Real dx1v = mbsize.d_view(m).dx1;
    Real dx2v = mbsize.d_view(m).dx2;
    Real dx3v = mbsize.d_view(m).dx3;

    for (int h = 0; h <= hmax; ++h) {
      if (ndim == 1) {
        Kokkos::parallel_for(Kokkos::TeamThreadRange(tmember, 1),
        [&](const int /*idx*/) {
          int i = (sx > 0) ? (is + h) : (ie - h);
          ir_(m,angg,ks,js,i) = UpdateCellSC(chi_,bb_,ir_,m,angg,
                                              i,js,ks, sx,0,0,
                                              mux,0.0,0.0,
                                              dx1v,0.0,0.0, ndim,ks,js);
        });
      } else if (ndim == 2) {
        Kokkos::parallel_for(Kokkos::TeamThreadRange(tmember, nx1),
        [&](const int li1) {
          int li2 = h - li1;
          if (li2 < 0 || li2 >= nx2) return;
          int i = (sx > 0) ? (is + li1) : (ie - li1);
          int j = (sy > 0) ? (js + li2) : (je - li2);
          ir_(m,angg,ks,j,i) = UpdateCellSC(chi_,bb_,ir_,m,angg,
                                             i,j,ks, sx,sy,0,
                                             mux,muy,0.0,
                                             dx1v,dx2v,0.0, ndim,ks,js);
        });
      } else {
        int max_cells = nx1 * nx2;
        Kokkos::parallel_for(Kokkos::TeamThreadRange(tmember, max_cells),
        [&](const int idx) {
          int li1 = idx / nx2;
          int li2 = idx - li1 * nx2;
          int li3 = h - li1 - li2;
          if (li1 >= nx1 || li3 < 0 || li3 >= nx3) return;
          int i = (sx > 0) ? (is + li1) : (ie - li1);
          int j = (sy > 0) ? (js + li2) : (je - li2);
          int k = (sz > 0) ? (ks + li3) : (ke - li3);
          ir_(m,angg,k,j,i) = UpdateCellSC(chi_,bb_,ir_,m,angg,
                                            i,j,k, sx,sy,sz,
                                            mux,muy,muz,
                                            dx1v,dx2v,dx3v, ndim,ks,js);
        });
      }
      tmember.team_barrier();
    }
  });
}

//----------------------------------------------------------------------------------------
//! \fn void VET::FormalSolutionJacobi
//! \brief All cells updated in a single par_for with no sweep ordering enforced.
//! Each cell reads upwind intensities from ir as-is (previous iteration's values for
//! cells that have not been updated yet in this kernel launch, current-iteration values
//! for cells that happen to have been scheduled first by the runtime). This is effectively
//! Gauss-Seidel in GPU thread execution order. The fixed point is the same as the
//! wavefront/diagonal result; convergence requires more outer iterations (iter_max).

void VET::FormalSolutionJacobi() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  int ndim = pang->ndim;
  int nang = pang->nang;
  int nangt1 = nang_tot - 1;
  auto &mu = pang->mu;
  auto &mbsize = pmy_pack->pmb->mb_size;
  auto ir_ = ir;
  auto chi_ = chi;
  auto bb_ = bb;

  par_for("vet_sweep_jacobi", DevExeSpace(), 0, nmb1, 0, nangt1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int angg, int k, int j, int i) {
    int oct = angg / nang;
    int a = angg - oct * nang;
    Real mux = mu.d_view(oct,a,0);
    Real muy = (ndim >= 2) ? mu.d_view(oct,a,1) : 0.0;
    Real muz = (ndim == 3) ? mu.d_view(oct,a,2) : 0.0;
    int sx = (mux > 0.0) ? 1 : -1;
    int sy = (ndim >= 2) ? ((muy > 0.0) ? 1 : -1) : 0;
    int sz = (ndim == 3) ? ((muz > 0.0) ? 1 : -1) : 0;
    Real dx1v = mbsize.d_view(m).dx1;
    Real dx2v = mbsize.d_view(m).dx2;
    Real dx3v = mbsize.d_view(m).dx3;
    ir_(m,angg,k,j,i) = UpdateCellSC(chi_,bb_,ir_,m,angg,
                                      i,j,k, sx,sy,sz,
                                      mux,muy,muz,
                                      dx1v,dx2v,dx3v, ndim,ks,js);
  });
}

}  // namespace nr_radiation
