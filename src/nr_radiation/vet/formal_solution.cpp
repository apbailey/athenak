//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file formal_solution.cpp
//! \brief Short-characteristics formal solution (Davis, Stone & Jiang 2012 Eq. 20)
//! with three dispatch modes selectable via nr_radiation/sweep. When use_ali is true,
//! also accumulates lamstr += wμ·Ψ⁰ (Olson & Kunasz diagonal Λ*) via atomic_add.

#include <algorithm>
#include <cmath>
#include <utility>   // std::swap

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "nr_radiation/nr_radiation.hpp"
#include "nr_radiation/vet/vet_interp.hpp"

namespace nr_radiation {

//----------------------------------------------------------------------------------------
//! \fn void VET::FormalSolution
//! \brief Dispatcher — delegates to the implementation selected by sweep_method.

void VET::FormalSolution() {
  if (use_ali) {
    Kokkos::deep_copy(DevExeSpace(), lamstr, 0.0);
  }
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
  auto &wmu = pang->wmu;
  auto &mbsize = pmy_pack->pmb->mb_size;
  auto ir_ = ir;
  auto chi_ = chi;
  auto bb_ = bb;
  auto lam_ = lamstr;
  bool accumulate = use_ali;

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
        Real a1 = 0.0;
        Real I = UpdateCellSC(chi_,bb_,ir_,m,angg,
                              i,js,ks, sx,0,0,
                              mux,0.0,0.0,
                              dx1v,0.0,0.0, ndim,ks,js, &a1);
        ir_(m,angg,ks,js,i) = I;
        if (accumulate) {
          Kokkos::atomic_add(&lam_(m,ks,js,i), wmu.d_view(a) * a1);
        }
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
        Real a1 = 0.0;
        Real I = UpdateCellSC(chi_,bb_,ir_,m,angg,
                              i,j,ks, sx,sy,0,
                              mux,muy,0.0,
                              dx1v,dx2v,0.0, ndim,ks,js, &a1);
        ir_(m,angg,ks,j,i) = I;
        if (accumulate) {
          Kokkos::atomic_add(&lam_(m,ks,j,i), wmu.d_view(a) * a1);
        }
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
        Real a1 = 0.0;
        Real I = UpdateCellSC(chi_,bb_,ir_,m,angg,
                              i,j,k, sx,sy,sz,
                              mux,muy,muz,
                              dx1v,dx2v,dx3v, ndim,ks,js, &a1);
        ir_(m,angg,k,j,i) = I;
        if (accumulate) {
          Kokkos::atomic_add(&lam_(m,k,j,i), wmu.d_view(a) * a1);
        }
      });
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn void VET::FormalSolutionDiagonal

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
  auto &wmu = pang->wmu;
  auto &mbsize = pmy_pack->pmb->mb_size;
  auto ir_ = ir;
  auto chi_ = chi;
  auto bb_ = bb;
  auto lam_ = lamstr;
  bool accumulate = use_ali;

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
    Real w = wmu.d_view(a);

    for (int h = 0; h <= hmax; ++h) {
      if (ndim == 1) {
        Kokkos::parallel_for(Kokkos::TeamThreadRange(tmember, 1),
        [&](const int /*idx*/) {
          int i = (sx > 0) ? (is + h) : (ie - h);
          Real a1 = 0.0;
          Real I = UpdateCellSC(chi_,bb_,ir_,m,angg,
                                i,js,ks, sx,0,0,
                                mux,0.0,0.0,
                                dx1v,0.0,0.0, ndim,ks,js, &a1);
          ir_(m,angg,ks,js,i) = I;
          if (accumulate) {
            Kokkos::atomic_add(&lam_(m,ks,js,i), w * a1);
          }
        });
      } else if (ndim == 2) {
        Kokkos::parallel_for(Kokkos::TeamThreadRange(tmember, nx1),
        [&](const int li1) {
          int li2 = h - li1;
          if (li2 < 0 || li2 >= nx2) return;
          int i = (sx > 0) ? (is + li1) : (ie - li1);
          int j = (sy > 0) ? (js + li2) : (je - li2);
          Real a1 = 0.0;
          Real I = UpdateCellSC(chi_,bb_,ir_,m,angg,
                                i,j,ks, sx,sy,0,
                                mux,muy,0.0,
                                dx1v,dx2v,0.0, ndim,ks,js, &a1);
          ir_(m,angg,ks,j,i) = I;
          if (accumulate) {
            Kokkos::atomic_add(&lam_(m,ks,j,i), w * a1);
          }
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
          Real a1 = 0.0;
          Real I = UpdateCellSC(chi_,bb_,ir_,m,angg,
                                i,j,k, sx,sy,sz,
                                mux,muy,muz,
                                dx1v,dx2v,dx3v, ndim,ks,js, &a1);
          ir_(m,angg,k,j,i) = I;
          if (accumulate) {
            Kokkos::atomic_add(&lam_(m,k,j,i), w * a1);
          }
        });
      }
      tmember.team_barrier();
    }
  });
}

//----------------------------------------------------------------------------------------
//! \fn void VET::FormalSolutionJacobi

void VET::FormalSolutionJacobi() {
  // Ping-pong double buffer. The jacobi sweep is an unordered par_for, so reading the upwind
  // intensity from the same `ir` that a sibling thread is writing would be a read-write DATA RACE.
  // Instead we read the previous sweep's field from `ir_prev` and write the new field into `ir`.
  // Swapping the two View handles up front is O(1) (no copy): it moves the bvals-filled previous
  // field into the read slot and leaves `ir` — the buffer every downstream consumer reads
  // (bvals/ComputeJ/output) — as the write target, matching wavefront/diagonal. `ir_prev` is
  // lazily allocated on first use; no seed copy is needed because the par_for overwrites the whole
  // interior every sweep and ghost zones are refilled by the next bvals exchange.
  if (ir_prev.size() == 0) {
    Kokkos::realloc(ir_prev, ir.extent_int(0), ir.extent_int(1),
                    ir.extent_int(2), ir.extent_int(3), ir.extent_int(4));
  }
  std::swap(ir, ir_prev);

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  int ndim = pang->ndim;
  int nang = pang->nang;
  int nangt1 = nang_tot - 1;
  auto &mu = pang->mu;
  auto &wmu = pang->wmu;
  auto &mbsize = pmy_pack->pmb->mb_size;
  auto ir_src = ir_prev;   // READ upwind intensity from the previous sweep's field
  auto ir_dst = ir;        // WRITE new field into ir (same target as wavefront/diagonal)
  auto chi_ = chi;
  auto bb_ = bb;
  auto lam_ = lamstr;
  bool accumulate = use_ali;

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
    Real a1 = 0.0;
    Real I = UpdateCellSC(chi_,bb_,ir_src,m,angg,
                          i,j,k, sx,sy,sz,
                          mux,muy,muz,
                          dx1v,dx2v,dx3v, ndim,ks,js, &a1);
    ir_dst(m,angg,k,j,i) = I;
    if (accumulate) {
      Kokkos::atomic_add(&lam_(m,k,j,i), wmu.d_view(a) * a1);
    }
  });
}

}  // namespace nr_radiation
