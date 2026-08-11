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
#include "nr_radiation/sc/sc_interp.hpp"

namespace nr_radiation {

//----------------------------------------------------------------------------------------
//! \fn void SC::FormalSolution
//! \brief Dispatcher — delegates to the implementation selected by sweep_method.

void SC::FormalSolution() {
  if (use_ali) {
    Kokkos::deep_copy(DevExeSpace(), lamstr, 0.0);
  }
  if (sweep_method == "diagonal" || sweep_method == "diagonal_compact") {
    FormalSolutionDiagonal();
  } else if (sweep_method == "jacobi") {
    FormalSolutionJacobi();
  } else if (sweep_method == "wavefront_coalesced") {
    FormalSolutionWavefrontCoalesced();
  } else if (ir_angle_inner) {
    FormalSolutionWavefrontAngleInner();   // native angle-innermost ir (I2)
  } else {
    FormalSolutionWavefront();
  }
}

//----------------------------------------------------------------------------------------
//! \fn void SC::BuildWavefrontIndex
//! \brief Precompute the compact per-hyperplane cell list for the 2D/3D wavefront sweep.
//! For each plane h (li1+li2+li3==h, li = distance from the upwind corner along each axis) we
//! enumerate exactly the interior cells that lie on it and store their packed linear index
//! lin=(li3*nx2+li2)*nx1+li1, grouped by plane via wf_plane_start_. This is a pure function of
//! the meshblock interior dims (identical for every meshblock and octant — the octant sign only
//! flips i=is+li1 vs ie-li1 in the kernel), so it is built once. Within a plane the cells are
//! causally independent (every footpoint sits on a strictly lower plane h-1..h-3, and the sweep
//! only writes ir while reading upwind ir + fixed srad/chi), so any ordering yields bit-identical
//! results; we iterate (li3 outer, li2 inner) for some transverse locality. Σ_h count == nx1·nx2·nx3.

void SC::BuildWavefrontIndex() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  int ndim = pang->ndim;
  int n2 = (ndim >= 2) ? nx2 : 1;
  int n3 = (ndim == 3) ? nx3 : 1;
  int ncells = nx1 * n2 * n3;
  int hmax;
  if (ndim == 1) hmax = nx1 - 1;
  else if (ndim == 2) hmax = nx1 + nx2 - 2;
  else hmax = nx1 + nx2 + nx3 - 3;

  HostArray1D<int> h_cell("wf_cell_host", ncells);
  wf_plane_start_.assign(hmax + 2, 0);
  int idx = 0;
  for (int h = 0; h <= hmax; ++h) {
    wf_plane_start_[h] = idx;
    for (int li3 = 0; li3 < n3; ++li3) {
      for (int li2 = 0; li2 < n2; ++li2) {
        int li1 = h - li2 - li3;
        if (li1 < 0 || li1 >= nx1) continue;
        h_cell(idx++) = (li3 * nx2 + li2) * nx1 + li1;   // packed linear interior index
      }
    }
  }
  wf_plane_start_[hmax + 1] = idx;   // == ncells

  Kokkos::realloc(wf_cell_, ncells);
  Kokkos::deep_copy(wf_cell_, h_cell);

  // Device mirror of the plane-offset table, for sweep=diagonal_compact (its h-loop runs on device
  // and must slice wf_cell_ per plane). Same one-time build; the host wavefront ignores it.
  int nstart = hmax + 2;
  HostArray1D<int> h_start("wf_plane_start_host", nstart);
  for (int h = 0; h < nstart; ++h) { h_start(h) = wf_plane_start_[h]; }
  Kokkos::realloc(wf_plane_start_dev_, nstart);
  Kokkos::deep_copy(wf_plane_start_dev_, h_start);
}

//----------------------------------------------------------------------------------------
//! \fn void SC::FormalSolutionWavefront

void SC::FormalSolutionWavefront() {
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
  auto srad_ = srad;
  auto lam_ = lamstr;
  bool accumulate = use_ali;

  if (ndim == 1) {
    for (int h = 0; h < nx1; ++h) {
      par_for("sc_sweep1d", DevExeSpace(), 0, nmb1, 0, nangt1,
      KOKKOS_LAMBDA(int m, int angg) {
        int oct = angg / nang;
        int a = angg - oct*nang;
        Real mux = mu.d_view(oct,a,0);
        int sx = (mux > 0.0) ? 1 : -1;
        int i  = (sx > 0) ? (is + h) : (ie - h);
        Real dx1v = mbsize.d_view(m).dx1;
        Real a1 = 0.0;
        Real I = UpdateCellSC(chi_,srad_,ir_,m,angg,
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
    if (wf_cell_.size() == 0) BuildWavefrontIndex();
    auto wfc_ = wf_cell_;
    int nx1_ = nx1;
    int hmax = nx1 + nx2 - 2;
    for (int h = 0; h <= hmax; ++h) {
      int lo = wf_plane_start_[h];
      int cnt = wf_plane_start_[h+1] - lo;   // exact cell count on plane h (no off-plane slots)
      if (cnt <= 0) continue;
      par_for("sc_sweep2d", DevExeSpace(), 0, nmb1, 0, nangt1, 0, cnt-1,
      KOKKOS_LAMBDA(int m, int angg, int c) {
        int lin = wfc_(lo + c);
        int li2 = lin / nx1_;
        int li1 = lin - li2*nx1_;
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
        Real I = UpdateCellSC(chi_,srad_,ir_,m,angg,
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
    if (wf_cell_.size() == 0) BuildWavefrontIndex();
    auto wfc_ = wf_cell_;
    int nx1_ = nx1;
    int nx12 = nx1*nx2;
    int hmax = nx1 + nx2 + nx3 - 3;
    for (int h = 0; h <= hmax; ++h) {
      int lo = wf_plane_start_[h];
      int cnt = wf_plane_start_[h+1] - lo;   // exact cell count on plane h (no off-plane slots)
      if (cnt <= 0) continue;
      par_for("sc_sweep3d", DevExeSpace(), 0, nmb1, 0, nangt1, 0, cnt-1,
      KOKKOS_LAMBDA(int m, int angg, int c) {
        int lin = wfc_(lo + c);
        int li3 = lin / nx12;
        int r = lin - li3*nx12;
        int li2 = r / nx1_;
        int li1 = r - li2*nx1_;
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
        Real I = UpdateCellSC(chi_,srad_,ir_,m,angg,
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
//! \fn void SC::FormalSolutionWavefrontCoalesced
//! \brief I2 gather-coalescing prototype (measurement only). Same wavefront sweep, but the intensity
//! is transposed into an angle-INNERMOST scratch `ir_t(m,k,j,i,angg)` and the plane kernel is reordered
//! so the WARP varies over angle (par_for(m,c,angg) => angg fastest). Consecutive lanes then read the
//! footpoint `ir` at stride 1 (coalesced) and the angle-independent chi/srad as a broadcast. Bit-identical
//! to `wavefront` (transpose is a pure copy; UpdateCellSC<true> runs the same FP ops via IrGet). The
//! transpose + doubled memory exist ONLY to isolate the gather prize without touching the global ir
//! layout or the shared bvals/AMR/moment code — they are not part of an eventual native design. 3D only
//! (1D/2D fall back to the plain wavefront). See rt-profiling/OPTIMIZATION_LEDGER.md I2.

void SC::FormalSolutionWavefrontCoalesced() {
  if (pang->ndim != 3) { FormalSolutionWavefront(); return; }
  if (wf_cell_.size() == 0) BuildWavefrontIndex();

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
  auto srad_ = srad;
  auto lam_ = lamstr;
  bool accumulate = use_ali;

  // lazily allocate the angle-innermost scratch: (nmb, nc3, nc2, nc1, nang_tot) = transpose of ir
  int nmb = ir_.extent_int(0), nc3 = ir_.extent_int(2);
  int nc2 = ir_.extent_int(3), nc1 = ir_.extent_int(4);
  if (ir_t.size() == 0) {
    Kokkos::realloc(ir_t, nmb, nc3, nc2, nc1, nang_tot);
  }
  auto ir_t_ = ir_t;

  // transpose IN: ir(m,angg,k,j,i) -> ir_t(m,k,j,i,angg), full array incl. ghosts
  par_for("sc_ir_transpose_in", DevExeSpace(), 0, nmb1, 0, nangt1, 0, nc3-1, 0, nc2-1, 0, nc1-1,
  KOKKOS_LAMBDA(int m, int angg, int k, int j, int i) {
    ir_t_(m,k,j,i,angg) = ir_(m,angg,k,j,i);
  });

  // reordered wavefront: warp varies over ANGLE (angg is the fastest par_for index); gather on ir_t
  int nx1_ = nx1;
  int nx12 = nx1*nx2;
  int hmax = nx1 + nx2 + nx3 - 3;
  auto wfc_ = wf_cell_;
  for (int h = 0; h <= hmax; ++h) {
    int lo = wf_plane_start_[h];
    int cnt = wf_plane_start_[h+1] - lo;
    if (cnt <= 0) continue;
    par_for("sc_sweep3d_coal", DevExeSpace(), 0, nmb1, 0, cnt-1, 0, nangt1,
    KOKKOS_LAMBDA(int m, int c, int angg) {
      int lin = wfc_(lo + c);
      int li3 = lin / nx12;
      int r = lin - li3*nx12;
      int li2 = r / nx1_;
      int li1 = r - li2*nx1_;
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
      Real I = UpdateCellSC<true>(chi_,srad_,ir_t_,m,angg,
                                  i,j,k, sx,sy,sz,
                                  mux,muy,muz,
                                  dx1v,dx2v,dx3v, ndim,ks,js, &a1);
      ir_t_(m,k,j,i,angg) = I;
      if (accumulate) {
        Kokkos::atomic_add(&lam_(m,k,j,i), wmu.d_view(a) * a1);
      }
    });
  }

  // transpose OUT: ir_t(m,k,j,i,angg) -> ir(m,angg,k,j,i), so downstream consumers see normal layout
  par_for("sc_ir_transpose_out", DevExeSpace(), 0, nmb1, 0, nangt1, 0, nc3-1, 0, nc2-1, 0, nc1-1,
  KOKKOS_LAMBDA(int m, int angg, int k, int j, int i) {
    ir_(m,angg,k,j,i) = ir_t_(m,k,j,i,angg);
  });
}

//----------------------------------------------------------------------------------------
//! \fn void SC::SyncIrNormal
//! \brief Bridge for ir_layout=angle_inner: transpose between the angle-INNERMOST `ir(m,k,j,i,angg)`
//! and the normal-layout companion `ir_normal(m,angg,k,j,i)` that the UNCHANGED CC exchange operates
//! on. to_normal=true feeds the exchange (ir -> ir_normal); false pulls fresh ghosts back (ir_normal
//! -> ir). Full-array copy (correctness first); a boundary-shell-only variant is the later speed opt.

void SC::SyncIrNormal(bool to_normal) {
  auto ir_ = ir;              // angle-inner (m,k,j,i,angg)
  auto irn_ = ir_normal;      // normal      (m,angg,k,j,i)
  int nmb1 = pmy_pack->nmb_thispack - 1;
  int nangt1 = nang_tot - 1;
  int nc3 = irn_.extent_int(2), nc2 = irn_.extent_int(3), nc1 = irn_.extent_int(4);
  if (to_normal) {
    par_for("sc_ir_to_normal", DevExeSpace(), 0, nmb1, 0, nangt1, 0, nc3-1, 0, nc2-1, 0, nc1-1,
    KOKKOS_LAMBDA(int m, int angg, int k, int j, int i) {
      irn_(m,angg,k,j,i) = ir_(m,k,j,i,angg);
    });
  } else {
    par_for("sc_ir_from_normal", DevExeSpace(), 0, nmb1, 0, nangt1, 0, nc3-1, 0, nc2-1, 0, nc1-1,
    KOKKOS_LAMBDA(int m, int angg, int k, int j, int i) {
      ir_(m,k,j,i,angg) = irn_(m,angg,k,j,i);
    });
  }
}

//----------------------------------------------------------------------------------------
//! \fn void SC::FormalSolutionWavefrontAngleInner
//! \brief I2 NATIVE: the coalesced angle-warp wavefront sweep run DIRECTLY on the angle-innermost
//! `ir` (no transpose). Identical math to FormalSolutionWavefront (bit-identical converged J), but
//! the warp varies over angle (`par_for(m,c,angg)`) so the footpoint `ir` gather coalesces and
//! chi/srad broadcast. 3D only (guaranteed by the ir_layout=angle_inner validator).

void SC::FormalSolutionWavefrontAngleInner() {
  if (wf_cell_.size() == 0) BuildWavefrontIndex();
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
  auto ir_ = ir;             // angle-innermost (m,k,j,i,angg)
  auto chi_ = chi;
  auto srad_ = srad;
  auto lam_ = lamstr;
  bool accumulate = use_ali;
  auto wfc_ = wf_cell_;
  int nx1_ = nx1;
  int nx12 = nx1*nx2;
  int hmax = nx1 + nx2 + nx3 - 3;
  for (int h = 0; h <= hmax; ++h) {
    int lo = wf_plane_start_[h];
    int cnt = wf_plane_start_[h+1] - lo;
    if (cnt <= 0) continue;
    par_for("sc_sweep3d_ai", DevExeSpace(), 0, nmb1, 0, cnt-1, 0, nangt1,
    KOKKOS_LAMBDA(int m, int c, int angg) {
      int lin = wfc_(lo + c);
      int li3 = lin / nx12;
      int r = lin - li3*nx12;
      int li2 = r / nx1_;
      int li1 = r - li2*nx1_;
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
      Real I = UpdateCellSC<true>(chi_,srad_,ir_,m,angg,
                                  i,j,k, sx,sy,sz,
                                  mux,muy,muz,
                                  dx1v,dx2v,dx3v, ndim,ks,js, &a1);
      ir_(m,k,j,i,angg) = I;
      if (accumulate) {
        Kokkos::atomic_add(&lam_(m,k,j,i), wmu.d_view(a) * a1);
      }
    });
  }
}

//----------------------------------------------------------------------------------------
//! \fn void SC::FormalSolutionDiagonal

void SC::FormalSolutionDiagonal() {
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
  auto srad_ = srad;
  auto lam_ = lamstr;
  bool accumulate = use_ali;

  // diagonal_compact: iterate ONLY the real interior cells per plane (reusing the wavefront's
  // compact list) instead of an nx1*nx2 candidate box with off-plane early-returns (~2/3 waste in
  // 3D). Bit-identical to "diagonal" (plane cells are causally independent; UpdateCellSC is a pure
  // function of strictly-lower planes). Only the 3D branch is compacted (1D is already 1 cell/plane).
  bool compact = (sweep_method == "diagonal_compact");
  if (compact && wf_cell_.size() == 0) { BuildWavefrontIndex(); }
  auto wfc_ = wf_cell_;
  auto wps_ = wf_plane_start_dev_;
  int nx12 = nx1 * nx2;

  int league_size = nmb * nang_tot_;
  int hmax;
  if (ndim == 1) hmax = nx1 - 1;
  else if (ndim == 2) hmax = nx1 + nx2 - 2;
  else hmax = nx1 + nx2 + nx3 - 3;

  // Team size: default Kokkos::AUTO; diagonal_compact may override via <nr_radiation>/diag_team_size
  // (>0) to raise resident threads on this latency-bound, team-starved kernel. Values must be within
  // the device/backend max (Kokkos errors otherwise); used only for the GPU A/B (AUTO on CPU).
  Kokkos::TeamPolicy<> policy = (compact && diag_team_size > 0)
      ? Kokkos::TeamPolicy<>(DevExeSpace(), league_size, diag_team_size)
      : Kokkos::TeamPolicy<>(DevExeSpace(), league_size, Kokkos::AUTO);
  Kokkos::parallel_for("sc_sweep_diag", policy,
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
          Real I = UpdateCellSC(chi_,srad_,ir_,m,angg,
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
          Real I = UpdateCellSC(chi_,srad_,ir_,m,angg,
                                i,j,ks, sx,sy,0,
                                mux,muy,0.0,
                                dx1v,dx2v,0.0, ndim,ks,js, &a1);
          ir_(m,angg,ks,j,i) = I;
          if (accumulate) {
            Kokkos::atomic_add(&lam_(m,ks,j,i), w * a1);
          }
        });
      } else if (compact) {
        // compact: exactly the plane's real interior cells (wf_cell_[lo..lo+cnt)), no early-return.
        int lo = wps_(h);
        int cnt = wps_(h + 1) - lo;
        Kokkos::parallel_for(Kokkos::TeamThreadRange(tmember, cnt),
        [&](const int c) {
          int lin = wfc_(lo + c);              // packed li: lin=(li3*nx2+li2)*nx1+li1 (matches build)
          int li3 = lin / nx12;
          int r = lin - li3 * nx12;
          int li2 = r / nx1;
          int li1 = r - li2 * nx1;
          int i = (sx > 0) ? (is + li1) : (ie - li1);
          int j = (sy > 0) ? (js + li2) : (je - li2);
          int k = (sz > 0) ? (ks + li3) : (ke - li3);
          Real a1 = 0.0;
          Real I = UpdateCellSC(chi_,srad_,ir_,m,angg,
                                i,j,k, sx,sy,sz,
                                mux,muy,muz,
                                dx1v,dx2v,dx3v, ndim,ks,js, &a1);
          ir_(m,angg,k,j,i) = I;
          if (accumulate) {
            Kokkos::atomic_add(&lam_(m,k,j,i), w * a1);
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
          Real I = UpdateCellSC(chi_,srad_,ir_,m,angg,
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
//! \fn void SC::FormalSolutionJacobi

void SC::FormalSolutionJacobi() {
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
  auto srad_ = srad;
  auto lam_ = lamstr;
  bool accumulate = use_ali;

  par_for("sc_sweep_jacobi", DevExeSpace(), 0, nmb1, 0, nangt1, ks, ke, js, je, is, ie,
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
    Real I = UpdateCellSC(chi_,srad_,ir_src,m,angg,
                          i,j,k, sx,sy,sz,
                          mux,muy,muz,
                          dx1v,dx2v,dx3v, ndim,ks,js, &a1);
    ir_dst(m,angg,k,j,i) = I;
    if (accumulate) {
      Kokkos::atomic_add(&lam_(m,k,j,i), wmu.d_view(a) * a1);
    }
  });
}

//----------------------------------------------------------------------------------------
//! \fn void SC::SweepUpdateGS
//! \brief Center-out Gauss-Seidel-ALI (Davis 2012 §3.4 / TF95; Option B, iteration/prototype/).
//! One fused wavefront pass. Per host plane h: Kernel A sweeps plane h for all octants,
//! accumulating ir, the mean intensity jmean (in-sweep) and the diagonal Λ* (lamstr); Kernel B
//! then finalizes every cell whose LAST octant just arrived — h == max(i-is,ie-i)[+max(j..)+..] —
//! via the Eq. 24 update S ← S + ω·ΔS IN PLACE, so downstream planes read the fresh S. Returns
//! max|ΔS/S| (from the UNRELAXED ΔS, like UpdateSourceALI). Wavefront (center-out) ordering only;
//! replaces FormalSolution+ComputeJ+UpdateSourceALI on the ali_mode=="gauss_seidel" path. The
//! ali_mode=="jacobi" path is untouched and bit-identical.
//!
//! Local scatter (gs_scatter; Option B's essential ≥2D increment, iteration/prototype/REPORT.md
//! "Consequence for Phase 2"). When a cell i finalizes with ΔS, Kernel B additionally scatters
//! ΔS into the mean intensity of each immediate Cartesian neighbour that has ALREADY had (some
//! of) its rays computed this sweep but has NOT itself finalized yet — the "psiint" idea (Olson &
//! Kunasz 1987; TF95 §2): a stale (old-S-based) contribution already sitting in that neighbour's
//! jmean is corrected forward by the same ΔS. Per axis direction the correction coefficient is
//! cpl_{axis}{p,m}(i) = Σ_{rays with that sign along that axis} w·(a0+e^{-Δτ}·a1)·(lmin/l_axis)
//! accumulated in Kernel A alongside lamstr (Eq. 24's Λ*); a1 is this cell's own local-source
//! weight and (a0+e^{-Δτ}·a1) is the one-step sensitivity of a downstream cell's intensity to
//! this cell's S (direct footprint term a0, plus the indirect term through the already-computed
//! upwind intensity that itself depends on this cell's S via a1). The GEOMETRIC weight lmin/l_axis
//! (l_axis = dx_axis/|mu_axis|, lmin = min over axes) is exactly sc_interp.hpp's own transverse
//! bilinear blend fraction (am_r/bm): a single 3D short-characteristic deposits its coupling as a
//! bilinear blend over the neighbours, FULL weight (=1) on the ray's dominant/shortest-path axis
//! and a FRACTIONAL weight (<1) on each of the two subordinate axes. Applying that same fraction
//! here spreads each ray's coupling across the axes exactly as the formal solver spreads its
//! footpoint stencil. gs_scatter_mode controls the ABSOLUTE scale: "normalized" (default) rescales
//! the three axis shares to sum to exactly c (the footpoint row-sum, correct GS, always stable);
//! "geometric" leaves them summing to c*(1+am_r+bm), an implicit over-relaxation that diverges in
//! isotropic 3D (see gs-scatter-3d-origin.md and nr_radiation.cpp's gs_scatter_mode comment).
//! AthenaK's two remaining departures from the exact form (still Q006-safe):
//!   (1) The coupling is evaluated at the FINALIZING cell i, not at the neighbour (whose exact
//!       a0/e^{-Δτ} would require re-deriving its own opacity stencil): a zeroth-order,
//!       locally-uniform-medium stand-in, exact in a homogeneous medium (matching the 1D/2D
//!       testbed, which used a single per-ray triple for exactly this reason).
//!   (2) The gate is "neighbour not yet finalized" (hl(neighbour) > h) rather than a genuine
//!       per-ray "already arrived" test (AthenaK's lockstep multi-octant Kernel A does not track
//!       per-ray arrival without extra storage). This can, for a bounded subset of octants/cells,
//!       apply the correction slightly before that specific ray has actually deposited at the
//!       neighbour — a bounded, self-limiting over-count (proportional to the already-small ΔS),
//!       not an unbounded error.
//! Neither departure changes the converged fixed point: at convergence ΔS→0 so the scatter
//! vanishes and S=(1-ε)Λ[S]+εB holds exactly regardless (Q006, design doc §6) — both are pure
//! *rate* approximations. Off by default (gs_scatter=false ⇒ bit-identical to the in-place-only
//! GS already shipped); opt in via <nr_radiation>/gs_scatter=true.
//!
//! Validated (2026-08, sc_atmosphere, Davis Eq. 30), gs_scatter_mode=normalized: 1D 300->204 iters
//! (single axis so W_x==1 either mode); 2D 342->248; 3D anisotropic 32x8x8 144->68; 3D ISOTROPIC
//! 8^3/16^3 converge (47/58) where the "geometric" mode diverges to NaN; thick isotropic
//! chi=100/200 converge in 10/2 iters; converged J/S match the scatter-off run (Q006). The old
//! "geometric fixes 3D" result held only on anisotropic grids (quasi-1D); normalized is what makes
//! genuine isotropic 3D stable at ali_omega=1. On stable problems the geometric mode's extra speed
//! is recoverable, under control, as normalized + ali_omega>1 (verified: normalized+omega=1.2 ==
//! geometric's 48 iters on the anisotropic 3D atmosphere). Full derivation + evidence:
//! iteration/gs-scatter-3d-origin.md.

void SC::SweepUpdateGS(Real &max_dS_rel) {
  Kokkos::deep_copy(DevExeSpace(), jmean, 0.0);
  Kokkos::deep_copy(DevExeSpace(), lamstr, 0.0);

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
  auto srad_ = srad;
  auto lam_ = lamstr;
  auto jmean_ = jmean;
  auto eps_ = eps;
  auto planck_ = planck;
  Real omega = ali_omega;
  bool do_scatter = gs_scatter;
  // Experimental (3D interrogation): renormalise the per-axis coupling so the three axis shares
  // sum to exactly c per ray (the footpoint-identity row-sum), instead of c·(1+am_r+bm). See
  // iteration/gs-scatter-3d-origin.md. do_norm=false ⇒ the shipped "geometric" weighting.
  bool do_norm = (gs_scatter_mode == "normalized");

  // Local-scatter coupling arrays: lazily allocated (like ir_prev for sweep=jacobi), only the
  // ones needed for ndim, only when gs_scatter is requested. Zeroed every call, like jmean/lamstr.
  if (do_scatter) {
    if (cpl_xp.size() == 0) {
      Kokkos::realloc(cpl_xp, jmean.extent_int(0), jmean.extent_int(1),
                      jmean.extent_int(2), jmean.extent_int(3));
      Kokkos::realloc(cpl_xm, jmean.extent_int(0), jmean.extent_int(1),
                      jmean.extent_int(2), jmean.extent_int(3));
      if (ndim >= 2) {
        Kokkos::realloc(cpl_yp, jmean.extent_int(0), jmean.extent_int(1),
                        jmean.extent_int(2), jmean.extent_int(3));
        Kokkos::realloc(cpl_ym, jmean.extent_int(0), jmean.extent_int(1),
                        jmean.extent_int(2), jmean.extent_int(3));
      }
      if (ndim == 3) {
        Kokkos::realloc(cpl_zp, jmean.extent_int(0), jmean.extent_int(1),
                        jmean.extent_int(2), jmean.extent_int(3));
        Kokkos::realloc(cpl_zm, jmean.extent_int(0), jmean.extent_int(1),
                        jmean.extent_int(2), jmean.extent_int(3));
      }
    }
    Kokkos::deep_copy(DevExeSpace(), cpl_xp, 0.0);
    Kokkos::deep_copy(DevExeSpace(), cpl_xm, 0.0);
    if (ndim >= 2) {
      Kokkos::deep_copy(DevExeSpace(), cpl_yp, 0.0);
      Kokkos::deep_copy(DevExeSpace(), cpl_ym, 0.0);
    }
    if (ndim == 3) {
      Kokkos::deep_copy(DevExeSpace(), cpl_zp, 0.0);
      Kokkos::deep_copy(DevExeSpace(), cpl_zm, 0.0);
    }
  }
  auto cxp_ = cpl_xp; auto cxm_ = cpl_xm;
  auto cyp_ = cpl_yp; auto cym_ = cpl_ym;
  auto czp_ = cpl_zp; auto czm_ = cpl_zm;

  int hmax;
  if (ndim == 1) hmax = nx1 - 1;
  else if (ndim == 2) hmax = nx1 + nx2 - 2;
  else hmax = nx1 + nx2 + nx3 - 3;

  int nx1a = ie - is + 1, nx2a = je - js + 1, nx3a = ke - ks + 1;
  int nmkji = (nmb1+1)*nx3a*nx2a*nx1a;

  Real gmax = 0.0;
  for (int h = 0; h <= hmax; ++h) {
    // ---- Kernel A: sweep plane h (all octants); accumulate ir, jmean, lamstr ----
    if (ndim == 1) {
      par_for("gs_sweepA1d", DevExeSpace(), 0, nmb1, 0, nangt1,
      KOKKOS_LAMBDA(int m, int angg) {
        int oct = angg / nang;
        int a = angg - oct*nang;
        Real mux = mu.d_view(oct,a,0);
        int sx = (mux > 0.0) ? 1 : -1;
        int i = (sx > 0) ? (is + h) : (ie - h);
        Real dx1v = mbsize.d_view(m).dx1;
        Real a1 = 0.0, a0 = 0.0, edtau = 0.0;
        Real I = UpdateCellSC(chi_,srad_,ir_,m,angg, i,js,ks, sx,0,0,
                              mux,0.0,0.0, dx1v,0.0,0.0, ndim,ks,js, &a1, &a0, &edtau);
        ir_(m,angg,ks,js,i) = I;
        Real w = wmu.d_view(a);
        Kokkos::atomic_add(&lam_(m,ks,js,i), w * a1);
        Kokkos::atomic_add(&jmean_(m,ks,js,i), w * I);
        if (do_scatter) {
          // 1D: x is the only (hence dominant) axis, so the geometric weight lmin/lx is exactly 1.
          Real c = w * (a0 + edtau * a1);
          if (sx > 0) { Kokkos::atomic_add(&cxp_(m,ks,js,i), c); }
          else        { Kokkos::atomic_add(&cxm_(m,ks,js,i), c); }
        }
      });
    } else if (ndim == 2) {
      par_for("gs_sweepA2d", DevExeSpace(), 0, nmb1, 0, nangt1, 0, (nx1-1),
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
        Real a1 = 0.0, a0 = 0.0, edtau = 0.0;
        Real I = UpdateCellSC(chi_,srad_,ir_,m,angg, i,j,ks, sx,sy,0,
                              mux,muy,0.0, dx1v,dx2v,0.0, ndim,ks,js, &a1, &a0, &edtau);
        ir_(m,angg,ks,j,i) = I;
        Real w = wmu.d_view(a);
        Kokkos::atomic_add(&lam_(m,ks,j,i), w * a1);
        Kokkos::atomic_add(&jmean_(m,ks,j,i), w * I);
        if (do_scatter) {
          Real c = w * (a0 + edtau * a1);
          // Geometric (exact fractional) weighting: split the diagonal coupling onto the two
          // axis-aligned neighbours by the path-length ratio lmin/l_axis (l_axis = dx_axis/|mu|,
          // exactly sc_interp.hpp's am/bm blend weights). The ray's dominant (shortest-path) axis
          // gets full weight 1; the subordinate axis gets lmin/l_axis < 1.
          Real lx = dx1v/fabs(mux), ly = dx2v/fabs(muy);
          Real lmin = fmin(lx,ly);
          Real wx = lmin/lx, wy = lmin/ly;
          if (do_norm) { Real s = wx + wy; wx /= s; wy /= s; }
          Real cx = c*wx, cy = c*wy;
          if (sx > 0) { Kokkos::atomic_add(&cxp_(m,ks,j,i), cx); }
          else        { Kokkos::atomic_add(&cxm_(m,ks,j,i), cx); }
          if (sy > 0) { Kokkos::atomic_add(&cyp_(m,ks,j,i), cy); }
          else        { Kokkos::atomic_add(&cym_(m,ks,j,i), cy); }
        }
      });
    } else {
      par_for("gs_sweepA3d", DevExeSpace(), 0, nmb1, 0, nangt1, 0,(nx1-1), 0,(nx2-1),
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
        Real a1 = 0.0, a0 = 0.0, edtau = 0.0;
        Real I = UpdateCellSC(chi_,srad_,ir_,m,angg, i,j,k, sx,sy,sz,
                              mux,muy,muz, dx1v,dx2v,dx3v, ndim,ks,js, &a1, &a0, &edtau);
        ir_(m,angg,k,j,i) = I;
        Real w = wmu.d_view(a);
        Kokkos::atomic_add(&lam_(m,k,j,i), w * a1);
        Kokkos::atomic_add(&jmean_(m,k,j,i), w * I);
        if (do_scatter) {
          Real c = w * (a0 + edtau * a1);
          // Geometric (exact fractional) weighting: split the diagonal coupling onto the three
          // axis-aligned neighbours by the path-length ratio lmin/l_axis (l_axis = dx_axis/|mu|,
          // exactly sc_interp.hpp's am_r/bm blend weights). The ray's dominant (shortest-path)
          // axis gets full weight 1; the two subordinate axes get lmin/l_axis < 1. The earlier
          // rule gave full weight to all 3 axes, over-counting the two subordinate ones and
          // pushing the 3D GS spectral radius past 1 (diverged to NaN — design doc §7).
          Real lx = dx1v/fabs(mux), ly = dx2v/fabs(muy), lz = dx3v/fabs(muz);
          Real lmin = fmin(fmin(lx,ly),lz);
          Real wx = lmin/lx, wy = lmin/ly, wz = lmin/lz;
          if (do_norm) { Real s = wx + wy + wz; wx /= s; wy /= s; wz /= s; }
          Real cx = c*wx, cy = c*wy, cz = c*wz;
          if (sx > 0) { Kokkos::atomic_add(&cxp_(m,k,j,i), cx); }
          else        { Kokkos::atomic_add(&cxm_(m,k,j,i), cx); }
          if (sy > 0) { Kokkos::atomic_add(&cyp_(m,k,j,i), cy); }
          else        { Kokkos::atomic_add(&cym_(m,k,j,i), cy); }
          if (sz > 0) { Kokkos::atomic_add(&czp_(m,k,j,i), cz); }
          else        { Kokkos::atomic_add(&czm_(m,k,j,i), cz); }
        }
      });
    }

    // ---- Kernel B: finalize cells whose LAST octant arrived at this plane (center-out) ----
    int hh = h;
    int ndim_ = ndim;
    Real plane_max = 0.0;
    Kokkos::parallel_reduce("gs_updateB", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
    KOKKOS_LAMBDA(const int idx, Real &lmax) {
      int m = idx / (nx3a*nx2a*nx1a);
      int kji = idx - m*(nx3a*nx2a*nx1a);
      int k = kji / (nx2a*nx1a);
      int ji = kji - k*(nx2a*nx1a);
      int j = ji / nx1a;
      int i = ji - j*nx1a + is;
      j += js;
      k += ks;
      int hl = (i-is > ie-i) ? (i-is) : (ie-i);
      if (ndim_ >= 2) hl += (j-js > je-j) ? (j-js) : (je-j);
      if (ndim_ == 3) hl += (k-ks > ke-k) ? (k-ks) : (ke-k);
      if (hl != hh) return;
      Real epsi = eps_(m,k,j,i);
      Real S = srad_(m,0,k,j,i);
      Real J = jmean_(m,k,j,i);
      Real B = planck_(m,k,j,i);
      Real lam = lam_(m,k,j,i);
      Real denom = 1.0 - (1.0 - epsi) * lam;
      if (fabs(denom) < 1.0e-14) denom = (denom >= 0.0) ? 1.0e-14 : -1.0e-14;
      Real Snew = (1.0 - epsi) * J + epsi * B;
      Real dS = (Snew - S) / denom;
      Real r = (fabs(S) > 0.0) ? fabs(dS / S) : fabs(dS);
      if (r != r) r = 1.0e300;
      Real dS_app = omega * dS;
      srad_(m,0,k,j,i) = S + dS_app;
      lmax = fmax(lmax, r);

      if (do_scatter) {
        // hl of an arbitrary cell (ii,jj,kk), same formula as above — used to test whether a
        // neighbour has already finalized (hl(neighbour) <= hh) or is still pending (> hh).
        auto hl_of = [&](int ii, int jj, int kk) {
          int v = (ii-is > ie-ii) ? (ii-is) : (ie-ii);
          if (ndim_ >= 2) v += (jj-js > je-jj) ? (jj-js) : (je-jj);
          if (ndim_ == 3) v += (kk-ks > ke-kk) ? (kk-ks) : (ke-kk);
          return v;
        };
        if (i+1 <= ie && hl_of(i+1,j,k) > hh) {
          Kokkos::atomic_add(&jmean_(m,k,j,i+1), cxp_(m,k,j,i) * dS_app);
        }
        if (i-1 >= is && hl_of(i-1,j,k) > hh) {
          Kokkos::atomic_add(&jmean_(m,k,j,i-1), cxm_(m,k,j,i) * dS_app);
        }
        if (ndim_ >= 2) {
          if (j+1 <= je && hl_of(i,j+1,k) > hh) {
            Kokkos::atomic_add(&jmean_(m,k,j+1,i), cyp_(m,k,j,i) * dS_app);
          }
          if (j-1 >= js && hl_of(i,j-1,k) > hh) {
            Kokkos::atomic_add(&jmean_(m,k,j-1,i), cym_(m,k,j,i) * dS_app);
          }
        }
        if (ndim_ == 3) {
          if (k+1 <= ke && hl_of(i,j,k+1) > hh) {
            Kokkos::atomic_add(&jmean_(m,k+1,j,i), czp_(m,k,j,i) * dS_app);
          }
          if (k-1 >= ks && hl_of(i,j,k-1) > hh) {
            Kokkos::atomic_add(&jmean_(m,k-1,j,i), czm_(m,k,j,i) * dS_app);
          }
        }
      }
    }, Kokkos::Max<Real>(plane_max));
    gmax = fmax(gmax, plane_max);
  }

#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &gmax, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
#endif
  max_dS_rel = gmax;
}

}  // namespace nr_radiation
