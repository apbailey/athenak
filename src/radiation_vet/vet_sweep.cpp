//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file vet_sweep.cpp
//! \brief Hyperplane-swept short-characteristics formal solution (Davis, Stone & Jiang
//! 2012 Eq. 20), for all discrete-ordinate directions and all MeshBlocks in the pack at
//! once. Direct, dimension-generalized port of Athena-C's radiation/jacobi_{1d,
//! 2d_linear,3d_linear}.c update_cell() kernels (Kunasz & Auer 1988 quadratic S
//! interpolation + Auer 2003 Bezier slope limiting; linear/bilinear transverse
//! interpolation of the upwind/downwind footpoints, per vet-athenak-plan-deepdive.md
//! Sec. D4's Level-3 "linear transverse interpolation is embarrassingly
//! hyperplane-parallel" argument).
//!
//! Rather than hand-branching on fixed octant index values 0..7 (as the C reference
//! does), the upwind/downwind stencil offsets are derived directly from sign(mu_x/y/z)
//! -- mathematically identical, but dimension/octant-agnostic, so one kernel per
//! dimensionality (1D/2D/3D) below covers all octants at once via the flattened
//! (octant,angle) index "angg".
//!
//! ir(m,angle,k,j,i) is persistent: cells not yet reached by a given ray's sweep still
//! hold last iteration's (or the initial) values, and ghost zones are never written
//! here -- they hold whatever the last ApplyPhysicalBCs/RecvAndUnpackCC call left there,
//! which is exactly the desired upwind boundary/neighbor-block data for Eq. 20's I_0.
//! chi/bb are fully precomputed (incl. ghost zones) by UpdateOpacityAndSource() before
//! any sweep begins, so they may be safely read anywhere in the stencil without a
//! read/write race; ir's per-direction upwind footpoints always lie strictly "before"
//! the current cell in the hyperplane order (h strictly smaller), so no race exists
//! there either -- see the file-level comment for the argument in full.

#include <cmath>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "radiation_vet.hpp"
#include "vet_interp.hpp"

namespace radiation_vet {

//----------------------------------------------------------------------------------------
//! \fn void RadiationVET::FormalSolution
//! \brief Sweeps every discrete ordinate across every MeshBlock in the pack, filling ir.
//! Host loop over hyperplane index h; device par_for flattens (MeshBlock x direction x
//! transverse-plane cell).

void RadiationVET::FormalSolution() {
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
    //--------------------------------------------------------------------------------
    // 1D: hyperplane = single cell; no transverse interpolation (jacobi_1d.c)
    for (int h = 0; h < nx1; ++h) {
      par_for("vet_sweep1d", DevExeSpace(), 0, nmb1, 0, nangt1,
      KOKKOS_LAMBDA(int m, int angg) {
        int oct = angg / nang;
        int a = angg - oct*nang;
        Real mux = mu.d_view(oct,a,0);
        int sx = (mux > 0.0) ? 1 : -1;
        int i  = (sx > 0) ? (is + h) : (ie - h);
        int im = i - sx;
        int ip = i + sx;
        Real dx1 = mbsize.d_view(m).dx1;

        Real chi0 = chi_(m,ks,js,im);
        Real chi1 = chi_(m,ks,js,i);
        Real chi2 = chi_(m,ks,js,ip);
        Real S0 = bb_(m,ks,js,im);
        Real S1 = bb_(m,ks,js,i);
        Real S2 = bb_(m,ks,js,ip);
        Real imu0 = ir_(m,angg,ks,js,im);

        Real dtaum = InterpQuadChi(chi0,chi1,chi2) * dx1 / fabs(mux);
        Real dtaup = InterpQuadChi(chi2,chi1,chi0) * dx1 / fabs(mux);
        Real edtau, a0, a1, a2;
        InterpQuadSourceSlopeLim(dtaum, dtaup, S0, S1, S2, &edtau, &a0, &a1, &a2);
        Real imu = a0*S0 + a1*S1 + a2*S2 + edtau*imu0;
        ir_(m,angg,ks,js,i) = imu;
      });
    }
  } else if (ndim == 2) {
    //--------------------------------------------------------------------------------
    // 2D: hyperplane h = li1' + li2', li1' in [0,nx1-1], li2' derived and masked
    // (jacobi_2d_linear.c update_cell(), am<=1/am>1 branches)
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
        int im = i - sx, ip = i + sx;
        int jm = j - sy, jp = j + sy;
        Real dx1 = mbsize.d_view(m).dx1;
        Real dx2 = mbsize.d_view(m).dx2;

        Real chi1 = chi_(m,ks,j,i);
        Real S1 = bb_(m,ks,j,i);
        Real S0, S2, chi0, chi2, imu0, dtaum, dtaup;

        Real am = fabs((dx2*mux) / (dx1*muy));
        if (am <= 1.0) {
          Real am1 = 1.0 - am;
          S0    = am*bb_(m,ks,jm,im)  + am1*bb_(m,ks,jm,i);
          S2    = am*bb_(m,ks,jp,ip)  + am1*bb_(m,ks,jp,i);
          chi0  = am*chi_(m,ks,jm,im) + am1*chi_(m,ks,jm,i);
          chi2  = am*chi_(m,ks,jp,ip) + am1*chi_(m,ks,jp,i);
          imu0  = am*ir_(m,angg,ks,jm,im) + am1*ir_(m,angg,ks,jm,i);
          dtaum = InterpQuadChi(chi0,chi1,chi2) * dx2/fabs(muy);
          dtaup = InterpQuadChi(chi2,chi1,chi0) * dx2/fabs(muy);
        } else {
          Real bm  = 1.0/am;
          Real bm1 = 1.0 - bm;
          S0    = bm*bb_(m,ks,jm,im)  + bm1*bb_(m,ks,j,im);
          S2    = bm*bb_(m,ks,jp,ip)  + bm1*bb_(m,ks,j,ip);
          chi0  = bm*chi_(m,ks,jm,im) + bm1*chi_(m,ks,j,im);
          chi2  = bm*chi_(m,ks,jp,ip) + bm1*chi_(m,ks,j,ip);
          imu0  = bm*ir_(m,angg,ks,jm,im) + bm1*ir_(m,angg,ks,j,im);
          dtaum = InterpQuadChi(chi0,chi1,chi2) * dx1/fabs(mux);
          dtaup = InterpQuadChi(chi2,chi1,chi0) * dx1/fabs(mux);
        }

        Real edtau, a0, a1, a2;
        InterpQuadSourceSlopeLim(dtaum, dtaup, S0, S1, S2, &edtau, &a0, &a1, &a2);
        Real imu = a0*S0 + a1*S1 + a2*S2 + edtau*imu0;
        ir_(m,angg,ks,j,i) = imu;
      });
    }
  } else {
    //--------------------------------------------------------------------------------
    // 3D: hyperplane h = li1'+li2'+li3'; upwind/downwind footpoints bilinearly
    // interpolated on whichever coordinate plane the ray exits first
    // (jacobi_3d_linear.c update_cell(), face=0/1/2 branches).
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
        int im = i - sx, ip = i + sx;
        int jm = j - sy, jp = j + sy;
        int km = k - sz, kp = k + sz;
        Real dx1 = mbsize.d_view(m).dx1;
        Real dx2 = mbsize.d_view(m).dx2;
        Real dx3 = mbsize.d_view(m).dx3;

        Real chi1 = chi_(m,k,j,i);
        Real S1 = bb_(m,k,j,i);

        Real lx = dx1/fabs(mux), ly = dx2/fabs(muy), lz = dx3/fabs(muz);
        Real lmin = fmin(fmin(lx,ly),lz);

        Real S0, S2, chi0, chi2, imu0, dtaum, dtaup;
        if (lmin == lx) {
          // ray exits through the x2-x3 (j,k) plane first
          Real am = lmin/ly, bm = lmin/lz;
          Real c0 = (1.0-am)*(1.0-bm), c1 = (1.0-am)*bm, c2 = am*bm, c3 = am*(1.0-bm);
          S0    = c0*bb_(m,k ,j ,im) + c1*bb_(m,km,j ,im) +
                  c2*bb_(m,km,jm,im) + c3*bb_(m,k ,jm,im);
          chi0  = c0*chi_(m,k ,j ,im) + c1*chi_(m,km,j ,im) +
                  c2*chi_(m,km,jm,im) + c3*chi_(m,k ,jm,im);
          imu0  = c0*ir_(m,angg,k ,j ,im) + c1*ir_(m,angg,km,j ,im) +
                  c2*ir_(m,angg,km,jm,im) + c3*ir_(m,angg,k ,jm,im);
          S2    = c0*bb_(m,k ,j ,ip) + c1*bb_(m,kp,j ,ip) +
                  c2*bb_(m,kp,jp,ip) + c3*bb_(m,k ,jp,ip);
          chi2  = c0*chi_(m,k ,j ,ip) + c1*chi_(m,kp,j ,ip) +
                  c2*chi_(m,kp,jp,ip) + c3*chi_(m,k ,jp,ip);
          dtaum = InterpQuadChi(chi0,chi1,chi2) * dx1/fabs(mux);
          dtaup = InterpQuadChi(chi2,chi1,chi0) * dx1/fabs(mux);
        } else if (lmin == ly) {
          // ray exits through the x1-x3 (i,k) plane first
          Real am = lmin/lx, bm = lmin/lz;
          Real c0 = (1.0-am)*(1.0-bm), c1 = (1.0-am)*bm, c2 = am*bm, c3 = am*(1.0-bm);
          S0    = c0*bb_(m,k ,jm,i ) + c1*bb_(m,km,jm,i ) +
                  c2*bb_(m,km,jm,im) + c3*bb_(m,k ,jm,im);
          chi0  = c0*chi_(m,k ,jm,i ) + c1*chi_(m,km,jm,i ) +
                  c2*chi_(m,km,jm,im) + c3*chi_(m,k ,jm,im);
          imu0  = c0*ir_(m,angg,k ,jm,i ) + c1*ir_(m,angg,km,jm,i ) +
                  c2*ir_(m,angg,km,jm,im) + c3*ir_(m,angg,k ,jm,im);
          S2    = c0*bb_(m,k ,jp,i ) + c1*bb_(m,kp,jp,i ) +
                  c2*bb_(m,kp,jp,ip) + c3*bb_(m,k ,jp,ip);
          chi2  = c0*chi_(m,k ,jp,i ) + c1*chi_(m,kp,jp,i ) +
                  c2*chi_(m,kp,jp,ip) + c3*chi_(m,k ,jp,ip);
          dtaum = InterpQuadChi(chi0,chi1,chi2) * dx2/fabs(muy);
          dtaup = InterpQuadChi(chi2,chi1,chi0) * dx2/fabs(muy);
        } else {
          // ray exits through the x1-x2 (i,j) plane first
          Real am = lmin/lx, bm = lmin/ly;
          Real c0 = (1.0-am)*(1.0-bm), c1 = (1.0-am)*bm, c2 = am*bm, c3 = am*(1.0-bm);
          S0    = c0*bb_(m,km,j ,i ) + c1*bb_(m,km,jm,i ) +
                  c2*bb_(m,km,jm,im) + c3*bb_(m,km,j ,im);
          chi0  = c0*chi_(m,km,j ,i ) + c1*chi_(m,km,jm,i ) +
                  c2*chi_(m,km,jm,im) + c3*chi_(m,km,j ,im);
          imu0  = c0*ir_(m,angg,km,j ,i ) + c1*ir_(m,angg,km,jm,i ) +
                  c2*ir_(m,angg,km,jm,im) + c3*ir_(m,angg,km,j ,im);
          S2    = c0*bb_(m,kp,j ,i ) + c1*bb_(m,kp,jp,i ) +
                  c2*bb_(m,kp,jp,ip) + c3*bb_(m,kp,j ,ip);
          chi2  = c0*chi_(m,kp,j ,i ) + c1*chi_(m,kp,jp,i ) +
                  c2*chi_(m,kp,jp,ip) + c3*chi_(m,kp,j ,ip);
          dtaum = InterpQuadChi(chi0,chi1,chi2) * dx3/fabs(muz);
          dtaup = InterpQuadChi(chi2,chi1,chi0) * dx3/fabs(muz);
        }

        Real edtau, a0, a1, a2;
        InterpQuadSourceSlopeLim(dtaum, dtaup, S0, S1, S2, &edtau, &a0, &a1, &a2);
        Real imu = a0*S0 + a1*S1 + a2*S2 + edtau*imu0;
        ir_(m,angg,k,j,i) = imu;
      });
    }
  }
}

}  // namespace radiation_vet
