//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file sc_qrad_bdry.cpp
//! \brief The physical-boundary face flux of the differential coupling (Davis Eq. 28).
//!
//! A uniform emitting and absorbing medium filling the domain, no fluid. Whatever the
//! boundary flags, the divergence telescopes to the boundary alone, so
//!
//!   Sum_cells Q dV = -crat prat * (net outward flux through the domain boundary)
//!
//! and the test recomputes that right-hand side on the host, independently of the
//! device kernel, by applying the face rule ray by ray to the intensity: each ray
//! contributes from whichever side of the face it arrived from. Agreement checks two
//! things at once -- that the flux difference telescopes, and that the boundary rule the
//! kernel implements is the one intended.
//!
//! The boundary flags are left to the deck, because each is a different and independently
//! informative case:
//!   reflect   nothing crosses a mirror, so the boundary flux is identically zero and the
//!             heating must integrate to zero. Exact, and free of any analytic input: the
//!             ghost holds each ray's mirror image, and mirroring flips mu at equal
//!             weight, so the two halves of the sum cancel term by term.
//!   vacuum    nothing enters, so the flux is purely what escapes and the gas can only
//!             cool on balance.
//!   inflow    in minus out, with the incident intensity set here on every physical face.
//!   outflow   the ghost is a copy of the last active cell, so both halves of the sum
//!             read the same numbers and the face flux reduces to the interior H.
//!
//! Two further quantities are reported. Sum_k w_k mu_k, per axis: it should vanish,
//! and the branch no longer checks the quadrature at startup, while the boundary rule
//! leans on it. And the naive alternative, a full-sphere quadrature of the ghost cell
//! averaged with the interior: it is what the rule is not, and on an inflow face with an
//! isotropic incident field it collapses to zero, losing the incoming radiation whole.
//!
//! Columns of sc_qrad_bdry-errs.dat: the relative difference between Sum Q dV and the
//! independent boundary flux, the two values, the naive flux for contrast,
//! max |Sum w mu|, and the sweep count. Thresholds live in the pytest wrapper.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "pgen/pgen.hpp"
#include "nr_radiation/nr_radiation.hpp"

namespace {

struct BdryVars {
  Real sigma, b0, amp, i_inc, tol;
  Real x1min, x2min, x3min, x1len, x2len, x3len;
  bool multi_d, three_d;
};
BdryVars bv;

//----------------------------------------------------------------------------------------
//! \fn void BdryOpacity() / BdryEmission()
//! \brief Uniform medium over the full arrays including ghosts, as the hook contract
//! requires (the sweep reads both at ghost footpoints).

void BdryOpacity(MeshBlockPack *pmbp) {
  auto &indcs = pmbp->pmesh->mb_indcs;
  const int ng = indcs.ng;
  const int n1 = indcs.nx1 + 2*ng;
  const int n2 = (indcs.nx2 > 1) ? indcs.nx2 + 2*ng : 1;
  const int n3 = (indcs.nx3 > 1) ? indcs.nx3 + 2*ng : 1;
  const int nmb1 = pmbp->nmb_thispack - 1;
  auto sigma_a = pmbp->pnrrad->sigma_a;
  const Real sig = bv.sigma;
  par_for("sc_qbdry_sigma", DevExeSpace(), 0,nmb1, 0,n3-1, 0,n2-1, 0,n1-1,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    sigma_a(m,0,k,j,i) = sig;
  });
}

void BdryEmission(MeshBlockPack *pmbp) {
  auto &indcs = pmbp->pmesh->mb_indcs;
  const int is = indcs.is, js = indcs.js, ks = indcs.ks;
  const int ng = indcs.ng;
  const int n1 = indcs.nx1 + 2*ng;
  const int n2 = (indcs.nx2 > 1) ? indcs.nx2 + 2*ng : 1;
  const int n3 = (indcs.nx3 > 1) ? indcs.nx3 + 2*ng : 1;
  const int nmb1 = pmbp->nmb_thispack - 1;
  auto brad = pmbp->pnrrad->brad;
  auto &size = pmbp->pmb->mb_size;
  const BdryVars v = bv;
  // Deliberately NOT uniform. A uniform source in a uniform medium is already in
  // equilibrium, so J = brad, Q = 0 and H = 0 everywhere, and every check below would
  // pass on nothing. One wavelength of the box gives a real interior flux while leaving
  // the boundary flux whatever the flags make it -- exactly zero for reflect.
  par_for("sc_qbdry_brad", DevExeSpace(), 0,nmb1, 0,n3-1, 0,n2-1, 0,n1-1,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real x1 = size.d_view(m).x1min + (i - is + 0.5)*size.d_view(m).dx1;
    Real ph = 2.0*M_PI*(x1 - v.x1min)/v.x1len;
    if (v.multi_d) {
      Real x2 = size.d_view(m).x2min + (j - js + 0.5)*size.d_view(m).dx2;
      ph += 2.0*M_PI*(x2 - v.x2min)/v.x2len;
    }
    if (v.three_d) {
      Real x3 = size.d_view(m).x3min + (k - ks + 0.5)*size.d_view(m).dx3;
      ph += 2.0*M_PI*(x3 - v.x3min)/v.x3len;
    }
    brad(m,k,j,i) = v.b0*(1.0 + v.amp*sin(ph));
  });
}

//----------------------------------------------------------------------------------------
//! \fn void SCQradBdryErrors
//! \brief Compare the domain heating with an independent host-side boundary flux.

void SCQradBdryErrors(ParameterInput *pin, Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  nr_radiation::SC *psc = pmbp->pnrrad;
  auto &indcs = pm->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb1 = pmbp->nmb_thispack - 1;
  const int nang = psc->pang->nang;
  const int nang_tot = psc->nang_tot;
  const bool multi_d = pm->multi_d;
  const bool three_d = pm->three_d;

  auto qrad_h = Kokkos::create_mirror_view(psc->qrad);
  Kokkos::deep_copy(qrad_h, psc->qrad);
  auto ir_h = Kokkos::create_mirror_view(psc->ir);
  Kokkos::deep_copy(ir_h, psc->ir);
  auto &mu = psc->pang->mu;
  auto &wmu = psc->pang->wmu;
  mu.template sync<HostMemSpace>();
  wmu.template sync<HostMemSpace>();
  auto &mb_bcs = pmbp->pmb->mb_bcs;
  auto &size = pmbp->pmb->mb_size;

  // the quadrature identity the boundary rule leans on, no longer checked at startup
  Real wmu_max = 0.0;
  for (int d = 0; d < 3; ++d) {
    if (d == 1 && !multi_d) continue;
    if (d == 2 && !three_d) continue;
    Real sw = 0.0;
    for (int oct = 0; oct < psc->pang->noct; ++oct) {
      for (int a = 0; a < nang; ++a) sw += wmu.h_view(a) * mu.h_view(oct, a, d);
    }
    wmu_max = std::fmax(wmu_max, std::fabs(sw));
  }

  Real sum_q = 0.0, sum_abs = 0.0, flux_out = 0.0, flux_naive = 0.0;
  for (int m = 0; m <= nmb1; ++m) {
    Real dx1 = size.h_view(m).dx1;
    Real dx2 = multi_d ? size.h_view(m).dx2 : 1.0;
    Real dx3 = three_d ? size.h_view(m).dx3 : 1.0;
    for (int k = ks; k <= ke; ++k) {
      for (int j = js; j <= je; ++j) {
        for (int i = is; i <= ie; ++i) {
          Real q = qrad_h(m,k,j,i) * dx1*dx2*dx3;
          sum_q += q;
          sum_abs += std::fabs(q);
        }
      }
    }
    // net OUTWARD flux over this block's physical faces, ray by ray from the intensity
    for (int f = 0; f < 6; ++f) {
      int d = f/2;
      if (d == 1 && !multi_d) continue;
      if (d == 2 && !three_d) continue;
      if (mb_bcs.h_view(m,f) == BoundaryFlag::block) continue;   // has a neighbour
      const bool hi_end = (f % 2 == 1);
      const Real nsign = hi_end ? 1.0 : -1.0;     // outward normal along +d
      const Real area = (d == 0) ? dx2*dx3 : ((d == 1) ? dx1*dx3 : dx1*dx2);
      // the two cells either side of that face, in increasing index order
      const int ilo = (d == 0) ? (hi_end ? ie : is-1) : is;
      const int jlo = (d == 1) ? (hi_end ? je : js-1) : js;
      const int klo = (d == 2) ? (hi_end ? ke : ks-1) : ks;
      const int i1 = (d == 0) ? 1 : 0, j1 = (d == 1) ? 1 : 0, k1 = (d == 2) ? 1 : 0;
      const int ni = (d == 0) ? 1 : (ie-is+1);
      const int nj = (d == 1) ? 1 : (je-js+1);
      const int nk = (d == 2) ? 1 : (ke-ks+1);
      for (int kk = 0; kk < nk; ++kk) {
        for (int jj = 0; jj < nj; ++jj) {
          for (int ii = 0; ii < ni; ++ii) {
            int kl = klo + kk, jl = jlo + jj, il = ilo + ii;
            Real h = 0.0, hlo = 0.0, hhi = 0.0;
            for (int angg = 0; angg < nang_tot; ++angg) {
              int oct = angg / nang;
              int a = angg - oct*nang;
              Real w = wmu.h_view(a);
              Real mun = mu.h_view(oct, a, d);
              Real i_lo = ir_h(m, angg, kl, jl, il);
              Real i_hi = ir_h(m, angg, kl+k1, jl+j1, il+i1);
              h   += w * mun * ((mun > 0.0) ? i_lo : i_hi);   // per-ray upwind
              hlo += w * mun * i_lo;
              hhi += w * mun * i_hi;
            }
            flux_out   += nsign * h * area;
            flux_naive += nsign * 0.5*(hlo + hhi) * area;     // the rule this is not
          }
        }
      }
    }
  }
  const Real expect = -psc->crat * psc->prat * flux_out;
  const Real naive = -psc->crat * psc->prat * flux_naive;

#if MPI_PARALLEL_ENABLED
  Real buf[4] = {sum_q, sum_abs, flux_out, flux_naive};
  MPI_Allreduce(MPI_IN_PLACE, buf, 4, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  sum_q = buf[0];
  sum_abs = buf[1];
#endif

  // Normalised by the total heating magnitude, not by the boundary flux itself. With
  // reflecting walls the flux is zero by construction and so is the sum, and dividing one
  // round-off residue by another would report a meaningless O(1) number; Sum |Q| dV is
  // the natural scale of the quantity being summed and is finite in every case.
  const Real rel = (sum_abs > 0.0) ? std::fabs(sum_q - expect)/sum_abs : 0.0;
  // normalised against the total heating magnitude: the meaningful number when the
  // boundary flux is itself zero, as it is for reflect
  const Real resid = (sum_abs > 0.0) ? std::fabs(sum_q)/sum_abs : std::fabs(sum_q);

  std::cout << "SC qrad bdry (" << psc->qrad_form_name << "): Sum Q dV = " << sum_q
            << ", independent boundary flux = " << expect << ", rel diff = " << rel
            << ", |Sum Q dV|/Sum |Q| dV = " << resid
            << "  (naive ghost-average flux " << naive << "; max |Sum w mu| = "
            << wmu_max << "; " << psc->niter_last << " sweeps)" << std::endl;

  if (global_variable::my_rank == 0) {
    std::string fname = "sc_qrad_bdry-errs.dat";
    FILE *pf = std::fopen(fname.c_str(), "r");
    if (pf != nullptr) {
      pf = std::freopen(fname.c_str(), "a", pf);
    } else {
      pf = std::fopen(fname.c_str(), "w");
      std::fprintf(pf, "# Nx1  Nx2  Nx3   Ncycle   rel-diff     residual     "
                       "SumQdV       Sum|Q|dV     bdry-flux    naive-flux   "
                       "sum-w-mu     niter\n");
    }
    std::fprintf(pf, "%04d  %04d  %04d  %05d  %e %e %e %e %e %e %e %d\n",
                 pm->mesh_indcs.nx1, pm->mesh_indcs.nx2, pm->mesh_indcs.nx3, pm->ncycle,
                 rel, resid, sum_q, sum_abs, expect, naive, wmu_max, psc->niter_last);
    std::fclose(pf);
  }

  // loose crash-guard only; the real thresholds live in the pytest wrapper
  if (!(rel < bv.tol)) {                        // also fails on NaN
    std::cout << "### SC qrad bdry FAILED (crash-guard): rel diff = " << rel
              << " not < " << bv.tol << std::endl;
    std::exit(EXIT_FAILURE);
  }
}
}  // namespace

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::SCQradBdry()
//! \brief Uniform medium, hooks enrolled before any restart return, incident intensity
//! set on every physical face.

void ProblemGenerator::SCQradBdry(ParameterInput *pin, const bool restart) {
  pgen_final_func = SCQradBdryErrors;
  bv.sigma = pin->GetOrAddReal("problem", "sigma", 1.0);
  bv.b0    = pin->GetOrAddReal("problem", "b0", 1.0);
  bv.amp   = pin->GetOrAddReal("problem", "amp", 0.5);
  bv.i_inc = pin->GetOrAddReal("problem", "i_inc", 1.0);
  bv.tol   = pin->GetOrAddReal("problem", "tol", 1.0e-2);

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->pnrrad == nullptr) {
    std::cout << "### FATAL ERROR: sc_qrad_bdry needs <nr_radiation>" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (pmbp->phydro != nullptr || pmbp->pmhd != nullptr) {
    std::cout << "### FATAL ERROR: sc_qrad_bdry takes no fluid; the check is on the "
              << "radiation bookkeeping alone" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (pmbp->pnrrad->qrad_form == nr_radiation::QradForm::integral) {
    std::cout << "### FATAL ERROR: sc_qrad_bdry needs a differential "
              << "<nr_radiation>/qrad_form; the integral form reads no face flux"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  auto &msz = pmy_mesh_->mesh_size;
  bv.x1min = msz.x1min;  bv.x1len = msz.x1max - msz.x1min;
  bv.x2min = msz.x2min;  bv.x2len = msz.x2max - msz.x2min;
  bv.x3min = msz.x3min;  bv.x3len = msz.x3max - msz.x3min;
  bv.multi_d = pmy_mesh_->multi_d;
  bv.three_d = pmy_mesh_->three_d;
  pmbp->pnrrad->EnrollOpacityFunction(BdryOpacity);
  pmbp->pnrrad->EnrollEmissionFunction(BdryEmission);
  if (restart) return;

  // isotropic incident intensity on every physical face; unused by flags that do not
  // read it (outflow, vacuum, reflect), and not allocated at all on a periodic mesh
  if (!pmy_mesh_->strictly_periodic) {
    auto &i_in = pmbp->pnrrad->pbval_ir->i_in;
    for (int n = 0; n < pmbp->pnrrad->nang_tot; ++n) {
      for (int f = 0; f < 6; ++f) i_in.h_view(n, f) = bv.i_inc;
    }
    i_in.template modify<HostMemSpace>();
    i_in.template sync<DevExeSpace>();
  }
}
