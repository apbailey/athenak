//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file sc_qrad_conserve.cpp
//! \brief Closed-box conservation of the differential coupling, Davis Eq. 28.
//!
//! A periodic box of gas with a smooth density and pressure perturbation, so that both
//! the opacity kappa_a*rho and the emission T^4 vary from cell to cell and the radiation
//! field has a genuine flux. There is no boundary, so the flux divergence telescopes to
//! nothing and the heating integrates to zero:
//!
//!   Sum_cells Q dV = 0
//!
//! exactly, for qrad_form = divh. The point of the test is that this holds whatever the
//! state of the solve: the cancellation is between face values shared by neighbouring
//! cells, so an unconverged iteration or a one-ray angular grid puts the energy in the
//! wrong place without creating or destroying any. Run it at iter_max = 1 and it must
//! still pass. The integral form has no such identity and does not pass; the test reports
//! its residual as the contrast rather than asserting on it.
//!
//! The run also checks the face flux against the cell-centred moments on interior faces,
//! where the two are the same quantity by two different routes (BuildHFlux averages
//! intensities then takes the moment; (H_i + H_i+1)/2 takes the moments then averages).
//! They agree there to round-off, and the check is deliberately confined to faces with
//! two interior neighbours: at a boundary face the routes differ, legitimately, and only
//! the per-ray one is defined.
//!
//! Columns of sc_qrad_conserve-errs.dat: the normalised residual |Sum Q dV| / Sum |Q| dV,
//! the two sums, the face-flux agreement, and the sweep count. Thresholds live in the
//! pytest wrapper; the crash-guard here is loose and catches only NaN or blow-up.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"
#include "eos/eos.hpp"
#include "pgen/pgen.hpp"
#include "nr_radiation/nr_radiation.hpp"

namespace {

struct ConsVars {
  Real d0, p0, amp, tol;
};
ConsVars cv;

//----------------------------------------------------------------------------------------
//! \fn void SCQradConserveErrors
//! \brief Reduce the heating over the domain and compare the face flux with the moments.

void SCQradConserveErrors(ParameterInput *pin, Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  nr_radiation::SC *psc = pmbp->pnrrad;
  auto &indcs = pm->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb1 = pmbp->nmb_thispack - 1;
  const bool multi_d = pm->multi_d;
  const bool three_d = pm->three_d;

  auto qrad_h = Kokkos::create_mirror_view(psc->qrad);
  Kokkos::deep_copy(qrad_h, psc->qrad);
  auto &size = pmbp->pmb->mb_size;

  Real sum_q = 0.0, sum_abs = 0.0;
  for (int m=0; m<=nmb1; ++m) {
    Real dv = size.h_view(m).dx1;
    if (multi_d) dv *= size.h_view(m).dx2;
    if (three_d) dv *= size.h_view(m).dx3;
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=is; i<=ie; ++i) {
          Real q = qrad_h(m,k,j,i) * dv;
          sum_q += q;
          sum_abs += std::fabs(q);
        }
      }
    }
  }

  // Two independent views of the x1 face flux. Zero unless a differential form ran,
  // since hflx is then unallocated.
  //   fdiff  vs the mean of the cell-centred MOMENTS, on faces with two interior
  //          neighbours. Those are the same quantity by two routes -- BuildHFlux
  //          averages intensities and then takes the moment, moments does the reverse --
  //          so this checks the route, and it can only run where moments is valid.
  //   gdiff  vs the mean of the INTENSITIES computed here on the host, over every face
  //          including the ones on a block edge. On a strictly periodic mesh every face
  //          is interior, ghosts and all, so every one of them must take the mean. This
  //          is what catches a face being misclassified as a physical boundary: the
  //          boundary rule also conserves and also agrees with the moments away from the
  //          block edge, so nothing else here would notice.
  Real fmax = 0.0, fdiff = 0.0, gdiff = 0.0;
  if (psc->qrad_form != nr_radiation::QradForm::integral) {
    auto mom_h = Kokkos::create_mirror_view(psc->moments);
    Kokkos::deep_copy(mom_h, psc->moments);
    auto f1_h = Kokkos::create_mirror_view(psc->hflx.x1f);
    Kokkos::deep_copy(f1_h, psc->hflx.x1f);
    auto ir_h = Kokkos::create_mirror_view(psc->ir);
    Kokkos::deep_copy(ir_h, psc->ir);
    auto &mu = psc->pang->mu;
    auto &wmu = psc->pang->wmu;
    mu.template sync<HostMemSpace>();
    wmu.template sync<HostMemSpace>();
    const int nang = psc->pang->nang;
    const int nang_tot = psc->nang_tot;
    for (int m=0; m<=nmb1; ++m) {
      for (int k=ks; k<=ke; ++k) {
        for (int j=js; j<=je; ++j) {
          for (int i=is; i<=ie+1; ++i) {
            Real mean_ir = 0.0;
            for (int angg = 0; angg < nang_tot; ++angg) {
              int oct = angg / nang;
              int a = angg - oct*nang;
              mean_ir += wmu.h_view(a) * mu.h_view(oct, a, 0)
                         * 0.5*(ir_h(m,angg,k,j,i-1) + ir_h(m,angg,k,j,i));
            }
            gdiff = std::fmax(gdiff, std::fabs(f1_h(m,0,k,j,i) - mean_ir));
            fmax = std::fmax(fmax, std::fabs(f1_h(m,0,k,j,i)));
            if (i > is && i <= ie) {
              Real mean = 0.5*(mom_h(m,1,k,j,i-1) + mom_h(m,1,k,j,i));
              fdiff = std::fmax(fdiff, std::fabs(f1_h(m,0,k,j,i) - mean));
            }
          }
        }
      }
    }
  }

#if MPI_PARALLEL_ENABLED
  Real sbuf[2] = {sum_q, sum_abs};
  MPI_Allreduce(MPI_IN_PLACE, sbuf, 2, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  sum_q = sbuf[0]; sum_abs = sbuf[1];
  Real mbuf[3] = {fdiff, gdiff, fmax};
  MPI_Allreduce(MPI_IN_PLACE, mbuf, 3, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
  fdiff = mbuf[0]; gdiff = mbuf[1]; fmax = mbuf[2];
#endif

  // normalised against the total magnitude, so the number is scale-free and a run in
  // which Q happens to be small everywhere is not flattered
  const Real resid = (sum_abs > 0.0) ? std::fabs(sum_q)/sum_abs : std::fabs(sum_q);
  const Real face_rel = (fmax > 0.0) ? fdiff/fmax : fdiff;
  const Real mean_rel = (fmax > 0.0) ? gdiff/fmax : gdiff;

  std::cout << "SC qrad conserve (" << psc->qrad_form_name << "): |Sum Q dV| / Sum |Q| dV"
            << " = " << resid << "  (Sum Q dV = " << sum_q << ", Sum |Q| dV = " << sum_abs
            << "; face vs moments " << face_rel << ", vs mean " << mean_rel
            << "; " << psc->niter_last << " sweeps)"
            << std::endl;

  if (global_variable::my_rank == 0) {
    std::string fname = "sc_qrad_conserve-errs.dat";
    FILE *pf = std::fopen(fname.c_str(), "r");
    if (pf != nullptr) {                       // exists -> append
      pf = std::freopen(fname.c_str(), "a", pf);
    } else {                                   // new -> write header
      pf = std::fopen(fname.c_str(), "w");
      std::fprintf(pf, "# Nx1  Nx2  Nx3   Ncycle   residual     SumQdV       "
                       "Sum|Q|dV     face-vs-mom  face-vs-mean niter\n");
    }
    std::fprintf(pf, "%04d  %04d  %04d  %05d  %e %e %e %e %e %d\n",
                 pm->mesh_indcs.nx1, pm->mesh_indcs.nx2, pm->mesh_indcs.nx3, pm->ncycle,
                 resid, sum_q, sum_abs, face_rel, mean_rel, psc->niter_last);
    std::fclose(pf);
  }

  // loose crash-guard only; the real thresholds live in the pytest wrapper
  if (!(resid < cv.tol)) {                     // also fails on NaN
    std::cout << "### SC qrad conserve FAILED (crash-guard): residual = " << resid
              << " not < " << cv.tol << std::endl;
    std::exit(EXIT_FAILURE);
  }
}
}  // namespace

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::SCQradConserve()
//! \brief Smooth periodic gas perturbation; the heating sum is checked at the end.

void ProblemGenerator::SCQradConserve(ParameterInput *pin, const bool restart) {
  pgen_final_func = SCQradConserveErrors;
  cv.d0  = pin->GetOrAddReal("problem", "d0", 1.0);
  cv.p0  = pin->GetOrAddReal("problem", "p0", 1.0);
  cv.amp = pin->GetOrAddReal("problem", "amp", 0.3);
  cv.tol = pin->GetOrAddReal("problem", "tol", 1.0e-2);
  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->pnrrad == nullptr || pmbp->phydro == nullptr) {
    std::cout << "### FATAL ERROR: sc_qrad_conserve needs <hydro> and <nr_radiation>"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (!pmy_mesh_->strictly_periodic) {
    std::cout << "### FATAL ERROR: sc_qrad_conserve needs a strictly periodic mesh; the "
              << "test is that there is no boundary flux to account for" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  auto &indcs = pmy_mesh_->mb_indcs;
  const int ng = indcs.ng;
  const int n1 = indcs.nx1 + 2*ng;
  const int n2 = (indcs.nx2 > 1) ? indcs.nx2 + 2*ng : 1;
  const int n3 = (indcs.nx3 > 1) ? indcs.nx3 + 2*ng : 1;
  const int nmb1 = pmbp->nmb_thispack - 1;
  const int is = indcs.is, js = indcs.js, ks = indcs.ks;
  const Real gm1 = pmbp->phydro->peos->eos_data.gamma - 1.0;

  auto u0 = pmbp->phydro->u0;
  auto &size = pmbp->pmb->mb_size;
  auto &msz = pmy_mesh_->mesh_size;
  const Real x1min = msz.x1min, x1len = msz.x1max - msz.x1min;
  const Real x2min = msz.x2min, x2len = msz.x2max - msz.x2min;
  const Real x3min = msz.x3min, x3len = msz.x3max - msz.x3min;
  const bool multi_d = pmy_mesh_->multi_d;
  const bool three_d = pmy_mesh_->three_d;
  const Real d0 = cv.d0, p0 = cv.p0, amp = cv.amp;

  // one wavelength of the box in each active direction; the pressure is perturbed out of
  // phase with the density so that T = p/rho, and so the emission, is not a function of
  // the opacity alone
  par_for("sc_qcons_init", DevExeSpace(), 0, nmb1, 0, (n3-1), 0, (n2-1), 0, (n1-1),
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real x1 = size.d_view(m).x1min + (i - is + 0.5)*size.d_view(m).dx1;
    Real ph = 2.0*M_PI*(x1 - x1min)/x1len;
    if (multi_d) {
      Real x2 = size.d_view(m).x2min + (j - js + 0.5)*size.d_view(m).dx2;
      ph += 2.0*M_PI*(x2 - x2min)/x2len;
    }
    if (three_d) {
      Real x3 = size.d_view(m).x3min + (k - ks + 0.5)*size.d_view(m).dx3;
      ph += 2.0*M_PI*(x3 - x3min)/x3len;
    }
    u0(m,IDN,k,j,i) = d0*(1.0 + amp*sin(ph));
    u0(m,IM1,k,j,i) = 0.0;
    u0(m,IM2,k,j,i) = 0.0;
    u0(m,IM3,k,j,i) = 0.0;
    u0(m,IEN,k,j,i) = p0*(1.0 + amp*cos(ph))/gm1;   // gas at rest: no kinetic part
  });
}
