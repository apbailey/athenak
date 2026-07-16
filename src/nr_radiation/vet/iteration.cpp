//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file iteration.cpp
//! \brief Task assembly and SolveTransfer: boundary-lag loop with optional Jacobi-ALI
//! (Davis 2012 Eq. 22–25). LTE (ε=1): converge on max|ΔJ/J|. Scattering: Eq. 24 ΔS.

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mesh/mesh_refinement.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "eos/eos.hpp"
#include "driver/driver.hpp"
#include "nr_radiation/nr_radiation.hpp"

namespace nr_radiation {

//----------------------------------------------------------------------------------------
//! \fn void VET::AssembleTasks

void VET::AssembleTasks(std::map<std::string, std::shared_ptr<TaskList>> tl) {
  TaskID none(0);
  hydro::Hydro *phyd = pmy_pack->phydro;
  mhd::MHD *pmhd = pmy_pack->pmhd;

  id.vet_solve = tl["before_timeintegrator"]->AddTask(&VET::SolveTransfer,
                                                      this, none);

  if (affect_fluid) {
    if (phyd != nullptr) {
      id.vet_qrad = tl["stagen"]->InsertTask(&VET::AddQrad, this,
                                             phyd->id.rkupdt, phyd->id.srctrms);
      id.vet_newdt = tl["stagen"]->AddTask(&VET::NewTimeStep, this, phyd->id.newdt);
    } else if (pmhd != nullptr) {
      id.vet_qrad = tl["stagen"]->InsertTask(&VET::AddQrad, this,
                                             pmhd->id.rkupdt, pmhd->id.srctrms);
      id.vet_newdt = tl["stagen"]->AddTask(&VET::NewTimeStep, this, pmhd->id.newdt);
    }
  }

  // vet_bvals: Restrict → Send → Recv → PhysBCs → Prolongate (GR radiation pattern)
  auto &vtl = tl["vet_bvals"];
  id.ir_irecv = vtl->AddTask(&VET::InitRecvIr, this, none);
  id.ir_rest  = vtl->AddTask(&VET::RestrictIr, this, id.ir_irecv);
  id.ir_send  = vtl->AddTask(&VET::SendIr,     this, id.ir_rest);
  id.ir_recv  = vtl->AddTask(&VET::RecvIr,     this, id.ir_send);
  id.ir_bcs   = vtl->AddTask(&VET::ApplyPhysicalBCsIr, this, id.ir_recv);
  id.ir_prol  = vtl->AddTask(&VET::ProlongateIr, this, id.ir_bcs);
  id.ir_csend = vtl->AddTask(&VET::ClearSendIr, this, id.ir_prol);
  id.ir_crecv = vtl->AddTask(&VET::ClearRecvIr, this, id.ir_csend);
}

//----------------------------------------------------------------------------------------
//! \fn void VET::UpdateOpacityAndSource
//! \brief Rebuild chi, eps, sigma_s, planck=B; warm-start S in bb.
//! chi = (opa+ops)*ρ; eps = opa/(opa+ops) or uniform override; B = T^4.

void VET::UpdateOpacityAndSource() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng) : 1;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  auto chi_ = chi;
  auto bb_ = bb;
  auto planck_ = planck;
  auto eps_ = eps;
  auto sigma_s_ = sigma_s;
  auto jmean_ = jmean;

  Real opa_ = opa;
  Real ops_ = ops;
  Real eps_u = eps_uniform;
  bool use_eps_u = use_eps_uniform;
  bool ali = use_ali;

  DvceArray5D<Real> w0;
  Real gm1 = 0.0;
  bool has_fluid = false;
  if (pmy_pack->phydro != nullptr) {
    w0 = pmy_pack->phydro->w0;
    gm1 = pmy_pack->phydro->peos->eos_data.gamma - 1.0;
    has_fluid = true;
  } else if (pmy_pack->pmhd != nullptr) {
    w0 = pmy_pack->pmhd->w0;
    gm1 = pmy_pack->pmhd->peos->eos_data.gamma - 1.0;
    has_fluid = true;
  }

  if (has_fluid) {
    Real gm1_ = gm1;
    bool affect_ = affect_fluid;
    par_for("vet_chi_bb", DevExeSpace(), 0, nmb1, 0, (n3-1), 0, (n2-1), 0, (n1-1),
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real dens = fmax(w0(m,IDN,k,j,i), 0.0);
      Real chi_tot = (opa_ + ops_) * dens;
      chi_(m,k,j,i) = chi_tot;
      sigma_s_(m,k,j,i) = ops_ * dens;
      Real epsi;
      if (use_eps_u) {
        epsi = eps_u;
      } else if (opa_ + ops_ > 0.0) {
        epsi = opa_ / (opa_ + ops_);
      } else {
        epsi = 1.0;
      }
      eps_(m,k,j,i) = epsi;

      Real B = 0.0;
      if (affect_) {
        Real temp = (dens > 0.0) ? (gm1_ * w0(m,IEN,k,j,i) / dens) : 0.0;
        B = temp * temp * temp * temp;
      }
      planck_(m,k,j,i) = B;
      if (ali) {
        // warm start: S = (1-ε)J + εB
        bb_(m,k,j,i) = (1.0 - epsi) * jmean_(m,k,j,i) + epsi * B;
      } else {
        bb_(m,k,j,i) = B;
      }
    });
  } else {
    // no-fluid path (beam tests): chi = opa+ops, B=0 unless pgen set fields
    par_for("vet_chi_nofluid", DevExeSpace(), 0, nmb1, 0, (n3-1), 0, (n2-1), 0, (n1-1),
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      chi_(m,k,j,i) = opa_ + ops_;
      sigma_s_(m,k,j,i) = ops_;
      Real epsi = use_eps_u ? eps_u
                            : ((opa_ + ops_ > 0.0) ? opa_ / (opa_ + ops_) : 1.0);
      eps_(m,k,j,i) = epsi;
      // leave planck/bb as set by pgen (or zero from ctor)
      if (ali) {
        bb_(m,k,j,i) = (1.0 - epsi) * jmean_(m,k,j,i) + epsi * planck_(m,k,j,i);
      } else if (planck_(m,k,j,i) == 0.0 && bb_(m,k,j,i) != 0.0) {
        // unit tests that only set bb: keep bb, sync planck
        planck_(m,k,j,i) = bb_(m,k,j,i);
      } else {
        bb_(m,k,j,i) = planck_(m,k,j,i);
      }
    });
  }
}

//----------------------------------------------------------------------------------------
//! \fn void VET::ApplyPhysicalBCs

void VET::ApplyPhysicalBCs() {
  if (!(pmy_pack->pmesh->strictly_periodic)) {
    pbval_ir->RadiationBCs(pmy_pack, i_in, ir);
  }
  // pgen may be null when unit tests call this from inside CallProblemGenerator
  if (pmy_pack->pmesh->pgen != nullptr && pmy_pack->pmesh->pgen->user_bcs) {
    (pmy_pack->pmesh->pgen->user_bcs_func)(pmy_pack->pmesh);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void VET::ComputeJ

void VET::ComputeJ() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  int nang = pang->nang;
  int nang_tot_ = nang_tot;
  auto &wmu = pang->wmu;
  auto ir_ = ir;
  auto jmean_ = jmean;

  par_for("vet_computeJ", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real J = 0.0;
    for (int angg = 0; angg < nang_tot_; ++angg) {
      int a = angg % nang;
      J += wmu.d_view(a) * ir_(m, angg, k, j, i);
    }
    jmean_(m, k, j, i) = J;
  });
}

//----------------------------------------------------------------------------------------
//! \fn void VET::UpdateSourceALI
//! \brief Jacobi ALI update (Davis Eq. 24 / Athena-C update_sfunc).
//! ΔS = [(1−ε)J + εB − S] / [1 − (1−ε)Λ*]; returns max|ΔS/S|.

void VET::UpdateSourceALI(Real &max_dS_rel) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  int nx1 = ie - is + 1, nx2 = je - js + 1, nx3 = ke - ks + 1;
  int nmkji = (nmb1+1)*nx3*nx2*nx1;

  auto bb_ = bb;
  auto planck_ = planck;
  auto jmean_ = jmean;
  auto eps_ = eps;
  auto lam_ = lamstr;

  Real dSmax = 0.0;
  Kokkos::parallel_reduce("vet_ali", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int idx, Real &lmax) {
    int m = idx / (nx3*nx2*nx1);
    int kji = idx - m*(nx3*nx2*nx1);
    int k = kji / (nx2*nx1);
    int ji = kji - k*(nx2*nx1);
    int j = ji / nx1;
    int i = ji - j*nx1 + is;
    j += js;
    k += ks;

    Real epsi = eps_(m,k,j,i);
    Real S = bb_(m,k,j,i);
    Real J = jmean_(m,k,j,i);
    Real B = planck_(m,k,j,i);
    Real lam = lam_(m,k,j,i);
    Real denom = 1.0 - (1.0 - epsi) * lam;
    if (fabs(denom) < 1.0e-14) denom = (denom >= 0.0) ? 1.0e-14 : -1.0e-14;
    Real Snew = (1.0 - epsi) * J + epsi * B;
    Real dS = (Snew - S) / denom;
    bb_(m,k,j,i) = S + dS;
    Real r = (fabs(S) > 0.0) ? fabs(dS / S) : fabs(dS);
    lmax = fmax(lmax, r);
  }, Kokkos::Max<Real>(dSmax));

#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &dSmax, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
#endif
  max_dS_rel = dSmax;
}

//----------------------------------------------------------------------------------------
//! Boundary-exchange wrapper tasks

TaskStatus VET::InitRecvIr(Driver *pdrive, int stage) {
  (void)pdrive; (void)stage;
  return pbval_ir->InitRecv(nang_tot);
}

TaskStatus VET::RestrictIr(Driver *pdrive, int stage) {
  (void)pdrive; (void)stage;
  if (pmy_pack->pmesh->multilevel) {
    pmy_pack->pmesh->pmr->RestrictCC(ir, coarse_ir);
  }
  return TaskStatus::complete;
}

TaskStatus VET::SendIr(Driver *pdrive, int stage) {
  (void)pdrive; (void)stage;
  return pbval_ir->PackAndSendCC(ir, coarse_ir);
}

TaskStatus VET::RecvIr(Driver *pdrive, int stage) {
  (void)pdrive; (void)stage;
  return pbval_ir->RecvAndUnpackCC(ir, coarse_ir);
}

TaskStatus VET::ApplyPhysicalBCsIr(Driver *pdrive, int stage) {
  (void)pdrive; (void)stage;
  ApplyPhysicalBCs();
  return TaskStatus::complete;
}

TaskStatus VET::ProlongateIr(Driver *pdrive, int stage) {
  (void)pdrive; (void)stage;
  if (pmy_pack->pmesh->multilevel) {
    pbval_ir->FillCoarseInBndryCC(ir, coarse_ir);
    pbval_ir->ProlongateCC(ir, coarse_ir);
  }
  return TaskStatus::complete;
}

TaskStatus VET::ClearSendIr(Driver *pdrive, int stage) {
  (void)pdrive; (void)stage;
  return pbval_ir->ClearSend();
}

TaskStatus VET::ClearRecvIr(Driver *pdrive, int stage) {
  (void)pdrive; (void)stage;
  return pbval_ir->ClearRecv();
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus VET::SolveTransfer

TaskStatus VET::SolveTransfer(Driver *pdrive, int stage) {
  (void)stage;
  UpdateOpacityAndSource();
  ApplyPhysicalBCs();
  Kokkos::deep_copy(DevExeSpace(), jmean, 0.0);

  last_niter = 0;
  cnv_flag = false;
  Real max_rel = std::numeric_limits<Real>::max();
  const Real tol = use_ali ? ali_tol : iter_tol;

  for (int it = 0; it < iter_max; ++it) {
    if (!use_ali) {
      Kokkos::deep_copy(DevExeSpace(), jmean_old, jmean);
    }

    FormalSolution();
    pdrive->ExecuteTaskList(pmy_pack->pmesh, "vet_bvals", 0);
    ComputeJ();

    if (use_ali) {
      UpdateSourceALI(max_rel);
    } else {
      auto &indcs = pmy_pack->pmesh->mb_indcs;
      int is = indcs.is, ie = indcs.ie;
      int js = indcs.js, je = indcs.je;
      int ks = indcs.ks, ke = indcs.ke;
      int nmb1 = pmy_pack->nmb_thispack - 1;
      int nx1 = ie - is + 1, nx2 = je - js + 1, nx3 = ke - ks + 1;
      int nmkji = (nmb1+1)*nx3*nx2*nx1;
      auto jmean_ = jmean;
      auto jold_ = jmean_old;
      Real dJmax = 0.0;
      Real dJabs_max = 0.0;
      Kokkos::parallel_reduce("vet_dj", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
      KOKKOS_LAMBDA(const int idx, Real &lmax, Real &labs) {
        int m = idx / (nx3*nx2*nx1);
        int kji = idx - m*(nx3*nx2*nx1);
        int k = kji / (nx2*nx1);
        int ji = kji - k*(nx2*nx1);
        int j = ji / nx1;
        int i = ji - j*nx1 + is;
        j += js;
        k += ks;
        Real Jo = jold_(m,k,j,i);
        Real Jn = jmean_(m,k,j,i);
        Real dJ = fabs(Jn - Jo);
        labs = fmax(labs, dJ);
        Real r = (Jo > 0.0) ? (dJ / Jo) : 0.0;
        lmax = fmax(lmax, r);
      }, Kokkos::Max<Real>(dJmax), Kokkos::Max<Real>(dJabs_max));

#if MPI_PARALLEL_ENABLED
      Real buf[2] = {dJmax, dJabs_max};
      MPI_Allreduce(MPI_IN_PLACE, buf, 2, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
      dJmax = buf[0];
      dJabs_max = buf[1];
#endif
      if (dJmax == 0.0 && dJabs_max > 0.0) dJmax = 1.0;
      max_rel = dJmax;
    }

    last_niter = it + 1;
    if (last_niter >= itermin && max_rel <= tol) {
      cnv_flag = true;
      break;
    }
  }
  if (!cnv_flag && last_niter >= iter_max) {
    if (global_variable::my_rank == 0) {
      std::cout << "### WARNING: VET iteration did not converge in " << iter_max
                << " iterations (max residual = " << max_rel
                << ", tol = " << tol
                << (use_ali ? ", ALI |dS/S|" : ", LTE |dJ/J|")
                << ")" << std::endl;
    }
  }

  CalculateMoments();
  ComputeQrad();
  return TaskStatus::complete;
}

}  // namespace nr_radiation
