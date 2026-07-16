//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file iteration.cpp
//! \brief Task assembly and SolveTransfer: LTE boundary-lag fixed-point loop
//! (Davis, Stone & Jiang 2012 Sec. 3.5 subdomain iteration with eps=1 => S=B fixed,
//! no ALI). Convergence criterion is max|ΔJ/J| (same as Athena-C lte=1 mode).

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "eos/eos.hpp"
#include "driver/driver.hpp"
#include "nr_radiation/nr_radiation.hpp"

namespace nr_radiation {

//----------------------------------------------------------------------------------------
//! \fn void VET::AssembleTasks
//! \brief Wire SolveTransfer / AddQrad without taking over hydro/mhd assembly (Rule 4).

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

  // "vet_bvals" task list: async boundary exchange for intensity array, driven
  // by ExecuteTaskList inside SolveTransfer's iteration loop.
  auto &vtl = tl["vet_bvals"];
  id.ir_irecv = vtl->AddTask(&VET::InitRecvIr, this, none);
  id.ir_send  = vtl->AddTask(&VET::SendIr,     this, id.ir_irecv);
  id.ir_recv  = vtl->AddTask(&VET::RecvIr,     this, id.ir_send);
  id.ir_bcs   = vtl->AddTask(&VET::ApplyPhysicalBCsIr, this, id.ir_recv);
  id.ir_csend = vtl->AddTask(&VET::ClearSendIr, this, id.ir_bcs);
  id.ir_crecv = vtl->AddTask(&VET::ClearRecvIr, this, id.ir_csend);
}

//----------------------------------------------------------------------------------------
//! \fn void VET::UpdateOpacityAndSource
//! \brief Rebuild chi and bb (=S, LTE) over the full MeshBlockPack including ghosts.
//! apb_rad convention: chi = opa * rho, S = T^4 where T = (gamma-1)*e/rho.
//! When no fluid exists (beam test with affect_fluid=false), chi = opa uniformly.

void VET::UpdateOpacityAndSource() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng) : 1;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  auto chi_ = chi;
  auto bb_ = bb;

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
    Real opa_ = opa;
    Real gm1_ = gm1;
    bool affect_ = affect_fluid;
    par_for("vet_chi_bb", DevExeSpace(), 0, nmb1, 0, (n3-1), 0, (n2-1), 0, (n1-1),
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real dens = w0(m,IDN,k,j,i);
      chi_(m,k,j,i) = opa_ * fmax(dens, 0.0);
      if (affect_) {
        Real temp = (dens > 0.0) ? (gm1_ * w0(m,IEN,k,j,i) / dens) : 0.0;
        Real T4 = temp * temp * temp * temp;
        bb_(m,k,j,i) = T4;
      } else {
        bb_(m,k,j,i) = 0.0;
      }
    });
  } else {
    Real opa_ = opa;
    par_for("vet_chi_nofluid", DevExeSpace(), 0, nmb1, 0, (n3-1), 0, (n2-1), 0, (n1-1),
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      chi_(m,k,j,i) = opa_;
      bb_(m,k,j,i) = 0.0;
    });
  }
}

//----------------------------------------------------------------------------------------
//! \fn void VET::ApplyPhysicalBCs

void VET::ApplyPhysicalBCs() {
  if (!(pmy_pack->pmesh->strictly_periodic)) {
    pbval_ir->RadiationBCs(pmy_pack, i_in, ir);
  }
  if (pmy_pack->pmesh->pgen->user_bcs) {
    (pmy_pack->pmesh->pgen->user_bcs_func)(pmy_pack->pmesh);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void VET::ComputeJ
//! \brief Single-kernel J = sum_n w_n I_n (Davis 2012 Eq. 17) into jmean.
//! Inner serial loop over all angles avoids launching nang_tot separate kernels.

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
//! \fn TaskStatus VET::InitRecvIr
//! \brief Wrapper task: post non-blocking receives for intensity boundary exchange.

TaskStatus VET::InitRecvIr(Driver *pdrive, int stage) {
  (void)pdrive; (void)stage;
  return pbval_ir->InitRecv(nang_tot);
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus VET::SendIr
//! \brief Wrapper task: pack and send intensity ghost-zone data.

TaskStatus VET::SendIr(Driver *pdrive, int stage) {
  (void)pdrive; (void)stage;
  return pbval_ir->PackAndSendCC(ir, coarse_ir);
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus VET::RecvIr
//! \brief Wrapper task: receive and unpack intensity ghost-zone data.
//! Returns TaskStatus::incomplete until all MPI receives complete.

TaskStatus VET::RecvIr(Driver *pdrive, int stage) {
  (void)pdrive; (void)stage;
  return pbval_ir->RecvAndUnpackCC(ir, coarse_ir);
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus VET::ApplyPhysicalBCsIr
//! \brief Wrapper task: apply physical and user BCs to intensity array.

TaskStatus VET::ApplyPhysicalBCsIr(Driver *pdrive, int stage) {
  (void)pdrive; (void)stage;
  ApplyPhysicalBCs();
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus VET::ClearSendIr
//! \brief Wrapper task: confirm all intensity MPI sends have completed.

TaskStatus VET::ClearSendIr(Driver *pdrive, int stage) {
  (void)pdrive; (void)stage;
  return pbval_ir->ClearSend();
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus VET::ClearRecvIr
//! \brief Wrapper task: confirm all intensity MPI receives have been cleared.

TaskStatus VET::ClearRecvIr(Driver *pdrive, int stage) {
  (void)pdrive; (void)stage;
  return pbval_ir->ClearRecv();
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus VET::SolveTransfer
//! \brief Opacity/B update, then boundary-lag loop until max|ΔJ/J| <= tol (or iter_max),
//! then moments + Q_rad (Eq. 27).

TaskStatus VET::SolveTransfer(Driver *pdrive, int stage) {
  (void)stage;
  UpdateOpacityAndSource();

  ApplyPhysicalBCs();
  Kokkos::deep_copy(DevExeSpace(), jmean, 0.0);

  last_niter = 0;
  cnv_flag = false;
  Real max_rel = std::numeric_limits<Real>::max();
  for (int it = 0; it < iter_max; ++it) {
    Kokkos::deep_copy(DevExeSpace(), jmean_old, jmean);

    FormalSolution();
    pdrive->ExecuteTaskList(pmy_pack->pmesh, "vet_bvals", 0);

    ComputeJ();

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
    last_niter = it + 1;
    // Require itermin iterations before early exit (apb_rad pattern)
    if (last_niter >= itermin && max_rel <= iter_tol) {
      cnv_flag = true;
      break;
    }
  }
  if (!cnv_flag && last_niter >= iter_max) {
    if (global_variable::my_rank == 0) {
      std::cout << "### WARNING: VET iteration did not converge in " << iter_max
                << " iterations (max|dJ/J| = " << max_rel
                << ", tol = " << iter_tol << ")" << std::endl;
    }
  }

  CalculateMoments();
  ComputeQrad();
  return TaskStatus::complete;
}

}  // namespace nr_radiation
