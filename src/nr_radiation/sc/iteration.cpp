//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file iteration.cpp
//! \brief Task assembly and SolveTransfer: exchange-before-sweep loop with optional
//! Jacobi-ALI (Davis 2012 Eq. 22–25). sc_bvals exchanges ir and srad (source S).

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
#include "utils/perf.hpp"

namespace nr_radiation {

//----------------------------------------------------------------------------------------
//! \fn void SC::AssembleTasks

void SC::AssembleTasks(std::map<std::string, std::shared_ptr<TaskList>> tl) {
  TaskID none(0);
  hydro::Hydro *phyd = pmy_pack->phydro;
  mhd::MHD *pmhd = pmy_pack->pmhd;

  id.sc_solve = tl["before_timeintegrator"]->AddTask(&SC::SolveTransfer,
                                                      this, none);

  if (affect_fluid) {
    if (phyd != nullptr) {
      id.sc_qrad = tl["stagen"]->InsertTask(&SC::AddQrad, this,
                                             phyd->id.rkupdt, phyd->id.srctrms);
      id.sc_newdt = tl["stagen"]->AddTask(&SC::NewTimeStep, this, phyd->id.newdt);
    } else if (pmhd != nullptr) {
      id.sc_qrad = tl["stagen"]->InsertTask(&SC::AddQrad, this,
                                             pmhd->id.rkupdt, pmhd->id.srctrms);
      id.sc_newdt = tl["stagen"]->AddTask(&SC::NewTimeStep, this, pmhd->id.newdt);
    }
  }

  // sc_bvals: dual CC exchange for ir (nvar=nang_tot) and srad (nvar=1)
  // InitRecv both from none → Restrict → Send → Recv → PhysBCs (after both Recvs)
  // → Prolong → Clear*
  auto &vtl = tl["sc_bvals"];
  id.srad_irecv = vtl->AddTask(&SC::InitRecvSrad, this, none);
  id.ir_irecv = vtl->AddTask(&SC::InitRecvIr, this, none);

  id.srad_rest = vtl->AddTask(&SC::RestrictSrad, this, id.srad_irecv);
  id.ir_rest = vtl->AddTask(&SC::RestrictIr, this, id.ir_irecv);

  id.srad_send = vtl->AddTask(&SC::SendSrad, this, id.srad_rest);
  id.ir_send = vtl->AddTask(&SC::SendIr, this, id.ir_rest);

  id.srad_recv = vtl->AddTask(&SC::RecvSrad, this, id.srad_send);
  id.ir_recv = vtl->AddTask(&SC::RecvIr, this, id.ir_send);

  TaskID both_recv = id.srad_recv | id.ir_recv;
  id.ir_bcs = vtl->AddTask(&SC::ApplyPhysicalBCsIr, this, both_recv);
  id.srad_bcs = vtl->AddTask(&SC::ApplyPhysicalBCsSrad, this, id.ir_bcs);

  id.srad_prol = vtl->AddTask(&SC::ProlongateSrad, this, id.srad_bcs);
  id.ir_prol = vtl->AddTask(&SC::ProlongateIr, this, id.srad_bcs);

  TaskID both_prol = id.srad_prol | id.ir_prol;
  id.srad_csend = vtl->AddTask(&SC::ClearSendSrad, this, both_prol);
  id.ir_csend = vtl->AddTask(&SC::ClearSendIr, this, id.srad_csend);
  id.srad_crecv = vtl->AddTask(&SC::ClearRecvSrad, this, id.ir_csend);
  id.ir_crecv = vtl->AddTask(&SC::ClearRecvIr, this, id.srad_crecv);
}

//----------------------------------------------------------------------------------------
//! \fn void SC::UpdateOpacityAndSource
//! \brief Rebuild chi, eps, sigma_s, planck=B; warm-start S in srad when J is available.
//! chi = (opa+ops)*ρ; eps = opa/(opa+ops) or uniform override; B = T^4.
//! If max|J|==0 (first call / post-regrid), leave existing srad (pgen or prolonged).

void SC::UpdateOpacityAndSource() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng) : 1;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  auto chi_ = chi;
  auto srad_ = srad;
  auto planck_ = planck;
  auto eps_ = eps;
  auto sigma_s_ = sigma_s;
  auto jmean_ = jmean;

  Real opa_ = opa;
  Real ops_ = ops;
  Real eps_u = eps_uniform;
  bool use_eps_u = use_eps_uniform;
  bool ali = use_ali;

  // Warm-start only once J has been established (skip when jmean is still all-zero)
  bool do_warm = false;
  if (ali) {
    Real jmax = 0.0;
    int nmkji = (nmb1+1)*n3*n2*n1;
    Kokkos::parallel_reduce("sc_jmax", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
    KOKKOS_LAMBDA(const int idx, Real &lmax) {
      int m = idx / (n3*n2*n1);
      int kji = idx - m*(n3*n2*n1);
      int k = kji / (n2*n1);
      int ji = kji - k*(n2*n1);
      int j = ji / n1;
      int i = ji - j*n1;
      lmax = fmax(lmax, fabs(jmean_(m,k,j,i)));
    }, Kokkos::Max<Real>(jmax));
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(MPI_IN_PLACE, &jmax, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
#endif
    do_warm = (jmax > 0.0);
  }
  bool do_warm_ = do_warm;

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
    par_for("sc_chi_srad", DevExeSpace(), 0, nmb1, 0, (n3-1), 0, (n2-1), 0, (n1-1),
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
        if (do_warm_) {
          srad_(m,0,k,j,i) = (1.0 - epsi) * jmean_(m,k,j,i) + epsi * B;
        }
        // else keep prolonged / pgen S
      } else {
        srad_(m,0,k,j,i) = B;
      }
    });
  } else {
    // no-fluid path (beam tests): chi = opa+ops, B=0 unless pgen set fields
    par_for("sc_chi_nofluid", DevExeSpace(), 0, nmb1, 0, (n3-1), 0, (n2-1), 0, (n1-1),
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      chi_(m,k,j,i) = opa_ + ops_;
      sigma_s_(m,k,j,i) = ops_;
      Real epsi = use_eps_u ? eps_u
                            : ((opa_ + ops_ > 0.0) ? opa_ / (opa_ + ops_) : 1.0);
      eps_(m,k,j,i) = epsi;
      if (ali) {
        if (do_warm_) {
          srad_(m,0,k,j,i) = (1.0 - epsi) * jmean_(m,k,j,i) + epsi * planck_(m,k,j,i);
        }
      } else if (planck_(m,k,j,i) == 0.0 && srad_(m,0,k,j,i) != 0.0) {
        // unit tests that only set srad: keep srad, sync planck
        planck_(m,k,j,i) = srad_(m,0,k,j,i);
      } else {
        srad_(m,0,k,j,i) = planck_(m,k,j,i);
      }
    });
  }
}

//----------------------------------------------------------------------------------------
//! \fn void SC::ApplyPhysicalBCs
//! \brief Intensity physical BCs only (RadiationBCs + optional user_bcs).

void SC::ApplyPhysicalBCs() {
  if (!(pmy_pack->pmesh->strictly_periodic)) {
    pbval_ir->RadiationBCs(pmy_pack, i_in, ir);
  }
  if (pmy_pack->pmesh->pgen != nullptr && pmy_pack->pmesh->pgen->user_bcs) {
    (pmy_pack->pmesh->pgen->user_bcs_func)(pmy_pack->pmesh);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void SC::ApplyPhysicalBCsSource
//! \brief Nearest-active copy of S (srad) into physical-boundary ghost layers.

void SC::ApplyPhysicalBCsSource() {
  auto &pm = pmy_pack->pmesh;
  if (pm->strictly_periodic) return;

  auto &indcs = pm->mb_indcs;
  int &ng = indcs.ng;
  auto &mb_bcs = pmy_pack->pmb->mb_bcs;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng) : 1;
  int nmb = pmy_pack->nmb_thispack;
  auto srad_ = srad;

  if (pm->mesh_bcs[BoundaryFace::inner_x1] != BoundaryFlag::periodic) {
    int &is = indcs.is;
    int &ie = indcs.ie;
    par_for("sc_srad_bc_x1", DevExeSpace(), 0, (nmb-1), 0, (n3-1), 0, (n2-1),
    KOKKOS_LAMBDA(int m, int k, int j) {
      auto f_in = mb_bcs.d_view(m, BoundaryFace::inner_x1);
      auto f_ox = mb_bcs.d_view(m, BoundaryFace::outer_x1);
      if (f_in == BoundaryFlag::outflow || f_in == BoundaryFlag::inflow) {
        for (int i = 0; i < ng; ++i) {
          srad_(m,0,k,j,is-i-1) = srad_(m,0,k,j,is);
        }
      }
      if (f_ox == BoundaryFlag::outflow || f_ox == BoundaryFlag::inflow) {
        for (int i = 0; i < ng; ++i) {
          srad_(m,0,k,j,ie+i+1) = srad_(m,0,k,j,ie);
        }
      }
    });
  }
  if (pm->one_d) return;

  if (pm->mesh_bcs[BoundaryFace::inner_x2] != BoundaryFlag::periodic) {
    int &js = indcs.js;
    int &je = indcs.je;
    par_for("sc_srad_bc_x2", DevExeSpace(), 0, (nmb-1), 0, (n3-1), 0, (n1-1),
    KOKKOS_LAMBDA(int m, int k, int i) {
      auto f_in = mb_bcs.d_view(m, BoundaryFace::inner_x2);
      auto f_ox = mb_bcs.d_view(m, BoundaryFace::outer_x2);
      if (f_in == BoundaryFlag::outflow || f_in == BoundaryFlag::inflow) {
        for (int j = 0; j < ng; ++j) {
          srad_(m,0,k,js-j-1,i) = srad_(m,0,k,js,i);
        }
      }
      if (f_ox == BoundaryFlag::outflow || f_ox == BoundaryFlag::inflow) {
        for (int j = 0; j < ng; ++j) {
          srad_(m,0,k,je+j+1,i) = srad_(m,0,k,je,i);
        }
      }
    });
  }
  if (pm->two_d) return;

  if (pm->mesh_bcs[BoundaryFace::inner_x3] == BoundaryFlag::periodic) return;
  int &ks = indcs.ks;
  int &ke = indcs.ke;
  par_for("sc_srad_bc_x3", DevExeSpace(), 0, (nmb-1), 0, (n2-1), 0, (n1-1),
  KOKKOS_LAMBDA(int m, int j, int i) {
    auto f_in = mb_bcs.d_view(m, BoundaryFace::inner_x3);
    auto f_ox = mb_bcs.d_view(m, BoundaryFace::outer_x3);
    if (f_in == BoundaryFlag::outflow || f_in == BoundaryFlag::inflow) {
      for (int k = 0; k < ng; ++k) {
        srad_(m,0,ks-k-1,j,i) = srad_(m,0,ks,j,i);
      }
    }
    if (f_ox == BoundaryFlag::outflow || f_ox == BoundaryFlag::inflow) {
      for (int k = 0; k < ng; ++k) {
        srad_(m,0,ke+k+1,j,i) = srad_(m,0,ke,j,i);
      }
    }
  });
}

//----------------------------------------------------------------------------------------
//! \fn void SC::ComputeJ

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
  auto jmean_ = jmean;

  par_for("sc_computeJ", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
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
//! \fn void SC::UpdateSourceALI
//! \brief Jacobi ALI update (Davis Eq. 24 / Athena-C update_sfunc).
//! ΔS = [(1−ε)J + εB − S] / [1 − (1−ε)Λ*]; returns max|ΔS/S|.

void SC::UpdateSourceALI(Real &max_dS_rel) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  int nx1 = ie - is + 1, nx2 = je - js + 1, nx3 = ke - ks + 1;
  int nmkji = (nmb1+1)*nx3*nx2*nx1;

  auto srad_ = srad;
  auto planck_ = planck;
  auto jmean_ = jmean;
  auto eps_ = eps;
  auto lam_ = lamstr;

  Real dSmax = 0.0;
  Kokkos::parallel_reduce("sc_ali", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
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
    Real S = srad_(m,0,k,j,i);
    Real J = jmean_(m,k,j,i);
    Real B = planck_(m,k,j,i);
    Real lam = lam_(m,k,j,i);
    Real denom = 1.0 - (1.0 - epsi) * lam;
    if (fabs(denom) < 1.0e-14) denom = (denom >= 0.0) ? 1.0e-14 : -1.0e-14;
    Real Snew = (1.0 - epsi) * J + epsi * B;
    Real dS = (Snew - S) / denom;
    srad_(m,0,k,j,i) = S + dS;
    Real r = (fabs(S) > 0.0) ? fabs(dS / S) : fabs(dS);
    lmax = fmax(lmax, r);
  }, Kokkos::Max<Real>(dSmax));

#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &dSmax, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
#endif
  max_dS_rel = dSmax;
}

//----------------------------------------------------------------------------------------
//! \fn void SC::ExchangeBoundariesSync
//! \brief Run the full ir+srad exchange sequence synchronously (polls Recv until done).
//! Matches sc_bvals ordering for unit tests that lack a Driver ExecuteTaskList.

void SC::ExchangeBoundariesSync() {
  (void)InitRecvSrad(nullptr, 0);
  (void)InitRecvIr(nullptr, 0);
  (void)RestrictSrad(nullptr, 0);
  (void)RestrictIr(nullptr, 0);
  (void)SendSrad(nullptr, 0);
  (void)SendIr(nullptr, 0);
  TaskStatus ssrad, sir;
  do {
    ssrad = RecvSrad(nullptr, 0);
  } while (ssrad == TaskStatus::incomplete);
  do {
    sir = RecvIr(nullptr, 0);
  } while (sir == TaskStatus::incomplete);
  (void)ApplyPhysicalBCsIr(nullptr, 0);
  (void)ApplyPhysicalBCsSrad(nullptr, 0);
  (void)ProlongateSrad(nullptr, 0);
  (void)ProlongateIr(nullptr, 0);
  (void)ClearSendSrad(nullptr, 0);
  (void)ClearSendIr(nullptr, 0);
  (void)ClearRecvSrad(nullptr, 0);
  (void)ClearRecvIr(nullptr, 0);
}

//----------------------------------------------------------------------------------------
//! Boundary-exchange wrapper tasks — intensity (ir)

TaskStatus SC::InitRecvIr(Driver *pdrive, int stage) {
  (void)pdrive; (void)stage;
  return pbval_ir->InitRecv(nang_tot);
}

TaskStatus SC::RestrictIr(Driver *pdrive, int stage) {
  (void)pdrive; (void)stage;
  if (pmy_pack->pmesh->multilevel) {
    pmy_pack->pmesh->pmr->RestrictCC(ir, coarse_ir);
  }
  return TaskStatus::complete;
}

TaskStatus SC::SendIr(Driver *pdrive, int stage) {
  (void)pdrive; (void)stage;
  return pbval_ir->PackAndSendCC(ir, coarse_ir);
}

TaskStatus SC::RecvIr(Driver *pdrive, int stage) {
  (void)pdrive; (void)stage;
  return pbval_ir->RecvAndUnpackCC(ir, coarse_ir);
}

TaskStatus SC::ApplyPhysicalBCsIr(Driver *pdrive, int stage) {
  (void)pdrive; (void)stage;
  ApplyPhysicalBCs();
  return TaskStatus::complete;
}

TaskStatus SC::ProlongateIr(Driver *pdrive, int stage) {
  (void)pdrive; (void)stage;
  if (pmy_pack->pmesh->multilevel) {
    pbval_ir->FillCoarseInBndryCC(ir, coarse_ir);
    pbval_ir->ProlongateCC(ir, coarse_ir);
  }
  return TaskStatus::complete;
}

TaskStatus SC::ClearSendIr(Driver *pdrive, int stage) {
  (void)pdrive; (void)stage;
  return pbval_ir->ClearSend();
}

TaskStatus SC::ClearRecvIr(Driver *pdrive, int stage) {
  (void)pdrive; (void)stage;
  return pbval_ir->ClearRecv();
}

//----------------------------------------------------------------------------------------
//! Boundary-exchange wrapper tasks — source S (srad)

TaskStatus SC::InitRecvSrad(Driver *pdrive, int stage) {
  (void)pdrive; (void)stage;
  return pbval_srad->InitRecv(1);
}

TaskStatus SC::RestrictSrad(Driver *pdrive, int stage) {
  (void)pdrive; (void)stage;
  if (pmy_pack->pmesh->multilevel) {
    pmy_pack->pmesh->pmr->RestrictCC(srad, coarse_srad);
  }
  return TaskStatus::complete;
}

TaskStatus SC::SendSrad(Driver *pdrive, int stage) {
  (void)pdrive; (void)stage;
  return pbval_srad->PackAndSendCC(srad, coarse_srad);
}

TaskStatus SC::RecvSrad(Driver *pdrive, int stage) {
  (void)pdrive; (void)stage;
  return pbval_srad->RecvAndUnpackCC(srad, coarse_srad);
}

TaskStatus SC::ApplyPhysicalBCsSrad(Driver *pdrive, int stage) {
  (void)pdrive; (void)stage;
  ApplyPhysicalBCsSource();
  return TaskStatus::complete;
}

TaskStatus SC::ProlongateSrad(Driver *pdrive, int stage) {
  (void)pdrive; (void)stage;
  if (pmy_pack->pmesh->multilevel) {
    pbval_srad->FillCoarseInBndryCC(srad, coarse_srad);
    pbval_srad->ProlongateCC(srad, coarse_srad);
  }
  return TaskStatus::complete;
}

TaskStatus SC::ClearSendSrad(Driver *pdrive, int stage) {
  (void)pdrive; (void)stage;
  return pbval_srad->ClearSend();
}

TaskStatus SC::ClearRecvSrad(Driver *pdrive, int stage) {
  (void)pdrive; (void)stage;
  return pbval_srad->ClearRecv();
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus SC::SolveTransfer
//! \brief Exchange (ir+srad) → FormalSolution → ComputeJ → ALI/LTE residual.

TaskStatus SC::SolveTransfer(Driver *pdrive, int stage) {
  (void)stage;
  UpdateOpacityAndSource();
  Kokkos::deep_copy(DevExeSpace(), jmean, 0.0);

  last_niter = 0;
  last_max_rel = 0.0;
  cnv_flag = false;
  Real max_rel = std::numeric_limits<Real>::max();
  const Real tol = use_ali ? ali_tol : iter_tol;

  for (int it = 0; it < iter_max; ++it) {
    if (!use_ali) {
      Kokkos::deep_copy(DevExeSpace(), jmean_old, jmean);
    }

    // Davis §3.5 / Athena-C order: refresh ghosts before the formal solution
    pdrive->ExecuteTaskList(pmy_pack->pmesh, "sc_bvals", 0);
    FormalSolution();
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
      Kokkos::parallel_reduce("sc_dj", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
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
    last_max_rel = max_rel;
    if (last_niter >= itermin && max_rel <= tol) {
      cnv_flag = true;
      break;
    }
  }
  if (!cnv_flag && last_niter >= iter_max) {
    if (global_variable::my_rank == 0) {
      std::cout << "### WARNING: SC iteration did not converge in " << iter_max
                << " iterations (max residual = " << max_rel
                << ", tol = " << tol
                << (use_ali ? ", ALI |dS/S|" : ", LTE |dJ/J|")
                << ")" << std::endl;
    }
  }

  // cells = active zones PER meshblock (nblocks passed separately -> total = cells*nblocks in
  // analyze); with nang, cum_niter and the sweep kernel time this yields gcaups downstream.
  auto &mbi = pmy_pack->pmesh->mb_indcs;
  int cells_per_mb = mbi.nx1 * mbi.nx2 * mbi.nx3;
  perf::EmitIteration(last_niter, last_max_rel, pmy_pack->pmesh->nmb_total, cells_per_mb, nang_tot);
  CalculateMoments();
  ComputeQrad();
  return TaskStatus::complete;
}

}  // namespace nr_radiation
