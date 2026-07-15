//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file vet_iteration.cpp
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
#include "radiation_vet.hpp"

namespace radiation_vet {

//----------------------------------------------------------------------------------------
//! \fn void RadiationVET::AssembleVETTasks
//! \brief Wire SolveTransfer / AddQrad without taking over hydro/mhd assembly (Rule 4).
//!
//! Davis 2012 §2 end-of-section sequence and plan D3: formal solution once per cycle from
//! the previous step's primitives, then apply Q_rad while advancing the fluid. Putting
//! SolveTransfer on `before_timeintegrator` (not appended to `stagen`) is essential —
//! `AddTask` to `stagen` lands after ConToPrim, so a stagen-only AddQrad would apply the
//! *previous* cycle's Q after the hydro update (explicit lag that blows up for stiff LTE
//! coupling). Match turb_driver's InsertTask pattern for the per-stage source.

void RadiationVET::AssembleVETTasks(std::map<std::string, std::shared_ptr<TaskList>> tl) {
  TaskID none(0);
  hydro::Hydro *phyd = pmy_pack->phydro;
  mhd::MHD *pmhd = pmy_pack->pmhd;

  // Once per hydro cycle, before RK stages (operator-split; Davis 2012 §2; plan D3)
  id.vet_solve = tl["before_timeintegrator"]->AddTask(&RadiationVET::SolveTransfer,
                                                      this, none);

  // Apply the same Q_rad each RK stage with beta_dt (turb_driver AddForcing pattern).
  // InsertTask after rkupdt, before srctrms; rewires srctrms's dependency cleanly.
  if (affect_fluid) {
    if (phyd != nullptr) {
      id.vet_qrad = tl["stagen"]->InsertTask(&RadiationVET::AddQrad, this,
                                             phyd->id.rkupdt, phyd->id.srctrms);
    } else if (pmhd != nullptr) {
      id.vet_qrad = tl["stagen"]->InsertTask(&RadiationVET::AddQrad, this,
                                             pmhd->id.rkupdt, pmhd->id.srctrms);
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn void RadiationVET::UpdateOpacityAndSource
//! \brief Rebuild chi and bb (=S, LTE) over the full MeshBlockPack including ghosts.
//! Constant chi fills everywhere; greybody B uses T=p/d (R_ideal=1) from primitives.

void RadiationVET::UpdateOpacityAndSource() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng) : 1;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  VETOpacity opac_ = opac;
  auto chi_ = chi;
  auto bb_ = bb;

  // constant opacity everywhere
  Real chi0 = opac_.Chi();
  par_for("vet_chi", DevExeSpace(), 0, nmb1, 0, (n3-1), 0, (n2-1), 0, (n1-1),
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    chi_(m,k,j,i) = chi0;
  });

  if (opac_.bb_type == VETBBType::zero) {
    par_for("vet_bb_zero", DevExeSpace(), 0, nmb1, 0, (n3-1), 0, (n2-1), 0, (n1-1),
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      bb_(m,k,j,i) = 0.0;
    });
    return;
  }

  // greybody: B = bb_norm * T^4 with T = p/(ρ R), R_ideal=1.
  // AthenaK hydro/mhd primitives store INTERNAL ENERGY DENSITY e at IEN(=IPR), not
  // pressure (see ideal_c2p_hyd.hpp: Primitive = (d,vx,vy,vz,e)). Therefore
  //   p = (γ−1) e ,   T = p/ρ = (γ−1) e/ρ .
  // Fill over the ENTIRE array including ghosts using primitive ghosts (already set by
  // hydro BCs / neighbor exchange from the previous stage).
  DvceArray5D<Real> w0;
  Real gm1;
  if (pmy_pack->phydro != nullptr) {
    w0 = pmy_pack->phydro->w0;
    gm1 = pmy_pack->phydro->peos->eos_data.gamma - 1.0;
  } else if (pmy_pack->pmhd != nullptr) {
    w0 = pmy_pack->pmhd->w0;
    gm1 = pmy_pack->pmhd->peos->eos_data.gamma - 1.0;
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
      << std::endl << "greybody bb_type requires <hydro> or <mhd>" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  par_for("vet_bb_all", DevExeSpace(), 0, nmb1, 0, (n3-1), 0, (n2-1), 0, (n1-1),
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real dens = w0(m,IDN,k,j,i);
    // w0(IEN) = internal energy density e; T = (γ−1) e / ρ
    Real temp = (dens > 0.0) ? (gm1 * w0(m,IEN,k,j,i) / dens) : 0.0;
    bb_(m,k,j,i) = opac_.BlackBody(temp);
  });
}

//----------------------------------------------------------------------------------------
//! \fn void RadiationVET::ApplyPhysicalBCs
//! \brief Physical (+ enrolled user) BCs on ir. Reuse MeshBoundaryValues::RadiationBCs
//! for outflow/uniform-inflow; problem generators enroll user_bcs_func for localized beams.

void RadiationVET::ApplyPhysicalBCs() {
  if (!(pmy_pack->pmesh->strictly_periodic)) {
    pbval_ir->RadiationBCs(pmy_pack, pbval_ir->i_in, ir);
  }
  if (pmy_pack->pmesh->pgen->user_bcs) {
    (pmy_pack->pmesh->pgen->user_bcs_func)(pmy_pack->pmesh);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void RadiationVET::ComputeJ
//! \brief Flat quadrature sum J = sum_n w_n I_n (Davis 2012 Eq. 17) into jmean.

void RadiationVET::ComputeJ() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  int nang = pang->nang;
  int noct = pang->noct;
  int nangt1 = nang_tot - 1;
  auto &wmu = pang->wmu;
  auto ir_ = ir;
  auto jmean_ = jmean;

  // zero, then accumulate (two kernels: simpler than atomics)
  par_for("vet_jzero", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    jmean_(m,k,j,i) = 0.0;
  });

  // host-side accumulate over angles would be slow; one kernel per angle batch
  for (int angg = 0; angg <= nangt1; ++angg) {
    int a = angg % nang;
    Real w = wmu.h_view(a);  // host view OK: DualArray already synced at construction
    par_for("vet_jacc", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      jmean_(m,k,j,i) += w * ir_(m,angg,k,j,i);
    });
    (void)noct;
  }
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus RadiationVET::SolveTransfer
//! \brief Opacity/B update, then boundary-lag loop until max|ΔJ/J| <= tol (or iter_max),
//! then moments + Q_rad (Eq. 27). Ghost exchange is private (InitRecv/Pack/Recv/Clear
//! spun to completion each iteration -- MPI nonblocking Recv may return incomplete).

TaskStatus RadiationVET::SolveTransfer(Driver *pdrive, int stage) {
  (void)pdrive; (void)stage;
  UpdateOpacityAndSource();

  // Clamp Mesh::dt now (before RK stages) using Athena-C radtrans_dt; also set dtnew so
  // Mesh::NewTimeStep at the end of the last stage carries the limit into the next cycle.
  if (affect_fluid) {
    UpdateTimeStep();
  }

  // apply physical BCs once before the first sweep so upwind ghosts are valid
  ApplyPhysicalBCs();
  Kokkos::deep_copy(DevExeSpace(), jmean, 0.0);

  last_niter = 0;
  Real max_rel = std::numeric_limits<Real>::max();
  for (int it = 0; it < iter_max; ++it) {
    // save previous J for residual
    Kokkos::deep_copy(DevExeSpace(), jmean_old, jmean);

    pbval_ir->InitRecv(nang_tot);
    FormalSolution();

    pbval_ir->PackAndSendCC(ir, coarse_ir);
    {
      TaskStatus tstat;
      do {
        tstat = pbval_ir->RecvAndUnpackCC(ir, coarse_ir);
      } while (tstat == TaskStatus::incomplete);
    }
    ApplyPhysicalBCs();
    pbval_ir->ClearSend();
    pbval_ir->ClearRecv();

    ComputeJ();

    // max |ΔJ/J| (Athena-C lte mode); if Jold==0 and |ΔJ|>0 treat as unconverged
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
    if (dJmax == 0.0 && dJabs_max > 0.0) dJmax = 1.0;  // Athena-C convention
    max_rel = dJmax;
    last_niter = it + 1;
    if (max_rel <= iter_tol) break;
  }

  CalculateMoments();
  ComputeQrad();
  return TaskStatus::complete;
}

}  // namespace radiation_vet
