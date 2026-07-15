#ifndef RADIATION_VET_RADIATION_VET_HPP_
#define RADIATION_VET_RADIATION_VET_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_vet.hpp
//! \brief definitions for the RadiationVET class: a minimal, hyperplane-swept, LTE-only
//! short-characteristics radiation solver, following Davis, Stone & Jiang (2012). See
//! Planning/vet-athenak-plan{,-deepdive,-internal}.md for the full design, and the plan
//! "LTE Short-Characteristics VET v1" for the scope of this specific implementation.
//!
//! Scope (v1): eps=1 hardcoded (LTE, S=B always) => no ALI/scattering iteration; only a
//! block/ghost-lag fixed-point loop over MeshBlocks is needed (Davis 2012 Sec. 3.5
//! subdomain iteration, specialized to the no-scattering case). Neither SMR nor AMR is
//! supported: the constructor aborts if pmesh->multilevel is true.

#include <map>
#include <memory>
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"
#include "bvals/bvals.hpp"
#include "vet_quadrature.hpp"
#include "vet_opacity.hpp"

// forward declarations
class Driver;

namespace radiation_vet {

//----------------------------------------------------------------------------------------
//! \struct VETTaskIDs
//! \brief container to hold TaskIDs of all radiation_vet tasks

struct VETTaskIDs {
  TaskID vet_solve;  // formal solution + boundary-lag loop + moments + Q_rad (Eq. 27)
  TaskID vet_qrad;   // apply Q_rad onto hydro/mhd conserved energy for this stage
};

//----------------------------------------------------------------------------------------
//! \class RadiationVET

class RadiationVET {
 public:
  RadiationVET(MeshBlockPack *ppack, ParameterInput *pin);
  ~RadiationVET();

  // angular quadrature (Bruls et al. 1999 type-A grid)
  VETAngularGrid *pang = nullptr;
  int nang_tot;  // = noct*nang, total number of discrete rays

  // opacity/emission specification (shared by sweep and Q_rad)
  VETOpacity opac;

  // flags controlling coupling to (M)HD
  bool is_hydro_enabled;
  bool is_mhd_enabled;
  bool affect_fluid;   // apply Q_rad back onto the fluid energy equation

  // iteration control for the boundary-lag fixed-point loop (Step 4)
  int iter_max;
  Real iter_tol;
  int last_niter;    // diagnostic: number of iterations used in the most recent solve

  // intensity array: (nmb, nang_tot, nx3, nx2, nx1) with ghost zones, persistent
  DvceArray5D<Real> ir;
  DvceArray5D<Real> coarse_ir;  // never allocated in v1 (multilevel disallowed)

  // per-zone radiation quantities, rebuilt every solve (nmb,nx3,nx2,nx1)
  DvceArray4D<Real> chi;     // total opacity chi^tot
  DvceArray4D<Real> bb;      // Planck function B(T) == source function S (LTE: eps=1)
  DvceArray4D<Real> jmean;   // mean intensity J (Eq. 17)
  DvceArray4D<Real> jmean_old;  // previous boundary-lag iteration's J (residual scratch)
  DvceArray4D<Real> qrad;    // radiative heating/cooling source term (Eq. 27)

  // radiation moments: n=0:J, 1-3:H_1,H_2,H_3, 4-9:K_11,K_12,K_13,K_22,K_23,K_33
  DvceArray5D<Real> moments;

  // boundary communication buffers/functions for ir
  MeshBoundaryValuesCC *pbval_ir = nullptr;

  Real dtnew;

  VETTaskIDs id;

  // functions...
  // SolveTransfer lives on `before_timeintegrator` (once per cycle): opacity update +
  // boundary-lag fixed-point (InitRecv/FormalSolution/PackAndSendCC/RecvAndUnpackCC/
  // ApplyPhysicalBCs/ClearSend/ClearRecv) + CalculateMoments + ComputeQrad + UpdateTimeStep.
  // AddQrad is InsertTask'd into `stagen` after rkupdt when affect_fluid.
  void AssembleVETTasks(std::map<std::string, std::shared_ptr<TaskList>> tl);
  TaskStatus SolveTransfer(Driver *pdrive, int stage);
  TaskStatus AddQrad(Driver *pdrive, int stage);

  // helpers called from SolveTransfer (not registered directly in any task list)
  void UpdateOpacityAndSource();
  void ApplyPhysicalBCs();
  void ComputeJ();
  void CalculateMoments();
  void ComputeQrad();
  // Athena-C radiation/radtrans_dt.c: radiation-relaxation CFL (needed for stable
  // operator-split heating/cooling; sets dtnew and clamps pmesh->dt for this cycle)
  void UpdateTimeStep();

  // formal solution driver (vet_sweep.cpp); computes ir over the whole MeshBlockPack
  // from the current chi/bb fields and existing ghost-zone intensities.
  void FormalSolution();

 private:
  MeshBlockPack* pmy_pack;
};

}  // namespace radiation_vet
#endif  // RADIATION_VET_RADIATION_VET_HPP_
