#ifndef NR_RADIATION_NR_RADIATION_HPP_
#define NR_RADIATION_NR_RADIATION_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file nr_radiation.hpp
//! \brief definitions for the VET class: a minimal, hyperplane-swept, LTE-only
//! short-characteristics radiation solver, following Davis, Stone & Jiang (2012). See
//! Planning/vet-athenak-plan{,-deepdive,-internal}.md for the full design, and the plan
//! "LTE Short-Characteristics VET v1" for the scope of this specific implementation.
//!
//! Scope (v1): eps=1 hardcoded (LTE, S=B always) => no ALI/scattering iteration; only a
//! block/ghost-lag fixed-point loop over MeshBlocks is needed (Davis 2012 Sec. 3.5
//! subdomain iteration, specialized to the no-scattering case). Neither SMR nor AMR is
//! supported: the constructor aborts if pmesh->multilevel is true. Reserved arrays
//! sigma_s, eps, lamstr (ALI/scattering) and coarse_ir (AMR scaffold) are allocated but
//! unused in v1; inflow intensities live in VET-owned DualArray2D i_in.
//!
//! Opacity/coupling convention (apb_rad / Athena++):
//!   chi = opa * rho        mass absorption coefficient times density
//!   S   = T^4              source function (LTE Planck in radiation units)
//!   Q   = chi * crat * prat * (J - S)   gas-radiation energy exchange

#include <map>
#include <memory>
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"
#include "bvals/bvals.hpp"
#include "nr_radiation/angular_grid.hpp"

// forward declarations
class Driver;

namespace nr_radiation {

//----------------------------------------------------------------------------------------
//! \struct VETTaskIDs
//! \brief container to hold TaskIDs of all nr_radiation tasks

struct VETTaskIDs {
  TaskID vet_solve;  // formal solution + boundary-lag loop + moments + Q_rad (Eq. 27)
  TaskID vet_qrad;   // apply Q_rad onto hydro/mhd conserved energy for this stage
  TaskID vet_newdt;  // radiation-relaxation timestep (stagen, after hydro/mhd newdt)
  // boundary-exchange tasks for the "vet_bvals" task list (driven by ExecuteTaskList)
  TaskID ir_irecv, ir_send, ir_recv, ir_bcs, ir_csend, ir_crecv;
};

//----------------------------------------------------------------------------------------
//! \class VET

class VET {
 public:
  VET(MeshBlockPack *ppack, ParameterInput *pin);
  ~VET();

  // angular quadrature (Bruls et al. 1999 type-A grid)
  VETAngularGrid *pang = nullptr;
  int nang_tot;  // = noct*nang, total number of discrete rays

  // opacity/coupling parameters (apb_rad convention)
  Real opa;    // mass absorption coefficient: chi = opa * rho
  Real prat;   // radiation-to-gas pressure ratio P_rad/P_gas
  Real crat;   // speed-of-light to sound-speed ratio c/a_gas

  // frequency scaffold (gray: nfreq=1, wfreq={1.0})
  int nfreq;
  DualArray1D<Real> wfreq;

  // flags controlling coupling to (M)HD
  bool is_hydro_enabled;
  bool is_mhd_enabled;
  bool affect_fluid;   // apply Q_rad back onto the fluid energy equation

  // sweep parallelization strategy: "wavefront" (default), "diagonal", or "jacobi"
  std::string sweep_method;

  // iteration control for the boundary-lag fixed-point loop (Step 4)
  int iter_max;
  int itermin;       // minimum iterations before allowing early exit (default 2)
  Real iter_tol;
  int last_niter;    // diagnostic: number of iterations used in the most recent solve
  bool cnv_flag;     // true if last SolveTransfer converged within iter_tol

  // intensity array: (nmb, nang_tot, nx3, nx2, nx1) with ghost zones, persistent
  DvceArray5D<Real> ir;
  DvceArray5D<Real> coarse_ir;  // AMR scaffold (allocated; multilevel still FATAL in v1)

  // per-zone radiation quantities, rebuilt every solve (nmb,nx3,nx2,nx1)
  DvceArray4D<Real> chi;     // absorption coefficient chi = opa*rho
  DvceArray4D<Real> bb;      // source function S = T^4 (LTE)
  DvceArray4D<Real> jmean;   // mean intensity J (Eq. 17)
  DvceArray4D<Real> jmean_old;  // previous boundary-lag iteration's J (residual scratch)
  DvceArray4D<Real> qrad;    // radiative heating/cooling source term (Eq. 27)

  // RESERVED for future ALI/scattering (v1 is LTE-only, eps=1 hardcoded)
  DvceArray4D<Real> sigma_s;  // scattering opacity (unused; default 0)
  DvceArray4D<Real> eps;      // photon destruction probability (unused; default 1)
  DvceArray4D<Real> lamstr;   // diagonal Lambda* operator (unused)

  // inflow intensity table (nang_tot, 6 faces): default 0 = vacuum edges
  DualArray2D<Real> i_in;

  // radiation moments: n=0:J, 1-3:H_1,H_2,H_3, 4-9:K_11,K_22,K_33,K_12,K_13,K_23
  // (Athena++ convention: diagonal K_ij first, then off-diagonal)
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
  void AssembleTasks(std::map<std::string, std::shared_ptr<TaskList>> tl);
  TaskStatus SolveTransfer(Driver *pdrive, int stage);
  TaskStatus AddQrad(Driver *pdrive, int stage);

  // boundary-exchange wrapper tasks for the "vet_bvals" task list
  TaskStatus InitRecvIr(Driver *pdrive, int stage);
  TaskStatus SendIr(Driver *pdrive, int stage);
  TaskStatus RecvIr(Driver *pdrive, int stage);
  TaskStatus ApplyPhysicalBCsIr(Driver *pdrive, int stage);
  TaskStatus ClearSendIr(Driver *pdrive, int stage);
  TaskStatus ClearRecvIr(Driver *pdrive, int stage);

  // stagen task: radiation-relaxation timestep
  TaskStatus NewTimeStep(Driver *pdrive, int stage);

  // helpers called from SolveTransfer (not registered directly in any task list)
  void UpdateOpacityAndSource();
  void ApplyPhysicalBCs();
  void ComputeJ();
  void CalculateMoments();
  void ComputeQrad();

  // formal solution driver (vet/formal_solution.cpp); computes ir over the whole
  // MeshBlockPack from the current chi/bb fields and existing ghost-zone intensities.
  void FormalSolution();

  // Sweep implementations called by FormalSolution(). Public (not private) because
  // Kokkos CUDA device lambdas cannot be defined inside private/protected members.
  void FormalSolutionWavefront();
  void FormalSolutionDiagonal();
  void FormalSolutionJacobi();

 private:
  MeshBlockPack* pmy_pack;
};

}  // namespace nr_radiation
#endif  // NR_RADIATION_NR_RADIATION_HPP_
