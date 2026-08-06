#ifndef NR_RADIATION_NR_RADIATION_HPP_
#define NR_RADIATION_NR_RADIATION_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file nr_radiation.hpp
//! \brief definitions for the SC class: short-characteristics radiation solver following
//! Davis, Stone & Jiang (2012). Supports LTE (ε=1) boundary-lag iteration and Jacobi-ALI
//! scattering (Eq. 22–25). SMR/AMR via CC Restrict/Prolong of ir and srad (source S).
//!
//! Opacity/coupling:
//!   chi = (opa + ops) * rho     total extinction
//!   eps = opa / (opa + ops)     (or uniform override)
//!   B   = T^4                   (planck); S iterate in srad
//!   Q   = eps * chi * crat * prat * (J - B)

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
//! \struct SCTaskIDs
//! \brief container to hold TaskIDs of all nr_radiation tasks

struct SCTaskIDs {
  TaskID sc_solve;  // formal solution + iteration + moments + Q_rad
  TaskID sc_qrad;   // apply Q_rad onto hydro/mhd conserved energy for this stage
  TaskID sc_newdt;  // radiation-relaxation timestep (stagen, after hydro/mhd newdt)
  // boundary-exchange tasks for the "sc_bvals" task list (driven by ExecuteTaskList)
  TaskID ir_irecv, ir_rest, ir_send, ir_recv, ir_bcs, ir_prol, ir_csend, ir_crecv;
  TaskID srad_irecv, srad_rest, srad_send, srad_recv, srad_bcs, srad_prol, srad_csend, srad_crecv;
};

//----------------------------------------------------------------------------------------
//! \class SC

class SC {
 public:
  SC(MeshBlockPack *ppack, ParameterInput *pin);
  ~SC();

  // angular quadrature (Bruls et al. 1999 type-A grid)
  SCAngularGrid *pang = nullptr;
  int nang_tot;  // = noct*nang, total number of discrete rays

  // opacity/coupling parameters
  Real opa;    // mass absorption opacity κ_a
  Real ops;    // mass scattering opacity κ_s (default 0)
  Real prat;   // radiation-to-gas pressure ratio P_rad/P_gas
  Real crat;   // speed-of-light to sound-speed ratio c/a_gas
  Real eps_uniform;  // if use_eps_uniform: zone eps forced to this value
  bool use_eps_uniform;
  bool use_ali;      // true when scattering ALI path is active

  // frequency scaffold (gray: nfreq=1, wfreq={1.0})
  int nfreq;
  DualArray1D<Real> wfreq;

  // flags controlling coupling to (M)HD
  bool is_hydro_enabled;
  bool is_mhd_enabled;
  bool affect_fluid;   // apply Q_rad back onto the fluid energy equation

  // sweep parallelization strategy: "wavefront" (default), "diagonal", or "jacobi"
  std::string sweep_method;

  // iteration control
  int iter_max;
  int itermin;       // minimum iterations before allowing early exit (default 2)
  Real iter_tol;     // LTE residual: max|ΔJ/J|
  Real ali_tol;      // ALI residual: max|ΔS/S| (Eq. 25)
  Real ali_omega;    // ALI over-relaxation (JOR/SOR) factor; 1.0 ≡ standard Jacobi-ALI (TF95 Eq. 25)
  std::string ali_mode;  // "jacobi" (default) | "gauss_seidel" (center-out fused GS, wavefront only)
  int last_niter;    // diagnostic: number of iterations used in the most recent solve
  Real last_max_rel; // diagnostic: final residual from most recent solve
  bool cnv_flag;     // true if last SolveTransfer converged

  // intensity array: (nmb, nang_tot, nx3, nx2, nx1) with ghost zones, persistent
  DvceArray5D<Real> ir;
  DvceArray5D<Real> coarse_ir;
  DvceArray5D<Real> ir_prev;   // sweep=jacobi ping-pong buffer (previous-sweep field); other
                               // sweeps leave it unallocated. Removes the read-write race that an
                               // unordered jacobi par_for would otherwise have on `ir`.

  // source iterate S: (nmb, 1, nx3, nx2, nx1) — nvar=1 for PackAndSendCC
  DvceArray5D<Real> srad;
  DvceArray5D<Real> coarse_srad;

  // per-zone radiation quantities (nmb,nx3,nx2,nx1)
  DvceArray4D<Real> chi;     // total opacity χ = (opa+ops)*ρ
  DvceArray4D<Real> planck;  // thermal Planck B = T^4
  DvceArray4D<Real> jmean;   // mean intensity J (Eq. 17)
  DvceArray4D<Real> jmean_old;  // previous iteration J (LTE residual scratch)
  DvceArray4D<Real> qrad;    // radiative heating/cooling source term

  // ALI / scattering
  DvceArray4D<Real> sigma_s;  // scattering opacity κ_s ρ
  DvceArray4D<Real> eps;      // photon destruction probability
  DvceArray4D<Real> lamstr;   // diagonal Λ* = Σ w Ψ⁰

  // inflow intensity table (nang_tot, 6 faces): default 0 = vacuum edges
  DualArray2D<Real> i_in;

  // radiation moments: n=0:J, 1-3:H_1,H_2,H_3, 4-9:K_11,K_22,K_33,K_12,K_13,K_23
  DvceArray5D<Real> moments;

  // boundary communication: separate MeshBoundaryValuesCC for ir and srad
  MeshBoundaryValuesCC *pbval_ir = nullptr;
  MeshBoundaryValuesCC *pbval_srad = nullptr;

  Real dtnew;

  SCTaskIDs id;

  void AssembleTasks(std::map<std::string, std::shared_ptr<TaskList>> tl);
  TaskStatus SolveTransfer(Driver *pdrive, int stage);
  TaskStatus AddQrad(Driver *pdrive, int stage);

  // boundary-exchange wrapper tasks for the "sc_bvals" task list
  TaskStatus InitRecvIr(Driver *pdrive, int stage);
  TaskStatus RestrictIr(Driver *pdrive, int stage);
  TaskStatus SendIr(Driver *pdrive, int stage);
  TaskStatus RecvIr(Driver *pdrive, int stage);
  TaskStatus ApplyPhysicalBCsIr(Driver *pdrive, int stage);
  TaskStatus ProlongateIr(Driver *pdrive, int stage);
  TaskStatus ClearSendIr(Driver *pdrive, int stage);
  TaskStatus ClearRecvIr(Driver *pdrive, int stage);

  TaskStatus InitRecvSrad(Driver *pdrive, int stage);
  TaskStatus RestrictSrad(Driver *pdrive, int stage);
  TaskStatus SendSrad(Driver *pdrive, int stage);
  TaskStatus RecvSrad(Driver *pdrive, int stage);
  TaskStatus ApplyPhysicalBCsSrad(Driver *pdrive, int stage);
  TaskStatus ProlongateSrad(Driver *pdrive, int stage);
  TaskStatus ClearSendSrad(Driver *pdrive, int stage);
  TaskStatus ClearRecvSrad(Driver *pdrive, int stage);

  // stagen task: radiation-relaxation timestep
  TaskStatus NewTimeStep(Driver *pdrive, int stage);

  // helpers called from SolveTransfer
  void UpdateOpacityAndSource();
  void ApplyPhysicalBCs();
  void ApplyPhysicalBCsSource();
  //! Synchronous ir+srad exchange (same ops as sc_bvals; for unit tests without Driver)
  void ExchangeBoundariesSync();
  void ComputeJ();
  void CalculateMoments();
  void ComputeQrad();
  void UpdateSourceALI(Real &max_dS_rel);

  //! Center-out Gauss-Seidel-ALI: fused wavefront sweep that accumulates J in-sweep and
  //! updates S per completion shell in place (+ local scatter). Replaces the
  //! FormalSolution+ComputeJ+UpdateSourceALI trio when ali_mode=="gauss_seidel". Wavefront only.
  //! Returns max|ΔS/S| (from the unrelaxed ΔS), like UpdateSourceALI. (sc/formal_solution.cpp)
  void SweepUpdateGS(Real &max_dS_rel);

  // formal solution driver (sc/formal_solution.cpp)
  void FormalSolution();

  // Sweep implementations. Public because Kokkos CUDA device lambdas cannot be
  // defined inside private/protected members.
  void FormalSolutionWavefront();
  void FormalSolutionDiagonal();
  void FormalSolutionJacobi();

 private:
  MeshBlockPack* pmy_pack;
};

}  // namespace nr_radiation
#endif  // NR_RADIATION_NR_RADIATION_HPP_
