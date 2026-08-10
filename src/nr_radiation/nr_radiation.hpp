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
#include <vector>

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

  // sweep parallelization strategy: "wavefront" (default), "diagonal", "diagonal_compact",
  // or "jacobi"
  std::string sweep_method;
  // sweep=diagonal_compact only: explicit TeamPolicy team size (0 = Kokkos::AUTO, the default).
  // The diagonal kernel is latency-bound with few teams (nmb*nang_tot); a larger team hides more
  // latency (capped by register-limited occupancy). Clamped to team_size_max at launch.
  int diag_team_size;

  // iteration control
  int iter_max;
  int itermin;       // minimum iterations before allowing early exit (default 2)
  Real iter_tol;     // LTE residual: max|ΔJ/J|
  Real ali_tol;      // ALI residual: max|ΔS/S| (Eq. 25)
  Real ali_omega;    // ALI over-relaxation (JOR/SOR) factor; 1.0 ≡ standard Jacobi-ALI (TF95 Eq. 25)
  std::string ali_mode;  // "jacobi" (default) | "gauss_seidel" (center-out fused GS, wavefront only)
  bool gs_scatter;   // GS local-scatter acceleration (design doc Option B); only read inside
                     // SweepUpdateGS (no effect unless ali_mode=="gauss_seidel"). Default false
                     // reproduces the in-place-only GS (Option 1) bit-for-bit; opt-in until
                     // validated end-to-end in AthenaK (iteration/prototype/REPORT.md Phase 2).
  std::string gs_scatter_mode;  // per-axis coupling decomposition (only when gs_scatter):
                     // "geometric" (default; c·lmin/l_axis per axis, total c·(1+am_r+bm)) |
                     // "normalized" (Fix B; the same three shares renormalised to total exactly c
                     // — restores the operator row-sum the footpoint identity requires; see
                     // iteration/gs-scatter-3d-origin.md). Experimental knob for the 3D interrogation.
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

  // GS local-scatter (Option B) coupling coefficients: cpl_Xp/cpl_Xm(cell) = Σ_{rays with
  // sign +/- along axis X} w·(a0+e^{-Δτ}·a1), evaluated at THIS cell (see UpdateCellSC /
  // SweepUpdateGS doc comments). One pair per active dimension. Lazily allocated (like
  // ir_prev) only on first use with gs_scatter==true; empty (size 0) otherwise.
  DvceArray4D<Real> cpl_xp, cpl_xm, cpl_yp, cpl_ym, cpl_zp, cpl_zm;

  // Compact wavefront plane index map (bit-identical throughput opt; sc/formal_solution.cpp).
  // The set of interior cells on hyperplane h (li1+li2+li3==h, li = distance from the upwind
  // corner) is a STATIC function of the meshblock interior dims only — identical for every
  // meshblock and every octant — so it is precomputed ONCE (lazily, like ir_prev) and reused
  // every sweep. wf_cell_ holds the plane-ordered packed linear interior index
  // lin=(li3*nx2+li2)*nx1+li1; wf_plane_start_[h] is where plane h begins (plane h ==
  // [start[h],start[h+1])). Lets the 2D/3D wavefront launch EXACTLY the plane's cells instead of
  // an nx1(*nx2)-shaped grid that early-returns off-plane slots (the ~1/2 in 2D, ~1/3 in 3D warp
  // over-issue). Empty (size 0) until BuildWavefrontIndex() runs; 1D is already compact (unused).
  DvceArray1D<int> wf_cell_;
  std::vector<int> wf_plane_start_;
  // Device mirror of wf_plane_start_ (offsets where each plane begins). The wavefront reads the
  // host std::vector between per-plane par_for launches, but sweep=diagonal_compact runs its h-loop
  // INSIDE one device kernel and needs the offsets on device to slice wf_cell_ per plane. Built once
  // alongside wf_cell_ in BuildWavefrontIndex(); empty until then.
  DvceArray1D<int> wf_plane_start_dev_;

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
  //! updates S per completion shell in place, optionally scattering ΔS into already-arrived
  //! immediate neighbours' J (gs_scatter; design doc Option B). Replaces the
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

  //! Precompute the compact per-hyperplane cell-index map used by the 2D/3D wavefront sweep
  //! (fills wf_cell_ / wf_plane_start_). Static in the meshblock interior dims, so built once.
  void BuildWavefrontIndex();

 private:
  MeshBlockPack* pmy_pack;
};

}  // namespace nr_radiation
#endif  // NR_RADIATION_NR_RADIATION_HPP_
