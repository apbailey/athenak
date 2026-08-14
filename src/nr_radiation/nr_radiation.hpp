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
//!   chi = sigma_a + sigma_s     total extinction (default: (opa + ops) * rho)
//!   eps = sigma_a / chi         (or uniform override)
//!   B   = T^4                   (planck); S iterate in srad
//!   Q   = eps * chi * crat * prat * (J - B)
//! Per-cell sigma_a/sigma_s and B may be supplied by user hooks enrolled from the pgen
//! (EnrollOpacityFunction / EnrollPlanckFunction; contract in sc/sc_opacity.hpp).

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"
#include "bvals/bvals.hpp"
#include "nr_radiation/angular_grid.hpp"
#include "nr_radiation/sc/sc_opacity.hpp"

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
  bool use_ali;      // true when scattering ALI path is active (re-derived each solve
                     // from per-cell sigma_s in UpdateOpacityAndSource)
  bool opa_specified;  // <nr_radiation>/opa present in input; opa is required unless an
                       // opacity hook is enrolled (checked post-pgen in pgen.cpp)

  // user-enrollable opacity / source hooks (contract in sc/sc_opacity.hpp);
  // dispatch is by null-check, no input flags
  SCOpacityFnPtr user_opacity_func = nullptr;
  SCPlanckFnPtr user_planck_func = nullptr;
  void EnrollOpacityFunction(SCOpacityFnPtr myfunc);
  void EnrollPlanckFunction(SCPlanckFnPtr myfunc);

  // frequency scaffold (gray: nfreq=1, wfreq={1.0})
  int nfreq;
  DualArray1D<Real> wfreq;

  // flags controlling coupling to (M)HD
  bool is_hydro_enabled;
  bool is_mhd_enabled;
  bool affect_fluid;   // apply Q_rad back onto the fluid energy equation

  // sweep parallelization strategy: "wavefront" (default), "diagonal", "diagonal_compact",
  // "tiled", or "jacobi"
  std::string sweep_method;
  // sweep=tiled only: tile edge in cells (0 = whole meshblock, which reduces the tiled sweep to
  // diagonal_compact exactly). Must divide the meshblock interior dims; validated at
  // construction. See rt-profiling/TILED_SWEEP_DESIGN.md.
  int tile_size;
  // sweep=tiled + ir_layout=angle_inner only: angles per team. Consecutive threads within a team
  // walk consecutive angles, so this is the coalescing width; 32 (a warp) is the default. Angles
  // are split across the league so the tile-derived team count is preserved.
  int tile_na;
  // sweep=diagonal_compact only: explicit TeamPolicy team size (0 = Kokkos::AUTO, the default).
  // The diagonal kernel is latency-bound with few teams (nmb*nang_tot); a larger team hides more
  // latency (capped by register-limited occupancy). Clamped to team_size_max at launch.
  int diag_team_size;
  // <nr_radiation>/ir_layout = normal (default) | angle_inner. When angle_inner, ir/coarse_ir are
  // stored ANGLE-INNERMOST (m,k,j,i,angg) so the wavefront sweep + moments run coalesced natively
  // (ledger I2). Opt-in; requires sweep=wavefront, ndim==3, uniform mesh (no SMR/AMR). The generic
  // CC exchange (which assumes angle=index1) is bridged SC-side via a normal-layout companion
  // ir_normal transposed around the UNCHANGED PackAndSendCC/RecvAndUnpackCC calls.
  bool ir_angle_inner;
  // <nr_radiation>/sc_hoist = false (default) | true. When true, the angle-only SC interpolation
  // invariants (dominant axis, bilinear weights c0..c3, path-length dx_dom/|mu_dom|) are precomputed
  // ONCE per ray into sc_inv_ and read by the wavefront sweep instead of being recomputed per cell
  // (ledger I3 — removes fabs/fmin/divides + the axis branch from the hot loop). Opt-in; requires
  // sweep=wavefront, ndim==3, uniform mesh, ir_layout=normal. The weights depend on dx (constant only
  // on a uniform mesh); the default (false) path is byte-identical (host-scope dispatch — a sibling
  // par_for launch). Bit-exact vs the recompute path (GatherSolveSC reproduces UpdateCellSC exactly).
  bool sc_hoist;

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
  DvceArray5D<Real> ir_t;      // sweep=wavefront_coalesced scratch: angle-INNERMOST transpose of ir,
                               // shape (nmb, nc3, nc2, nc1, nang_tot). Lazily allocated; isolates the
                               // I2 gather-coalescing measurement (global ir layout stays unchanged).
  DvceArray5D<Real> ir_normal; // ir_layout=angle_inner ONLY: normal-layout (m,nang_tot,k,j,i) companion
                               // used solely to bridge the UNCHANGED CC exchange (ir<->ir_normal sync
                               // around PackAndSendCC/RecvAndUnpackCC). Empty when ir_layout=normal.

  // source iterate S: (nmb, 1, nx3, nx2, nx1) — nvar=1 for PackAndSendCC
  DvceArray5D<Real> srad;
  DvceArray5D<Real> coarse_srad;

  // per-zone radiation quantities (nmb,nx3,nx2,nx1)
  DvceArray4D<Real> chi;     // total opacity χ = (opa+ops)*ρ
  DvceArray4D<Real> planck;  // thermal Planck B = T^4
  DvceArray4D<Real> jmean;   // mean intensity J (Eq. 17)
  DvceArray4D<Real> jmean_old;  // previous iteration J (LTE residual scratch)
  DvceArray4D<Real> qrad;    // radiative heating/cooling source term

  // ALI / scattering. sigma_a/sigma_s are (nmb, 1, nx3, nx2, nx1) — nvar=1 5D so they are
  // directly registrable as stored output variables (basetype_output requires DvceArray5D).
  DvceArray5D<Real> sigma_a;  // absorption opacity κ_a ρ
  DvceArray5D<Real> sigma_s;  // scattering opacity κ_s ρ
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

  // I3 precomputed per-ray SC interpolation invariants (sc_hoist=true only): shape (nang_tot, 10),
  // columns [sx,sy,sz,axis (stored as Real, cast to int), c0,c1,c2,c3, pdx,pamu]. Packed layout of
  // SCRayInv (sc_interp.hpp) so the header need not include it. Built once by BuildAngleInvTable();
  // read by the wavefront sweep's hoisted variant. Empty (size 0) unless sc_hoist.
  DvceArray2D<Real> sc_inv_;
  // Tiled (KBA) sweep index maps — two nested applications of the same hyperplane rule, both
  // pure functions of the meshblock/tile dims, so both are built once (BuildTileIndex()).
  //   tile_cell_ / tile_plane_start_dev_ : the compact per-plane cell list WITHIN one tile,
  //     packed as lin=(l3*tx2+l2)*tx1+l1. Device-side: the h-loop runs inside the kernel.
  //   tp_cell_ / tp_start_ : the map over TILES, packed as (tc*nt2+tb)*nt1+ta, grouped by
  //     tile-plane H=ta+tb+tc. tp_start_ is host-side: it drives the launch loop, one kernel
  //     per tile-plane, and that kernel boundary IS the cross-tile barrier.
  DvceArray1D<int> tile_cell_;
  DvceArray1D<int> tile_plane_start_dev_;
  DvceArray1D<int> tp_cell_;
  std::vector<int> tp_start_;
  int tx1_ = 0, tx2_ = 0, tx3_ = 0;      // tile dims in cells
  int nt1_ = 0, nt2_ = 0, nt3_ = 0;      // tiles per axis
  int hmax_tile_ = 0;                    // last cell-plane index within a tile
  int hmax_tplane_ = 0;                  // last tile-plane index

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
  void FormalSolutionWavefrontCoalesced();  // I2 prototype: angle-warp sweep on transposed ir_t
  void FormalSolutionWavefrontAngleInner(); // I2 native: angle-warp sweep directly on angle-inner ir
  void SyncIrNormal(bool to_normal);        // ir<->ir_normal transpose (CC-exchange bridge)
  void FormalSolutionDiagonal();
  void FormalSolutionJacobi();

  //! Tiled (KBA) sweep: partition the meshblock into tiles and run a wavefront over TILE-planes,
  //! one kernel launch per tile-plane, each team sweeping one tile with the diagonal's inner
  //! h-loop. The cross-tile barrier is the kernel boundary, so unlike splitting a single cell-
  //! plane across teams (a data race) this is safe: a tile reads only its upwind corner
  //! neighbours, all of which sit on strictly lower tile-planes and so completed in an earlier
  //! launch. Bit-identical to the other sweeps. tile_size=0 (whole block) degenerates to exactly
  //! diagonal_compact. See rt-profiling/TILED_SWEEP_DESIGN.md.
  void FormalSolutionTiled();

  //! Precompute the compact per-hyperplane cell-index map used by the 2D/3D wavefront sweep
  //! (fills wf_cell_ / wf_plane_start_). Static in the meshblock interior dims, so built once.
  void BuildWavefrontIndex();

  //! Precompute the per-ray SC interpolation invariants into sc_inv_ (ledger I3; sc_hoist=true).
  //! One device par_for over angg calling the same ComputeSCAngleInv the recompute path uses, so
  //! the table is bit-identical to the inline computation. Uniform-mesh only (uses meshblock-0 dx).
  void BuildAngleInvTable();
  //! Tiled sweep on the angle-innermost `ir` (I7 x I2). Same tile-plane launch structure as
  //! FormalSolutionTiled, but each team owns a BLOCK of `tile_na` angles instead of one, and the
  //! team's inner range runs (plane cells x angles) with ANGLE FASTEST -- so consecutive threads
  //! read consecutive `angg`, which is contiguous under angle_inner. That is what makes the two
  //! optimisations composable: tiling supplies the teams, angle-major lanes supply the coalescing.
  //! Team count is preserved by splitting angles across the league (ntile*nmb*ceil(nang_tot/na))
  //! rather than collapsing them into one team. Bit-identical to every other sweep.
  void FormalSolutionTiledAngleInner();

  //! Precompute the two index maps the tiled sweep needs (tile-local cells, and tiles). Both are
  //! pure functions of the meshblock/tile dims, so this runs once on first use.
  void BuildTileIndex();

 private:
  MeshBlockPack* pmy_pack;
};

}  // namespace nr_radiation
#endif  // NR_RADIATION_NR_RADIATION_HPP_
