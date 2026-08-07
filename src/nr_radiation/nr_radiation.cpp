//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file nr_radiation.cpp
//! \brief implementation of the SC class constructor/destructor

#include <float.h>

#include <algorithm>
#include <iostream>
#include <limits>
#include <string>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/coordinates.hpp"
#include "bvals/bvals.hpp"
#include "nr_radiation/nr_radiation.hpp"

namespace nr_radiation {
//----------------------------------------------------------------------------------------
// constructor, initializes data structures and parameters

SC::SC(MeshBlockPack *ppack, ParameterInput *pin) :
    ir("sc_ir",1,1,1,1,1),
    coarse_ir("sc_coarse_ir",1,1,1,1,1),
    srad("sc_srad",1,1,1,1,1),
    coarse_srad("sc_coarse_srad",1,1,1,1,1),
    chi("sc_chi",1,1,1,1),
    planck("sc_planck",1,1,1,1),
    jmean("sc_jmean",1,1,1,1),
    jmean_old("sc_jmean_old",1,1,1,1),
    qrad("sc_qrad",1,1,1,1),
    sigma_s("sc_sigma_s",1,1,1,1),
    eps("sc_eps",1,1,1,1),
    lamstr("sc_lamstr",1,1,1,1),
    i_in("sc_i_in",1,1),
    moments("sc_moments",1,1,1,1,1),
    wfreq("sc_wfreq",1),
    pmy_pack(ppack) {
  // straight-line rays require flat, Cartesian spacetime
  if (pmy_pack->pcoord->is_general_relativistic) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
      << std::endl << "<nr_radiation> requires flat, Cartesian coordinates; the "
      << "short-characteristics straight-ray formal solution is not valid in GR"
      << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // Check for hydro/mhd coupling (mirrors radiation::Radiation)
  is_hydro_enabled = pin->DoesBlockExist("hydro");
  is_mhd_enabled = pin->DoesBlockExist("mhd");
  if (is_hydro_enabled && is_mhd_enabled) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
      << std::endl << "<nr_radiation> does not support two fluid calculations, yet "
      << "both <hydro> and <mhd> blocks exist in input file" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  affect_fluid = pin->GetOrAddBoolean("nr_radiation", "affect_fluid", true);

  // Sweep parallelization strategy
  sweep_method = pin->GetOrAddString("nr_radiation", "sweep", "wavefront");
  if (sweep_method != "wavefront" && sweep_method != "diagonal"
      && sweep_method != "jacobi") {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
      << std::endl << "<nr_radiation>/sweep = '" << sweep_method << "' is not recognised; "
      << "valid values are 'wavefront', 'diagonal', 'jacobi'" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // Iteration control
  iter_max = pin->GetOrAddInteger("nr_radiation", "iter_max", 100);
  itermin  = pin->GetOrAddInteger("nr_radiation", "itermin", 2);
  iter_tol = pin->GetOrAddReal("nr_radiation", "iter_tol", 1.0e-6);
  ali_tol  = pin->GetOrAddReal("nr_radiation", "ali_tol", 1.0e-5);
  // ALI over-relaxation factor (Phase I1 / Option 0). S <- S + omega*dS (TF95 Eq. 25).
  // Default 1.0 reproduces standard Jacobi-ALI bit-for-bit; (0,2) is the SOR convergence range.
  ali_omega = pin->GetOrAddReal("nr_radiation", "ali_omega", 1.0);
  if (ali_omega <= 0.0 || ali_omega >= 2.0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
      << std::endl << "<nr_radiation>/ali_omega must be in (0,2); got " << ali_omega
      << " (over-relaxation is unstable for omega>=2, non-advancing for omega<=0)" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  // ALI update ordering (Phase I2). "jacobi" (default): one full formal solve -> global J ->
  // update all S (standard hyperplane Jacobi-ALI, bit-identical to before). "gauss_seidel":
  // center-out fused sweep — J accumulated in-sweep, S updated per completion shell in place,
  // with a local (immediate-neighbour) scatter (Davis 2012 §3.4 / TF95; Option B, validated in
  // iteration/prototype/). GS needs the wavefront host-plane ordering.
  ali_mode = pin->GetOrAddString("nr_radiation", "ali_mode", "jacobi");
  if (ali_mode != "jacobi" && ali_mode != "gauss_seidel") {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
      << std::endl << "<nr_radiation>/ali_mode = '" << ali_mode << "' is not recognised; "
      << "valid values are 'jacobi', 'gauss_seidel'" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (ali_mode == "gauss_seidel" && sweep_method != "wavefront") {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
      << std::endl << "<nr_radiation>/ali_mode=gauss_seidel requires sweep=wavefront "
      << "(the center-out GS uses the wavefront host-plane ordering); got sweep='"
      << sweep_method << "'" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  // GS local-scatter acceleration (Phase I2, Option B; iteration/prototype/REPORT.md). Only
  // read inside SweepUpdateGS — a no-op unless ali_mode=="gauss_seidel". Default false
  // reproduces the existing in-place-only GS (Option 1) bit-for-bit.
  gs_scatter = pin->GetOrAddBoolean("nr_radiation", "gs_scatter", false);
  gs_scatter_mode = pin->GetOrAddString("nr_radiation", "gs_scatter_mode", "normalized");
  if (gs_scatter_mode != "geometric" && gs_scatter_mode != "normalized") {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
      << std::endl << "<nr_radiation>/gs_scatter_mode = '" << gs_scatter_mode
      << "' is not recognised; use 'geometric' or 'normalized'" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  last_niter = 0;
  last_max_rel = 0.0;
  cnv_flag = false;

  if (itermin < 1) itermin = 1;
  if (itermin > iter_max) itermin = iter_max;

  // Opacity/coupling parameters
  opa  = pin->GetReal("nr_radiation", "opa");
  ops  = pin->GetOrAddReal("nr_radiation", "ops", 0.0);
  prat = pin->GetOrAddReal("nr_radiation", "prat", 1.0);
  crat = pin->GetOrAddReal("nr_radiation", "crat", 1.0);
  if (opa < 0.0 || ops < 0.0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
      << std::endl << "<nr_radiation> opa and ops must be non-negative" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  use_eps_uniform = pin->DoesParameterExist("nr_radiation", "eps");
  eps_uniform = use_eps_uniform ? pin->GetReal("nr_radiation", "eps") : 1.0;
  if (use_eps_uniform && (eps_uniform < 0.0 || eps_uniform > 1.0)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
      << std::endl << "<nr_radiation>/eps must be in [0,1]" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // Activate ALI when scattering is present. All three sweeps (wavefront, diagonal,
  // jacobi) support ALI: the jacobi sweep reads its upwind intensities from the previous
  // iterate held in a ping-pong double buffer (ir_prev), so its unordered par_for no
  // longer races on ir and is safe with ALI just like the ordered sweeps.
  use_ali = (ops > 0.0) || (use_eps_uniform && eps_uniform < 1.0);

  // Frequency scaffold (gray default)
  nfreq = pin->GetOrAddInteger("nr_radiation", "nfreq", 1);
  Kokkos::realloc(wfreq, nfreq);
  auto wfreq_h = wfreq.h_view;
  for (int f = 0; f < nfreq; ++f) wfreq_h(f) = 1.0 / static_cast<Real>(nfreq);
  wfreq.template modify<HostMemSpace>();
  wfreq.template sync<DevMemSpace>();

  // Angular quadrature
  int nmu = pin->GetInteger("nr_radiation", "nmu");
  Mesh *pm = pmy_pack->pmesh;
  int ndim = (pm->three_d) ? 3 : ((pm->two_d) ? 2 : 1);
  pang = new SCAngularGrid(ndim, nmu);
  nang_tot = pang->noct * pang->nang;

  // gs_scatter's per-cell coupling (cpl_xp/xm/yp/ym/zp/zm) splits each ray's one-step coupling
  // w·(a0+e^{-Δτ}a1) across the axis-aligned neighbours (formal_solution.cpp SweepUpdateGS Kernel
  // A). Two decompositions are available via gs_scatter_mode:
  //   "normalized" (DEFAULT): the three axis shares lmin/l_axis are renormalised to sum to
  //     exactly the true one-step coupling c — the row-sum the short-characteristic footpoint
  //     identity (Σ bilinear weights = 1) requires. This is the correct Gauss-Seidel off-diagonal
  //     and is unconditionally stable at ali_omega=1 in every dimension (verified 1D/2D/3D incl.
  //     isotropic thick atmospheres). See iteration/gs-scatter-3d-origin.md.
  //   "geometric" (opt-in): the un-renormalised shares lmin/l_axis, total c·(1+am_r+bm). This
  //     equals the normalized coupling times an implicit, uncontrolled over-relaxation factor
  //     (1+am_r+bm) ∈ [1,3] — free speed when it happens to be stable (quasi-1D/2D and anisotropic
  //     grids) but, in genuinely isotropic 3D, that factor reaches ~3 and pushes the GS spectral
  //     radius past 1 → diverges to NaN at ali_omega=1. Use "normalized" + explicit ali_omega>1
  //     (SOR) instead to get the same acceleration under control (verified: normalized+ω=1.2 ≡
  //     geometric's iteration count on the anisotropic 3D atmosphere).
  // Warn only for the opt-in over-relaxed mode in 3D (the default is safe):
  if (gs_scatter && gs_scatter_mode == "geometric" && ndim == 3
      && global_variable::my_rank == 0) {
    std::cout << "### WARNING in " << __FILE__ << ": <nr_radiation>/gs_scatter_mode=geometric in "
      << "3D is an implicitly over-relaxed coupling (factor up to ~3) that can diverge to NaN on "
      << "isotropic scattering problems. Prefer gs_scatter_mode=normalized (default, stable) with "
      << "ali_omega>1 for controlled SOR; see iteration/gs-scatter-3d-origin.md." << std::endl;
  }

  // Array allocation ----------------------------------------------------------------
  int nmb = std::max((ppack->nmb_thispack), (ppack->pmesh->nmb_maxperrank));
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int ncells2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*(indcs.ng)) : 1;
  int ncells3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*(indcs.ng)) : 1;

  Kokkos::realloc(ir, nmb, nang_tot, ncells3, ncells2, ncells1);
  Kokkos::realloc(srad, nmb, 1, ncells3, ncells2, ncells1);
  Kokkos::realloc(chi, nmb, ncells3, ncells2, ncells1);
  Kokkos::realloc(planck, nmb, ncells3, ncells2, ncells1);
  Kokkos::realloc(jmean, nmb, ncells3, ncells2, ncells1);
  Kokkos::realloc(jmean_old, nmb, ncells3, ncells2, ncells1);
  Kokkos::realloc(qrad, nmb, ncells3, ncells2, ncells1);
  Kokkos::realloc(moments, nmb, 10, ncells3, ncells2, ncells1);
  Kokkos::realloc(sigma_s, nmb, ncells3, ncells2, ncells1);
  Kokkos::realloc(eps,     nmb, ncells3, ncells2, ncells1);
  Kokkos::realloc(lamstr,  nmb, ncells3, ncells2, ncells1);
  Kokkos::deep_copy(ir, 0.0);
  Kokkos::deep_copy(srad, 0.0);
  Kokkos::deep_copy(jmean, 0.0);
  Kokkos::deep_copy(qrad, 0.0);
  Kokkos::deep_copy(moments, 0.0);
  Kokkos::deep_copy(sigma_s, 0.0);
  Kokkos::deep_copy(eps, 1.0);
  Kokkos::deep_copy(lamstr, 0.0);
  Kokkos::deep_copy(planck, 0.0);

  // coarse_ir / coarse_srad for SMR/AMR (CC Restrict/Prolong)
  {
    int nccells1 = indcs.cnx1 + 2*(indcs.ng);
    int nccells2 = (indcs.cnx2 > 1) ? (indcs.cnx2 + 2*(indcs.ng)) : 1;
    int nccells3 = (indcs.cnx3 > 1) ? (indcs.cnx3 + 2*(indcs.ng)) : 1;
    if (nccells1 < 1) nccells1 = ncells1;
    if (nccells2 < 1) nccells2 = ncells2;
    if (nccells3 < 1) nccells3 = ncells3;
    Kokkos::realloc(coarse_ir, nmb, nang_tot, nccells3, nccells2, nccells1);
    Kokkos::realloc(coarse_srad, nmb, 1, nccells3, nccells2, nccells1);
    Kokkos::deep_copy(coarse_ir, 0.0);
    Kokkos::deep_copy(coarse_srad, 0.0);
  }

  // Inflow BC table (nang_tot, 6 faces): default vacuum
  Kokkos::realloc(i_in, nang_tot, 6);
  for (int n = 0; n < nang_tot; ++n) {
    for (int f = 0; f < 6; ++f) i_in.h_view(n, f) = 0.0;
  }
  i_in.template modify<HostMemSpace>();
  i_in.template sync<DevMemSpace>();

  pbval_ir = new MeshBoundaryValuesCC(ppack, pin, false);
  pbval_ir->InitializeBuffers(nang_tot);
  pbval_srad = new MeshBoundaryValuesCC(ppack, pin, false);
  pbval_srad->InitializeBuffers(1);

  dtnew = std::numeric_limits<Real>::max();
}

//----------------------------------------------------------------------------------------
// destructor

SC::~SC() {
  delete pbval_srad;
  delete pbval_ir;
  delete pang;
}

}  // namespace nr_radiation
