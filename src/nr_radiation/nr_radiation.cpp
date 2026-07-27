//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file nr_radiation.cpp
//! \brief implementation of the VET class constructor/destructor

#include <float.h>

#include <algorithm>
#include <iostream>
#include <limits>
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/coordinates.hpp"
#include "bvals/bvals.hpp"
#include "nr_radiation/nr_radiation.hpp"

namespace nr_radiation {
//----------------------------------------------------------------------------------------
// constructor, initializes data structures and parameters

VET::VET(MeshBlockPack *ppack, ParameterInput *pin) :
    ir("vet_ir",1,1,1,1,1),
    coarse_ir("vet_coarse_ir",1,1,1,1,1),
    bb("vet_bb",1,1,1,1,1),
    coarse_bb("vet_coarse_bb",1,1,1,1,1),
    chi("vet_chi",1,1,1,1),
    planck("vet_planck",1,1,1,1),
    jmean("vet_jmean",1,1,1,1),
    jmean_old("vet_jmean_old",1,1,1,1),
    qrad("vet_qrad",1,1,1,1),
    sigma_s("vet_sigma_s",1,1,1,1),
    eps("vet_eps",1,1,1,1),
    lamstr("vet_lamstr",1,1,1,1),
    i_in("vet_i_in",1,1),
    moments("vet_moments",1,1,1,1,1),
    wfreq("vet_wfreq",1),
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

  // Activate Jacobi-ALI when scattering is present
  use_ali = (ops > 0.0) || (use_eps_uniform && eps_uniform < 1.0);

  // Unordered jacobi sweep races on upwind I — not safe for ALI accuracy claims.
  // Opt-in bypass (nr_radiation/allow_jacobi_ali=true) for experiments.
  if (use_ali && sweep_method == "jacobi") {
    if (pin->GetOrAddBoolean("nr_radiation", "allow_jacobi_ali", false)) {
      std::cout << "### WARNING: jacobi+ALI enabled (experimental, unordered sweep)"
                << std::endl;
    } else {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
        << std::endl << "<nr_radiation>/sweep = 'jacobi' is incompatible with ALI "
        << "(eps < 1 or ops > 0); use 'wavefront' or 'diagonal'" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

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
  pang = new VETAngularGrid(ndim, nmu);
  nang_tot = pang->noct * pang->nang;

  // Array allocation ----------------------------------------------------------------
  int nmb = std::max((ppack->nmb_thispack), (ppack->pmesh->nmb_maxperrank));
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int ncells2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*(indcs.ng)) : 1;
  int ncells3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*(indcs.ng)) : 1;

  Kokkos::realloc(ir, nmb, nang_tot, ncells3, ncells2, ncells1);
  Kokkos::realloc(bb, nmb, 1, ncells3, ncells2, ncells1);
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
  Kokkos::deep_copy(bb, 0.0);
  Kokkos::deep_copy(jmean, 0.0);
  Kokkos::deep_copy(qrad, 0.0);
  Kokkos::deep_copy(moments, 0.0);
  Kokkos::deep_copy(sigma_s, 0.0);
  Kokkos::deep_copy(eps, 1.0);
  Kokkos::deep_copy(lamstr, 0.0);
  Kokkos::deep_copy(planck, 0.0);

  // coarse_ir / coarse_bb for SMR/AMR (CC Restrict/Prolong)
  {
    int nccells1 = indcs.cnx1 + 2*(indcs.ng);
    int nccells2 = (indcs.cnx2 > 1) ? (indcs.cnx2 + 2*(indcs.ng)) : 1;
    int nccells3 = (indcs.cnx3 > 1) ? (indcs.cnx3 + 2*(indcs.ng)) : 1;
    if (nccells1 < 1) nccells1 = ncells1;
    if (nccells2 < 1) nccells2 = ncells2;
    if (nccells3 < 1) nccells3 = ncells3;
    Kokkos::realloc(coarse_ir, nmb, nang_tot, nccells3, nccells2, nccells1);
    Kokkos::realloc(coarse_bb, nmb, 1, nccells3, nccells2, nccells1);
    Kokkos::deep_copy(coarse_ir, 0.0);
    Kokkos::deep_copy(coarse_bb, 0.0);
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
  pbval_bb = new MeshBoundaryValuesCC(ppack, pin, false);
  pbval_bb->InitializeBuffers(1);

  dtnew = std::numeric_limits<Real>::max();
}

//----------------------------------------------------------------------------------------
// destructor

VET::~VET() {
  delete pbval_bb;
  delete pbval_ir;
  delete pang;
}

}  // namespace nr_radiation
