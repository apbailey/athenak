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
    chi("vet_chi",1,1,1,1),
    bb("vet_bb",1,1,1,1),
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
  // Rule 2/Step 1 safety gates -----------------------------------------------------
  // (a) straight-line rays require flat, Cartesian spacetime -- refuse GR/curved coords
  if (pmy_pack->pcoord->is_general_relativistic) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
      << std::endl << "<nr_radiation> requires flat, Cartesian coordinates; the "
      << "short-characteristics straight-ray formal solution is not valid in GR"
      << std::endl;
    std::exit(EXIT_FAILURE);
  }
  // (b) v1 is LTE-only (eps==1 hardcoded): refuse any input that tries to set eps!=1
  if (pin->DoesParameterExist("nr_radiation", "eps")) {
    Real eps_in = pin->GetReal("nr_radiation", "eps");
    if (eps_in != 1.0) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
        << std::endl << "<nr_radiation>/eps = " << eps_in << " requested, but this is "
        << "an LTE-only (eps=1) v1 implementation; the ALI/scattering iteration is not "
        << "implemented" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
  // (c) v1 supports neither SMR nor AMR (see plan's "Modularity check" section)
  if (pmy_pack->pmesh->multilevel) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
      << std::endl << "<nr_radiation> does not yet support SMR or AMR (mesh has "
      << "multilevel=true); coarse-fine prolongation of the intensity array is not "
      << "implemented" << std::endl;
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

  // Iteration control for the boundary-lag fixed-point loop (Step 4)
  iter_max = pin->GetOrAddInteger("nr_radiation", "iter_max", 100);
  itermin  = pin->GetOrAddInteger("nr_radiation", "itermin", 2);
  iter_tol = pin->GetOrAddReal("nr_radiation", "iter_tol", 1.0e-6);
  last_niter = 0;
  cnv_flag = false;

  if (itermin < 1) itermin = 1;
  if (itermin > iter_max) itermin = iter_max;

  // Opacity/coupling parameters (apb_rad convention)
  opa  = pin->GetReal("nr_radiation", "opa");
  prat = pin->GetOrAddReal("nr_radiation", "prat", 1.0);
  crat = pin->GetOrAddReal("nr_radiation", "crat", 1.0);

  // Frequency scaffold (gray default)
  nfreq = pin->GetOrAddInteger("nr_radiation", "nfreq", 1);
  Kokkos::realloc(wfreq, nfreq);
  auto wfreq_h = wfreq.h_view;
  for (int f = 0; f < nfreq; ++f) wfreq_h(f) = 1.0 / static_cast<Real>(nfreq);
  wfreq.template modify<HostMemSpace>();
  wfreq.template sync<DevMemSpace>();

  // Angular quadrature: Bruls et al. (1999) type-A grid, dimensionality set by mesh
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
  Kokkos::realloc(chi, nmb, ncells3, ncells2, ncells1);
  Kokkos::realloc(bb, nmb, ncells3, ncells2, ncells1);
  Kokkos::realloc(jmean, nmb, ncells3, ncells2, ncells1);
  Kokkos::realloc(jmean_old, nmb, ncells3, ncells2, ncells1);
  Kokkos::realloc(qrad, nmb, ncells3, ncells2, ncells1);
  Kokkos::realloc(moments, nmb, 10, ncells3, ncells2, ncells1);
  Kokkos::realloc(sigma_s, nmb, ncells3, ncells2, ncells1);
  Kokkos::realloc(eps,     nmb, ncells3, ncells2, ncells1);
  Kokkos::realloc(lamstr,  nmb, ncells3, ncells2, ncells1);
  Kokkos::deep_copy(ir, 0.0);
  Kokkos::deep_copy(jmean, 0.0);
  Kokkos::deep_copy(qrad, 0.0);
  Kokkos::deep_copy(moments, 0.0);
  Kokkos::deep_copy(sigma_s, 0.0);
  Kokkos::deep_copy(eps, 1.0);      // LTE default
  Kokkos::deep_copy(lamstr, 0.0);

  // AMR scaffold: allocate coarse_ir using coarse-cell counts even though multilevel
  // is currently FATAL. This prepares for future SMR/AMR support without changing
  // runtime behaviour (the FATAL above still fires if multilevel is true).
  {
    int nccells1 = indcs.cnx1 + 2*(indcs.ng);
    int nccells2 = (indcs.cnx2 > 1) ? (indcs.cnx2 + 2*(indcs.ng)) : 1;
    int nccells3 = (indcs.cnx3 > 1) ? (indcs.cnx3 + 2*(indcs.ng)) : 1;
    // On uniform meshes cnx* is still set (nx*/2); if somehow invalid, fall back
    // to fine-grid sizes so the array is well-formed.
    if (nccells1 < 1) nccells1 = ncells1;
    if (nccells2 < 1) nccells2 = ncells2;
    if (nccells3 < 1) nccells3 = ncells3;
    Kokkos::realloc(coarse_ir, nmb, nang_tot, nccells3, nccells2, nccells1);
    Kokkos::deep_copy(coarse_ir, 0.0);
  }

  // Inflow BC table (nang_tot, 6 faces): default vacuum
  Kokkos::realloc(i_in, nang_tot, 6);
  for (int n = 0; n < nang_tot; ++n) {
    for (int f = 0; f < 6; ++f) i_in.h_view(n, f) = 0.0;
  }
  i_in.template modify<HostMemSpace>();
  i_in.template sync<DevMemSpace>();

  // Boundary communication buffers for the intensity array
  pbval_ir = new MeshBoundaryValuesCC(ppack, pin, false);
  pbval_ir->InitializeBuffers(nang_tot);

  // radiation-relaxation timestep (Athena-C radtrans_dt) is recomputed each
  // SolveTransfer when affect_fluid; start unconstrained so beam/non-coupled runs
  // keep the hydro CFL.
  dtnew = std::numeric_limits<Real>::max();
}

//----------------------------------------------------------------------------------------
// destructor

VET::~VET() {
  delete pbval_ir;
  delete pang;
}

}  // namespace nr_radiation
