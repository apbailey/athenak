//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_vet.cpp
//! \brief implementation of the RadiationVET class constructor/destructor

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
#include "radiation_vet.hpp"

namespace radiation_vet {
//----------------------------------------------------------------------------------------
// constructor, initializes data structures and parameters

RadiationVET::RadiationVET(MeshBlockPack *ppack, ParameterInput *pin) :
    ir("vet_ir",1,1,1,1,1),
    coarse_ir("vet_coarse_ir",1,1,1,1,1),
    chi("vet_chi",1,1,1,1),
    bb("vet_bb",1,1,1,1),
    jmean("vet_jmean",1,1,1,1),
    jmean_old("vet_jmean_old",1,1,1,1),
    qrad("vet_qrad",1,1,1,1),
    moments("vet_moments",1,1,1,1,1),
    pmy_pack(ppack) {
  // Rule 2/Step 1 safety gates -----------------------------------------------------
  // (a) straight-line rays require flat, Cartesian spacetime -- refuse GR/curved coords
  if (pmy_pack->pcoord->is_general_relativistic) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
      << std::endl << "<radiation_vet> requires flat, Cartesian coordinates; the "
      << "short-characteristics straight-ray formal solution is not valid in GR"
      << std::endl;
    std::exit(EXIT_FAILURE);
  }
  // (b) v1 is LTE-only (eps==1 hardcoded): refuse any input that tries to set eps!=1
  if (pin->DoesParameterExist("radiation_vet", "eps")) {
    Real eps_in = pin->GetReal("radiation_vet", "eps");
    if (eps_in != 1.0) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
        << std::endl << "<radiation_vet>/eps = " << eps_in << " requested, but this is "
        << "an LTE-only (eps=1) v1 implementation; the ALI/scattering iteration is not "
        << "implemented" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
  // (c) v1 supports neither SMR nor AMR (see plan's "Modularity check" section)
  if (pmy_pack->pmesh->multilevel) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
      << std::endl << "<radiation_vet> does not yet support SMR or AMR (mesh has "
      << "multilevel=true); coarse-fine prolongation of the intensity array is not "
      << "implemented" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // Check for hydro/mhd coupling (mirrors radiation::Radiation)
  is_hydro_enabled = pin->DoesBlockExist("hydro");
  is_mhd_enabled = pin->DoesBlockExist("mhd");
  if (is_hydro_enabled && is_mhd_enabled) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
      << std::endl << "<radiation_vet> does not support two fluid calculations, yet "
      << "both <hydro> and <mhd> blocks exist in input file" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  affect_fluid = pin->GetOrAddBoolean("radiation_vet", "affect_fluid", true);

  // Iteration control for the boundary-lag fixed-point loop (Step 4)
  iter_max = pin->GetOrAddInteger("radiation_vet", "iter_max", 100);
  iter_tol = pin->GetOrAddReal("radiation_vet", "iter_tol", 1.0e-10);
  last_niter = 0;

  // Opacity/emission specification (D6: single point of specification)
  {
    std::string chi_str = pin->GetOrAddString("radiation_vet", "chi_type", "constant");
    if (chi_str.compare("constant") == 0) {
      opac.chi_type = VETChiType::constant;
      opac.chi0 = pin->GetReal("radiation_vet", "chi0");
    } else {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
        << std::endl << "<radiation_vet>/chi_type = '" << chi_str << "' not implemented "
        << "(only 'constant' in v1)" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    std::string bb_str = pin->GetOrAddString("radiation_vet", "bb_type", "zero");
    if (bb_str.compare("zero") == 0) {
      opac.bb_type = VETBBType::zero;
    } else if (bb_str.compare("greybody") == 0) {
      opac.bb_type = VETBBType::greybody;
      opac.bb_norm = pin->GetReal("radiation_vet", "bb_norm");
    } else {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
        << std::endl << "<radiation_vet>/bb_type = '" << bb_str << "' not implemented "
        << "(only 'zero' or 'greybody' in v1)" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  // Angular quadrature: Bruls et al. (1999) type-A grid, dimensionality set by mesh
  int nmu = pin->GetInteger("radiation_vet", "nmu");
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
  // coarse_ir intentionally left as its (1,1,1,1,1)-sized default: multilevel is
  // asserted false above, so it is never read or written.
  Kokkos::deep_copy(ir, 0.0);
  Kokkos::deep_copy(jmean, 0.0);
  Kokkos::deep_copy(qrad, 0.0);
  Kokkos::deep_copy(moments, 0.0);

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

RadiationVET::~RadiationVET() {
  delete pbval_ir;
  delete pang;
}

}  // namespace radiation_vet
