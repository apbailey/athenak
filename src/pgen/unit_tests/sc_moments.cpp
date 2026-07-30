//========================================================================================
// AthenaK astrophysical plasma code
// Copyright(C) 2024 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file sc_moments.cpp
//! \brief Unit test for SC::CalculateMoments (angular moments of the radiation field).
//!
//! Sets the intensity to I(n) = 1 + mu_x(n) (a linearly anisotropic field), computes the
//! moments, and checks them against Carlson quadrature identities:
//!   J = 1,  H = (1/3, 0, 0),  K_ii = 1/3,  K_ij = 0 (i!=j).
//! Adapted from apb_rad for the rt-vet API (octant-indexed mu, moments array).

#include <cmath>
#include <cstdlib>
#include <iostream>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "pgen/pgen.hpp"
#include "nr_radiation/nr_radiation.hpp"

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::SCMoments()

void ProblemGenerator::SCMoments(ParameterInput *pin, const bool restart) {
  (void)restart;
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->pnrrad == nullptr) {
    std::cout << "### FATAL ERROR in sc_moments: requires a <nr_radiation> block"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  nr_radiation::SC *psc = pmbp->pnrrad;

  auto &indcs = pmy_mesh_->mb_indcs;
  const int is = indcs.is, ie = indcs.ie, js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb1 = pmbp->nmb_thispack - 1;
  const int nang = psc->pang->nang;
  const int nang_tot = psc->nang_tot;
  const int nangt1 = nang_tot - 1;

  // set I(n) = 1 + mu_x(n) over active cells; angg = oct*nang + a
  auto ir = psc->ir;
  auto mu = psc->pang->mu;
  const int nang_loc = nang;
  par_for("sc_mom_init", DevExeSpace(), 0, nmb1, 0, nangt1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int angg, int k, int j, int i) {
    int oct = angg / nang_loc;
    int a   = angg - oct * nang_loc;
    ir(m, angg, k, j, i) = 1.0 + mu.d_view(oct, a, 0);
  });
  Kokkos::fence();

  psc->CalculateMoments();
  Kokkos::fence();

  auto mom = psc->moments;
  auto mom_h = Kokkos::create_mirror_view(mom);
  Kokkos::deep_copy(mom_h, mom);

  const Real third = 1.0/3.0;
  // Athena++ ordering: J, H1,H2,H3, K11,K22,K33, K12,K13,K23
  const Real expected[10] = {1.0, third, 0.0, 0.0, third, third, third, 0.0, 0.0, 0.0};
  const char* names[10] = {"J","H1","H2","H3","K11","K22","K33","K12","K13","K23"};

  Real max_err = 0.0;
  int bad = -1;
  for (int m=0; m<=nmb1; ++m)
  for (int k=ks; k<=ke; ++k) for (int j=js; j<=je; ++j) for (int i=is; i<=ie; ++i)
  for (int q=0; q<10; ++q) {
    Real err = std::fabs(mom_h(m,q,k,j,i) - expected[q]);
    if (err > max_err) { max_err = err; bad = q; }
  }

  const Real tol = pin->GetOrAddReal("problem", "tol", 1.0e-12);
  if (max_err > tol) {
    std::cout << "### SC moments test FAILED: " << names[bad] << " off by "
              << max_err << " > tol " << tol << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::cout << "SC moments test PASSED: max abs err = " << max_err << std::endl;
}
