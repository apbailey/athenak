//========================================================================================
// AthenaK astrophysical plasma code
// Copyright(C) 2024 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file vet_uniform.cpp
//! \brief Machine-precision structural test for the multi-D VET sweep (FormalSolution).
//!
//! Sets a uniform medium: constant absorption chi, constant source S = b everywhere
//! (including ghosts), and the intensity initialized to b everywhere. One sweep must
//! leave I == b to machine precision for every cell and angle. This holds iff the SC
//! quadrature reproduces a constant source (edtau + a0 + a1 + a2 = 1) AND the bilinear
//! upwind-face weights form a partition of unity. Adapted from apb_rad for the rt-vet
//! API (4D chi/bb, FormalSolution dispatcher).

#include <cmath>
#include <cstdlib>
#include <iostream>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "pgen/pgen.hpp"
#include "nr_radiation/nr_radiation.hpp"

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::VETUniform()

void ProblemGenerator::VETUniform(ParameterInput *pin, const bool restart) {
  (void)restart;
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->pnrrad == nullptr) {
    std::cout << "### FATAL ERROR in vet_uniform: requires a <nr_radiation> block"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  nr_radiation::VET *pvet = pmbp->pnrrad;

  auto &indcs = pmy_mesh_->mb_indcs;
  const int is = indcs.is, ie = indcs.ie, js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int ng = indcs.ng;
  const int n1 = indcs.nx1 + 2*ng;
  const int n2 = (indcs.nx2 > 1) ? indcs.nx2 + 2*ng : 1;
  const int n3 = (indcs.nx3 > 1) ? indcs.nx3 + 2*ng : 1;
  const int nmb1 = pmbp->nmb_thispack - 1;
  const int nang_tot = pvet->nang_tot;
  const int nangt1 = nang_tot - 1;

  const Real chi = pin->GetOrAddReal("problem", "chi", 5.0);
  const Real b   = pin->GetOrAddReal("problem", "source", 1.0);

  auto chi_a = pvet->chi;
  auto bb_a  = pvet->bb;
  auto ir    = pvet->ir;

  // uniform opacity and source over the full array (including ghosts)
  par_for("vet_uni_setup", DevExeSpace(), 0, nmb1, 0, (n3-1), 0, (n2-1), 0, (n1-1),
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    chi_a(m,k,j,i) = chi;
    bb_a(m,k,j,i)  = b;
  });
  // intensity initialized to b everywhere (incl. ghosts -> upwind boundary is also b)
  par_for("vet_uni_ir", DevExeSpace(), 0, nmb1, 0, nangt1, 0, (n3-1), 0, (n2-1), 0, (n1-1),
  KOKKOS_LAMBDA(int m, int a, int k, int j, int i) { ir(m,a,k,j,i) = b; });
  Kokkos::fence();

  pvet->FormalSolution();
  Kokkos::fence();

  auto ir_h = Kokkos::create_mirror_view(ir);
  Kokkos::deep_copy(ir_h, ir);

  Real max_err = 0.0;
  for (int m=0; m<=nmb1; ++m)
  for (int a=0; a<nang_tot; ++a)
  for (int k=ks; k<=ke; ++k) for (int j=js; j<=je; ++j) for (int i=is; i<=ie; ++i) {
    Real err = std::fabs(ir_h(m,a,k,j,i) - b)/b;
    if (err > max_err) max_err = err;
  }

  const Real tol = pin->GetOrAddReal("problem", "tol", 1.0e-12);
  if (max_err > tol) {
    std::cout << "### VET uniform test FAILED: max rel err = " << max_err
              << " > tol " << tol << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::cout << "VET uniform test PASSED: max rel err = " << max_err << std::endl;
}
