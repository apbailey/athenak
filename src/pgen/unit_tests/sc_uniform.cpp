//========================================================================================
// AthenaK astrophysical plasma code
// Copyright(C) 2024 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file sc_uniform.cpp
//! \brief Machine-precision structural test for the multi-D SC sweep (FormalSolution).
//!
//! Sets a uniform medium: constant absorption chi, constant source S = b everywhere
//! (including ghosts), and the intensity initialized to b everywhere. One sweep must
//! leave I == b to machine precision for every cell and angle. This holds iff the SC
//! quadrature reproduces a constant source (edtau + a0 + a1 + a2 = 1) AND the bilinear
//! upwind-face weights form a partition of unity. Adapted from apb_rad for the rt-vet
//! API (4D chi/srad, FormalSolution dispatcher).
//!
//! With multilevel meshes, Driver::Finalize also exercises RestrictCC/ProlongateCC of ir
//! (MeshRefinement exists only after ProblemGenerator returns).

#include <cmath>
#include <cstdlib>
#include <iostream>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mesh/mesh_refinement.hpp"
#include "pgen/pgen.hpp"
#include "nr_radiation/nr_radiation.hpp"

namespace {

Real MaxRelErrIr(Mesh *pm, nr_radiation::SC *psc, Real b) {
  auto &indcs = pm->mb_indcs;
  const int is = indcs.is, ie = indcs.ie, js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb1 = pm->pmb_pack->nmb_thispack - 1;
  const int nang_tot = psc->nang_tot;
  auto ir_h = Kokkos::create_mirror_view(psc->ir);
  Kokkos::deep_copy(ir_h, psc->ir);
  Real max_err = 0.0;
  for (int m=0; m<=nmb1; ++m)
  for (int a=0; a<nang_tot; ++a)
  for (int k=ks; k<=ke; ++k) for (int j=js; j<=je; ++j) for (int i=is; i<=ie; ++i) {
    Real err = std::fabs(ir_h(m,a,k,j,i) - b)/b;
    if (err > max_err) max_err = err;
  }
  return max_err;
}

void SCUniformFinal(ParameterInput *pin, Mesh *pm) {
  if (!pm->multilevel || pm->pmr == nullptr || pm->pmb_pack->pnrrad == nullptr) return;
  nr_radiation::SC *psc = pm->pmb_pack->pnrrad;
  const Real b = pin->GetOrAddReal("problem", "source", 1.0);
  const Real tol = pin->GetOrAddReal("problem", "tol", 1.0e-12);

  // Restrict → fill coarse ghosts → prolongate for ir and srad (same order as sc_bvals)
  pm->pmr->RestrictCC(psc->ir, psc->coarse_ir);
  psc->pbval_ir->FillCoarseInBndryCC(psc->ir, psc->coarse_ir);
  psc->pbval_ir->ProlongateCC(psc->ir, psc->coarse_ir);
  pm->pmr->RestrictCC(psc->srad, psc->coarse_srad);
  psc->pbval_srad->FillCoarseInBndryCC(psc->srad, psc->coarse_srad);
  psc->pbval_srad->ProlongateCC(psc->srad, psc->coarse_srad);
  Kokkos::fence();

  Real max_err = MaxRelErrIr(pm, psc, b);
  if (max_err > tol) {
    std::cout << "### SC uniform SMR FAILED after Restrict/Prolong: max rel err = "
              << max_err << " > tol " << tol << std::endl;
    std::exit(EXIT_FAILURE);
  }
  // constant-S identity for srad after Restrict/Prolong
  auto &indcs = pm->mb_indcs;
  const int is = indcs.is, ie = indcs.ie, js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb1 = pm->pmb_pack->nmb_thispack - 1;
  auto srad_h = Kokkos::create_mirror_view(psc->srad);
  Kokkos::deep_copy(srad_h, psc->srad);
  Real max_err_bb = 0.0;
  for (int m = 0; m <= nmb1; ++m)
  for (int k = ks; k <= ke; ++k)
  for (int j = js; j <= je; ++j)
  for (int i = is; i <= ie; ++i) {
    max_err_bb = std::max(max_err_bb, std::fabs(srad_h(m,0,k,j,i) - b) / b);
  }
  if (max_err_bb > tol) {
    std::cout << "### SC uniform SMR FAILED srad Restrict/Prolong: max rel err = "
              << max_err_bb << " > tol " << tol << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::cout << "SC uniform SMR Restrict/Prolong PASSED: max rel err(ir) = "
            << max_err << " max rel err(srad) = " << max_err_bb << std::endl;
}

}  // namespace

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::SCUniform()

void ProblemGenerator::SCUniform(ParameterInput *pin, const bool restart) {
  (void)restart;
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->pnrrad == nullptr) {
    std::cout << "### FATAL ERROR in sc_uniform: requires a <nr_radiation> block"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  nr_radiation::SC *psc = pmbp->pnrrad;

  auto &indcs = pmy_mesh_->mb_indcs;
  const int ng = indcs.ng;
  const int n1 = indcs.nx1 + 2*ng;
  const int n2 = (indcs.nx2 > 1) ? indcs.nx2 + 2*ng : 1;
  const int n3 = (indcs.nx3 > 1) ? indcs.nx3 + 2*ng : 1;
  const int nmb1 = pmbp->nmb_thispack - 1;
  const int nang_tot = psc->nang_tot;
  const int nangt1 = nang_tot - 1;

  const Real chi = pin->GetOrAddReal("problem", "chi", 5.0);
  const Real b   = pin->GetOrAddReal("problem", "source", 1.0);

  auto chi_a = psc->chi;
  auto srad_a  = psc->srad;
  auto ir    = psc->ir;

  // uniform opacity and source over the full array (including ghosts)
  par_for("sc_uni_setup", DevExeSpace(), 0, nmb1, 0, (n3-1), 0, (n2-1), 0, (n1-1),
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    chi_a(m,k,j,i) = chi;
    srad_a(m,0,k,j,i)  = b;
  });
  // intensity initialized to b everywhere (incl. ghosts -> upwind boundary is also b)
  par_for("sc_uni_ir", DevExeSpace(), 0, nmb1, 0, nangt1, 0, (n3-1), 0, (n2-1), 0, (n1-1),
  KOKKOS_LAMBDA(int m, int a, int k, int j, int i) { ir(m,a,k,j,i) = b; });
  Kokkos::fence();

  psc->FormalSolution();
  Kokkos::fence();

  const Real tol = pin->GetOrAddReal("problem", "tol", 1.0e-12);
  Real max_err = MaxRelErrIr(pmy_mesh_, psc, b);
  if (max_err > tol) {
    std::cout << "### SC uniform test FAILED: max rel err = " << max_err
              << " > tol " << tol << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::cout << "SC uniform test PASSED: max rel err = " << max_err << std::endl;

  // MeshRefinement is constructed after ProblemGenerator; exercise multilevel in Finalize
  if (pmy_mesh_->multilevel) {
    pgen_final_func = SCUniformFinal;
  }
}
