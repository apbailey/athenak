//========================================================================================
// AthenaK astrophysical plasma code
// Copyright(C) 2024 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file vet_attenuation.cpp
//! \brief Unit test for the nr_radiation VET 1D short-characteristics sweep.
//!
//! Sets up a 1D uniform slab with constant absorption opacity chi, zero source, and an
//! incoming beam I_bc at the upwind boundary. Pure absorption has the exact solution
//! I(x) = I_bc * exp(-tau), tau = chi * (distance from boundary) / |mu|. Adapted from
//! apb_rad for the rt-vet API (4D chi/bb, octant-indexed mu, FormalSolution).

#include <cmath>
#include <cstdlib>
#include <iostream>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "pgen/pgen.hpp"
#include "nr_radiation/nr_radiation.hpp"

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::VETAttenuation()

void ProblemGenerator::VETAttenuation(ParameterInput *pin, const bool restart) {
  (void)restart;
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->pnrrad == nullptr) {
    std::cout << "### FATAL ERROR in vet_attenuation: requires a <nr_radiation> block"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  nr_radiation::VET *pvet = pmbp->pnrrad;

  auto &indcs = pmy_mesh_->mb_indcs;
  const int is = indcs.is, ie = indcs.ie, js = indcs.js, ks = indcs.ks;
  const int ng = indcs.ng;
  const int n1 = indcs.nx1 + 2*ng;
  const int n2 = (indcs.nx2 > 1) ? indcs.nx2 + 2*ng : 1;
  const int n3 = (indcs.nx3 > 1) ? indcs.nx3 + 2*ng : 1;
  const int nmb1 = pmbp->nmb_thispack - 1;
  const int nang = pvet->pang->nang;
  const int nang_tot = pvet->nang_tot;
  const int nangt1 = nang_tot - 1;

  const Real chi  = pin->GetOrAddReal("problem", "chi", 2.0);
  const Real i_bc = pin->GetOrAddReal("problem", "i_bc", 1.0);

  auto chi_a = pvet->chi;
  auto bb_a  = pvet->bb;
  auto ir    = pvet->ir;
  auto mu    = pvet->pang->mu;

  // constant opacity, zero source over the full array (including ghosts)
  par_for("vet_att_setup", DevExeSpace(), 0, nmb1, 0, (n3-1), 0, (n2-1), 0, (n1-1),
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    chi_a(m,k,j,i) = chi;
    bb_a(m,0,k,j,i)  = 0.0;
  });
  // zero the intensity field
  par_for("vet_att_ir0", DevExeSpace(), 0, nmb1, 0, nangt1, 0, (n3-1), 0, (n2-1), 0, (n1-1),
  KOKKOS_LAMBDA(int m, int a, int k, int j, int i) { ir(m,a,k,j,i) = 0.0; });
  // incoming beam in the upwind ghost cell for each angle (1D: j=js, k=ks)
  // angg = oct*nang + a; mux = mu(oct,a,0)
  const int nang_loc = nang;
  par_for("vet_att_bc", DevExeSpace(), 0, nmb1, 0, nangt1,
  KOKKOS_LAMBDA(int m, int angg) {
    int oct = angg / nang_loc;
    int a   = angg - oct * nang_loc;
    Real mux = mu.d_view(oct, a, 0);
    if (mux > 0.0) { ir(m,angg,ks,js,is-1) = i_bc; }
    else           { ir(m,angg,ks,js,ie+1) = i_bc; }
  });
  Kokkos::fence();

  pvet->FormalSolution();
  Kokkos::fence();

  auto ir_h = Kokkos::create_mirror_view(ir);
  Kokkos::deep_copy(ir_h, ir);
  auto &size = pmbp->pmb->mb_size;
  auto mu_h  = mu.h_view;

  Real max_err = 0.0;
  for (int m=0; m<=nmb1; ++m) {
    const Real dx = size.h_view(m).dx1;
    for (int angg=0; angg<nang_tot; ++angg) {
      int oct = angg / nang;
      int a   = angg - oct * nang;
      const Real mux = mu_h(oct, a, 0);
      for (int i=is; i<=ie; ++i) {
        Real tau = (mux > 0.0) ? chi*(i-is+1)*dx/std::fabs(mux)
                               : chi*(ie+1-i)*dx/std::fabs(mux);
        Real exact = i_bc*std::exp(-tau);
        Real err = std::fabs(ir_h(m,angg,ks,js,i) - exact)/exact;
        if (err > max_err) max_err = err;
      }
    }
  }

  const Real tol = pin->GetOrAddReal("problem", "tol", 1.0e-10);
  if (max_err > tol) {
    std::cout << "### VET attenuation test FAILED: max rel err = " << max_err
              << " > tol " << tol << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::cout << "VET attenuation test PASSED: max rel err = " << max_err << std::endl;
}
