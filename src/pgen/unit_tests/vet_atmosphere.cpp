//========================================================================================
// AthenaK astrophysical plasma code
// Copyright(C) 2024 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file vet_atmosphere.cpp
//! \brief Non-LTE atmosphere validation of Jacobi-ALI (Davis 2012 Eq. 30).
//!
//! 1D semi-infinite setup following Athena-C radtest: constant χ and ε, B=1,
//! vacuum at τ=0 (inner_x1 inflow I=0), thermalized deep boundary (outer_x1 inflow I=B).
//! Solves S=(1−ε)Λ[S]+εB via Jacobi-ALI and compares J/B to:
//!   J/B = 1 − exp(−√(3ε) τ) / (1+√ε) ,  τ measured from the left surface.

#include <cmath>
#include <cstdlib>
#include <iostream>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "bvals/bvals.hpp"
#include "coordinates/cell_locations.hpp"
#include "pgen/pgen.hpp"
#include "nr_radiation/nr_radiation.hpp"

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::VETAtmosphere()

void ProblemGenerator::VETAtmosphere(ParameterInput *pin, const bool restart) {
  (void)restart;
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->pnrrad == nullptr) {
    std::cout << "### FATAL ERROR in vet_atmosphere: requires <nr_radiation>"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  nr_radiation::VET *pvet = pmbp->pnrrad;
  if (!pvet->use_ali) {
    std::cout << "### FATAL ERROR in vet_atmosphere: requires eps < 1 (ALI path)"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  auto &indcs = pmy_mesh_->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, ks = indcs.ks;
  const int ng = indcs.ng;
  const int n1 = indcs.nx1 + 2*ng;
  const int n2 = (indcs.nx2 > 1) ? indcs.nx2 + 2*ng : 1;
  const int n3 = (indcs.nx3 > 1) ? indcs.nx3 + 2*ng : 1;
  const int nmb1 = pmbp->nmb_thispack - 1;
  const int nang_tot = pvet->nang_tot;
  const int nangt1 = nang_tot - 1;

  const Real chi0 = pin->GetOrAddReal("problem", "chi", 40.0);
  const Real B0   = pin->GetOrAddReal("problem", "source", 1.0);
  const Real eps0 = pin->GetReal("nr_radiation", "eps");
  const Real tol  = pin->GetOrAddReal("problem", "tol", 5.0e-2);
  const int maxit = pin->GetOrAddInteger("problem", "max_ali_iters", 5000);

  auto chi_a = pvet->chi;
  auto bb_a  = pvet->bb;
  auto pl_a  = pvet->planck;
  auto eps_a = pvet->eps;
  auto ir    = pvet->ir;

  par_for("vet_atm_setup", DevExeSpace(), 0, nmb1, 0, (n3-1), 0, (n2-1), 0, (n1-1),
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    chi_a(m,k,j,i) = chi0;
    pl_a(m,k,j,i)  = B0;
    bb_a(m,k,j,i)  = B0;   // cold start S=B
    eps_a(m,k,j,i) = eps0;
  });
  par_for("vet_atm_ir0", DevExeSpace(), 0, nmb1, 0, nangt1, 0, (n3-1), 0, (n2-1), 0, (n1-1),
  KOKKOS_LAMBDA(int m, int a, int k, int j, int i) { ir(m,a,k,j,i) = 0.0; });

  // Athena-C radtest BCs: τ=0 vacuum (I=0), deep end thermalized (I=B)
  auto &i_in = pvet->i_in;
  for (int n = 0; n < nang_tot; ++n) {
    i_in.h_view(n, BoundaryFace::inner_x1) = 0.0;
    i_in.h_view(n, BoundaryFace::outer_x1) = B0;
  }
  i_in.template modify<HostMemSpace>();
  i_in.template sync<DevMemSpace>();
  Kokkos::fence();

  // Jacobi-ALI loop (single MeshBlock: physical BCs only, no MPI)
  Real max_rel = 1.0;
  int niter = 0;
  pvet->ApplyPhysicalBCs();
  for (niter = 0; niter < maxit; ++niter) {
    pvet->FormalSolution();
    pvet->ApplyPhysicalBCs();
    pvet->ComputeJ();
    pvet->UpdateSourceALI(max_rel);
    if (niter + 1 >= pvet->itermin && max_rel <= pvet->ali_tol) break;
  }
  Kokkos::fence();

  // Compare J to Eq. 30 (τ from left surface at x1min)
  auto jmean_h = Kokkos::create_mirror_view(pvet->jmean);
  Kokkos::deep_copy(jmean_h, pvet->jmean);
  auto &size = pmbp->pmb->mb_size;
  Real x1min = size.h_view(0).x1min;
  Real x1max = size.h_view(0).x1max;
  Real tau_max = chi0 * (x1max - x1min);

  Real max_err = 0.0;
  Real sum_err2 = 0.0;
  int npts = 0;
  const Real sqrt_eps = std::sqrt(eps0);
  const Real kth = std::sqrt(3.0 * eps0);
  for (int i = is; i <= ie; ++i) {
    Real x = CellCenterX(i - is, indcs.nx1, x1min, x1max);
    Real tau = chi0 * (x - x1min);
    // skip near the free surface; keep away from deep boundary (≥2/√ε from τ_max)
    Real skip_deep = 2.0 / std::max(sqrt_eps, 1.0e-3);
    if (tau < 0.5 || tau > tau_max - skip_deep) continue;
    Real J_an = B0 * (1.0 - std::exp(-kth * tau) / (1.0 + sqrt_eps));
    Real J_num = jmean_h(0, ks, js, i);
    Real err = std::fabs(J_num - J_an) / B0;
    max_err = std::max(max_err, err);
    sum_err2 += err * err;
    ++npts;
  }
  Real rms = (npts > 0) ? std::sqrt(sum_err2 / npts) : 1.0;

  std::cout << "VET atmosphere (Eq. 30): niter=" << (niter+1)
            << " max|dS/S|=" << max_rel
            << " rms|J-Jan|/B=" << rms
            << " max|J-Jan|/B=" << max_err
            << " (npts=" << npts << ")" << std::endl;

  if (rms > tol || npts < 4) {
    std::cout << "### VET atmosphere FAILED: rms err = " << rms
              << " > tol " << tol << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::cout << "VET atmosphere test PASSED" << std::endl;
}
