//========================================================================================
// AthenaK astrophysical plasma code
// Copyright(C) 2024 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file vet_bench.cpp
//! \brief Throughput benchmark for the short-characteristics sweep.
//!
//! Times VET::FormalSolution() (selected via <nr_radiation>/sweep = wavefront | diagonal |
//! jacobi) on a uniform medium (constant chi, S=b, I=b steady state). Reports ms/sweep and
//! cell*angle updates/s. Also compares wavefront vs diagonal on a non-uniform field by
//! temporarily switching sweep_method. Adapted from apb_rad for the rt-vet API.
//! Run with <time>/evolution=static, nlim=0 — all timing happens here in the pgen.

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "pgen/pgen.hpp"
#include "nr_radiation/nr_radiation.hpp"

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::VETBench()

void ProblemGenerator::VETBench(ParameterInput *pin, const bool restart) {
  (void)restart;
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->pnrrad == nullptr) {
    std::cout << "### FATAL ERROR in vet_bench: requires a <nr_radiation> block"
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
  const int nmb  = pmbp->nmb_thispack;
  const int nmb1 = nmb - 1;
  const int nang_tot = pvet->nang_tot;
  const int nangt1 = nang_tot - 1;

  const Real chi = pin->GetOrAddReal("problem", "chi", 1.0);
  const Real b   = pin->GetOrAddReal("problem", "source", 1.0);
  const int  nbench = pin->GetOrAddInteger("problem", "nbench", 20);

  auto chi_a = pvet->chi;
  auto bb_a  = pvet->bb;
  auto ir    = pvet->ir;

  // uniform opacity and source over the full array (including ghosts)
  par_for("vet_bench_setup", DevExeSpace(), 0, nmb1, 0, (n3-1), 0, (n2-1), 0, (n1-1),
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    chi_a(m,k,j,i) = chi;
    bb_a(m,0,k,j,i)  = b;
  });
  // intensity = b everywhere (incl. ghosts) -> I == b is the exact steady state
  par_for("vet_bench_ir", DevExeSpace(), 0, nmb1, 0, nangt1, 0, (n3-1), 0, (n2-1), 0, (n1-1),
  KOKKOS_LAMBDA(int m, int a, int k, int j, int i) { ir(m,a,k,j,i) = b; });
  Kokkos::fence();

  // --- warmup ---
  pvet->FormalSolution();
  Kokkos::fence();

  // --- timed loop ---
  Kokkos::Timer timer;
  for (int it=0; it<nbench; ++it) {
    pvet->FormalSolution();
  }
  Kokkos::fence();
  const double sec = timer.seconds();

  const int nx1 = indcs.nx1;
  const int nx2 = indcs.nx2;
  const int nx3 = indcs.nx3;
  const double active_cells = static_cast<double>(nmb)*nx1*nx2*nx3;
  const double updates = active_cells*nang_tot*static_cast<double>(nbench);
  const double ms_per_sweep = 1.0e3*sec/nbench;
  const double gcaups = updates/sec/1.0e9;
  const long threads = static_cast<long>(nmb)*nang_tot;

  // --- correctness sanity: I must still equal b ---
  auto ir_h = Kokkos::create_mirror_view(ir);
  Kokkos::deep_copy(ir_h, ir);
  Real max_err = 0.0;
  for (int m=0; m<=nmb1; ++m)
  for (int a=0; a<nang_tot; ++a)
  for (int k=ks; k<=ke; ++k) for (int j=js; j<=je; ++j) for (int i=is; i<=ie; ++i) {
    Real err = std::fabs(ir_h(m,a,k,j,i) - b)/b;
    if (err > max_err) max_err = err;
  }

  // --- wavefront vs diagonal equivalence on a NON-uniform field ---
  par_for("vet_bench_nonuni", DevExeSpace(), 0, nmb1, 0, (n3-1), 0, (n2-1), 0, (n1-1),
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    chi_a(m,k,j,i) = chi*(1.0 + 0.10*(i%3));
    bb_a(m,0,k,j,i)  = b*(1.0 + 0.25*((i%5) + (j%7) + (k%4)));
  });
  auto zero_ir = [&]() {
    par_for("vet_bench_zero", DevExeSpace(), 0, nmb1, 0, nangt1, 0, (n3-1), 0, (n2-1), 0, (n1-1),
    KOKKOS_LAMBDA(int m, int a, int k, int j, int i) { ir(m,a,k,j,i) = 0.0; });
    Kokkos::fence();
  };

  std::string sweep_saved = pvet->sweep_method;
  zero_ir();
  pvet->sweep_method = "wavefront";
  pvet->FormalSolution();
  Kokkos::fence();
  auto irA_h = Kokkos::create_mirror_view(ir);
  Kokkos::deep_copy(irA_h, ir);

  zero_ir();
  pvet->sweep_method = "diagonal";
  pvet->FormalSolution();
  Kokkos::fence();
  auto irW_h = Kokkos::create_mirror_view(ir);
  Kokkos::deep_copy(irW_h, ir);
  pvet->sweep_method = sweep_saved;

  Real maxdiff = 0.0, maxval = 0.0;
  for (int m=0; m<=nmb1; ++m)
  for (int a=0; a<nang_tot; ++a)
  for (int k=ks; k<=ke; ++k) for (int j=js; j<=je; ++j) for (int i=is; i<=ie; ++i) {
    Real d = std::fabs(irA_h(m,a,k,j,i) - irW_h(m,a,k,j,i));
    if (d > maxdiff) maxdiff = d;
    if (std::fabs(irA_h(m,a,k,j,i)) > maxval) maxval = std::fabs(irA_h(m,a,k,j,i));
  }
  const Real aw_reldiff = (maxval > 0.0) ? maxdiff/maxval : 0.0;

  std::cout << "\n============================ VET sweep benchmark ============================\n"
            << "  kernel timed = " << sweep_saved << "\n"
            << "  nmb="    << nmb
            << "  block="  << nx1 << "x" << nx2 << "x" << nx3
            << "  nang_tot=" << nang_tot
            << "  nbench=" << nbench << "\n"
            << "  active cells = " << active_cells
            << "   updates/sweep = " << active_cells*nang_tot << "\n"
            << "  threads exposed (nmb*nang_tot) = " << threads << "\n"
            << std::setprecision(6)
            << "  time/sweep   = " << ms_per_sweep << " ms\n"
            << "  throughput   = " << gcaups << " G(cell*ang)-updates/s\n"
            << "  steady-state max rel err = " << std::scientific << max_err << "\n"
            << "  wavefront-vs-diagonal diff = " << aw_reldiff
            << std::defaultfloat << "\n"
            << "============================================================================\n"
            << std::endl;

  if (!std::isfinite(max_err) || max_err > 1.0e-8) {
    std::cout << "### VET bench FAILED: sweep did not preserve the uniform steady state "
              << "(max rel err = " << max_err << ")" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (!std::isfinite(aw_reldiff) || aw_reldiff > 1.0e-12) {
    std::cout << "### VET bench FAILED: wavefront and diagonal sweep kernels disagree "
              << "(max rel diff = " << aw_reldiff << ")" << std::endl;
    std::exit(EXIT_FAILURE);
  }
}
