//========================================================================================
// AthenaK astrophysical plasma code
// Copyright(C) 2024 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file sc_sphere.cpp
//! \brief Homogeneous emitting/absorbing sphere test for the multi-D SC formal solution.
//!
//! Classic short-characteristics benchmark (Davis, Stone & Jiang 2012, ApJS 199, 9): a
//! uniform sphere of radius R with constant absorption chi and constant source S = b sits
//! in vacuum (chi = S = 0 outside, no incoming radiation). For a point at radius r < R the
//! exact specific intensity along a ray whose direction makes cosine mu_r = n.rhat with the
//! outward radial is
//!     I(r, mu_r) = b [ 1 - exp(-chi * s) ],   s = r*mu_r + sqrt(R^2 - r^2 (1 - mu_r^2)),
//! s being the chord length from the sphere entry point to the cell (all inside, since the
//! sphere is convex). The mean intensity J(r) = sum_n wmu_n I(r, mu_r(n)) is compared to
//! the numeric moment. Because the analytic J uses the SAME angular quadrature (identical
//! octant-indexed mu/wmu as SC::CalculateMoments), the angle integration cancels and this
//! isolates the transport + bilinear-interpolation error. S is fixed, so a single sweep on
//! one meshblock is the exact formal solution (no iteration / no block-Jacobi halo). Interior
//! cells (r < r_mask*R) are checked, avoiding the staircased sphere surface. The relative L2
//! and L-infty errors are written to <basename>-errs.dat (RMS-L1 / L-infty columns) so the
//! pytest wrapper runs a ladder of resolutions and checks the error CONVERGES (the SC bilinear
//! sweep is ~1st order here, limited by the chi discontinuity at the staircased surface), not
//! just a single-resolution threshold. A loose in-pgen tol only guards against a grossly
//! broken (or NaN) sweep. Adapted from apb_rad for the rt-vet API (4D chi, srad source iterate,
//! octant-indexed mu/wmu, FormalSolution / CalculateMoments, J from jmean).

#include <cmath>
#include <cstdio>
#include <iostream>
#include <cstdlib>
#include <string>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/cell_locations.hpp"
#include "pgen/pgen.hpp"
#include "nr_radiation/nr_radiation.hpp"

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::SCSphere()

void ProblemGenerator::SCSphere(ParameterInput *pin, const bool restart) {
  (void)restart;
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->pnrrad == nullptr) {
    std::cout << "### FATAL ERROR in sc_sphere: requires a <nr_radiation> block"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  nr_radiation::SC *psc = pmbp->pnrrad;

  auto &indcs = pmy_mesh_->mb_indcs;
  const int is = indcs.is, ie = indcs.ie, js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int ng = indcs.ng;
  const int n1 = indcs.nx1 + 2*ng;
  const int n2 = (indcs.nx2 > 1) ? indcs.nx2 + 2*ng : 1;
  const int n3 = (indcs.nx3 > 1) ? indcs.nx3 + 2*ng : 1;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  const int nmb1 = pmbp->nmb_thispack - 1;
  const int nang = psc->pang->nang;
  const int nang_tot = psc->nang_tot;
  const int nangt1 = nang_tot - 1;

  const Real rsph  = pin->GetOrAddReal("problem", "radius", 0.6);
  const Real chi   = pin->GetOrAddReal("problem", "chi", 5.0);
  const Real b     = pin->GetOrAddReal("problem", "source", 1.0);
  const Real cx    = pin->GetOrAddReal("problem", "x0", 0.0);
  const Real cy    = pin->GetOrAddReal("problem", "y0", 0.0);
  const Real cz    = pin->GetOrAddReal("problem", "z0", 0.0);

  auto chi_a = psc->chi;
  auto srad_a  = psc->srad;
  auto ir    = psc->ir;
  auto &size = pmbp->pmb->mb_size;

  // ---- set opacity/source: uniform sphere in vacuum (over full array incl. ghosts) ----
  const Real rsph2 = rsph*rsph;
  par_for("sc_sph_setup", DevExeSpace(), 0,nmb1, 0,n3-1, 0,n2-1, 0,n1-1,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real x = CellCenterX(i-is, nx1, size.d_view(m).x1min, size.d_view(m).x1max) - cx;
    Real y = (nx2 > 1) ? CellCenterX(j-js, nx2, size.d_view(m).x2min,
                                     size.d_view(m).x2max) - cy : 0.0;
    Real z = (nx3 > 1) ? CellCenterX(k-ks, nx3, size.d_view(m).x3min,
                                     size.d_view(m).x3max) - cz : 0.0;
    bool inside = (x*x + y*y + z*z) < rsph2;
    chi_a(m,k,j,i)   = inside ? chi : 0.0;
    srad_a(m,0,k,j,i)  = inside ? b   : 0.0;
  });
  // vacuum: no incoming radiation (interior + ghosts start at zero)
  par_for("sc_sph_ir0", DevExeSpace(), 0,nmb1, 0,nangt1, 0,n3-1, 0,n2-1, 0,n1-1,
  KOKKOS_LAMBDA(int m, int a, int k, int j, int i) { ir(m,a,k,j,i) = 0.0; });
  Kokkos::fence();

  // ---- one formal solution (fixed source -> exact on a single block) + moments ----
  psc->FormalSolution();
  Kokkos::fence();
  psc->CalculateMoments();
  Kokkos::fence();

  // ---- compare numeric J(r) to the analytic angular quadrature over interior cells ----
  auto jmean_h = Kokkos::create_mirror_view(psc->jmean);
  Kokkos::deep_copy(jmean_h, psc->jmean);
  auto mu_h  = psc->pang->mu.h_view;
  auto wmu_h = psc->pang->wmu.h_view;

  const Real r_mask = pin->GetOrAddReal("problem", "r_mask", 0.6);
  const Real rchk2 = (r_mask*rsph)*(r_mask*rsph);

  Real max_err = 0.0, l2_num = 0.0, l2_den = 0.0;
  int ncheck = 0;
  for (int m=0; m<=nmb1; ++m) {
    for (int k=ks; k<=ke; ++k) for (int j=js; j<=je; ++j) for (int i=is; i<=ie; ++i) {
      Real x = CellCenterX(i-is, nx1, size.h_view(m).x1min, size.h_view(m).x1max) - cx;
      Real y = (nx2 > 1) ? CellCenterX(j-js, nx2, size.h_view(m).x2min,
                                       size.h_view(m).x2max) - cy : 0.0;
      Real z = (nx3 > 1) ? CellCenterX(k-ks, nx3, size.h_view(m).x3min,
                                       size.h_view(m).x3max) - cz : 0.0;
      Real r2 = x*x + y*y + z*z;
      if (r2 >= rchk2) continue;                       // interior mask
      Real r = std::sqrt(r2);

      // analytic mean intensity: quadrature of I(r, mu_r) over the SAME octant-indexed angles
      // as CalculateMoments (wmu[a], mu[oct,a,:], angg = oct*nang + a), so the angle
      // integration cancels. rt-vet keeps mu a full 3D unit vector even in 2D (|mu| = 1), but
      // the sweep only marches the ACTIVE spatial dims, so the transport path scales with the
      // in-plane speed sqrt(mux^2+muy^2). Hence mnorm and mu_r are taken over the active dims
      // only (mu_z excluded when nx3==1); in 3D all dims are active and this reduces to the
      // plain sphere. tau = chi * (unit-dir chord) / mnorm (the sweep path is chord/mnorm).
      Real j_ana = 0.0;
      for (int angg=0; angg<nang_tot; ++angg) {
        int oct = angg / nang;
        int a   = angg - oct * nang;
        Real nx = mu_h(oct, a, 0);
        Real ny = (nx2 > 1) ? mu_h(oct, a, 1) : 0.0;
        Real nz = (nx3 > 1) ? mu_h(oct, a, 2) : 0.0;
        Real w  = wmu_h(a);
        Real mnorm = std::sqrt(nx*nx + ny*ny + nz*nz);  // norm over ACTIVE spatial dims
        Real mu_r = (r > 1.0e-12) ? (nx*x + ny*y + nz*z)/(r*mnorm) : 0.0;
        Real s = r*mu_r + std::sqrt(rsph2 - r2*(1.0 - mu_r*mu_r));  // unit-dir chord
        Real tau = chi*s/mnorm;                                    // sweep path = chord/mnorm
        j_ana += w * b*(1.0 - std::exp(-tau));
      }
      Real j_num = jmean_h(m,k,j,i);
      Real err = std::fabs(j_num - j_ana)/j_ana;
      if (err > max_err) max_err = err;
      l2_num += (j_num - j_ana)*(j_num - j_ana);
      l2_den += j_ana*j_ana;
      ncheck++;
    }
  }
  Real l2_err = (l2_den > 0.0) ? std::sqrt(l2_num/l2_den) : 0.0;

  std::cout << "SC sphere: checked " << ncheck << " interior cells (r < " << r_mask
            << "R), J rel err  max = " << max_err << "  L2 = " << l2_err << std::endl;

  // write relative L2 (RMS-L1 column) and L-infty errors for the convergence ladder
  if (global_variable::my_rank == 0) {
    std::string fname = pin->GetString("job", "basename") + "-errs.dat";
    FILE *pf = std::fopen(fname.c_str(), "r");
    if (pf != nullptr) {                       // exists -> append
      pf = std::freopen(fname.c_str(), "a", pf);
    } else {                                   // new -> write header
      pf = std::fopen(fname.c_str(), "w");
      std::fprintf(pf, "# Nx1  Nx2  Nx3   Ncycle   RMS-L1       L-infty\n");
    }
    std::fprintf(pf, "%04d  %04d  %04d  %05d  %e %e\n", indcs.nx1, indcs.nx2, indcs.nx3,
                 0, l2_err, max_err);
    std::fclose(pf);
  }

  // loose crash-guard only; the real convergence check lives in the pytest wrapper
  const Real tol = pin->GetOrAddReal("problem", "tol", 1.0e-1);
  if (!(l2_err < tol)) {                        // also fails on NaN
    std::cout << "### SC sphere test FAILED (crash-guard): L2 = " << l2_err
              << " not < " << tol << std::endl;
    std::exit(EXIT_FAILURE);
  }
  return;
}
