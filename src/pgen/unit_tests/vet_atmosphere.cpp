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
//! Default: ALI loop in pgen with exchange→FS→J→ALI (matches SolveTransfer).
//! Optional use_solve_transfer: pgen sets ICs only; driver SolveTransfer runs ALI;
//! results checked in Finalize.

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <algorithm>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "bvals/bvals.hpp"
#include "coordinates/cell_locations.hpp"
#include "pgen/pgen.hpp"
#include "nr_radiation/nr_radiation.hpp"

namespace {

struct AtmParams {
  Real chi0, B0, eps0, tol;
  bool require_analytic;
  bool exp_profile;   // false: uniform chi (finite slab); true: chi ~ exp(depth/H)
  Real H;             // scale height for the exponential (Davis 2012 Fig. 4) atmosphere
};

// chi(x) and cumulative optical depth tau(x) measured from the surface at x1min.
//   uniform:     chi = chi0,                        tau = chi0*(x-x1min)
//   exponential: chi = chi0*exp((x-x1min)/H),       tau = chi0*H*(exp((x-x1min)/H)-1)
KOKKOS_INLINE_FUNCTION
Real ChiOfX(bool exp_profile, Real chi0, Real H, Real x, Real x1min) {
  if (exp_profile) return chi0 * Kokkos::exp((x - x1min) / H);
  return chi0;
}
KOKKOS_INLINE_FUNCTION
Real TauOfX(bool exp_profile, Real chi0, Real H, Real x, Real x1min) {
  if (exp_profile) return chi0 * H * (Kokkos::exp((x - x1min) / H) - 1.0);
  return chi0 * (x - x1min);
}

void ReportAtmosphere(Mesh *pm, nr_radiation::VET *pvet, const AtmParams &ap,
                      int niter, Real max_rel, bool converged) {
  auto &indcs = pm->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  MeshBlockPack *pmbp = pm->pmb_pack;
  const int nmb1 = pmbp->nmb_thispack - 1;

  auto jmean_h = Kokkos::create_mirror_view(pvet->jmean);
  auto bb_h = Kokkos::create_mirror_view(pvet->bb);
  Kokkos::deep_copy(jmean_h, pvet->jmean);
  Kokkos::deep_copy(bb_h, pvet->bb);
  auto &size = pmbp->pmb->mb_size;

  Real max_err_J = 0.0, max_err_S = 0.0;
  Real sum_err2_J = 0.0, sum_err2_S = 0.0;
  int npts = 0;
  Real sumJ_dx = 0.0;
  const Real sqrt_eps = std::sqrt(ap.eps0);
  const Real kth = std::sqrt(3.0 * ap.eps0);
  Real x1min_g = pm->mesh_size.x1min;
  Real x1max_g = pm->mesh_size.x1max;
  Real tau_max = TauOfX(ap.exp_profile, ap.chi0, ap.H, x1max_g, x1min_g);
  Real skip_deep = 2.0 / std::max(sqrt_eps, 1.0e-3);
  // surface (min-tau active cell) source function, and a full S/B(tau) profile dump
  Real surf_tau = 1.0e300, surf_SB = 0.0;
  std::vector<std::array<Real,4>> prof;  // (tau, J/B, S/B, San/B)

  for (int m = 0; m <= nmb1; ++m) {
    Real x1min = size.h_view(m).x1min;
    Real x1max = size.h_view(m).x1max;
    Real dx1 = size.h_view(m).dx1;
    for (int k = ks; k <= ke; ++k)
    for (int j = js; j <= je; ++j)
    for (int i = is; i <= ie; ++i) {
      Real x = CellCenterX(i - is, indcs.nx1, x1min, x1max);
      Real tau = TauOfX(ap.exp_profile, ap.chi0, ap.H, x, x1min_g);
      Real J_num = jmean_h(m, k, j, i);
      Real S_num = bb_h(m, 0, k, j, i);
      sumJ_dx += J_num * dx1;

      // analytic (Eddington semi-infinite scattering atmosphere): S/B -> sqrt(eps) surface
      Real J_an = ap.B0 * (1.0 - std::exp(-kth * tau) / (1.0 + sqrt_eps));
      Real S_an = ap.eps0 * ap.B0 + (1.0 - ap.eps0) * J_an;
      if (tau > 0.0) prof.push_back({tau, J_num/ap.B0, S_num/ap.B0, S_an/ap.B0});
      if (tau > 0.0 && tau < surf_tau) { surf_tau = tau; surf_SB = S_num/ap.B0; }

      if (tau < 0.5 || tau > tau_max - skip_deep) continue;
      Real errJ = std::fabs(J_num - J_an) / ap.B0;
      Real errS = std::fabs(S_num - S_an) / ap.B0;
      max_err_J = std::max(max_err_J, errJ);
      max_err_S = std::max(max_err_S, errS);
      sum_err2_J += errJ * errJ;
      sum_err2_S += errS * errS;
      ++npts;
    }
  }
  Real rms_J = (npts > 0) ? std::sqrt(sum_err2_J / npts) : 1.0;
  Real rms_S = (npts > 0) ? std::sqrt(sum_err2_S / npts) : 1.0;

  std::cout << "VET atmosphere (Eq. 30): niter=" << niter
            << " max|dS/S|=" << max_rel
            << " converged=" << (converged ? 1 : 0)
            << " rms|J-Jan|/B=" << rms_J
            << " max|J-Jan|/B=" << max_err_J
            << " rms|S-San|/B=" << rms_S
            << " max|S-San|/B=" << max_err_S
            << " sumJ_dx=" << sumJ_dx
            << " nmb=" << (nmb1+1)
            << " (npts=" << npts << ")" << std::endl;
  std::cout << "  note: Eq. 30 is Eddington analytic; DO SC error is characterization"
            << std::endl;
  // surface sqrt(eps) thermalization law + full S/B(tau) profile dump
  std::cout << "  surface: tau=" << surf_tau << " S/B=" << surf_SB
            << " sqrt(eps)=" << sqrt_eps
            << " ratio(S/B)/sqrt(eps)=" << (sqrt_eps > 0.0 ? surf_SB/sqrt_eps : 0.0)
            << std::endl;
  {
    std::sort(prof.begin(), prof.end(),
      [](const std::array<Real,4>&a, const std::array<Real,4>&b){ return a[0] < b[0]; });
    std::ofstream f("vet_atm_profile.dat");
    f << "# tau  J/B  S/B  San/B   eps=" << ap.eps0 << "\n";
    for (auto &r : prof) f << r[0] << " " << r[1] << " " << r[2] << " " << r[3] << "\n";
  }

  if (!converged) {
    std::cout << "### VET atmosphere FAILED: ALI did not converge (max|dS/S|="
              << max_rel << " tol=" << pvet->ali_tol << ")" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (ap.require_analytic && (rms_J > ap.tol || npts < 4)) {
    std::cout << "### VET atmosphere FAILED: rms|J-Jan|/B = " << rms_J
              << " > tol " << ap.tol << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::cout << "VET atmosphere test PASSED" << std::endl;
}

AtmParams g_atm;
bool g_use_st = false;

void VETAtmosphereFinal(ParameterInput *pin, Mesh *pm) {
  (void)pin;
  if (!g_use_st || pm->pmb_pack->pnrrad == nullptr) return;
  nr_radiation::VET *pvet = pm->pmb_pack->pnrrad;
  bool converged = pvet->cnv_flag;
  ReportAtmosphere(pm, pvet, g_atm, pvet->last_niter, pvet->last_max_rel, converged);
}

}  // namespace

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
  const int ng = indcs.ng;
  const int n1 = indcs.nx1 + 2*ng;
  const int n2 = (indcs.nx2 > 1) ? indcs.nx2 + 2*ng : 1;
  const int n3 = (indcs.nx3 > 1) ? indcs.nx3 + 2*ng : 1;
  const int nmb1 = pmbp->nmb_thispack - 1;
  const int nang_tot = pvet->nang_tot;
  const int nangt1 = nang_tot - 1;

  g_atm.chi0 = pin->GetOrAddReal("problem", "chi", 40.0);
  g_atm.B0   = pin->GetOrAddReal("problem", "source", 1.0);
  g_atm.eps0 = pin->GetReal("nr_radiation", "eps");
  g_atm.tol  = pin->GetOrAddReal("problem", "tol", 5.0e-2);
  g_atm.require_analytic = pin->GetOrAddBoolean("problem", "require_analytic", true);
  g_atm.exp_profile = (pin->GetOrAddString("problem", "profile", "uniform") == "exponential");
  g_atm.H = pin->GetOrAddReal("problem", "scale_height", 1.0);
  g_use_st = pin->GetOrAddBoolean("problem", "use_solve_transfer", false);
  const int maxit = pin->GetOrAddInteger("problem", "max_ali_iters", 5000);

  auto chi_a = pvet->chi;
  auto bb_a  = pvet->bb;
  auto pl_a  = pvet->planck;
  auto eps_a = pvet->eps;
  auto ir    = pvet->ir;
  const Real chi0 = g_atm.chi0;
  const Real B0 = g_atm.B0;
  const Real eps0 = g_atm.eps0;
  const bool exp_p = g_atm.exp_profile;
  const Real Hsc = g_atm.H;
  const Real x1min_g = pmy_mesh_->mesh_size.x1min;
  const int is_l = indcs.is;
  auto &size = pmbp->pmb->mb_size;

  par_for("vet_atm_setup", DevExeSpace(), 0, nmb1, 0, (n3-1), 0, (n2-1), 0, (n1-1),
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real x = size.d_view(m).x1min + (i - is_l + 0.5) * size.d_view(m).dx1;
    chi_a(m,k,j,i) = ChiOfX(exp_p, chi0, Hsc, x, x1min_g);
    pl_a(m,k,j,i)  = B0;
    bb_a(m,0,k,j,i)  = B0;   // cold start S=B
    eps_a(m,k,j,i) = eps0;
  });
  par_for("vet_atm_ir0", DevExeSpace(), 0, nmb1, 0, nangt1, 0, (n3-1), 0, (n2-1), 0, (n1-1),
  KOKKOS_LAMBDA(int m, int a, int k, int j, int i) { ir(m,a,k,j,i) = 0.0; });

  auto &i_in = pvet->i_in;
  for (int n = 0; n < nang_tot; ++n) {
    i_in.h_view(n, BoundaryFace::inner_x1) = 0.0;
    i_in.h_view(n, BoundaryFace::outer_x1) = B0;
  }
  i_in.template modify<HostMemSpace>();
  i_in.template sync<DevMemSpace>();
  Kokkos::fence();

  if (g_use_st) {
    // Production path: SolveTransfer in before_timeintegrator; check in Finalize
    pgen_final_func = VETAtmosphereFinal;
    return;
  }

  // Pgen ALI loop — same ordering as SolveTransfer
  Real max_rel = 1.0;
  int niter = 0;
  for (niter = 0; niter < maxit; ++niter) {
    pvet->ExchangeBoundariesSync();
    pvet->FormalSolution();
    pvet->ComputeJ();
    pvet->UpdateSourceALI(max_rel);
    if (niter + 1 >= pvet->itermin && max_rel <= pvet->ali_tol) break;
  }
  Kokkos::fence();
  bool converged = (niter + 1 >= pvet->itermin && max_rel <= pvet->ali_tol);
  ReportAtmosphere(pmy_mesh_, pvet, g_atm, niter + 1, max_rel, converged);
}
