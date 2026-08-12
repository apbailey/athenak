//========================================================================================
// AthenaK astrophysical plasma code
// Copyright(C) 2024 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file sc_lag.cpp
//! \brief Boundary-lag probe: how many SC iterations does a domain decomposition cost?
//!
//! The SC sweep is meshblock-LOCAL. Light crosses a block boundary only through the ghost
//! exchange at the top of each iteration (iteration.cpp SolveTransfer), so information
//! advances at most ONE meshblock per iteration. Every throughput number in rt-profiling was
//! taken at iter_max=1, which makes that cost invisible. This pgen measures it.
//!
//! Setup: uniform emitting medium (chi, S = b const) in a cube with VACUUM boundaries
//! (ix/ox_bc = inflow with the default i_in = 0). The intensity is initialised to b — the
//! infinite-medium answer, which is wrong near the boundaries. The true solution is
//! I = b(1 - e^{-tau}) with tau measured from the boundary along the ray, so a "deficit wave"
//! has to propagate inward from every face. With the sweep confined to a meshblock, that wave
//! advances one block per iteration.
//!
//! Prediction: niter ~ min(blocks a ray must cross, the optical-depth horizon in blocks),
//! i.e. the decomposition penalty should be real when the medium is thin and should saturate
//! when it is thick (light is absorbed before it can cross many blocks). The controlling
//! parameter is the optical depth PER MESHBLOCK, tau_blk = chi * L / nblk_per_side.
//!
//! LTE only (eps = 1, ops = 0): the source function is fixed, so the ONLY reason to iterate is
//! ghost propagation. That isolates the mechanism from scattering convergence.
//!
//! The iteration loop below reproduces SolveTransfer's LTE branch exactly — same ordering
//! (exchange -> FormalSolution -> ComputeJ), same residual max|dJ/J| from the same arrays, same
//! itermin/tol stopping test — so the reported count is the count a production run would pay.
//! Emits one machine-readable RESULT line for rt-profiling/lag_study.py.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "pgen/pgen.hpp"
#include "nr_radiation/nr_radiation.hpp"

namespace {

//! \brief max|dJ/J| between jmean and jmean_old — a copy of the LTE residual in
//! SolveTransfer (iteration.cpp), including its dJmax==0 && dJabs>0 -> 1.0 guard.
Real LTEResidual(Mesh *pm, nr_radiation::SC *psc) {
  auto &indcs = pm->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb1 = pm->pmb_pack->nmb_thispack - 1;
  const int nx1 = ie - is + 1, nx2 = je - js + 1, nx3 = ke - ks + 1;
  const int nmkji = (nmb1 + 1) * nx3 * nx2 * nx1;
  auto jmean_ = psc->jmean;
  auto jold_ = psc->jmean_old;

  Real dJmax = 0.0, dJabs_max = 0.0;
  Kokkos::parallel_reduce("lag_dj", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int idx, Real &lmax, Real &labs) {
    int m = idx / (nx3*nx2*nx1);
    int kji = idx - m*(nx3*nx2*nx1);
    int k = kji / (nx2*nx1);
    int ji = kji - k*(nx2*nx1);
    int j = ji / nx1;
    int i = ji - j*nx1 + is;
    j += js;
    k += ks;
    Real Jo = jold_(m,k,j,i);
    Real Jn = jmean_(m,k,j,i);
    Real dJ = fabs(Jn - Jo);
    labs = fmax(labs, dJ);
    lmax = fmax(lmax, (Jo > 0.0) ? (dJ / Jo) : 0.0);
  }, Kokkos::Max<Real>(dJmax), Kokkos::Max<Real>(dJabs_max));

#if MPI_PARALLEL_ENABLED
  Real buf[2] = {dJmax, dJabs_max};
  MPI_Allreduce(MPI_IN_PLACE, buf, 2, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
  dJmax = buf[0];
  dJabs_max = buf[1];
#endif
  if (dJmax == 0.0 && dJabs_max > 0.0) dJmax = 1.0;
  return dJmax;
}

//! \brief Volume-averaged J over the active zones (a decomposition-invariant physics check:
//! the converged answer must not depend on how the domain was cut up).
Real MeanJ(Mesh *pm, nr_radiation::SC *psc) {
  auto &indcs = pm->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb1 = pm->pmb_pack->nmb_thispack - 1;
  const int nx1 = ie - is + 1, nx2 = je - js + 1, nx3 = ke - ks + 1;
  const int nmkji = (nmb1 + 1) * nx3 * nx2 * nx1;
  auto jmean_ = psc->jmean;

  Real sum = 0.0;
  Kokkos::parallel_reduce("lag_meanj", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int idx, Real &ls) {
    int m = idx / (nx3*nx2*nx1);
    int kji = idx - m*(nx3*nx2*nx1);
    int k = kji / (nx2*nx1);
    int ji = kji - k*(nx2*nx1);
    int j = ji / nx1;
    int i = ji - j*nx1 + is;
    ls += jmean_(m,k+ks,j+js,i);
  }, sum);

  Real cnt = static_cast<Real>(nmkji);
#if MPI_PARALLEL_ENABLED
  Real buf[2] = {sum, cnt};
  MPI_Allreduce(MPI_IN_PLACE, buf, 2, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  sum = buf[0];
  cnt = buf[1];
#endif
  return (cnt > 0.0) ? sum / cnt : 0.0;
}

}  // namespace

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::SCLag

void ProblemGenerator::SCLag(ParameterInput *pin, const bool restart) {
  (void)restart;
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->pnrrad == nullptr) {
    std::cout << "### FATAL ERROR in sc_lag: requires a <nr_radiation> block" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  nr_radiation::SC *psc = pmbp->pnrrad;
  if (psc->use_ali) {
    std::cout << "### FATAL ERROR in sc_lag: LTE probe requires ops=0 and eps=1 "
              << "(use_ali must be false, so the only reason to iterate is ghost propagation)"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  auto &indcs = pmy_mesh_->mb_indcs;
  const int ng = indcs.ng;
  const int n1 = indcs.nx1 + 2*ng;
  const int n2 = (indcs.nx2 > 1) ? indcs.nx2 + 2*ng : 1;
  const int n3 = (indcs.nx3 > 1) ? indcs.nx3 + 2*ng : 1;
  const int nmb1 = pmbp->nmb_thispack - 1;
  const int nangt1 = psc->nang_tot - 1;

  const Real chi = pin->GetOrAddReal("problem", "chi", 1.0);
  const Real b = pin->GetOrAddReal("problem", "source", 1.0);

  auto chi_a = psc->chi;
  auto srad_a = psc->srad;
  auto planck_a = psc->planck;
  auto ir_a = psc->ir;

  // Uniform medium over the whole array incl. ghosts. planck is set alongside srad so that
  // any later UpdateOpacityAndSource call would be a no-op on this state.
  par_for("sc_lag_setup", DevExeSpace(), 0, nmb1, 0, (n3-1), 0, (n2-1), 0, (n1-1),
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    chi_a(m,k,j,i) = chi;
    srad_a(m,0,k,j,i) = b;
    planck_a(m,k,j,i) = b;
  });
  // Intensity starts at the infinite-medium answer; the vacuum boundaries make it wrong near
  // every face, and that error is what has to propagate inward one meshblock per iteration.
  par_for("sc_lag_ir", DevExeSpace(), 0, nmb1, 0, nangt1, 0, (n3-1), 0, (n2-1), 0, (n1-1),
  KOKKOS_LAMBDA(int m, int a, int k, int j, int i) { ir_a(m,a,k,j,i) = b; });
  Kokkos::fence();

  // ---- LTE iteration, identical in ordering and stopping test to SolveTransfer ----
  const int iter_max = psc->iter_max;
  const int itermin = psc->itermin;
  const Real tol = psc->iter_tol;

  Kokkos::deep_copy(DevExeSpace(), psc->jmean, 0.0);
  int niter = 0;
  Real max_rel = std::numeric_limits<Real>::max();
  bool converged = false;

  for (int it = 0; it < iter_max; ++it) {
    Kokkos::deep_copy(DevExeSpace(), psc->jmean_old, psc->jmean);
    psc->ExchangeBoundariesSync();
    psc->FormalSolution();
    psc->ComputeJ();
    max_rel = LTEResidual(pmy_mesh_, psc);
    niter = it + 1;
    if (niter >= itermin && max_rel <= tol) { converged = true; break; }
  }

  const Real jbar = MeanJ(pmy_mesh_, psc);

  if (global_variable::my_rank == 0) {
    const int nb1 = pmy_mesh_->mesh_indcs.nx1 / indcs.nx1;   // meshblocks per side (x1)
    const Real len = pmy_mesh_->mesh_size.x1max - pmy_mesh_->mesh_size.x1min;
    const Real tau_dom = chi * len;
    std::printf("SC_LAG RESULT nx1=%d B=%d nblk_side=%d nmb=%d nmu=%d nang=%d "
                "chi=%.6g tau_dom=%.6g tau_blk=%.6g niter=%d converged=%d "
                "res=%.6e jbar=%.10e tol=%.3g\n",
                pmy_mesh_->mesh_indcs.nx1, indcs.nx1, nb1, pmy_mesh_->nmb_total,
                pin->GetInteger("nr_radiation", "nmu"), psc->nang_tot,
                chi, tau_dom, tau_dom / static_cast<Real>(nb1),
                niter, converged ? 1 : 0, max_rel, jbar, tol);
    std::fflush(stdout);
    if (!converged) {
      std::cout << "### WARNING in sc_lag: not converged in " << iter_max
                << " iterations (residual " << max_rel << " > tol " << tol << ")" << std::endl;
    }
  }
}
