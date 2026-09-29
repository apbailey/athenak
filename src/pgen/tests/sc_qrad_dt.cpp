//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file sc_qrad_dt.cpp
//! \brief Does the timestep bound the gas-energy update that Q_rad drives?
//!
//! Cold uniform gas at rest, lit from one face by a strong incident intensity, with an
//! opacity small enough that the radiation-relaxation limit of sc/newdt.cpp is inert.
//! That limit is
//!
//!   nu_rad = 4(gamma-1) T^3 (sigma_a/rho) crat prat / (1 + 3(chi dx/pi)^2),
//!
//! so it vanishes as sigma_a -> 0 and as T -> 0. It bounds the INTEGRAL form regardless,
//! because there |Q| <= crat prat sigma_a T^4 vanishes with it -- in the thin cooling
//! limit 1/nu_rad = (1/4) e_int/|Q^int| identically. The DIFFERENTIAL form has no such
//! relation: Q = -crat prat div(H) is set by the radiation field, and here the field is
//! the incident beam. So nu_rad -> 0 while |Q| stays large, and one RK stage can take the
//! internal energy negative. See theory/timestep-constraints.md Sec. 4.
//!
//! With <nr_radiation>/cfl_qrad = 0 that is exactly what happens; with the default 0.25
//! for a differential form, dt is cut to cfl_no*cfl_qrad*e_int/|Q| and the energy stays
//! positive. The test asserts both directions, so it fails if either the disease or the
//! cure stops working.
//!
//! Columns of sc_qrad_dt-errs.dat: min internal energy over the domain at the end of the
//! run, min dt taken, max |Q|, and the cycle count. There is deliberately NO crash-guard
//! assertion here: one configuration of this pgen is meant to drive the energy negative.
//! Thresholds live in the pytest wrapper.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"
#include "eos/eos.hpp"
#include "pgen/pgen.hpp"
#include "nr_radiation/nr_radiation.hpp"

namespace {

struct DtVars {
  Real d0, p0, i_inc;
};
DtVars dv;

//----------------------------------------------------------------------------------------
//! \fn void SCQradDtErrors
//! \brief Report the minimum internal energy reached, the minimum dt, and max |Q|.

void SCQradDtErrors(ParameterInput *pin, Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  nr_radiation::SC *psc = pmbp->pnrrad;
  auto &indcs = pm->mb_indcs;
  const int is = indcs.is, js = indcs.js, ks = indcs.ks;
  const int nx1 = indcs.ie - is + 1;
  const int nx2 = indcs.je - js + 1;
  const int nx3 = indcs.ke - ks + 1;
  const int nkji = nx3*nx2*nx1;
  const int nji = nx2*nx1;
  const int nmkji = (pmbp->nmb_thispack)*nkji;

  auto w0 = pmbp->phydro->w0;
  auto qrad_ = psc->qrad;
  Real emin = std::numeric_limits<Real>::max();
  Real qmax = 0.0;
  int nbad = 0;
  Kokkos::parallel_reduce("sc_qdt_err", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int idx, Real &le, Real &lq, int &lb) {
    int m = idx/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    j += js;
    k += ks;
    Real e = w0(m,IEN,k,j,i);
    // NaN-safe: fmin(x, NaN) returns x, so a blown-up run would report a perfectly
    // healthy minimum. Count anything that is not strictly positive instead -- the
    // predicate is false for NaN as well as for e <= 0.
    if (!(e > 0.0)) { lb++; }
    le = fmin(le, e);
    lq = fmax(lq, fabs(qrad_(m,k,j,i)));
  }, Kokkos::Min<Real>(emin), Kokkos::Max<Real>(qmax), Kokkos::Sum<int>(nbad));

#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &emin, 1, MPI_ATHENA_REAL, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &qmax, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &nbad, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#endif

  std::cout << "SC qrad dt (" << psc->qrad_form_name << ", cfl_qrad = " << psc->cfl_qrad
            << "): min e_int = " << emin << ", non-positive/NaN cells = " << nbad
            << ", dt = " << pm->dt << ", max |Q| = " << qmax
            << ", " << pm->ncycle << " cycles" << std::endl;

  if (global_variable::my_rank == 0) {
    std::string fname = "sc_qrad_dt-errs.dat";
    FILE *pf = std::fopen(fname.c_str(), "r");
    if (pf != nullptr) {                       // exists -> append
      pf = std::freopen(fname.c_str(), "a", pf);
    } else {                                   // new -> write header
      pf = std::fopen(fname.c_str(), "w");
      std::fprintf(pf, "# Nx1  Nx2  Nx3   Ncycle   min_eint     dt           "
                       "max_absQ     cfl_qrad     nbad\n");
    }
    std::fprintf(pf, "%04d  %04d  %04d  %05d  %e %e %e %e %d\n",
                 pm->mesh_indcs.nx1, pm->mesh_indcs.nx2, pm->mesh_indcs.nx3, pm->ncycle,
                 emin, pm->dt, qmax, psc->cfl_qrad, nbad);
    std::fclose(pf);
  }
  // no crash-guard: the unguarded configuration is supposed to fail positivity
}
}  // namespace

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::SCQradDt()
//! \brief Cold uniform gas, lit from one face; the energy floor is the observable.

void ProblemGenerator::SCQradDt(ParameterInput *pin, const bool restart) {
  pgen_final_func = SCQradDtErrors;
  dv.d0    = pin->GetOrAddReal("problem", "d0", 1.0);
  dv.p0    = pin->GetOrAddReal("problem", "p0", 1.0e-6);
  dv.i_inc = pin->GetOrAddReal("problem", "i_inc", 1.0);

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->pnrrad == nullptr || pmbp->phydro == nullptr) {
    std::cout << "### FATAL ERROR: sc_qrad_dt needs <hydro> and <nr_radiation>"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (pmy_mesh_->strictly_periodic) {
    std::cout << "### FATAL ERROR: sc_qrad_dt needs a lit physical boundary; a periodic "
              << "mesh has no incident flux to drive div(H)" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  const Real gam = pmbp->phydro->peos->eos_data.gamma;

  // Every face is `inflow`, for both physics. For the gas that means a quiescent ambient
  // state equal to the interior, so the boundary adds no gradient of its own and no
  // pressure floor fires from the ghosts -- `vacuum` would zero the conserved variables
  // and `outflow` would copy whatever the interior has already become. For the radiation
  // it means the ghost intensity is i_in: the incident beam on ix1, dark elsewhere. The
  // heating is then unambiguously driven by the beam.
  auto &u_in = pmbp->phydro->pbval_u->u_in;
  for (int f = 0; f < 6; ++f) {
    u_in.h_view(IDN,f) = dv.d0;
    u_in.h_view(IM1,f) = 0.0;
    u_in.h_view(IM2,f) = 0.0;
    u_in.h_view(IM3,f) = 0.0;
    u_in.h_view(IEN,f) = dv.p0/(gam - 1.0);
  }
  u_in.template modify<HostMemSpace>();
  u_in.template sync<DevExeSpace>();

  auto &i_in = pmbp->pnrrad->pbval_ir->i_in;
  for (int n = 0; n < pmbp->pnrrad->nang_tot; ++n) {
    for (int f = 0; f < 6; ++f) {
      i_in.h_view(n, f) = (f == BoundaryFace::inner_x1) ? dv.i_inc : 0.0;
    }
  }
  i_in.template modify<HostMemSpace>();
  i_in.template sync<DevExeSpace>();

  // u_in and i_in are boundary state, not grid state, and are not written to the restart
  // file -- so they are re-established above on a restart too, before this early return
  if (restart) return;

  auto &indcs = pmy_mesh_->mb_indcs;
  const int ng = indcs.ng;
  const int n1 = indcs.nx1 + 2*ng;
  const int n2 = (indcs.nx2 > 1) ? indcs.nx2 + 2*ng : 1;
  const int n3 = (indcs.nx3 > 1) ? indcs.nx3 + 2*ng : 1;
  const int nmb1 = pmbp->nmb_thispack - 1;
  const Real gm1 = gam - 1.0;
  const Real d0 = dv.d0, p0 = dv.p0;
  auto u0 = pmbp->phydro->u0;

  // uniform and at rest: every gradient in Q comes from the radiation field, so the
  // failure cannot be blamed on the initial hydro state
  par_for("sc_qdt_init", DevExeSpace(), 0, nmb1, 0, (n3-1), 0, (n2-1), 0, (n1-1),
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    u0(m,IDN,k,j,i) = d0;
    u0(m,IM1,k,j,i) = 0.0;
    u0(m,IM2,k,j,i) = 0.0;
    u0(m,IM3,k,j,i) = 0.0;
    u0(m,IEN,k,j,i) = p0/gm1;        // gas at rest: no kinetic part
  });
}
