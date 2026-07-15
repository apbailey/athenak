//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file vet_newdt.cpp
//! \brief Radiation-relaxation timestep for operator-split Q_rad coupling.
//! Direct port of Athena-C radiation/radtrans_dt.c (used with Davis 2012 Eq. 26
//! heating/cooling). Sets RadiationVET::dtnew = 1/nu_rad (Mesh::NewTimeStep multiplies
//! by cfl_no) and clamps pmesh->dt for the current cycle so SolveTransfer → AddQrad
//! see a stable step.

#include <algorithm>
#include <cmath>
#include <limits>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "eos/eos.hpp"
#include "radiation_vet.hpp"

namespace radiation_vet {

//----------------------------------------------------------------------------------------
//! \fn void RadiationVET::UpdateTimeStep
//! \brief Athena-C radtrans_dt: Courant-like constraint on the radiation thermal
//! relaxation rate
//!   nu_rad = [16*pi*Gamma_1*eps*B*chi/(R_ideal*T*rho)] / (1 + 3*(dxmin*chi/pi)^2)
//! with CPrat=1 (Davis units). Without this limit, explicit Eq. 26 coupling is stiff
//! for Bo~O(1), tau~O(1) linear waves and the hydro CFL alone is unsafe.

void RadiationVET::UpdateTimeStep() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  int nx1 = ie - is + 1, nx2 = je - js + 1, nx3 = ke - ks + 1;
  int nmkji = (nmb1 + 1) * nx3 * nx2 * nx1;

  auto &mbsize = pmy_pack->pmb->mb_size;
  Real dxmin = std::numeric_limits<Real>::max();
  for (int m = 0; m <= nmb1; ++m) {
    dxmin = std::min(dxmin, mbsize.h_view(m).dx1);
    if (pmy_pack->pmesh->multi_d) {
      dxmin = std::min(dxmin, mbsize.h_view(m).dx2);
    }
    if (pmy_pack->pmesh->three_d) {
      dxmin = std::min(dxmin, mbsize.h_view(m).dx3);
    }
  }
  const Real qa = 3.0 * (dxmin * dxmin) / (M_PI * M_PI);

  // Gamma_1 = gamma - 1, R_ideal = 1 (Athena-C linear_wave_rad2d units)
  Real gamma = 5.0 / 3.0;
  if (pmy_pack->phydro != nullptr) {
    gamma = pmy_pack->phydro->peos->eos_data.gamma;
  } else if (pmy_pack->pmhd != nullptr) {
    gamma = pmy_pack->pmhd->peos->eos_data.gamma;
  }
  const Real gm1 = gamma - 1.0;
  const Real nu_con = 16.0 * M_PI * gm1;  // CPrat=1, R_ideal=1

  DvceArray5D<Real> w0;
  if (pmy_pack->phydro != nullptr) {
    w0 = pmy_pack->phydro->w0;
  } else if (pmy_pack->pmhd != nullptr) {
    w0 = pmy_pack->pmhd->w0;
  } else {
    dtnew = std::numeric_limits<Real>::max();
    return;
  }

  auto chi_ = chi;
  auto bb_ = bb;
  Real dt_min = std::numeric_limits<Real>::max();
  Kokkos::parallel_reduce("vet_rad_dt",
    Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
    KOKKOS_LAMBDA(const int idx, Real &ldt) {
      int m = idx / (nx3 * nx2 * nx1);
      int kji = idx - m * (nx3 * nx2 * nx1);
      int k = kji / (nx2 * nx1);
      int ji = kji - k * (nx2 * nx1);
      int j = ji / nx1;
      int i = ji - j * nx1 + is;
      j += js;
      k += ks;
      Real dens = w0(m, IDN, k, j, i);
      if (dens <= 0.0) return;
      // primitives store internal energy density e at IEN; T = (γ−1) e / ρ
      Real temp = gm1 * w0(m, IEN, k, j, i) / dens;
      if (temp <= 0.0) return;
      Real chiv = chi_(m, k, j, i);
      Real Bv = bb_(m, k, j, i);
      // eps=1 (LTE)
      Real nu_rad = nu_con * Bv * chiv / (temp * dens);
      nu_rad /= (1.0 + qa * chiv * chiv);
      if (nu_rad > 0.0) {
        ldt = fmin(ldt, 1.0 / nu_rad);
      }
    }, Kokkos::Min<Real>(dt_min));

  dtnew = dt_min;

  // Current-cycle clamp (Athena-C updates Mesh.dt before rad_to_hydro)
  Real cfl = pmy_pack->pmesh->cfl_no;
  if (dt_min < std::numeric_limits<Real>::max()) {
    pmy_pack->pmesh->dt = std::min(pmy_pack->pmesh->dt, cfl * dt_min);
  }
}

}  // namespace radiation_vet
