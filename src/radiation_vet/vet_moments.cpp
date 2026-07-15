//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file vet_moments.cpp
//! \brief Radiation moment quadrature (Davis 2012 Eq. 17-19) and integral-form Q_rad
//! (Eq. 27): Q^int = 4*pi*chi_abs*(J-B) with chi_abs = chi_tot (LTE eps=1).

#include <cmath>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "radiation_vet.hpp"

namespace radiation_vet {

//----------------------------------------------------------------------------------------
//! \fn void RadiationVET::CalculateMoments
//! \brief Fill moments array: n=0 J; 1-3 H_i; 4-9 K_11,K_12,K_13,K_22,K_23,K_33.
//! Also syncs jmean from moments(0) for consistency with ComputeJ.

void RadiationVET::CalculateMoments() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  int nang = pang->nang;
  int nangt1 = nang_tot - 1;
  auto &mu = pang->mu;
  auto &wmu = pang->wmu;
  auto ir_ = ir;
  auto mom_ = moments;
  auto jmean_ = jmean;

  par_for("vet_mom_zero", DevExeSpace(), 0, nmb1, 0, 9, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
    mom_(m,n,k,j,i) = 0.0;
  });

  // accumulate over discrete ordinates on host-driven angle loop (weights from host view)
  for (int angg = 0; angg <= nangt1; ++angg) {
    int oct = angg / nang;
    int a = angg - oct*nang;
    Real w = wmu.h_view(a);
    Real mux = mu.h_view(oct,a,0);
    Real muy = mu.h_view(oct,a,1);
    Real muz = mu.h_view(oct,a,2);
    par_for("vet_mom_acc", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real wi = w * ir_(m,angg,k,j,i);
      mom_(m,0,k,j,i) += wi;                 // J
      mom_(m,1,k,j,i) += mux * wi;           // H_1
      mom_(m,2,k,j,i) += muy * wi;           // H_2
      mom_(m,3,k,j,i) += muz * wi;           // H_3
      mom_(m,4,k,j,i) += mux * mux * wi;     // K_11
      mom_(m,5,k,j,i) += mux * muy * wi;     // K_12
      mom_(m,6,k,j,i) += mux * muz * wi;     // K_13
      mom_(m,7,k,j,i) += muy * muy * wi;     // K_22
      mom_(m,8,k,j,i) += muy * muz * wi;     // K_23
      mom_(m,9,k,j,i) += muz * muz * wi;     // K_33
    });
  }

  // keep jmean consistent with moments(:,0,:,:,:)
  par_for("vet_j_from_mom", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    jmean_(m,k,j,i) = mom_(m,0,k,j,i);
  });
}

//----------------------------------------------------------------------------------------
//! \fn void RadiationVET::ComputeQrad
//! \brief Integral-form heating/cooling (Davis 2012 Eq. 27):
//!   Q^int = 4*pi*chi_abs*(J - B),  chi_abs = eps*chi_tot = chi_tot (eps=1).

void RadiationVET::ComputeQrad() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  auto chi_ = chi;
  auto bb_ = bb;
  auto jmean_ = jmean;
  auto qrad_ = qrad;
  const Real four_pi = 4.0 * M_PI;

  par_for("vet_qrad", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    // sign: energy equation gets +Q_rad; fluid cools when J < B if Q ~ (J-B)?
    // Davis Eq. 27/26: gas energy change from absorption-emission. Athena-C linear_wave
    // uses Q = 4*pi*kappa*(J-B) added to gas (J>B heats). Match that.
    qrad_(m,k,j,i) = four_pi * chi_(m,k,j,i) * (jmean_(m,k,j,i) - bb_(m,k,j,i));
  });
}

}  // namespace radiation_vet
