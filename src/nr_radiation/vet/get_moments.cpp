//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file get_moments.cpp
//! \brief Radiation moment quadrature (Davis 2012 Eq. 17-19) and apb_rad-convention
//! coupling: Q = chi * crat * prat * (J - S) with S = T^4 (LTE).
//! Moments array ordering (Athena++ convention):
//!   n=0:J, 1-3:H_1,H_2,H_3, 4:K_11, 5:K_22, 6:K_33, 7:K_12, 8:K_13, 9:K_23

#include <cmath>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "nr_radiation/nr_radiation.hpp"

namespace nr_radiation {

//----------------------------------------------------------------------------------------
//! \fn void VET::CalculateMoments
//! \brief Single-kernel moment quadrature: one par_for over (m,k,j,i) with an inner serial
//! loop over all nang_tot angles, accumulating J, H_i, K_ij into local registers.
//! K_ij stored in Athena++ ordering: K11(4), K22(5), K33(6), K12(7), K13(8), K23(9).
//! For multi-frequency (nfreq>1), outer frequency loop with wfreq(f) weight goes here.

void VET::CalculateMoments() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  int nang = pang->nang;
  int nang_tot_ = nang_tot;
  auto &mu = pang->mu;
  auto &wmu = pang->wmu;
  auto ir_ = ir;
  auto mom_ = moments;
  auto jmean_ = jmean;

  par_for("vet_moments", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real J = 0.0, H1 = 0.0, H2 = 0.0, H3 = 0.0;
    Real K11 = 0.0, K22 = 0.0, K33 = 0.0;
    Real K12 = 0.0, K13 = 0.0, K23 = 0.0;

    for (int angg = 0; angg < nang_tot_; ++angg) {
      int oct = angg / nang;
      int a = angg - oct * nang;
      Real w = wmu.d_view(a);
      Real nx = mu.d_view(oct, a, 0);
      Real ny = mu.d_view(oct, a, 1);
      Real nz = mu.d_view(oct, a, 2);
      Real wI = w * ir_(m, angg, k, j, i);

      J   += wI;
      H1  += nx * wI;
      H2  += ny * wI;
      H3  += nz * wI;
      K11 += nx * nx * wI;
      K22 += ny * ny * wI;
      K33 += nz * nz * wI;
      K12 += nx * ny * wI;
      K13 += nx * nz * wI;
      K23 += ny * nz * wI;
    }

    mom_(m, 0, k, j, i) = J;
    mom_(m, 1, k, j, i) = H1;
    mom_(m, 2, k, j, i) = H2;
    mom_(m, 3, k, j, i) = H3;
    mom_(m, 4, k, j, i) = K11;
    mom_(m, 5, k, j, i) = K22;
    mom_(m, 6, k, j, i) = K33;
    mom_(m, 7, k, j, i) = K12;
    mom_(m, 8, k, j, i) = K13;
    mom_(m, 9, k, j, i) = K23;

    jmean_(m, k, j, i) = J;
  });
}

//----------------------------------------------------------------------------------------
//! \fn void VET::ComputeQrad
//! \brief Gas-radiation coupling (Davis Eq. 27 / absorption form):
//!   Q = χ_abs * crat * prat * (J - B) = eps * chi * crat * prat * (J - B)

void VET::ComputeQrad() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  auto chi_ = chi;
  auto eps_ = eps;
  auto planck_ = planck;
  auto jmean_ = jmean;
  auto qrad_ = qrad;
  const Real crat_prat = crat * prat;

  par_for("vet_qrad", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real chi_abs = eps_(m,k,j,i) * chi_(m,k,j,i);
    qrad_(m,k,j,i) = crat_prat * chi_abs * (jmean_(m,k,j,i) - planck_(m,k,j,i));
  });
}

}  // namespace nr_radiation
