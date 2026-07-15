#ifndef RADIATION_VET_VET_OPACITY_HPP_
#define RADIATION_VET_VET_OPACITY_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file vet_opacity.hpp
//! \brief Single point of specification of opacity chi^tot and the Planck function B,
//! shared by both the formal-solution sweep and the Q_rad coupling (D6 "specify opacity
//! once" lesson from Planning/vet-athenak-plan-internal.md). Photon destruction
//! probability eps = chi^abs/chi^tot is NOT exposed as an input in v1: it is hardcoded
//! to 1 (pure LTE, S=B everywhere, Davis, Stone & Jiang 2012 Eq. 9 with eps->1), matching
//! <radiation>/lte=1 in both Athena-C reference inputs (athinput.beam2d,
//! athinput.linear_wave_rad2d).
//!
//! chi_type=constant matches both validation targets: beam2d's const_opacity() (fixed
//! chi0 = tau0/dx2) and linear_wave_rad2d's const_chi() (fixed kappa = tau*2*pi).
//! bb_type=greybody matches linear_wave_rad2d's grey_B(): B(T) = B0*(T/T0)^4.

#include "athena.hpp"

namespace radiation_vet {

enum class VETChiType {constant};
enum class VETBBType {zero, greybody};

struct VETOpacity {
  VETChiType chi_type = VETChiType::constant;
  VETBBType  bb_type  = VETBBType::zero;
  Real chi0 = 0.0;      // constant total opacity chi^tot (chi_type=constant)
  Real bb_norm = 0.0;   // B0 normalization for bb_type=greybody: B(T) = bb_norm*T^4

  KOKKOS_INLINE_FUNCTION
  Real Chi() const {
    return chi0;  // only chi_type=constant implemented in v1
  }

  KOKKOS_INLINE_FUNCTION
  Real BlackBody(Real temp) const {
    if (bb_type == VETBBType::greybody) return bb_norm * SQR(SQR(temp));
    return 0.0;
  }
};

}  // namespace radiation_vet

#endif  // RADIATION_VET_VET_OPACITY_HPP_
