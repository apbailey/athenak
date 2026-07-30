#ifndef NR_RADIATION_SC_SC_OPACITY_HPP_
#define NR_RADIATION_SC_SC_OPACITY_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file sc_opacity.hpp
//! \brief Placeholder kept for build compatibility. The SCOpacity struct has been
//! replaced by per-class members (opa, prat, crat) following the apb_rad convention:
//!   chi = opa * rho   (absorption coefficient)
//!   S   = T^4         (source function in radiation units)
//!   Q   = chi * crat * prat * (J - S)   (gas-radiation coupling)
//! See Davis, Stone & Jiang (2012) and the Athena++ radiation module convention.

#include "athena.hpp"

namespace nr_radiation {
// Empty — retained only so that #include directives compile without error until
// all references are cleaned up in downstream phases.
}  // namespace nr_radiation

#endif  // NR_RADIATION_SC_SC_OPACITY_HPP_
