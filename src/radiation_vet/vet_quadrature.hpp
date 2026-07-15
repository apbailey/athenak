#ifndef RADIATION_VET_VET_QUADRATURE_HPP_
#define RADIATION_VET_VET_QUADRATURE_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file vet_quadrature.hpp
//! \brief Angular quadrature grid for the LTE short-characteristics VET solver.
//!
//! Implements the Bruls et al. (1999, A&A 348, 233) "type-A" discrete-ordinate grid
//! (Carlson 1963 symmetric S_N method), as described in Davis, Stone & Jiang (2012)
//! Sec. 3.2. The construction (permutation-family weight solve for n_mu<=6, equal
//! weights above that) follows the reference implementation validated in Athena-C's
//! radiation/angles.c::carlson(), cross-checked against Planning/theory-davis-2012.md.

#include "athena.hpp"

namespace radiation_vet {

//----------------------------------------------------------------------------------------
//! \class VETAngularGrid
//! \brief discrete-ordinate angular grid: direction cosines and quadrature weights

class VETAngularGrid {
 public:
  VETAngularGrid(int ndim, int nmu);

  int ndim;   // 1, 2, or 3
  int nmu;    // input <radiation_vet>/nmu parameter (number of polar levels)
  int nang;   // number of unique rays per octant
  int noct;   // number of octants: 2 (1D), 4 (2D), 8 (3D)

  // mu(oct,ang,0..2) = direction cosines (n-hat . x1hat, x2hat, x3hat) of ray 'ang' in
  // octant 'oct'; wmu(ang) = quadrature weight (same for every octant, already includes
  // the 1/noct octant-degeneracy factor, so moments are plain sums over (oct,ang) with
  // no extra prefactor -- Eq. 17-19 of Davis 2012).
  DualArray3D<Real> mu;
  DualArray1D<Real> wmu;

 private:
  void BuildCarlson1D();
  void BuildCarlsonND();
  void CheckNormalization();
};

}  // namespace radiation_vet

#endif  // RADIATION_VET_VET_QUADRATURE_HPP_
