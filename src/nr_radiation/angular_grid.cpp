//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file angular_grid.cpp
//! \brief Construction of the Bruls et al. (1999) type-A angular quadrature grid.
//!
//! This is a direct, validated port (host-side, run-once at startup) of the Carlson
//! symmetric S_N method implemented in Athena-C's radiation/angles.c::carlson()
//! (itself following Bruls et al. 1999, A&A 348, 233, as cited by Davis, Stone & Jiang
//! 2012 Sec. 3.2). For n_mu<=6 the polar weights are distributed in azimuth by solving
//! a linear system over "permutation families" of direction-cosine triplets (exact type-A
//! grid). Values n_mu>6 are rejected at construction: the equal-weight fallback can
//! produce negative quadrature weights that violate intensity positivity and invalidate
//! the Eddington tensor (Bruls et al. 1999 define the exact grid only for n_mu<=6).

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "athena.hpp"
#include "nr_radiation/angular_grid.hpp"

namespace nr_radiation {

namespace {

//----------------------------------------------------------------------------------------
// Host-only linear algebra helpers, ported from Athena-C radiation/angles.c.
// Only ever called once, at startup, for small (n_mu-1)x(n_mu-1) systems -- plain
// std::vector-based implementation is sufficient (not Kokkos/device code).

//! \brief Gauss-Legendre abscissae/weights on [x1,x2] (Numerical Recipes gauleg())
void GaussLegendre(Real x1, Real x2, std::vector<Real> *x, std::vector<Real> *w, int n) {
  const Real eps = 3.0e-14;
  int m = (n + 1) / 2;
  Real xm = 0.5 * (x2 + x1);
  Real xl = 0.5 * (x2 - x1);
  for (int i = 1; i <= m; i++) {
    Real z = std::cos(M_PI * ((Real)i - 0.25) / ((Real)n + 0.5));
    Real z1, pp, p1, p2, p3;
    do {
      p1 = 1.0;
      p2 = 0.0;
      for (int j = 1; j <= n; j++) {
        p3 = p2;
        p2 = p1;
        p1 = ((2.0 * (Real)j - 1.0) * z * p2 - ((Real)j - 1.0) * p3) / (Real)j;
      }
      pp = (Real)n * (z * p1 - p2) / (z * z - 1.0);
      z1 = z;
      z = z1 - p1 / pp;
    } while (std::abs(z - z1) > eps);
    (*x)[i-1] = xm - xl * z;
    (*x)[n-i] = xm + xl * z;
    (*w)[i-1] = 2.0 * xl / ((1.0 - z * z) * pp * pp);
    (*w)[n-i] = (*w)[i-1];
  }
}

//! \brief LU decomposition with partial pivoting (Numerical Recipes ludcmp())
void LUDecompose(std::vector<std::vector<Real>> *a, int n, std::vector<int> *indx) {
  std::vector<Real> rowscale(n);
  for (int i = 0; i < n; i++) {
    Real big = 0.0;
    for (int j = 0; j < n; j++) big = std::max(big, std::abs((*a)[i][j]));
    if (big == 0.0) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Singular matrix in SC quadrature setup" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    rowscale[i] = 1.0 / big;
  }
  for (int j = 0; j < n; j++) {
    for (int i = 0; i < j; i++) {
      Real sum = (*a)[i][j];
      for (int k = 0; k < i; k++) sum -= (*a)[i][k] * (*a)[k][j];
      (*a)[i][j] = sum;
    }
    Real big = 0.0;
    int imax = j;
    for (int i = j; i < n; i++) {
      Real sum = (*a)[i][j];
      for (int k = 0; k < j; k++) sum -= (*a)[i][k] * (*a)[k][j];
      (*a)[i][j] = sum;
      Real dum = rowscale[i] * std::abs(sum);
      if (dum >= big) { big = dum; imax = i; }
    }
    if (j != imax) {
      std::swap((*a)[imax], (*a)[j]);
      rowscale[imax] = rowscale[j];
    }
    (*indx)[j] = imax;
    if ((*a)[j][j] == 0.0) (*a)[j][j] = 1.0e-30;
    Real dum = 1.0 / (*a)[j][j];
    for (int i = j+1; i < n; i++) (*a)[i][j] *= dum;
  }
}

//! \brief Backward substitution (Numerical Recipes lubksb())
void LUBackSub(const std::vector<std::vector<Real>> &a, int n,
               const std::vector<int> &indx, std::vector<Real> *b) {
  int ii = -1;
  for (int i = 0; i < n; i++) {
    int ip = indx[i];
    Real sum = (*b)[ip];
    (*b)[ip] = (*b)[i];
    if (ii >= 0) {
      for (int j = ii; j <= i-1; j++) sum -= a[i][j] * (*b)[j];
    } else if (sum != 0.0) {
      ii = i;
    }
    (*b)[i] = sum;
  }
  for (int i = n-1; i >= 0; i--) {
    Real sum = (*b)[i];
    for (int j = i+1; j < n; j++) sum -= a[i][j] * (*b)[j];
    (*b)[i] = sum / a[i][i];
  }
}

//! \brief Full matrix inverse via LU decomposition (destroys input copy)
void InvertMatrix(std::vector<std::vector<Real>> a, int n,
                   std::vector<std::vector<Real>> *ainv) {
  std::vector<int> indx(n);
  LUDecompose(&a, n, &indx);
  for (int j = 0; j < n; j++) {
    std::vector<Real> col(n, 0.0);
    col[j] = 1.0;
    LUBackSub(a, n, indx, &col);
    for (int i = 0; i < n; i++) (*ainv)[i][j] = col[i];
  }
}

//! \brief returns index of matching permutation family in pl[0..np-1], or -1
int MatchPermutation(int i, int j, int k, const std::vector<std::array<int,3>> &pl,
                     int np) {
  for (int l = 0; l < np; l++) {
    for (int m = 0; m < 3; m++) {
      if (i == pl[l][m]) {
        for (int n = 0; n < 3; n++) {
          if (n != m && j == pl[l][n]) {
            for (int o = 0; o < 3; o++) {
              if (o != m && o != n && k == pl[l][o]) return l;
            }
          }
        }
      }
    }
  }
  return -1;
}

}  // anonymous namespace

//----------------------------------------------------------------------------------------
// SCAngularGrid constructor

SCAngularGrid::SCAngularGrid(int ndim_in, int nmu_in) :
    mu("sc_mu",1,1,1),
    wmu("sc_wmu",1),
    ndim(ndim_in), nmu(nmu_in) {
  if (nmu < 1) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
      << std::endl << "<nr_radiation>/nmu = " << nmu << " must be >= 1" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (nmu > 6) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
      << std::endl << "<nr_radiation>/nmu = " << nmu
      << " exceeds the Bruls type-A exact-grid limit of 6. "
      << "Values nmu>6 produce negative quadrature weights which violate "
      << "positivity of the intensity and invalidate the Eddington tensor. "
      << "Use nmu <= 6." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (ndim == 1) {
    noct = 2;
    nang = nmu;
  } else if (ndim == 2) {
    noct = 4;
    nang = nmu * (nmu + 1) / 2;
  } else {
    noct = 8;
    nang = nmu * (nmu + 1) / 2;
  }
  Kokkos::realloc(mu, noct, nang, 3);
  Kokkos::realloc(wmu, nang);

  if (ndim == 1) {
    BuildCarlson1D();
  } else {
    BuildCarlsonND();
  }

  mu.template modify<HostMemSpace>();
  mu.template sync<DevExeSpace>();
  wmu.template modify<HostMemSpace>();
  wmu.template sync<DevExeSpace>();

  CheckNormalization();
}

//----------------------------------------------------------------------------------------
//! \fn SCAngularGrid::BuildCarlson1D
//! \brief 1D angular grid: plain Gauss-Legendre quadrature on mu in [-1,1], split into
//! two octants (mu>0 outgoing, mu<0 incoming) -- Athena-C angles.c::carlson(), nDim==1
//! branch.

void SCAngularGrid::BuildCarlson1D() {
  std::vector<Real> mutmp1d(2*nmu), wtmp(2*nmu);
  GaussLegendre(-1.0, 1.0, &mutmp1d, &wtmp, 2*nmu);
  for (int i = nmu; i < 2*nmu; i++) {
    wmu.h_view(i-nmu) = 0.5 * wtmp[i];
    mu.h_view(0, i-nmu, 0) = mutmp1d[i];
    mu.h_view(0, i-nmu, 1) = 0.0;
    mu.h_view(0, i-nmu, 2) = 0.0;
    mu.h_view(1, i-nmu, 0) = -mutmp1d[i];
    mu.h_view(1, i-nmu, 1) = 0.0;
    mu.h_view(1, i-nmu, 2) = 0.0;
  }
}

//----------------------------------------------------------------------------------------
//! \fn SCAngularGrid::BuildCarlsonND
//! \brief 2D/3D angular grid following Bruls et al. (1999) type-A construction --
//! Athena-C angles.c::carlson(), nDim>1 branch, ported verbatim.

void SCAngularGrid::BuildCarlsonND() {
  // --- polar weights/angles (mu2tmp, wtmp); nmu<=6 enforced in ctor ---
  std::vector<Real> mu2tmp(nmu);
  Real deltamu = 2.0 / (2*nmu - 1);
  mu2tmp[0] = 1.0 / (3.0 * (2*nmu - 1));
  for (int i = 1; i < nmu; i++) mu2tmp[i] = mu2tmp[i-1] + deltamu;

  std::vector<Real> Wtmp(std::max(nmu-1,1)), wtmp(nmu);
  Real W2 = 4.0 * mu2tmp[0];
  Real Wsum = Wtmp[0] = std::sqrt(W2);
  for (int i = 1; i < nmu-2; i++) {
    W2 += deltamu;
    Wsum += Wtmp[i] = std::sqrt(W2);
  }
  if (nmu > 2) Wtmp[nmu-2] = 2.0*(nmu-1)/3.0 - Wsum;

  Real wsum = wtmp[0] = Wtmp[0];
  for (int i = 1; i < nmu-1; i++) wsum += wtmp[i] = Wtmp[i] - Wtmp[i-1];
  if (nmu > 1) wtmp[nmu-1] = 1.0 - wsum;
  else wtmp[0] = 1.0;

  // --- direction cosines: all (i,j,k)>=0 with i+j+k=nmu-1, plus the
  // permutation-family incidence matrix pmat(i,fam) needed to solve for azimuthal
  // weight distribution (Bruls et al. 1999; angles.c::carlson() single combined pass) ---
  std::vector<std::array<Real,3>> mutmp(nang);
  std::vector<int> plab(nang);
  int nfam = std::max(nmu - 1, 1);
  std::vector<std::array<int,3>> pl(nmu, {0,0,0});
  std::vector<std::vector<Real>> pmat(nfam, std::vector<Real>(nfam, 0.0));
  int np = 0, iang = 0;
  for (int i = 0; i < nmu; i++) {
    for (int j = 0; j < nmu; j++) {
      for (int k = 0; k < nmu; k++) {
        if (i + j + k == nmu - 1) {
          mutmp[iang][0] = std::sqrt(mu2tmp[j]);
          mutmp[iang][1] = std::sqrt(mu2tmp[k]);
          mutmp[iang][2] = std::sqrt(mu2tmp[i]);
          int ip = MatchPermutation(i, j, k, pl, np);
          if (ip == -1) {
            pl[np] = {i, j, k};
            if (i < nfam) pmat[i][np] += 1.0;
            plab[iang] = np;
            np++;
          } else {
            if (i < nfam) pmat[i][ip] += 1.0;
            plab[iang] = ip;
          }
          iang++;
        }
      }
    }
  }

  // --- weights: exact type-A permutation-family solve (nmu<=6, enforced in ctor) ---
  std::vector<Real> wang(nang);
  if (nmu > 1) {
    std::vector<std::vector<Real>> pinv(nfam, std::vector<Real>(nfam, 0.0));
    InvertMatrix(pmat, nfam, &pinv);
    std::vector<Real> wpf(nfam, 0.0);
    for (int i = 0; i < nfam; i++) {
      for (int j = 0; j < nfam; j++) wpf[i] += pinv[i][j] * wtmp[j];
    }
    for (int i = 0; i < nang; i++) wang[i] = wpf[plab[i]];
  } else {
    wang[0] = 1.0;
  }

  // --- assign signed direction cosines to octants, and finalize weights ---
  Real octfac = (ndim == 2) ? 0.25 : 0.125;
  for (int i = 0; i < nang; i++) {
    if (ndim == 2) {
      for (int j = 0; j < 2; j++) {
        for (int k = 0; k < 2; k++) {
          int l = 2*j + k;
          mu.h_view(l, i, 0) = (k == 0 ?  mutmp[i][0] : -mutmp[i][0]);
          mu.h_view(l, i, 1) = (j == 0 ?  mutmp[i][1] : -mutmp[i][1]);
          mu.h_view(l, i, 2) = mutmp[i][2];
        }
      }
    } else {  // ndim == 3
      for (int j = 0; j < 2; j++) {
        for (int k = 0; k < 2; k++) {
          for (int l = 0; l < 2; l++) {
            int m = 4*j + 2*k + l;
            mu.h_view(m, i, 0) = (l == 0 ?  mutmp[i][0] : -mutmp[i][0]);
            mu.h_view(m, i, 1) = (k == 0 ?  mutmp[i][1] : -mutmp[i][1]);
            mu.h_view(m, i, 2) = (j == 0 ?  mutmp[i][2] : -mutmp[i][2]);
          }
        }
      }
    }
    wmu.h_view(i) = wang[i] * octfac;
  }
}

//----------------------------------------------------------------------------------------
//! \fn SCAngularGrid::CheckNormalization
//! \brief Startup asserts: Sum_k w_k = 1, Sum_k w_k mu_ik = 0, Sum_k w_k mu_ik mu_jk =
//! delta_ij/3 (Davis 2012 Sec. 3.2 normalisation check; theory-davis-2012.md checklist).

void SCAngularGrid::CheckNormalization() {
  const Real tol = 1.0e-10;
  Real sumw = 0.0, summu[3] = {0.0,0.0,0.0};
  Real summumu[3][3] = {{0.0,0.0,0.0},{0.0,0.0,0.0},{0.0,0.0,0.0}};
  for (int oct = 0; oct < noct; oct++) {
    for (int n = 0; n < nang; n++) {
      Real w = wmu.h_view(n);
      sumw += w;
      Real m0 = mu.h_view(oct,n,0), m1 = mu.h_view(oct,n,1), m2 = mu.h_view(oct,n,2);
      Real mvec[3] = {m0, m1, m2};
      for (int a = 0; a < 3; a++) {
        summu[a] += w * mvec[a];
        for (int b = 0; b < 3; b++) summumu[a][b] += w * mvec[a] * mvec[b];
      }
    }
  }
  bool ok = (std::abs(sumw - 1.0) < tol);
  for (int a = 0; a < 3; a++) {
    // Only require odd-moment nullity for active dimensions (in 2D mu_z is unsigned)
    if (a < ndim && std::abs(summu[a]) > tol) ok = false;
    for (int b = 0; b < 3; b++) {
      Real target = (a == b) ? (1.0/3.0) : 0.0;
      // In 1D/2D the "unused" transverse component(s) are identically zero by
      // construction, so K_ii for those components is 0, not 1/3 -- only check the
      // active dimensions (ndim of them).
      if (a < ndim && b < ndim) {
        if (std::abs(summumu[a][b] - target) > tol) ok = false;
      }
    }
  }
  if (!ok) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
      << std::endl << "SC angular quadrature (ndim=" << ndim << ", nmu=" << nmu
      << ") failed normalization checks: Sum(w)=" << sumw
      << " Sum(w*mu)=(" << summu[0] << "," << summu[1] << "," << summu[2] << ")"
      << " Sum(w*mu*mu) diag=(" << summumu[0][0] << "," << summumu[1][1] << ","
      << summumu[2][2] << ")" << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

}  // namespace nr_radiation
