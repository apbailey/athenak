//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file radvet_linwave.cpp
//! \brief Radiatively damped acoustic waves (Davis, Stone & Jiang 2012 §5.4 / Eq. 38;
//! Stein & Spiegel 1967). Faithful 1D port of Athena-C linear_wave_rad1d.c for Figs. 9–10,
//! with optional 2D non-grid-aligned extension (Athena-C linear_wave_rad2d.c).
//!
//! §5.4 protocol:
//!   ρ0=1, γ=5/3, v0=0, a=1 (adiabatic sound speed), A=10^{-6}, periodic, tf=L/a.
//!   χ0 = τ0·k = τ0·2π/λ  (λ = domain length along the wave for 1D).
//!   B0 = γ Egas0 / (Bo π), bb_norm = B0/T0^4 with T0 = p0/ρ0 = 1/γ.
//! Final diagnostics: Fourier-fitted ωR/(ka), ωI/(ka) (Fig. 9) and L1 / RMS error
//! against the analytic eigenmode at tf (Fig. 10 / Eq. 44).

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <iostream>
#include <limits>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"
#include "eos/eos.hpp"
#include "pgen/pgen.hpp"
#include "radiation_vet/radiation_vet.hpp"

namespace {
using Complex = std::complex<Real>;

struct WaveVars {
  Real amp, vflow, Bo, tau;
  Real sin_a, cos_a, lambda;
  Real vph, rdamp;           // analytic: vph = Re(ω)/(k a) with a=1; rdamp = 2π Im(ω)
  Real omega_r, omega_i;     // analytic ω/(k a)
  Real V0R, V0I, E0R, E0I;
  Real dens, pgas;
  Real gamma;
  Real B0, T0, chi0;
  bool is_1d;
};
WaveVars wv;

void Laguer(Complex *a, int m, Complex *x, int *its) {
  const Real epss = 2.0e-7;
  static Real frac[9] = {0.0,0.5,0.25,0.75,0.13,0.38,0.62,0.88,1.0};
  for (int iter=1; iter<=18; ++iter) {
    *its = iter;
    Complex b = a[m];
    Real err = std::abs(b);
    Complex d = 0.0, f = 0.0;
    Real abx = std::abs(*x);
    for (int j=m-1; j>=0; --j) {
      f = (*x)*f + d;
      d = (*x)*d + b;
      b = (*x)*b + a[j];
      err = std::abs(b) + abx*err;
    }
    err *= epss;
    if (std::abs(b) <= err) return;
    Complex g = d/b;
    Complex g2 = g*g;
    Complex h = g2 - 2.0*f/b;
    Complex sq = std::sqrt(Complex(m-1)*(Complex(m)*h - g2));
    Complex gp = g + sq;
    Complex gm = g - sq;
    Real abp = std::abs(gp), abm = std::abs(gm);
    if (abp < abm) gp = gm;
    Complex dx;
    if (std::max(abp,abm) > 0.0) dx = Complex(m)/gp;
    else dx = std::exp(std::log(1.0+abx) + Complex(0,iter));
    Complex x1 = (*x) - dx;
    if ((*x) == x1) return;
    if (iter % 10) *x = x1;
    else *x -= frac[iter/10]*dx;
  }
  std::cout << "too many laguer iterations" << std::endl;
}

void Zroots(Complex *a, int m, Complex *roots, int polish) {
  Complex ad[4];
  const Real eps = 1.0e-8;
  for (int i=0; i<=m; ++i) ad[i] = a[i];
  for (int j=m; j>0; --j) {
    Complex x = 0.0;
    int its;
    Laguer(ad, j, &x, &its);
    if (std::abs(x.imag()) <= 2.0*eps*std::abs(x.real())) x = x.real();
    roots[j-1] = x;
    Complex b = ad[j];
    for (int jj=j-1; jj>=0; --jj) {
      Complex c = ad[jj];
      ad[jj] = b;
      b = x*b + c;
    }
  }
  if (polish) {
    for (int j=0; j<m; ++j) {
      int its;
      Laguer(a, m, &(roots[j]), &its);
    }
  }
}

//! Davis 2012 Eq. 38 cubic for radiatively modified acoustic modes
//! (Athena-C acoustic_wave_rad). Returns dimensionless ω/(k a) with a=cs=1.
void AcousticWaveRad(Real Bo, Real tau, Real cs, Real d0,
                     Real *vph, Real *rdamp,
                     Real *V0R, Real *V0I, Real *E0R, Real *E0I,
                     Real *omega_r, Real *omega_i) {
  // Ξ = 1 - τ arctan(1/τ)  (Eq. 39 with τ=χ/k); θ ≡ ν0 Ξ0/(k a)
  Real mu = 1.0 - tau * std::atan(1.0/tau);
  Real theta = 16.0 * tau * mu / Bo;
  Complex coeff[4], roots[3];
  coeff[0] = Complex(0.0, theta);
  coeff[1] = -1.0;
  coeff[2] = Complex(0.0, -theta * wv.gamma);
  coeff[3] = 1.0;
  Zroots(coeff, 3, roots, 1);
  Complex omega = roots[0];
  for (int i=0; i<3; ++i) {
    if (roots[i].real() > 0.5) omega = roots[i];
  }
  Complex V0 = cs * omega / d0;
  Complex E0 = cs*cs * omega*omega / (wv.gamma - 1.0);
  *V0R = V0.real(); *V0I = V0.imag();
  *E0R = E0.real(); *E0I = E0.imag();
  *omega_r = omega.real();
  *omega_i = omega.imag();
  *vph = omega.real();                 // phase speed / a  (a=1)
  *rdamp = 2.0 * M_PI * omega.imag();  // amp ∝ exp(-rdamp t) for sin(2π(x/λ - vph t))
}

//----------------------------------------------------------------------------------------
//! Fourier fit of the fundamental density mode (Davis §5.4 Fig. 9):
//!   δρ = A exp(+ω_I k a t) sin(k x - ω_R k a t + φ0)   [paper time convention e^{iωt}]
//! With k=2π/λ, a=1, and IC δρ = A sin(2π x/λ) so φ0=0 at t=0:
//!   ω_I/(k a) = ln(A_meas/A) / (k a t) = ln(A_meas/A)/(2π t/λ)
//!   ω_R/(k a) = -Δφ / (k a t)
//! Note Stein/Davis Im(ω)>0 for growth of e^{iωt}; our eigenmode uses damping with
//! amp_t = A exp(-rdamp t) and rdamp = 2π Im(ω_cubic) where the cubic root has Im>0 for
//! damping under the e^{-iωt} convention equivalently. We report ωI_norm = Im(ω)/ (k a)
//! with the SAME sign convention as the analytic cubic root (positive = damping rate of
//! the wave amplitude when plotted as in Fig. 9 bottom panel of Davis 2012).

void FitFourierOmega(Mesh *pm, Real *omega_r_fit, Real *omega_i_fit, Real *amp_meas) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  auto &size = pmbp->pmb->mb_size;
  auto u0 = pmbp->phydro->u0;
  auto u_h = Kokkos::create_mirror_view(u0);
  Kokkos::deep_copy(u_h, u0);
  auto size_h = size.h_view;

  Real sum_s = 0.0, sum_c = 0.0;
  int ncell = 0;
  int nmb = pmbp->nmb_thispack;
  for (int m=0; m<nmb; ++m) {
    Real x1min = size_h(m).x1min, x1max = size_h(m).x1max;
    Real x2min = size_h(m).x2min, x2max = size_h(m).x2max;
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);
        for (int i=is; i<=ie; ++i) {
          Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);
          Real r = (x1v*wv.cos_a + x2v*wv.sin_a) / wv.lambda;
          Real dpert = u_h(m,IDN,k,j,i) - wv.dens;
          sum_s += dpert * std::sin(2.0*M_PI*r);
          sum_c += dpert * std::cos(2.0*M_PI*r);
          ++ncell;
        }
      }
    }
  }
#if MPI_PARALLEL_ENABLED
  Real buf[2] = {sum_s, sum_c};
  int nbuf = ncell;
  MPI_Allreduce(MPI_IN_PLACE, buf, 2, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &nbuf, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  sum_s = buf[0]; sum_c = buf[1]; ncell = nbuf;
#endif
  // Projection onto orthonormal basis: <sin,sin>=N/2 for discrete periodic sum
  Real As = 2.0 * sum_s / static_cast<Real>(ncell);
  Real Ac = 2.0 * sum_c / static_cast<Real>(ncell);
  *amp_meas = std::sqrt(As*As + Ac*Ac);
  // Initial: As=A, Ac=0. At time t the mode is A e^{-rdamp t} * sin(2π(r - vph t))
  //        = A_t [sin(2πr)cos(φ) - cos(2πr)sin(φ)], φ=2π vph t
  // so As = A_t cos(φ), Ac = -A_t sin(φ) ⇒ φ = atan2(-Ac, As)
  Real phase = std::atan2(-Ac, As);  // in (-π, π]
  Real t = pm->time;
  if (t <= 0.0 || *amp_meas <= 0.0) {
    *omega_r_fit = 0.0;
    *omega_i_fit = 0.0;
    return;
  }
  // Unwrap onto the branch nearest the analytic expectation. Critical for the
  // adiabatic regime where vph≈a and tf=L/a ⇒ φ_an≈2π, which otherwise wraps to 0
  // and spuriously yields ωR≈0 (Davis §5.4 Fig. 9, high-Bo curves).
  Real phase_an = 2.0*M_PI * wv.vph * t / wv.lambda;
  while (phase < phase_an - M_PI) phase += 2.0*M_PI;
  while (phase > phase_an + M_PI) phase -= 2.0*M_PI;
  *omega_r_fit = phase * wv.lambda / (2.0*M_PI * t);
  *omega_i_fit = -std::log((*amp_meas)/wv.amp) * wv.lambda / (2.0*M_PI * t);
}

void RadvetLinwaveErrors(ParameterInput *pin, Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  if (pmbp->phydro == nullptr) return;
  auto &indcs = pm->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  auto &size = pmbp->pmb->mb_size;
  auto u0 = pmbp->phydro->u0;
  Real gm1 = pmbp->phydro->peos->eos_data.gamma - 1.0;
  Real time = pm->time;
  Real amp_t = wv.amp * std::exp(-time * wv.rdamp);

  Real l1_d=0, l1_e=0, l1_m1=0, l1_m2=0, l1_m3=0;
  int nmb = pmbp->nmb_thispack;
  auto u_h = Kokkos::create_mirror_view(u0);
  Kokkos::deep_copy(u_h, u0);
  auto size_h = size.h_view;
  for (int m=0; m<nmb; ++m) {
    Real x1min = size_h(m).x1min, x1max = size_h(m).x1max;
    Real x2min = size_h(m).x2min, x2max = size_h(m).x2max;
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);
        for (int i=is; i<=ie; ++i) {
          Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);
          Real r = (x1v*wv.cos_a + x2v*wv.sin_a)/wv.lambda - wv.vph*time;
          Real sinr = std::sin(2.0*M_PI*r);
          Real cosr = std::cos(2.0*M_PI*r);
          Real d0 = wv.dens;
          Real p0 = wv.pgas;
          Real uflow = wv.vflow;
          Real d_ex = d0 + amp_t*sinr;
          Real E_ex = p0/gm1 + 0.5*d0*uflow*uflow
                    + amp_t*(sinr*wv.E0R - cosr*wv.E0I);
          Real M1_ex = wv.cos_a*d0*uflow
                     + wv.cos_a*amp_t*(sinr*wv.V0R - cosr*wv.V0I);
          Real M2_ex = wv.sin_a*d0*uflow
                     + wv.sin_a*amp_t*(sinr*wv.V0R - cosr*wv.V0I);
          l1_d  += std::abs(u_h(m,IDN,k,j,i) - d_ex);
          l1_e  += std::abs(u_h(m,IEN,k,j,i) - E_ex);
          l1_m1 += std::abs(u_h(m,IM1,k,j,i) - M1_ex);
          l1_m2 += std::abs(u_h(m,IM2,k,j,i) - M2_ex);
          l1_m3 += std::abs(u_h(m,IM3,k,j,i));
        }
      }
    }
  }
  Real ncell = static_cast<Real>(pm->nmb_total * (ie-is+1)*(je-js+1)*(ke-ks+1));
#if MPI_PARALLEL_ENABLED
  Real buf[5] = {l1_d,l1_e,l1_m1,l1_m2,l1_m3};
  MPI_Allreduce(MPI_IN_PLACE, buf, 5, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  l1_d=buf[0]; l1_e=buf[1]; l1_m1=buf[2]; l1_m2=buf[3]; l1_m3=buf[4];
#endif
  // Davis Eq. 44: per-variable mean absolute error; RMS over conserved variables
  Real d_err = l1_d/ncell, e_err = l1_e/ncell;
  Real m1_err = l1_m1/ncell, m2_err = l1_m2/ncell, m3_err = l1_m3/ncell;
  Real rms_err = std::sqrt(d_err*d_err + e_err*e_err + m1_err*m1_err
                           + m2_err*m2_err + m3_err*m3_err);

  Real omega_r_fit=0, omega_i_fit=0, amp_meas=0;
  FitFourierOmega(pm, &omega_r_fit, &omega_i_fit, &amp_meas);

  if (global_variable::my_rank == 0) {
    int nx1_tot = pm->mesh_indcs.nx1;
    std::cout << "radvet_linwave §5.4: Bo=" << wv.Bo << " tau=" << wv.tau
              << " N=" << nx1_tot << " t=" << time
              << " 1D=" << (wv.is_1d?1:0) << std::endl;
    std::cout << "  analytic: ωR/(ka)=" << wv.omega_r
              << " ωI/(ka)=" << wv.omega_i
              << " vph=" << wv.vph << " rdamp=" << wv.rdamp << std::endl;
    std::cout << "  fitted:   ωR/(ka)=" << omega_r_fit
              << " ωI/(ka)=" << omega_i_fit
              << " A_meas=" << amp_meas << std::endl;
    std::cout << "  L1: dens=" << d_err << " E=" << e_err
              << " M1=" << m1_err << " M2=" << m2_err
              << " RMS=" << rms_err << std::endl;

    // Machine-readable row for Fig. 9 / Fig. 10 post-processing
    FILE *fp = std::fopen("RadvetLinWave-davis54.dat", "a");
    if (fp) {
      // Bo tau N nx2 omegaR_an omegaI_an omegaR_fit omegaI_fit RMS L1d L1E L1M1
      std::fprintf(fp,
        "%.6g %.6g %d %d %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e\n",
        wv.Bo, wv.tau, nx1_tot, pm->mesh_indcs.nx2,
        wv.omega_r, wv.omega_i, omega_r_fit, omega_i_fit,
        rms_err, d_err, e_err, m1_err);
      std::fclose(fp);
    }
  }
  (void)pin;
}
}  // namespace

void ProblemGenerator::RadvetLinwave(ParameterInput *pin, const bool restart) {
  pgen_final_func = RadvetLinwaveErrors;
  wv.amp   = pin->GetOrAddReal("problem", "amp", 1.0e-6);
  wv.vflow = pin->GetOrAddReal("problem", "vflow", 0.0);
  wv.Bo    = pin->GetReal("problem", "Bo");
  wv.tau   = pin->GetReal("problem", "tau");
  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->pradvet == nullptr || pmbp->phydro == nullptr) {
    std::cout << "### FATAL ERROR: radvet_linwave needs <hydro> and <radiation_vet>"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  wv.gamma = pmbp->phydro->peos->eos_data.gamma;
  Real gm1 = wv.gamma - 1.0;

  Real x1size = pmy_mesh_->mesh_size.x1max - pmy_mesh_->mesh_size.x1min;
  Real x2size = pmy_mesh_->mesh_size.x2max - pmy_mesh_->mesh_size.x2min;
  wv.is_1d = pmy_mesh_->one_d || (pmy_mesh_->mesh_indcs.nx2 <= 1);

  if (wv.is_1d) {
    // Davis §5.4 / Athena-C linear_wave_rad1d: grid-aligned wave, λ = Lx
    wv.cos_a = 1.0;
    wv.sin_a = 0.0;
    wv.lambda = x1size;
  } else {
    // Athena-C linear_wave_rad2d non-grid-aligned geometry
    Real angle = std::atan(x1size/x2size);
    wv.sin_a = std::sin(angle);
    wv.cos_a = std::cos(angle);
    wv.lambda = (wv.cos_a >= wv.sin_a) ? (x1size*wv.cos_a) : (x2size*wv.sin_a);
  }

  wv.dens = 1.0;
  wv.pgas = wv.dens / wv.gamma;  // a = √(γ p/ρ) = 1

  AcousticWaveRad(wv.Bo, wv.tau, 1.0, wv.dens,
                  &wv.vph, &wv.rdamp, &wv.V0R, &wv.V0I, &wv.E0R, &wv.E0I,
                  &wv.omega_r, &wv.omega_i);

  // Opacity / Planck normalization from Bo, τ (Davis Eqs. 40–41; Athena-C)
  // Egas0 = p0/(γ-1) = 1/(γ(γ-1)); T0 = p0/(ρ0 R) = 1/γ; χ0 = τ0·k = τ0·2π/λ
  Real Etherm0 = 1.0 / (wv.gamma * gm1);
  wv.T0 = Etherm0 * gm1 / wv.dens;
  wv.B0 = wv.gamma * Etherm0 / (wv.Bo * M_PI);
  wv.chi0 = wv.tau * 2.0 * M_PI / wv.lambda;
  Real bb_norm = wv.B0 / (wv.T0*wv.T0*wv.T0*wv.T0);

  // Override radiation_vet opacity from problem Bo/τ so sweep scripts need not
  // hardcode chi0/bb_norm (still constructed from pin; values updated here).
  pmbp->pradvet->opac.chi0 = wv.chi0;
  pmbp->pradvet->opac.bb_norm = bb_norm;
  pmbp->pradvet->opac.bb_type = radiation_vet::VETBBType::greybody;

  std::cout << "radvet_linwave: Bo=" << wv.Bo << " tau=" << wv.tau
            << " lambda=" << wv.lambda << " 1D=" << (wv.is_1d?1:0) << std::endl;
  std::cout << "  omega/(ka)=(" << wv.omega_r << "," << wv.omega_i
            << ") vph=" << wv.vph << " rdamp=" << wv.rdamp << std::endl;
  std::cout << "  T0=" << wv.T0 << " B0=" << wv.B0 << " chi0=" << wv.chi0
            << " bb_norm=" << bb_norm << std::endl;

  auto &indcs = pmy_mesh_->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  auto &size = pmbp->pmb->mb_size;
  auto u0 = pmbp->phydro->u0;
  int nmb1 = pmbp->nmb_thispack - 1;
  auto wv_ = wv;

  par_for("radvet_linwave_ic", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real x1v = CellCenterX(i-is, indcs.nx1, size.d_view(m).x1min, size.d_view(m).x1max);
    Real x2v = CellCenterX(j-js, indcs.nx2, size.d_view(m).x2min, size.d_view(m).x2max);
    Real r = (x1v*wv_.cos_a + x2v*wv_.sin_a)/wv_.lambda;
    Real sinr = sin(2.0*M_PI*r);
    Real cosr = cos(2.0*M_PI*r);
    Real d0 = wv_.dens;
    Real p0 = wv_.pgas;
    Real uflow = wv_.vflow;
    u0(m,IDN,k,j,i) = d0 + wv_.amp*sinr;
    u0(m,IEN,k,j,i) = p0/gm1 + 0.5*d0*uflow*uflow
                    + wv_.amp*(sinr*wv_.E0R - cosr*wv_.E0I);
    u0(m,IM1,k,j,i) = wv_.cos_a*d0*uflow
                    + wv_.cos_a*wv_.amp*(sinr*wv_.V0R - cosr*wv_.V0I);
    u0(m,IM2,k,j,i) = wv_.sin_a*d0*uflow
                    + wv_.sin_a*wv_.amp*(sinr*wv_.V0R - cosr*wv_.V0I);
    u0(m,IM3,k,j,i) = 0.0;
  });

  Kokkos::deep_copy(pmbp->pradvet->ir, wv.B0);
}
