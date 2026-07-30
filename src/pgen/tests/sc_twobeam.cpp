//========================================================================================
// AthenaK astrophysical plasma code
// Copyright(C) 2024 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file sc_twobeam.cpp
//! \brief Two crossing pencil beams -- mirror-symmetry test for the SC formal solution.
//!
//! Port of the Athena++ beam.cpp TwoBeams test (via apb_rad). Two collimated pencil beams are
//! injected into the lower-x2 ghosts of a 2D vacuum box using the first ("a0") ordinate of the
//! two upward octants -- octant 0 (up-right, mu_x>0) at x=-x0, octant 1 (up-left, mu_x<0) at
//! x=+x0 -- and stream in straight lines, crossing in an X. The two ordinates differ ONLY in
//! the sign of mu_x (equal weights), so the beams are exact mirror images: the resulting
//! E_r = J must be symmetric under x -> -x. Because short characteristics commutes with
//! reflection, the asymmetry is machine-zero -- a teeth-y invariant needing NO analytic (the
//! spread beam itself, Davis+2012 Fig. 6, has no closed form). nmu>=2 gives a non-diagonal
//! ordinate whose upwind point falls between cells, so the bilinear interp spreads the beam
//! (the physically interesting cross-diffusion); nmu=1 would give the 45-degree grid diagonal
//! (diffusion-free). The pgen std::exit(EXIT_FAILURE)s if the x-mirror asymmetry exceeds tol.

#include <cmath>
#include <cstdlib>
#include <iostream>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/cell_locations.hpp"
#include "pgen/pgen.hpp"
#include "nr_radiation/nr_radiation.hpp"

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::SCTwoBeam()

void ProblemGenerator::SCTwoBeam(ParameterInput *pin, const bool restart) {
  (void)restart;
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->pnrrad == nullptr) {
    std::cout << "### FATAL ERROR in sc_twobeam: requires a <nr_radiation> block"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  nr_radiation::SC *psc = pmbp->pnrrad;

  auto &indcs = pmy_mesh_->mb_indcs;
  const int is = indcs.is, ie = indcs.ie, js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int ng = indcs.ng;
  const int n1 = indcs.nx1 + 2*ng;
  const int n2 = (indcs.nx2 > 1) ? indcs.nx2 + 2*ng : 1;
  const int n3 = (indcs.nx3 > 1) ? indcs.nx3 + 2*ng : 1;
  const int nx1 = indcs.nx1;
  const int nmb1 = pmbp->nmb_thispack - 1;
  const int nang = psc->pang->nang;
  const int nang_tot = psc->nang_tot;
  const int nangt1 = nang_tot - 1;

  if (indcs.nx2 == 1) {
    std::cout << "### FATAL ERROR in sc_twobeam: needs a 2D mesh (nx2 > 1)" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  const Real i_beam = pin->GetOrAddReal("problem", "i_beam", 10.0);
  const Real x0     = pin->GetOrAddReal("problem", "x0", 0.5);
  const Real hw     = pin->GetOrAddReal("problem", "halfwidth", 0.04);

  auto chi_a = psc->chi;
  auto srad_a  = psc->srad;
  auto ir    = psc->ir;
  auto &size = pmbp->pmb->mb_size;

  // vacuum, zero source, zero intensity everywhere (including ghosts)
  par_for("sc_2beam_zero", DevExeSpace(), 0,nmb1, 0,n3-1, 0,n2-1, 0,n1-1,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    chi_a(m,k,j,i)  = 0.0;
    srad_a(m,0,k,j,i) = 0.0;
  });
  par_for("sc_2beam_ir0", DevExeSpace(), 0,nmb1, 0,nangt1, 0,n3-1, 0,n2-1, 0,n1-1,
  KOKKOS_LAMBDA(int m, int a, int k, int j, int i) { ir(m,a,k,j,i) = 0.0; });
  Kokkos::fence();

  // inject two pencil beams into the lower-x2 ghosts using the first ("a0") ordinate of the
  // two upward octants: octant 0 (up-right, mu_x>0) -> angg = 0*nang + 0 = 0 at x=-x0;
  // octant 1 (up-left, mu_x<0) -> angg = 1*nang + 0 = nang at x=+x0. Exact mirror images.
  const int angg_up_right = 0;      // octant 0, angle a=0
  const int angg_up_left  = nang;   // octant 1, angle a=0
  par_for("sc_2beam_bc", DevExeSpace(), 0,nmb1, 0,n1-1,
  KOKKOS_LAMBDA(int m, int i) {
    Real x = CellCenterX(i-is, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
    for (int jg = 0; jg < js; ++jg) {
      for (int kk = ks; kk <= ke; ++kk) {
        if (std::fabs(x + x0) < hw) ir(m, angg_up_right, kk, jg, i) = i_beam;  // up-right
        if (std::fabs(x - x0) < hw) ir(m, angg_up_left,  kk, jg, i) = i_beam;  // up-left
      }
    }
  });
  Kokkos::fence();

  psc->FormalSolution();
  Kokkos::fence();
  psc->CalculateMoments();
  Kokkos::fence();

  // E_r (= jmean J) must be mirror-symmetric under x -> -x (domain symmetric about x=0)
  auto jmean_h = Kokkos::create_mirror_view(psc->jmean);
  Kokkos::deep_copy(jmean_h, psc->jmean);

  Real max_er = 0.0, max_asym = 0.0;
  for (int m=0; m<=nmb1; ++m)
  for (int k=ks; k<=ke; ++k) for (int j=js; j<=je; ++j) for (int i=is; i<=ie; ++i) {
    Real er  = jmean_h(m,k,j,i);
    Real erm = jmean_h(m,k,j,is+ie-i);      // mirror column
    if (std::fabs(er) > max_er) max_er = std::fabs(er);
    if (std::fabs(er - erm) > max_asym) max_asym = std::fabs(er - erm);
  }
  Real rel_asym = (max_er > 0.0) ? max_asym/max_er : 0.0;

  const Real tol = pin->GetOrAddReal("problem", "tol", 1.0e-12);
  std::cout << "SC twobeam: max E_r = " << max_er << ", x-mirror asymmetry = "
            << rel_asym << std::endl;
  if (rel_asym > tol) {
    std::cout << "### SC twobeam test FAILED: x-mirror asymmetry " << rel_asym
              << " > tol " << tol << std::endl;
    std::exit(EXIT_FAILURE);
  }
  return;
}
