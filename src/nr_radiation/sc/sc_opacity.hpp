#ifndef NR_RADIATION_SC_SC_OPACITY_HPP_
#define NR_RADIATION_SC_SC_OPACITY_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file sc_opacity.hpp
//! \brief User-enrollable opacity and radiation-source hooks for the SC solver.
//!
//! Both hooks are HOST function pointers, enrolled from the problem generator via
//! SC::EnrollOpacityFunction / SC::EnrollPlanckFunction (pmbp->pnrrad->Enroll...).
//! The enrolled function is called on the host once per cycle at the top of
//! SC::UpdateOpacityAndSource (i.e. before the iteration loop of each SolveTransfer)
//! and must launch its own par_for over the mesh. Parameters reach the kernel via the
//! usual pgen idiom: a file-scope POD struct copied to a local and captured by value
//! (see src/pgen/sbox.cpp for the canonical example).
//!
//! Contract:
//!  - Fill ALL cells INCLUDING ghosts (loop bounds 0..ncells-1 in every active
//!    dimension). sigma_a/sigma_s/planck have no ghost-zone communication; the sweep
//!    reads chi at ghost footpoints, so stale ghosts corrupt the formal solution.
//!  - Opacity hook: fill sigma_a(m,0,k,j,i) = kappa_a*rho and
//!    sigma_s(m,0,k,j,i) = kappa_s*rho (per-volume opacities). The module then derives
//!    chi = sigma_a + sigma_s and eps = sigma_a/chi (unless <nr_radiation>/eps is set),
//!    and auto-enables ALI when any cell has sigma_s > 0. Do NOT write chi/eps directly.
//!    When an opacity hook is enrolled, <nr_radiation>/opa is optional and ignored.
//!  - Source hook: fill planck(m,k,j,i) = B (the thermal/emission source function, in
//!    intensity units; default is B = T^4 from fluid prims). With ALI, S iterates via
//!    S = (1-eps)J + eps*B; in the LTE path S = B exactly. A "fixed S" problem is
//!    therefore an enrolled B with eps = 1 (no scattering).
//!  - Enroll BEFORE any `if (restart) return;` in the pgen so restarted runs
//!    re-enroll (the pgen runs on the restart path with restart=true).

class MeshBlockPack;

namespace nr_radiation {

// user-enrollable opacity function: fills sigma_a, sigma_s over all cells incl ghosts
using SCOpacityFnPtr = void (*)(MeshBlockPack *pmbp);
// user-enrollable radiation source function: fills planck (B) over all cells incl ghosts
using SCPlanckFnPtr = void (*)(MeshBlockPack *pmbp);

}  // namespace nr_radiation

#endif  // NR_RADIATION_SC_SC_OPACITY_HPP_
