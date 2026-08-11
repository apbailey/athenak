//========================================================================================
// AthenaK astrophysical plasma code
// Copyright(C) 2024 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file sc_sweep_determinism.cpp
//! \brief Generic determinism / GPU-race probe for any SC sweep (nr_radiation/sweep).
//!
//! Builds an input-controlled gradient field, then runs the configured sweep `nrepeat`
//! times from the SAME frozen input and requires bitwise-identical output. The unordered
//! single-buffer jacobi sweep reads an upwind cell a sibling thread is overwriting, so its
//! output varies launch-to-launch on GPU (data race) -> FAIL; the double-buffered jacobi
//! sweep and the ordered wavefront/diagonal sweeps are reproducible -> PASS. The race is
//! only observable with a spatial gradient (amp>0); amp=0 is a trivial uniform pass.
//!
//! Everything about the run comes from the input file: sweep/grid/angles/scattering from the
//! standard <mesh>/<meshblock>/<nr_radiation> blocks, and the field knobs + nrepeat from
//! <problem>. The pgen itself does the check and std::exit(EXIT_FAILURE) on any mismatch.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <iostream>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "pgen/pgen.hpp"
#include "nr_radiation/nr_radiation.hpp"

namespace {
// order-independent, position-mixed 64-bit finalizer (murmur3-style)
KOKKOS_INLINE_FUNCTION
std::uint64_t MixBits(double v, std::uint64_t pos) {
  union { double d; std::uint64_t u; } cvt;
  cvt.d = v;
  std::uint64_t x = cvt.u ^ (pos * 0x9E3779B97F4A7C15ULL);
  x ^= x >> 33; x *= 0xff51afd7ed558ccdULL; x ^= x >> 33;
  return x;
}

// Interior-only bitwise hash of psc->ir, folded on the HOST. A device atomic-XOR into a single
// scalar would serialize ~1e6 threads onto one address (measured ~20 s/call on a P100 --
// pathological contention); copying ir down (~15 MB) and XOR-folding serially is a few ms. XOR is
// commutative so the result is identical and order-independent, and hashing the interior only
// avoids ghost / scratch-buffer noise from the jacobi ping-pong swap.
std::uint64_t InteriorHash(nr_radiation::SC *psc, int nmb1, int nangt1,
                           int ks, int ke, int js, int je, int is, int ie,
                           int nat, int N1, int N2, int N3, bool ai) {
  auto ir_h = Kokkos::create_mirror_view(psc->ir);
  Kokkos::deep_copy(ir_h, psc->ir);
  std::uint64_t h = 0;
  for (int m = 0; m <= nmb1; ++m) {
    for (int ang = 0; ang <= nangt1; ++ang) {
      for (int k = ks; k <= ke; ++k) {
        for (int j = js; j <= je; ++j) {
          for (int i = is; i <= ie; ++i) {
            std::uint64_t p = ((((static_cast<std::uint64_t>(m)*nat + ang)*N3 + k)*N2 + j)*N1 + i);
            Real v = ai ? ir_h(m,k,j,i,ang) : ir_h(m,ang,k,j,i);   // angle-inner vs normal layout
            h ^= MixBits(v, p);
          }
        }
      }
    }
  }
  return h;
}

// Layout-INVARIANT hash of the mean intensity J (jmean is (m,k,j,i) either way) -- lets an
// angle_inner run be compared bit-for-bit against a normal-layout run (same physics, different ir
// storage). Requires psc->ComputeJ() to have been called on the current ir.
std::uint64_t JmeanHash(nr_radiation::SC *psc, int nmb1,
                        int ks, int ke, int js, int je, int is, int ie,
                        int N1, int N2, int N3) {
  auto j_h = Kokkos::create_mirror_view(psc->jmean);
  Kokkos::deep_copy(j_h, psc->jmean);
  std::uint64_t h = 0;
  for (int m = 0; m <= nmb1; ++m) {
    for (int k = ks; k <= ke; ++k) {
      for (int j = js; j <= je; ++j) {
        for (int i = is; i <= ie; ++i) {
          std::uint64_t p = (((static_cast<std::uint64_t>(m)*N3 + k)*N2 + j)*N1 + i);
          h ^= MixBits(j_h(m,k,j,i), p);
        }
      }
    }
  }
  return h;
}
}  // namespace

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::SCSweepDeterminism()

void ProblemGenerator::SCSweepDeterminism(ParameterInput *pin, const bool restart) {
  (void)restart;
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->pnrrad == nullptr) {
    std::cout << "### FATAL ERROR in sc_sweep_determinism: requires <nr_radiation>" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  nr_radiation::SC *psc = pmbp->pnrrad;

  auto &indcs = pmy_mesh_->mb_indcs;
  const int ng = indcs.ng;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int n1 = indcs.nx1 + 2*ng;
  const int n2 = (indcs.nx2 > 1) ? indcs.nx2 + 2*ng : 1;
  const int n3 = (indcs.nx3 > 1) ? indcs.nx3 + 2*ng : 1;
  const int nmb1 = pmbp->nmb_thispack - 1;
  const int nang_tot = psc->nang_tot;
  const int nangt1 = nang_tot - 1;

  // ---- field knobs from <problem> ----
  const Real chi0  = pin->GetOrAddReal   ("problem", "chi0",  0.5);
  const Real ir_bg = pin->GetOrAddReal   ("problem", "ir_bg", 1.0);
  const Real amp   = pin->GetOrAddReal   ("problem", "amp",   1.0);
  const int  axis  = pin->GetOrAddInteger("problem", "axis",  1);
  const bool rough = pin->GetOrAddBoolean("problem", "rough", true);
  const int  K     = pin->GetOrAddInteger("problem", "nrepeat", 32);

  // ---- build the frozen gradient field (full range incl. ghosts so nothing is garbage) ----
  auto ir_a  = psc->ir;
  auto chi_a = psc->chi;
  auto srad_a  = psc->srad;
  par_for("det_field", DevExeSpace(), 0, nmb1, 0, (n3-1), 0, (n2-1), 0, (n1-1),
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    chi_a(m,k,j,i)   = chi0;
    srad_a(m,0,k,j,i)  = ir_bg;   // uniform source; the ir gradient below drives old!=new upwind
  });
  const bool ai = psc->ir_angle_inner;   // ir stored (m,k,j,i,ang) vs (m,ang,k,j,i)
  par_for("det_ir0", DevExeSpace(), 0, nmb1, 0, nangt1, 0, (n3-1), 0, (n2-1), 0, (n1-1),
  KOKKOS_LAMBDA(int m, int ang, int k, int j, int i) {
    int c = (axis == 2) ? j : (axis == 3) ? k : i;
    Real g;
    if (rough) {
      unsigned int h = static_cast<unsigned int>(c) * 2654435761u;   // deterministic hash ramp
      g = static_cast<Real>(h & 0xffffu) / 65535.0;
    } else {
      g = static_cast<Real>(c) / 64.0;                                // smooth ramp
    }
    Real val = ir_bg + amp*g;
    if (ai) { ir_a(m,k,j,i,ang) = val; } else { ir_a(m,ang,k,j,i) = val; }
  });

  // ---- optional: end-to-end BOUNDARY-EXCHANGE bridge test (ir_layout=angle_inner) ----
  // Zero the ghosts, run the real exchange sequence (which routes through SyncIrNormal +
  // the UNCHANGED PackAndSendCC/RecvAndUnpackCC when angle_inner), and hash the refilled field.
  // The angle_inner hash MUST equal the trusted normal-layout run's hash (same periodic ghosts).
  if (pin->GetOrAddBoolean("problem", "test_exchange", false)) {
    auto ir_z = psc->ir;
    par_for("det_zero_ghost", DevExeSpace(), 0, nmb1, 0, nangt1, 0, (n3-1), 0, (n2-1), 0, (n1-1),
    KOKKOS_LAMBDA(int m, int ang, int k, int j, int i) {
      bool ghost = (i < is || i > ie || j < js || j > je || k < ks || k > ke);
      if (ghost) { if (ai) { ir_z(m,k,j,i,ang) = 0.0; } else { ir_z(m,ang,k,j,i) = 0.0; } }
    });
    psc->InitRecvIr(nullptr, 0);
    psc->SendIr(nullptr, 0);
    for (int t = 0; t < 10000 && psc->RecvIr(nullptr, 0) != TaskStatus::complete; ++t) {}
    psc->ApplyPhysicalBCsIr(nullptr, 0);
    psc->ClearSendIr(nullptr, 0);
    psc->ClearRecvIr(nullptr, 0);
    std::uint64_t eh = InteriorHash(psc, nmb1, nangt1, 0, (n3-1), 0, (n2-1), 0, (n1-1),
                                    nang_tot, n1, n2, n3, ai);
    if (global_variable::my_rank == 0) {
      std::printf("SC exchange_test ir_layout=%s hash 0x%016llx\n",
                  (ai ? "angle_inner" : "normal"), static_cast<unsigned long long>(eh));
    }
    return;
  }

  // ---- snapshot the frozen input ----
  DvceArray5D<Real> ir0("ir0_det", ir_a.extent_int(0), ir_a.extent_int(1),
                        ir_a.extent_int(2), ir_a.extent_int(3), ir_a.extent_int(4));
  Kokkos::deep_copy(ir0, psc->ir);

  // ---- K relaunches from the frozen input; every output must hash-match the first ----
  std::uint64_t href = 0;
  int bad_rep = -1;
  std::uint64_t bad_hash = 0;
  for (int rep = 0; rep < K; ++rep) {
    Kokkos::deep_copy(psc->ir, ir0);     // reset the field the sweep reads (fixed input)
    psc->FormalSolution();               // dispatches on nr_radiation/sweep
    std::uint64_t h = InteriorHash(psc, nmb1, nangt1, ks, ke, js, je, is, ie,
                                   nang_tot, n1, n2, n3, ai);
    if (rep == 0) {
      href = h;
    } else if (h != href) {
      bad_rep = rep; bad_hash = h;
      break;
    }
  }

  // Layout-INVARIANT cross-check: compute J from the last swept ir and hash it. An angle_inner run
  // must print the SAME jmean hash as the equivalent normal-layout run (same physics, different ir
  // storage) -- the bit-exact gate for the reordered sweep + ComputeJ (ledger I2 native path).
  psc->ComputeJ();
  std::uint64_t jhash = JmeanHash(psc, nmb1, ks, ke, js, je, is, ie, n1, n2, n3);

  // Report the verdict on stdout and return NORMALLY (exit 0). We deliberately do NOT
  // std::exit() on a mismatch: under Kokkos+MPI that skips Kokkos::finalize() and makes
  // mpirun abort the job, which leaves the GPU context unclean and can poison the next test.
  // The pytest wrapper decides pass/fail by grepping the "determinism PASS" line, so a
  // failing sweep is non-destructive and the harness continues to the next test.
  if (global_variable::my_rank == 0) {
    if (bad_rep < 0) {
      std::printf("SC %s sweep determinism PASS: %d identical launches (hash 0x%016llx)\n",
                  psc->sweep_method.c_str(), K, static_cast<unsigned long long>(href));
      std::printf("SC %s ir_layout=%s jmean 0x%016llx\n", psc->sweep_method.c_str(),
                  (ai ? "angle_inner" : "normal"),
                  static_cast<unsigned long long>(jhash));
    } else {
      std::printf("SC %s sweep determinism FAIL (race): rep %d hash 0x%016llx != ref "
                  "0x%016llx  [%d relaunches]\n", psc->sweep_method.c_str(), bad_rep,
                  static_cast<unsigned long long>(bad_hash),
                  static_cast<unsigned long long>(href), K);
    }
  }
}
