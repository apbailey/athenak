//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file sc_beam.cpp
//! \brief Pencil-beam test for the LTE short-characteristics SC solver.
//! Ported from Athena-C beam2d.c. Injects unit intensity at selected boundary cells along
//! a chosen discrete-ordinate angle; chi=const (= opa since affect_fluid=false means
//! UpdateOpacityAndSource uses opa directly), B=0 (Davis 2012 Fig. 6 linear-interp).

#include <cmath>
#include <cstdio>
#include <iostream>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"
#include "eos/eos.hpp"
#include "pgen/pgen.hpp"
#include "nr_radiation/nr_radiation.hpp"

namespace {
struct BeamVars {
  int iang;
  int ihor1, ihor2;
  int ivert1, ivert2;
  Real dens, pgas;
};
BeamVars beamvars;

void SCBeamBCs(Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  if (pmbp->pnrrad == nullptr) return;
  auto &indcs = pm->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks;
  int ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng) : 1;
  int nmb = pmbp->nmb_thispack;
  int nang = pmbp->pnrrad->pang->nang;
  int nang_tot = pmbp->pnrrad->nang_tot;
  int nx1 = indcs.nx1, nx2 = indcs.nx2;
  int iang = beamvars.iang;
  int ihor1 = beamvars.ihor1, ihor2 = beamvars.ihor2;
  int ivert1 = beamvars.ivert1, ivert2 = beamvars.ivert2;
  LogicalLocation *lloc = pm->lloc_eachmb;

  if (pmbp->phydro != nullptr) {
    auto u0 = pmbp->phydro->u0;
    int nvar = u0.extent_int(1);
    auto &mb_bcs = pmbp->pmb->mb_bcs;
    par_for("beam_hyd_x1", DevExeSpace(), 0, (nmb-1), 0, (nvar-1), 0, (n3-1), 0, (n2-1),
    KOKKOS_LAMBDA(int m, int n, int k, int j) {
      if (mb_bcs.d_view(m,BoundaryFace::inner_x1) == BoundaryFlag::user) {
        for (int i=0; i<ng; ++i) u0(m,n,k,j,is-i-1) = u0(m,n,k,j,is);
      }
      if (mb_bcs.d_view(m,BoundaryFace::outer_x1) == BoundaryFlag::user) {
        for (int i=0; i<ng; ++i) u0(m,n,k,j,ie+i+1) = u0(m,n,k,j,ie);
      }
    });
    if (n2 > 1) {
      par_for("beam_hyd_x2", DevExeSpace(), 0, (nmb-1), 0, (nvar-1), 0, (n3-1), 0, (n1-1),
      KOKKOS_LAMBDA(int m, int n, int k, int i) {
        if (mb_bcs.d_view(m,BoundaryFace::inner_x2) == BoundaryFlag::user) {
          for (int j=0; j<ng; ++j) u0(m,n,k,js-j-1,i) = u0(m,n,k,js,i);
        }
        if (mb_bcs.d_view(m,BoundaryFace::outer_x2) == BoundaryFlag::user) {
          for (int j=0; j<ng; ++j) u0(m,n,k,je+j+1,i) = u0(m,n,k,je,i);
        }
      });
    }
  }

  auto ir = pmbp->pnrrad->ir;
  auto ir_h = Kokkos::create_mirror_view(ir);
  Kokkos::deep_copy(ir_h, ir);
  auto mb_bcs_h = pmbp->pmb->mb_bcs.h_view;
  auto mb_gid_h = pmbp->pmb->mb_gid.h_view;

  for (int m=0; m<nmb; ++m) {
    int gid = mb_gid_h(m);
    int g0 = lloc[gid].lx1 * nx1;
    int h0 = lloc[gid].lx2 * nx2;

    if (mb_bcs_h(m,BoundaryFace::inner_x2) == BoundaryFlag::user) {
      for (int ig=0; ig<ng; ++ig)
        for (int i=0; i<n1; ++i)
          for (int a=0; a<nang_tot; ++a) ir_h(m,a,ks,js-ig-1,i) = 0.0;
      for (int i=is; i<=ie; ++i) {
        int gi = g0 + (i - is);
        for (int ig=0; ig<ng; ++ig) {
          if (ihor1 >= 0 && gi == ihor1) ir_h(m,0*nang+iang,ks,js-ig-1,i) = 1.0;
          if (ihor2 >= 0 && gi == ihor2) ir_h(m,1*nang+iang,ks,js-ig-1,i) = 1.0;
        }
      }
    }
    if (mb_bcs_h(m,BoundaryFace::outer_x2) == BoundaryFlag::user) {
      for (int ig=0; ig<ng; ++ig)
        for (int i=0; i<n1; ++i)
          for (int a=0; a<nang_tot; ++a) ir_h(m,a,ks,je+ig+1,i) = 0.0;
    }
    if (mb_bcs_h(m,BoundaryFace::inner_x1) == BoundaryFlag::user) {
      for (int ig=0; ig<ng; ++ig)
        for (int j=0; j<n2; ++j)
          for (int a=0; a<nang_tot; ++a) ir_h(m,a,ks,j,is-ig-1) = 0.0;
      for (int j=js; j<=je; ++j) {
        int gj = h0 + (j - js);
        for (int ig=0; ig<ng; ++ig) {
          if (ivert1 >= 0 && gj == ivert1) ir_h(m,0*nang+iang,ks,j,is-ig-1) = 1.0;
          if (ivert2 >= 0 && gj == ivert2) ir_h(m,2*nang+iang,ks,j,is-ig-1) = 1.0;
        }
      }
    }
    if (mb_bcs_h(m,BoundaryFace::outer_x1) == BoundaryFlag::user) {
      for (int ig=0; ig<ng; ++ig)
        for (int j=0; j<n2; ++j)
          for (int a=0; a<nang_tot; ++a) ir_h(m,a,ks,j,ie+ig+1) = 0.0;
    }
  }
  Kokkos::deep_copy(ir, ir_h);
}

void SCBeamErrors(ParameterInput *pin, Mesh *pm) {
  (void)pin;
  auto *prv = pm->pmb_pack->pnrrad;
  if (prv == nullptr) return;
  std::cout << "sc_beam: last_niter=" << prv->last_niter << std::endl;
  auto j_h = Kokkos::create_mirror_view(prv->jmean);
  Kokkos::deep_copy(j_h, prv->jmean);
  auto &indcs = pm->mb_indcs;
  Real jmx = 0.0;
  int nmb = pm->pmb_pack->nmb_thispack;
  for (int m=0; m<nmb; ++m)
    for (int j=indcs.js; j<=indcs.je; ++j)
      for (int i=indcs.is; i<=indcs.ie; ++i)
        jmx = std::max(jmx, j_h(m,indcs.ks,j,i));
  std::cout << "sc_beam: max(J)=" << jmx << std::endl;

  FILE *fp = std::fopen("jmean_dump.bin", "wb");
  if (fp) {
    int nx1 = indcs.nx1, nx2 = indcs.nx2;
    std::fwrite(&nmb, sizeof(int), 1, fp);
    std::fwrite(&nx1, sizeof(int), 1, fp);
    std::fwrite(&nx2, sizeof(int), 1, fp);
    for (int m=0; m<nmb; ++m)
      for (int j=indcs.js; j<=indcs.je; ++j)
        for (int i=indcs.is; i<=indcs.ie; ++i) {
          float val = static_cast<float>(j_h(m,indcs.ks,j,i));
          std::fwrite(&val, sizeof(float), 1, fp);
        }
    std::fclose(fp);
  }
}
}  // namespace

void ProblemGenerator::SCBeam(ParameterInput *pin, const bool restart) {
  user_bcs_func = SCBeamBCs;
  pgen_final_func = SCBeamErrors;

  beamvars.iang   = pin->GetOrAddInteger("problem", "iang", 2);
  beamvars.ihor1  = pin->GetOrAddInteger("problem", "ihor1", 96);
  beamvars.ihor2  = pin->GetOrAddInteger("problem", "ihor2", -1);
  beamvars.ivert1 = pin->GetOrAddInteger("problem", "ivert1", -1);
  beamvars.ivert2 = pin->GetOrAddInteger("problem", "ivert2", -1);
  beamvars.dens   = pin->GetOrAddReal("problem", "dens", 1.0);
  beamvars.pgas   = pin->GetOrAddReal("problem", "pgas", 1.0);
  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->pnrrad == nullptr || pmbp->phydro == nullptr) {
    std::cout << "### FATAL ERROR: sc_beam needs <hydro> and <nr_radiation>"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  auto &pang = *pmbp->pnrrad->pang;
  if (beamvars.iang < 0 || beamvars.iang >= pang.nang) {
    std::cout << "### FATAL ERROR: problem/iang out of range" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::cout << "sc_beam: iang=" << beamvars.iang
            << " mu0=(" << pang.mu.h_view(0,beamvars.iang,0) << ","
            << pang.mu.h_view(0,beamvars.iang,1) << ")" << std::endl;

  auto &indcs = pmy_mesh_->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  auto u0 = pmbp->phydro->u0;
  Real gm1 = pmbp->phydro->peos->eos_data.gamma - 1.0;
  Real dens = beamvars.dens, pgas = beamvars.pgas;
  int nmb1 = pmbp->nmb_thispack - 1;
  par_for("sc_beam_ic", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    u0(m,IDN,k,j,i) = dens;
    u0(m,IM1,k,j,i) = 0.0;
    u0(m,IM2,k,j,i) = 0.0;
    u0(m,IM3,k,j,i) = 0.0;
    u0(m,IEN,k,j,i) = pgas / gm1;
  });
}
