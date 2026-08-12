# Vista GH200 — rt-profiling run metadata

- **Cluster:** TACC Vista, partition `gh`, account `AST22008`
- **GPU:** NVIDIA GH200 120GB, 97871 MiB (~95.6 GiB), driver 590.48.01
- **Module:** `nvhpc-hpcx-cuda13/26.1`
- **Binary:** `build_gh200/` (`Kokkos_ARCH_HOPPER90=ON`, Kokkos 4.7.02)
- **Launcher:** `ibrun` (1 MPI rank / 1 GPU)
- **Phase 1 N★:** 224³ (16³ blocks, nmu=6, 168 rays; peak 73.6 GiB ≈ 77% fill; 256³ OOM)
- **Phase 2 block ladder:** divisors of 224 ≥16 → {16, 28, 56, 112, 224}
- **Jobs:** phase1=903230; sweeps=903247 (wavefront), 903248 (diagonal), 903249 (diagonal_compact)
- **Run date:** 2026-08-11
