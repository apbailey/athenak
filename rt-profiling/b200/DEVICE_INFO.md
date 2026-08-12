# Vista B200 — rt-profiling run metadata

- **Cluster:** TACC Vista, partition `gb`, account `AST22008`
- **GPU:** NVIDIA GB200, 189471 MiB (~185 GiB) ×4 per node (1 GPU used), driver 590.48.01
- **Module:** `nvhpc-hpcx-cuda13/26.1`
- **Binary:** `build_b200/` (`Kokkos_ARCH_BLACKWELL100=ON`, Kokkos 4.7.02)
- **Launcher:** `ibrun` (1 MPI rank / 1 GPU)
- **Phase 1 N★:** 256³ (16³ blocks, nmu=6, 168 rays; peak 109.7 GiB ≈ 59% fill vs 160 GiB ceiling; 320³ OOM)
- **Phase 2 block ladder:** divisors of 256 ≥16 → {16, 32, 64, 128, 256}
- **Jobs:** phase1=903411; sweeps=903471 (wavefront), 903472 (diagonal), 903473 (diagonal_compact)
- **Run date:** 2026-08-11
- **Note:** `kernel_resources.py` patched on-cluster to add `sm_100` occupancy limits
