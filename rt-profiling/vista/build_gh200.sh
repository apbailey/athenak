#!/bin/bash
# Build AthenaK for Vista GH200 (Hopper sm_90). Run on Vista login node.
set -euo pipefail
ROOT="${1:-$HOME/athenak-rt-sc}"
cd "$ROOT"

ml purge
ml use /home1/apps/nvidia/modulefiles
ml nvhpc-hpcx-cuda13/26.1

# Init kokkos if needed
if [ ! -f kokkos/CMakeLists.txt ]; then
  git submodule update --init kokkos
fi

# Patch nvcc_wrapper for Hopper
cd kokkos
git checkout -- bin/nvcc_wrapper 2>/dev/null || true
sed -i 's/sm_80/sm_90/g; s/sm_70/sm_90/g; s|g++|nvc++|g; s/default_arch="sm_70"/default_arch="sm_90"/' bin/nvcc_wrapper
cd "$ROOT"

cmake -S . -B build_gh200 \
  -DCMAKE_CXX_COMPILER="$ROOT/kokkos/bin/nvcc_wrapper" \
  -DKokkos_ENABLE_CUDA=ON \
  -DKokkos_ARCH_HOPPER90=ON \
  -DAthena_ENABLE_MPI=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build_gh200 -j16
echo "Built: $ROOT/build_gh200/src/athena"
