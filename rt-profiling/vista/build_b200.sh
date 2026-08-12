#!/bin/bash
# Build AthenaK for Vista B200 (Blackwell sm_100). Run on Vista login node.
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

# Patch nvcc_wrapper for Blackwell
cd kokkos
git checkout -- bin/nvcc_wrapper 2>/dev/null || true
sed -i 's/sm_80/sm_100/g; s/sm_70/sm_100/g; s/sm_90/sm_100/g; s|g++|nvc++|g; s/default_arch="sm_70"/default_arch="sm_100"/' bin/nvcc_wrapper
cd "$ROOT"

cmake -S . -B build_b200 \
  -DCMAKE_CXX_COMPILER="$ROOT/kokkos/bin/nvcc_wrapper" \
  -DKokkos_ENABLE_CUDA=ON \
  -DKokkos_ARCH_BLACKWELL100=ON \
  -DAthena_ENABLE_MPI=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build_b200 -j16
echo "Built: $ROOT/build_b200/src/athena"
