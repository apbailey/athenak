#!/bin/bash
# One-shot: submit Phase 1 for both devices, then Phase 2 sweeps after N* is known.
# Usage:
#   bash rt-profiling/vista/setup_and_submit.sh phase1
#   bash rt-profiling/vista/setup_and_submit.sh phase2 <N_star_gh200> <N_star_b200>
#   bash rt-profiling/vista/setup_and_submit.sh phase2_gh200 <N_star>
#   bash rt-profiling/vista/setup_and_submit.sh phase2_b200 <N_star>
set -euo pipefail
ROOT="${ATHENAK:-$HOME/athenak-rt-sc}"
WORK="${WORK:-/work/11121/davela/vista}/rt_profiling"
cd "$ROOT"

case "${1:-}" in
  phase1)
    mkdir -p "$WORK/gh200" "$WORK/b200"
    cd "$WORK/gh200" && sbatch "$ROOT/rt-profiling/gh200/submit_phase1.sbatch"
    cd "$WORK/b200" && sbatch "$ROOT/rt-profiling/b200/submit_phase1.sbatch"
    echo "Phase 1 submitted. Monitor: sacct -u \$USER --starttime today"
    ;;
  phase2_gh200)
    N="${2:?set N_star for GH200}"
    mkdir -p "$WORK/gh200"
    cd "$WORK/gh200"
    for sw in wavefront diagonal diagonal_compact; do
      RT_N_STAR="$N" RT_SWEEPS="$sw" sbatch "$ROOT/rt-profiling/gh200/submit_sweep.sbatch"
      sleep 2
    done
    echo "GH200 Phase 2 submitted (N*=$N)"
    ;;
  phase2_b200)
    N="${2:?set N_star for B200}"
    mkdir -p "$WORK/b200"
    cd "$WORK/b200"
    for sw in wavefront diagonal diagonal_compact; do
      RT_N_STAR="$N" RT_SWEEPS="$sw" sbatch "$ROOT/rt-profiling/b200/submit_sweep.sbatch"
      sleep 5
    done
    echo "B200 Phase 2 submitted (N*=$N)"
    ;;
  phase2)
    NGH="${2:?set N_star_gh200}"
    NB200="${3:?set N_star_b200}"
    "$0" phase2_gh200 "$NGH"
    "$0" phase2_b200 "$NB200"
    ;;
  *)
    echo "Usage: $0 phase1 | phase2 <N_gh200> <N_b200> | phase2_gh200 <N> | phase2_b200 <N>"
    exit 1
    ;;
esac
