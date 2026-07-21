#ifndef UTILS_PERF_HPP_
#define UTILS_PERF_HPP_
//========================================================================================
// AthenaK astrophysical plasma code
// Copyright(C) 2024 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file perf.hpp
//! \brief athinput-<output>-driven perf-diagnostics framework: probe classes + registry.
//!
//! Perf is configured via <output file_type=perf> blocks (there is NO <perf> block). Example:
//!     <output3>
//!     file_type  = perf
//!     perf_probe = kernels,memory   # comma-list of {kernels,memory,comms,iteration}; OMIT = all
//!     kernels    = vet_sweep*       # optional collection filter (only these kernels are timed)
//!     dcycle     = 50               # cadence (or dt=..)
//! perf::Enable() scans those blocks EARLY (before any kernel), builds a registry of PerfProbe
//! instances, and registers the Kokkos callbacks / hooks. The PerfOutput athenak output type then
//! queries a probe via GetProbe() and writes its cumulative Rows() to <basename>.<probe> at the
//! block's cadence. The binary emits only RAW reductions; derived quantities (avg_ms, pct, gcaups)
//! are computed downstream in analyze.py from successive snapshots.

#include <string>
#include <utility>
#include <vector>

class ParameterInput;

namespace perf {

//! One row of a probe's wide output table: ordered string identity columns + numeric metric columns.
struct Row {
  std::vector<std::pair<std::string, std::string>> str_cols;
  std::vector<std::pair<std::string, double>> num_cols;
};

//! Abstract instrumentation probe. Subclasses own their accumulator state and register their own
//! Kokkos callbacks / hooks in Setup(). Rows() returns the CUMULATIVE, rank-local snapshot of raw
//! reductions; PerfOutput writes it at each cadence trigger (per-interval deltas derived downstream).
class PerfProbe {
 public:
  virtual ~PerfProbe() = default;
  virtual const char *Name() const = 0;                       //!< file suffix, e.g. "kernels"
  virtual void Setup(ParameterInput *pin, const std::string &block) = 0;
  virtual std::vector<Row> Rows() const = 0;
  virtual void Reduce(std::vector<Row> & /*rows*/) const {}   //!< optional cross-rank MPI reduce
};

//! Scan <output file_type=perf> blocks, build the probe registry, and register callbacks/hooks.
//! Call ONCE, after ModifyFromCmdline and BEFORE the mesh is built (callbacks must be live before
//! the first kernel). pin==nullptr registers nothing (zero overhead when no perf output block).
void Enable(ParameterInput *pin);

//! Look up an enabled probe by name (== its Name()); nullptr if not enabled.
PerfProbe *GetProbe(const std::string &name);

//! Block name of the FIRST <output> block that named a probe (its owner/writer); empty if the probe
//! is not enabled. Lets PerfOutput write each probe from exactly one block.
std::string ProbeOwner(const std::string &probe);

//! Parse a `perf_probe` spec (comma-list; empty -> all four) into canonical probe names.
std::vector<std::string> ProbeNames(const std::string &spec);

//! Comms exposed-wait timing hook (active only if a comms probe is enabled). Wrap an MPI_Wait:
//!   double t0 = perf::CommWaitStart(); <wait loop> perf::CommWaitStop("recv_wait", t0);
//! Non-perturbing: it times the exposed wait that blocks anyway, not the overlapped transfer.
double CommWaitStart();
void CommWaitStop(const char *bucket, double t0);

//! Per-solve iteration/convergence hook (active only if an iteration probe is enabled). Called once
//! per VET solve; accumulates n_solves / cum_niter and records last resid/nblocks and cells/nang
//! (cells x nang x cum_niter / sweep-time -> gcaups is derived in analyze.py).
void EmitIteration(int niter, double resid, int nblocks, int cells, int nang);

}  // namespace perf

#endif  // UTILS_PERF_HPP_
