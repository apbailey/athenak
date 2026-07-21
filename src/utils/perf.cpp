//========================================================================================
// AthenaK astrophysical plasma code
// Copyright(C) 2024 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file perf.cpp
//! \brief Probe class hierarchy + registry for the <output file_type=perf> diagnostics framework.
//!
//! Each probe owns its accumulator state. The Kokkos parallel-region / allocation callbacks are C
//! function pointers, so they stay free functions that forward to the active probe instance via the
//! file-static g_*_probe pointers (set in each probe's Setup()). Enable() scans the <output> blocks
//! and builds the registry; PerfOutput (outputs/perf_output.cpp) reads a probe's Rows() at cadence.

#include "utils/perf.hpp"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"

namespace {

// Shared wall-clock timer for kernel + comms timing (constructed at static init).
Kokkos::Timer g_timer;

// Forward decls so the Kokkos C-callback free functions and the g_*_probe pointers resolve.
class KernelsProbe;
class MemoryProbe;
class CommsProbe;
class IterationProbe;
KernelsProbe   *g_kernels_probe = nullptr;
MemoryProbe    *g_memory_probe  = nullptr;
CommsProbe     *g_comms_probe   = nullptr;
IterationProbe *g_iter_probe    = nullptr;

void PerfBegin(const char *name, const uint32_t devid, uint64_t *kID);
void PerfEnd(uint64_t kID);
void PerfAlloc(const Kokkos::Tools::SpaceHandle space, const char *label,
               const void *ptr, const uint64_t size);
void PerfDealloc(const Kokkos::Tools::SpaceHandle space, const char *label,
                 const void *ptr, const uint64_t size);

std::string Sanitize(std::string s) {
  for (char &c : s) if (c == ' ' || c == '\t') c = '_';   // keep whitespace-delimited columns intact
  return s;
}

// Match a name against a comma-list of globs, each an exact string or a trailing-'*' prefix.
bool MatchFilter(const std::string &filter, const std::string &name) {
  if (filter.empty()) return true;                       // no filter -> track everything
  std::size_t i = 0;
  while (i < filter.size()) {
    std::size_t j = filter.find(',', i);
    std::string pat = filter.substr(i, (j == std::string::npos) ? std::string::npos : j - i);
    if (!pat.empty()) {
      if (pat.back() == '*') {
        if (name.compare(0, pat.size() - 1, pat, 0, pat.size() - 1) == 0) return true;
      } else if (pat == name) {
        return true;
      }
    }
    if (j == std::string::npos) break;
    i = j + 1;
  }
  return false;
}

//----------------------------------------------------------------------------------------
struct MemAcc { long long cur = 0; long long peak = 0; };

class KernelsProbe : public perf::PerfProbe {
 public:
  const char *Name() const override { return "kernels"; }
  void Setup(ParameterInput *pin, const std::string &block) override {
    filter_ = pin->GetOrAddString(block, "kernels", "");
    all_ = pin->GetOrAddBoolean(block, "kernels_all", false);
    g_kernels_probe = this;
    namespace KTE = Kokkos::Tools::Experimental;
    KTE::set_begin_parallel_for_callback(PerfBegin);
    KTE::set_end_parallel_for_callback(PerfEnd);
    KTE::set_begin_parallel_reduce_callback(PerfBegin);
    KTE::set_end_parallel_reduce_callback(PerfEnd);
    KTE::set_begin_parallel_scan_callback(PerfBegin);
    KTE::set_end_parallel_scan_callback(PerfEnd);
  }
  // Filter applied HERE so a non-matching kernel skips the fence (the real perturbation).
  void OnBegin(const char *name) {
    const char *nm = (name != nullptr && name[0] != '\0') ? name : "(unnamed)";
    active_ = MatchFilter(filter_, nm);
    if (active_) { name_ = nm; start_ = g_timer.seconds(); }
  }
  void OnEnd() {
    if (!active_) return;
    Kokkos::fence();  // force device completion so the delta is real kernel wall time
    auto &e = acc_[name_];
    e.first += g_timer.seconds() - start_;
    e.second += 1;
  }
  std::vector<perf::Row> Rows() const override {
    std::vector<perf::Row> rows;
    for (const auto &kv : acc_) {
      if (!all_ && kv.first.rfind("Kokkos::", 0) == 0) continue;   // skip internal regions
      perf::Row r;
      r.str_cols.emplace_back("kernel", Sanitize(kv.first));
      r.num_cols.emplace_back("count", static_cast<double>(kv.second.second));
      r.num_cols.emplace_back("total_ms", kv.second.first * 1.0e3);
      rows.push_back(std::move(r));
    }
    return rows;
  }
 private:
  std::map<std::string, std::pair<double, long>> acc_;   // label -> {accumulated sec, launch count}
  std::string filter_;
  bool all_ = false;
  bool active_ = false;
  std::string name_;
  double start_ = 0.0;
};

class MemoryProbe : public perf::PerfProbe {
 public:
  const char *Name() const override { return "memory"; }
  void Setup(ParameterInput * /*pin*/, const std::string & /*block*/) override {
    g_memory_probe = this;
    namespace KTE = Kokkos::Tools::Experimental;
    KTE::set_allocate_data_callback(PerfAlloc);
    KTE::set_deallocate_data_callback(PerfDealloc);
  }
  void OnAlloc(const char *space, const char *label, uint64_t size) {
    Add(space_, space, static_cast<long long>(size));
    Add(label_, (label != nullptr && label[0] != '\0') ? label : "(unnamed)",
        static_cast<long long>(size));
  }
  void OnDealloc(const char *space, const char *label, uint64_t size) {
    space_[space].cur -= static_cast<long long>(size);
    label_[(label != nullptr && label[0] != '\0') ? label : "(unnamed)"].cur -=
        static_cast<long long>(size);
  }
  std::vector<perf::Row> Rows() const override {
    std::vector<perf::Row> rows;
    for (const auto &kv : space_) {                    // per-space high-water-mark (GPU-OOM metric)
      perf::Row r;
      r.str_cols.emplace_back("kind", "space");
      r.str_cols.emplace_back("name", Sanitize(kv.first));
      r.num_cols.emplace_back("hwm_mb", kv.second.peak / 1.0e6);
      rows.push_back(std::move(r));
    }
    for (const auto &kv : label_) {                    // per-View-label peak (>= 1 MB)
      if (kv.second.peak < (1LL << 20)) continue;
      perf::Row r;
      r.str_cols.emplace_back("kind", "label");
      r.str_cols.emplace_back("name", Sanitize(kv.first));
      r.num_cols.emplace_back("hwm_mb", kv.second.peak / 1.0e6);
      rows.push_back(std::move(r));
    }
    return rows;
  }
 private:
  static void Add(std::map<std::string, MemAcc> &m, const std::string &k, long long d) {
    auto &e = m[k];
    e.cur += d;
    if (e.cur > e.peak) e.peak = e.cur;
  }
  std::map<std::string, MemAcc> space_, label_;
};

class CommsProbe : public perf::PerfProbe {
 public:
  const char *Name() const override { return "comms"; }
  void Setup(ParameterInput * /*pin*/, const std::string & /*block*/) override {
    g_comms_probe = this;
  }
  void Add(const char *bucket, double sec) { comm_[bucket] += sec; }
  std::vector<perf::Row> Rows() const override {
    std::vector<perf::Row> rows;
    for (const auto &kv : comm_) {
      perf::Row r;
      r.str_cols.emplace_back("bucket", Sanitize(kv.first));
      r.num_cols.emplace_back("wait_ms", kv.second * 1.0e3);
      rows.push_back(std::move(r));
    }
    return rows;
  }
  // recv_wait/send_wait are present on every rank -> reduce positionally to a global picture.
  // (Kernel/View-label maps can diverge across ranks, so those probes stay rank-0-representative.)
  void Reduce(std::vector<perf::Row> &rows) const override {
#if MPI_PARALLEL_ENABLED
    for (auto &r : rows) {
      for (auto &c : r.num_cols) {
        double v = c.second;
        MPI_Allreduce(MPI_IN_PLACE, &v, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        c.second = v;
      }
    }
#else
    (void)rows;
#endif
  }
 private:
  std::map<std::string, double> comm_;   // bucket -> accumulated exposed-wait seconds
};

class IterationProbe : public perf::PerfProbe {
 public:
  const char *Name() const override { return "iteration"; }
  void Setup(ParameterInput * /*pin*/, const std::string & /*block*/) override {
    g_iter_probe = this;
  }
  void OnSolve(int niter, double resid, int nblocks, int cells, int nang) {
    n_solves_ += 1;
    cum_niter_ += niter;
    last_resid_ = resid;
    last_nblocks_ = nblocks;
    cells_ = cells;
    nang_ = nang;
  }
  std::vector<perf::Row> Rows() const override {
    perf::Row r;
    r.num_cols.emplace_back("n_solves", static_cast<double>(n_solves_));
    r.num_cols.emplace_back("cum_niter", static_cast<double>(cum_niter_));
    r.num_cols.emplace_back("last_resid", last_resid_);
    r.num_cols.emplace_back("nblocks", static_cast<double>(last_nblocks_));
    r.num_cols.emplace_back("cells", static_cast<double>(cells_));
    r.num_cols.emplace_back("nang", static_cast<double>(nang_));
    return {r};
  }
 private:
  long n_solves_ = 0;
  long long cum_niter_ = 0;
  double last_resid_ = 0.0;
  int last_nblocks_ = 0, cells_ = 0, nang_ = 0;
};

// ---- Kokkos C-callback free functions: forward to the active probe instance ----
void PerfBegin(const char *name, const uint32_t /*devid*/, uint64_t * /*kID*/) {
  if (g_kernels_probe != nullptr) g_kernels_probe->OnBegin(name);
}
void PerfEnd(uint64_t /*kID*/) {
  if (g_kernels_probe != nullptr) g_kernels_probe->OnEnd();
}
void PerfAlloc(const Kokkos::Tools::SpaceHandle space, const char *label,
               const void * /*ptr*/, const uint64_t size) {
  if (g_memory_probe != nullptr) g_memory_probe->OnAlloc(space.name, label, size);
}
void PerfDealloc(const Kokkos::Tools::SpaceHandle space, const char *label,
                 const void * /*ptr*/, const uint64_t size) {
  if (g_memory_probe != nullptr) g_memory_probe->OnDealloc(space.name, label, size);
}

// ---- registry + factory ----
std::vector<std::unique_ptr<perf::PerfProbe>> g_registry;
std::map<std::string, std::string> g_owner;   // probe name -> owning <output> block name

std::string Canon(const std::string &n) { return (n == "iter") ? "iteration" : n; }

std::unique_ptr<perf::PerfProbe> MakeProbe(const std::string &name) {
  if (name == "kernels")   return std::unique_ptr<perf::PerfProbe>(new KernelsProbe());
  if (name == "memory")    return std::unique_ptr<perf::PerfProbe>(new MemoryProbe());
  if (name == "comms")     return std::unique_ptr<perf::PerfProbe>(new CommsProbe());
  if (name == "iteration") return std::unique_ptr<perf::PerfProbe>(new IterationProbe());
  std::cout << "### FATAL ERROR perf: unknown perf_probe '" << name
            << "' (expected kernels|memory|comms|iteration)" << std::endl;
  std::exit(EXIT_FAILURE);
}

}  // namespace

namespace perf {

std::vector<std::string> ProbeNames(const std::string &spec) {
  if (spec.empty()) return {"kernels", "memory", "comms", "iteration"};
  std::vector<std::string> out;
  std::string cur;
  for (char c : spec) {
    if (c == ',') {
      if (!cur.empty()) out.push_back(Canon(cur));
      cur.clear();
    } else if (c != ' ' && c != '\t') {
      cur.push_back(c);
    }
  }
  if (!cur.empty()) out.push_back(Canon(cur));
  return out;
}

void Enable(ParameterInput *pin) {
  if (pin == nullptr) return;   // -n dump / no input: register nothing (zero overhead)
  std::set<std::string> claimed;
  for (const auto &blk : pin->block) {                     // pin->block is a public std::list
    if (blk.block_name.compare(0, 6, "output") != 0) continue;
    if (!pin->DoesParameterExist(blk.block_name, "file_type")) continue;
    if (pin->GetString(blk.block_name, "file_type") != "perf") continue;
    for (const auto &nm : ProbeNames(pin->GetOrAddString(blk.block_name, "perf_probe", ""))) {
      if (!claimed.insert(nm).second) {                    // one probe -> one block (R1)
        if (global_variable::my_rank == 0) {
          std::cout << "### WARNING perf: probe '" << nm << "' named in more than one <output> "
                    << "block; keeping the first." << std::endl;
        }
        continue;
      }
      g_owner[nm] = blk.block_name;
      auto p = MakeProbe(nm);
      p->Setup(pin, blk.block_name);
      g_registry.push_back(std::move(p));
    }
  }
}

PerfProbe *GetProbe(const std::string &name) {
  for (auto &p : g_registry) {
    if (name == p->Name()) return p.get();
  }
  return nullptr;
}

std::string ProbeOwner(const std::string &probe) {
  auto it = g_owner.find(probe);
  return (it != g_owner.end()) ? it->second : std::string();
}

double CommWaitStart() {
  return (g_comms_probe != nullptr) ? g_timer.seconds() : 0.0;
}

void CommWaitStop(const char *bucket, double t0) {
  if (g_comms_probe != nullptr) g_comms_probe->Add(bucket, g_timer.seconds() - t0);
}

void EmitIteration(int niter, double resid, int nblocks, int cells, int nang) {
  if (g_iter_probe != nullptr) g_iter_probe->OnSolve(niter, resid, nblocks, cells, nang);
}

}  // namespace perf
