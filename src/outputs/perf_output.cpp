//========================================================================================
// AthenaK astrophysical plasma code
// Copyright(C) 2024 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file perf_output.cpp
//! \brief PerfOutput: writes each perf probe's cumulative snapshot to a wide time-series file
//! <basename>.<probe> at the block's cadence. One block writes a file per probe it OWNS (perf::
//! ProbeOwner). Raw reductions only; derived quantities (avg_ms/pct/gcaups) are computed in
//! analyze.py from successive snapshots. Snapshots are cumulative -- no per-interval reset.

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "utils/perf.hpp"
#include "outputs.hpp"

//----------------------------------------------------------------------------------------
// ctor: also calls BaseTypeOutput base class constructor

PerfOutput::PerfOutput(ParameterInput *pin, Mesh *pm, OutputParameters op) :
    BaseTypeOutput(pin, pm, op) {
  // write only the probes this block OWNS (the first <output> block that named each), so two
  // blocks never append to the same <basename>.<probe> file.
  for (const auto &nm : perf::ProbeNames(out_params.perf_probe)) {
    if (perf::ProbeOwner(nm) == out_params.block_name) probe_names_.push_back(nm);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void PerfOutput::LoadOutputData()
//! \brief snapshot each owned probe's cumulative rows (rank-local), reducing across ranks if needed
//! (all ranks call this -- Reduce() is collective for the comms probe).

void PerfOutput::LoadOutputData(Mesh *pm) {
  snapshot_.clear();
  snap_cycle_ = pm->ncycle;
  snap_time_ = pm->time;
  for (const auto &nm : probe_names_) {
    perf::PerfProbe *p = perf::GetProbe(nm);
    std::vector<perf::Row> rows = (p != nullptr) ? p->Rows() : std::vector<perf::Row>();
    if (p != nullptr) p->Reduce(rows);
    snapshot_.push_back(std::move(rows));
  }
}

//----------------------------------------------------------------------------------------
//! \fn void PerfOutput::WriteOutputFile()
//! \brief append the cumulative snapshot (one line per row, tagged cycle/time) to each probe's file

void PerfOutput::WriteOutputFile(Mesh *pm, ParameterInput *pin) {
  // only the master rank writes (probes are rank-0-representative or already reduced in LoadData)
  if (global_variable::my_rank == 0) {
    for (std::size_t ip = 0; ip < probe_names_.size(); ++ip) {
      const std::string &probe = probe_names_[ip];
      const std::vector<perf::Row> &rows = snapshot_[ip];
      bool have_header = (std::find(header_written_.begin(), header_written_.end(), probe)
                          != header_written_.end());
      // tolerate an empty snapshot (Finalize / pre-kernel): defer until we know the columns
      if (rows.empty() && !have_header) continue;

      std::string fname = out_params.file_basename + "." + probe;
      FILE *pfile = std::fopen(fname.c_str(), "a");
      if (pfile == nullptr) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                  << "Output file '" << fname << "' could not be opened" << std::endl;
        std::exit(EXIT_FAILURE);
      }

      // first-seen-order union of column names across this snapshot's rows (schema is stable per
      // probe, so recomputing each write matches the header written on the first non-empty write)
      std::vector<std::string> cols;
      for (const auto &r : rows) {
        for (const auto &c : r.str_cols) {
          if (std::find(cols.begin(), cols.end(), c.first) == cols.end()) cols.push_back(c.first);
        }
        for (const auto &c : r.num_cols) {
          if (std::find(cols.begin(), cols.end(), c.first) == cols.end()) cols.push_back(c.first);
        }
      }

      if (!have_header) {
        std::fprintf(pfile, "# cycle time");
        for (const auto &c : cols) std::fprintf(pfile, " %s", c.c_str());
        std::fprintf(pfile, "\n");
        header_written_.push_back(probe);
      }

      for (const auto &r : rows) {
        std::fprintf(pfile, "%d", snap_cycle_);
        std::fprintf(pfile, out_params.data_format.c_str(), static_cast<double>(snap_time_));
        for (const auto &col : cols) {
          bool found = false;
          for (const auto &c : r.str_cols) {
            if (c.first == col) {
              std::fprintf(pfile, " %s", c.second.c_str());
              found = true;
              break;
            }
          }
          if (!found) {
            for (const auto &c : r.num_cols) {
              if (c.first == col) {
                std::fprintf(pfile, out_params.data_format.c_str(), static_cast<double>(c.second));
                found = true;
                break;
              }
            }
          }
          if (!found) std::fprintf(pfile, " -");   // missing-value sentinel
        }
        std::fprintf(pfile, "\n");
      }
      std::fclose(pfile);
    }
  }

  // increment output time for restart continuity (mirrors history/eventlog; dcycle -> dt==0 no-op)
  if (out_params.last_time < 0.0) {
    out_params.last_time = pm->time;
  } else {
    out_params.last_time += out_params.dt;
  }
  pin->SetReal(out_params.block_name, "last_time", out_params.last_time);
  return;
}
