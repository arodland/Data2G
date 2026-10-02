// data2g/arq/modes.py: every mode the ARQ can use, the frozen OFDM ladder
// (config::SUBMODES) then the CPM data modes (tables::CPM_SPECS), with what
// differs by family.
#pragma once

#include <span>
#include <string_view>

#include "generated/config.hpp"
#include "tables/tables.hpp"

namespace data2g::arq {

struct Mode {
    std::string_view name, band;  // band: an OFDM band or a CPM grid
    const config::Submode* ofdm = nullptr;
    const tables::CpmSpec* cpm = nullptr;
    bool is_cpm() const { return cpm != nullptr; }
};

std::span<const Mode> modes();  // MODES' order
const Mode* mode(std::string_view name);  // nullptr: no such mode
const Mode& mode_at(std::string_view name);  // throws std::out_of_range

double burst_seconds(const Mode& m, int n_cw, bool dup = false);
int payload_bytes(const Mode& m);
int ctl_payload_bytes(const Mode& m);  // its control codeword's (CPM: the grid's short one)
int max_ctl(const Mode& m);
int min_cw(const Mode& m, bool data);
int rv_cycle(const Mode& m);

}  // namespace data2g::arq
