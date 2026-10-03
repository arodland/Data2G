#include "arq/modes.hpp"

#include <stdexcept>
#include <string>
#include <vector>

#include "codes/codes.hpp"
#include "cpm/cpm.hpp"
#include "modem/timing.hpp"

namespace data2g::arq {

std::span<const Mode> modes() {
    static const std::vector<Mode> all = [] {
        std::vector<Mode> v;
        for (const auto& s : config::SUBMODES) v.push_back({s.name, s.band, &s, nullptr});
        for (const auto& s : tables::CPM_SPECS) v.push_back({s.name, s.grid, nullptr, &s});
        return v;
    }();
    return all;
}

const Mode* mode(std::string_view name) {
    for (const auto& m : modes())
        if (m.name == name) return &m;
    return nullptr;
}

const Mode& mode_at(std::string_view name) {
    const Mode* m = mode(name);
    if (!m) throw std::out_of_range("no mode " + std::string(name));
    return *m;
}

double burst_seconds(const Mode& m, int n_cw, bool dup) {
    return m.is_cpm() ? cpm::burst_seconds(*m.cpm, n_cw, dup) : modem::burst_seconds(*m.ofdm, n_cw);
}

int payload_bytes(const Mode& m) { return m.is_cpm() ? codes::payload_bytes(*m.cpm) : codes::payload_bytes(*m.ofdm); }

int ctl_payload_bytes(const Mode& m) {
    return m.is_cpm() ? codes::payload_bytes(cpm::ctl(cpm::grid_of(*m.cpm))) : codes::payload_bytes(*m.ofdm);
}

int max_ctl(const Mode& m) { return m.is_cpm() ? 1 : 4; }

int min_cw(const Mode& m, bool data) { return data && m.is_cpm() ? 2 : 1; }

int rv_cycle(const Mode& m) { return (m.is_cpm() ? m.cpm->code : m.ofdm->code) == "ldpc" ? 4 : 1; }

}  // namespace data2g::arq
