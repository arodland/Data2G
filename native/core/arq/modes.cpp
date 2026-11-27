#include "arq/modes.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "arq/frames.hpp"
#include "codes/codes.hpp"
#include "cpm/cpm.hpp"
#include "modem/timing.hpp"

namespace data2g::arq {

namespace {

// frames.BANDS_CODE: sync bands, 2 bits. FM: w and n10 (floor, ACKs), f55 and
// f108; no w48, n4 or CPM, so every code names a band (plan M6, Q2).
constexpr bool FM = config::PROFILE == "fm";
constexpr std::pair<std::string_view, int> HF_BANDS_CODE[] = {{"w", 0}, {"n10", 1}, {"w48", 2}};
constexpr std::pair<std::string_view, int> FM_BANDS_CODE[] = {{"w", 0}, {"n10", 1}, {"f55", 2}, {"f108", 3}};

std::span<const std::pair<std::string_view, int>> bands_code() {
    if constexpr (FM) return FM_BANDS_CODE;
    return HF_BANDS_CODE;
}

constexpr std::array<std::string_view, 53> HF_RECOMMEND = {
    // 0..5: 500 Hz CPM floor (all caps; r1/3<->r1/2 = +-1, cross-grid same rate = +-2)
    "fsk8w50-r1/3", "fsk8w50-r1/2",
    "fsk16r25-r1/3", "fsk16r25-r1/2",
    "fsk8r50-r1/3", "fsk8r50-r1/2",
    // 6..8: 2300 Hz CPM (fsk8r50-r1/2 <-> fsk32r62-r1/2 = +-2; r1/2 <-> r2/3 = +-1)
    "fsk32r62-r1/3", "fsk32r62-r1/2", "fsk32r62-r2/3",
    // 9..19: w + low w48 interleaved by threshold (fsk32r62-r1/2 <-> qpsk-r1/5 = +-2)
    "qpsk-r1/5", "qpsk-r1/3", "w48-qpsk-r1/5", "qpsk-r1/2", "w48-qpsk-r1/3",
    "16qam-r1/3", "w48-qpsk-r1/2", "qpsk-r3/4", "w48-16qam-r1/3", "16qam-r1/2", "w48-qpsk-r2/3",
    // 20..29: high w48
    "w48-qpsk-r3/4", "w48-16qam-r1/2", "w48-16qam-r2/3", "w48-64l-r1/2",
    "w48-16qam-r3/4", "w48-64l-r7/12", "w48-16qam-r5/6", "w48-64l-r2/3",
    "w48-64l-r3/4", "w48-256l-r5/8",
    // 30..39: n10 (500 Hz OFDM, contiguous)
    "n10-qpsk-r1/5", "n10-qpsk-r1/3", "n10-qpsk-r1/2", "n10-16qam-r1/3",
    "n10-qpsk-r3/4", "n10-16qam-r1/2", "n10-16qam-r2/3", "n10-16qam-r3/4",
    "n10-64l-r3/4", "n10-256l-r3/4",
    // 40..44: n4 (200 Hz OFDM, contiguous)
    "n4-qpsk-r1/5", "n4-qpsk-r1/3", "n4-qpsk-r1/2", "n4-qpsk-r2/3", "n4-16qam-r1/3",
    // 45..52: polar & ACK-only
    "polar-k96-f8", "polar-k96-f4", "polar-k192-f8",
    "ack-4f", "ack-1f", "n10-ack-4f", "n4-ack-8f", "n4-ack-2f",
};

constexpr std::array<std::string_view, 48> FM_RECOMMEND = {
    // 0..7: n10 data
    "n10-qpsk-r1/5", "n10-qpsk-r1/3", "n10-qpsk-r1/2", "n10-16qam-r1/3",
    "n10-qpsk-r3/4", "n10-16qam-r1/2", "n10-16qam-r2/3", "n10-16qam-r3/4",
    // 8..13: w data
    "qpsk-r1/5", "qpsk-r1/3", "qpsk-r1/2", "16qam-r1/3", "qpsk-r3/4", "16qam-r1/2",
    // 14..27: f55 data
    "f55-qpsk-r21", "f55-qpsk-r40", "f55-qpsk-r60", "f55-16qam-r47",
    "f55-64l-r50", "f55-64l-r67", "f55-64l-r75", "f55-64l-r80", "f55-64l-r83",
    "f55-256l-r67", "f55-256l-r72", "f55-256l-r80", "f55-256l-r83", "f55-256l-r88",
    // 28..41: f108 data
    "f108-qpsk-r21", "f108-qpsk-r41", "f108-qpsk-r60", "f108-16qam-r41",
    "f108-64l-r40", "f108-64l-r50", "f108-64l-r59", "f108-64l-r67", "f108-64l-r75", "f108-64l-r80",
    "f108-256l-r67", "f108-256l-r75", "f108-256l-r80", "f108-256l-r87",
    // 42..47: polar & ACK-only
    "polar-k96-f8", "polar-k96-f4", "polar-k192-f8",
    "ack-4f", "ack-1f", "n10-ack-4f",
};

std::span<const std::string_view> recommend_table() {
    if constexpr (FM) return FM_RECOMMEND;
    return HF_RECOMMEND;
}

}  // namespace

std::span<const Mode> modes() {
    static const std::vector<Mode> all = [] {
        std::vector<Mode> v;
        // modes.py MODES: every mode compiled in (HF: on a band with a recommendation code)
        const auto codes = bands_code();
        for (const auto& s : config::SUBMODES)
            if (FM || std::any_of(codes.begin(), codes.end(), [&](const auto& b) { return b.first == s.sync_band; }))
                v.push_back({s.name, s.band, &s, nullptr});
        for (const auto* s : cpm::specs()) v.push_back({s->name, s->grid, nullptr, s});
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

int max_ctl(const Mode& m) { return m.is_cpm() ? 1 : MAX_CTL; }

int min_cw(const Mode& m, bool data) { return data && m.is_cpm() ? 2 : 1; }

int rv_cycle(const Mode& m) { return m.is_cpm() ? codes::rv_cycle(*m.cpm) : codes::rv_cycle(*m.ofdm); }

int cap_hz(int cap) {
    // modes.py CAP_HZ. FM: 500 Hz (the n10 floor), narrow (w, f55), wide (f108)
    switch (cap) {
        case 0: return 500;
        case 1: return FM ? 2700 : 1200;
        case 2: return FM ? 5350 : 2400;
    }
    throw std::out_of_range("no cap " + std::to_string(cap));
}

std::string bw_name(int cap) {
    static const std::array<const char*, 3> names =
        FM ? std::array<const char*, 3>{"500", "NARROW", "WIDE"} : std::array<const char*, 3>{"500", "1200", "2300"};
    return names.at(static_cast<std::size_t>(cap));
}

double width_hz(const Mode& m) {
    if (m.is_cpm()) {
        const auto& g = cpm::grid_of(*m.cpm);
        return g.m * g.rate + 2 * g.bp;
    }
    if (m.band == "n4") return 200;
    if (m.band == "n10") return 500;
    if (m.band == "w") return 1200;
    if (m.band == "w48") return 2400;
    if (m.band == "f55") return 2700;
    if (m.band == "f108") return 5350;
    throw std::out_of_range("no width for band " + std::string(m.band));
}

bool selectable(const Mode& m) {
    // modes.py SELECTABLE. FM (plan M6, Q2): its coded bands, indices under 16; no w48, no CPM
    if (!FM) return true;
    if (m.is_cpm() || m.ofdm->index >= 16) return false;
    const auto codes = bands_code();
    return std::any_of(codes.begin(), codes.end(), [&](const auto& b) { return b.first == m.ofdm->sync_band; });
}

std::vector<const Mode*> allowed(int cap) {
    const int hz = cap_hz(cap);
    std::vector<const Mode*> out;
    for (const auto& m : modes())
        if (width_hz(m) <= hz && selectable(m)) out.push_back(&m);
    return out;
}

int encode(std::string_view submode) {
    (void)mode_at(submode);
    const auto table = recommend_table();
    for (std::size_t i = 0; i < table.size(); ++i)
        if (table[i] == submode) return static_cast<int>(i);
    throw std::out_of_range(std::string(submode) + (FM ? ": no recommendation code on FM" : ": no recommendation code on HF"));
}

const Mode* decode(int rec) {
    const auto table = recommend_table();
    if (rec < 0 || static_cast<std::size_t>(rec) >= table.size()) return nullptr;
    return mode(table[static_cast<std::size_t>(rec)]);
}

}  // namespace data2g::arq
