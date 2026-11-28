// Lookups on data2g/config.py's tables (generated/config.hpp, hand-written
// here beside them): what Python writes as BANDS[name], SUBMODES[name],
// config.BY_INDEX[(sync band, index)] and BandSpec.preamble_samples. Every
// area asks config, not modem or codes, which submode or band it is given.
#pragma once

#include <cstddef>
#include <string_view>

#include "generated/config.hpp"

namespace data2g::config {

const Band& band(std::string_view name);                         // throws std::out_of_range
const Submode* submode(std::string_view name);                   // nullptr if none
std::size_t row(const Submode& spec);                            // its index in SUBMODES; throws unless an element
const Submode* by_index(std::string_view sync_band, int index);  // the header's index -> submode; nullptr if none
inline int preamble_samples(const Band& b) { return PREAMBLE_CP + b.preamble_repeats * M; }

// Scattered-pilot bands (config.SC_BANDS): HF 6-symbol OFDM bands (w, w48, n10, n4); none on FM.
bool sc_band(std::string_view band);  // band in config.SC_BANDS

// Callee-callsign hash carried in a CONNECT burst's header (config.connect_hash, docs/pending-header.md):
// FNV-1a 32-bit over upper-cased trimmed ASCII, folded to `bits` bits.
int connect_hash(std::string_view call, int bits);

}  // namespace data2g::config
