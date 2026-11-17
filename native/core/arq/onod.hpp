// data2g/arq/onod.py: OFDM nods (docs/arq.md §3c), a nod (§3b) for the OFDM
// regime. A band's preamble, then K keyed QPSK symbols where a header would
// be, detected where the listener expects the reply, not decoded.
#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include "arq/nod.hpp"
#include "util/mat.hpp"

namespace data2g::arq::onod {

inline constexpr int M = 32;                         // sequences in a family
inline constexpr int N_MEANINGS = nod::N_MEANINGS;   // nod's codebook; the rest reserved
inline constexpr double SCREEN = 0.25;               // no valid header word matches a sequence's head over this

int k_of(std::string_view band);                     // K[band]: the keyed symbols of a nod there; 0: none
std::optional<double> thr(std::string_view band, int k);  // THR[(band, k)]
// The sync band a control-only reply in `submode` nods on (onod.band_of): an OFDM ACK mode's; empty: none.
std::string_view band_of(std::string_view submode);

std::vector<std::uint8_t> layout(int k);             // per symbol after the preamble: 1 for a pilot
std::int64_t samples(std::string_view band, int k);  // the preamble's first sample to the last pilot's end
double seconds(std::string_view band, int k);

using Family = Mat<float>;  // (M, 2 * nc * K) +-1 (+1 = bit 0)
// The keyed family (onod.family): splitmix64 seeded with the key and the turn, each sequence redrawn
// until no valid header word of the band matches it over SCREEN.
Family family(std::string_view band, int k, const nod::FamilyKey& key);
// A sequence -> unit-RMS audio, conditioned as an ACK mode's burst.
std::vector<double> waveform(std::string_view band, int k, std::span<const float> seq);

struct Scores {
    std::array<double, M> score;
    std::array<std::int64_t, M> start;
    std::array<double, M> cfo;
};
// Each sequence's best normalized correlation over the reads of a nod whose preamble starts in [lo, hi]
// at a CFO near cfo (modem::prime_ofdm's search); nullopt: the window is outside x.
std::optional<Scores> scores(std::span<const double> x, std::int64_t lo, std::int64_t hi, double cfo,
                             std::string_view band, int k, const Family& fam);
struct Detection {
    int meaning;
    std::int64_t start;
    double score, cfo;
};
// A nod of `fam` in the slot, or nullopt: none over the threshold (thr: THR's unless given), a runner-up
// over it too, or a reserved meaning.
std::optional<Detection> detect(std::span<const double> x, std::int64_t lo, std::int64_t hi, double cfo,
                                std::string_view band, int k, const Family& fam, std::optional<double> thr = {});

}  // namespace data2g::arq::onod
