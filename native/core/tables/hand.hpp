// The Hand burst's frozen random bits, compiled in: generated/hand.cpp.
#pragma once

#include <cstdint>
#include <span>

namespace data2g::tables {

extern const std::span<const std::uint8_t> HAND_INTERLEAVER;  // hand.interleaver(): coded bit i goes out at [i]
extern const std::span<const std::uint8_t> HAND_SYNC;         // hand.SYNC, the Costas order-6 tones

}  // namespace data2g::tables
