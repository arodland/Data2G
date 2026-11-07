// data2g/hand.py: the Hand burst (the D2G net's show of hands, docs/arq.md
// §9). 8-FSK at 6.25 Hz, 63 payload bits + CRC-16 seeded with the request
// tag, CA-SCL polar K=79/E=162, two Costas blocks; 40 channels at 60 Hz.
// Python is normative; tests/test_native_hand.py checks parity.
#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "config/config.hpp"

namespace data2g::hand {

inline constexpr int M = 8, PAYLOAD = 63, K = PAYLOAD + 16, E = 162, N_DATA = E / 3, N_SYNC = 6;
inline constexpr int N_SYM = 2 * N_SYNC + N_DATA;  // 66
inline constexpr double RATE = 6.25, LEN_S = N_SYM / RATE;
inline constexpr int N_CHANNELS = 40;
inline constexpr double PITCH = 60.0, CENTER = 1500.0;
inline constexpr double SEARCH_S = 0.6, SEARCH_HZ = 5.0, LATE_S = 0.5;
inline constexpr int TOP = 3;
inline constexpr std::int64_t SEARCH_SAMPLES = 4800;  // round(SEARCH_S * FS)
inline constexpr std::int64_t LEN_SAMPLES = 84480;    // round(LEN_S * FS)
inline constexpr std::int64_t WINDOW_SAMPLES = 2 * SEARCH_SAMPLES + LEN_SAMPLES;
static_assert(config::FS == 8000, "SEARCH_SAMPLES / LEN_SAMPLES assume 8 kHz");

double channel_freq(int c);
std::vector<std::uint8_t> payload_bits(std::uint64_t p);  // 63 bits, MSB first
std::uint64_t payload_int(std::span<const std::uint8_t> bits);
std::vector<std::uint8_t> crc16(std::span<const std::uint8_t> bits, unsigned init);

// 63 payload bits -> unit-RMS Hand audio at `freq` (tone centre).
std::vector<double> modulate(int tag, std::span<const std::uint8_t> payload, double freq);
// A Hand as it goes out: channel c, shifted by the ref's cfo, envelope peak 1.
std::vector<double> tx_audio(int tag, std::uint64_t payload, int channel, double cfo);

struct Decoded {
    int c;
    std::uint64_t payload;
    double snr, dt_ms;
};
// Net control's round: every channel of the captured window y; `nominal`: a
// Hand's expected start sample in y. Thread-safe.
std::vector<Decoded> decode_window(std::span<const double> y, int tag, std::span<const int> channels, std::int64_t nominal);

}  // namespace data2g::hand
