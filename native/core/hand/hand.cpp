#include "hand/hand.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numbers>
#include <tuple>

#include "codes/codes.hpp"
#include "cpm/cpm.hpp"
#include "dsp/dsp.hpp"
#include "polar/polar.hpp"
#include "tables/hand.hpp"

namespace data2g::hand {
namespace {

constexpr int SYNC_AT[2] = {0, N_SYNC + N_DATA / 2};  // the two Costas blocks' first rows

bool is_sync_row(int r) { return (r >= SYNC_AT[0] && r < SYNC_AT[0] + N_SYNC) || (r >= SYNC_AT[1] && r < SYNC_AT[1] + N_SYNC); }

const std::vector<int>& data_rows() {
    static const std::vector<int> rows = [] {
        std::vector<int> v;
        for (int r = 0; r < N_SYM; ++r)
            if (!is_sync_row(r)) v.push_back(r);
        return v;
    }();
    return rows;
}

// cpm.Grid(f"hand{freq:g}", 8, 6.25, center=freq): only m, rate, T, bits and f0 are read.
const cpm::Grid& grid(int c) {
    static const std::array<cpm::Grid, N_CHANNELS> grids = [] {
        std::array<cpm::Grid, N_CHANNELS> gs{};
        for (int i = 0; i < N_CHANNELS; ++i) {
            auto& g = gs[static_cast<std::size_t>(i)];
            g.name = "hand";
            g.m = M;
            g.rate = RATE;
            g.center = channel_freq(i);
            g.bp = 50.0;
            g.T = static_cast<int>(std::lround(config::FS / RATE));
            g.bits = 3;
            // Python's round(): half to even, as nearbyint under the default rounding mode
            g.f0 = std::nearbyint((g.center - (M - 1) * RATE / 2) / RATE) * RATE;
        }
        return gs;
    }();
    return grids.at(static_cast<std::size_t>(c));
}

const polar::SCLDecoder& decoder() {
    static const polar::SCLDecoder dec(polar::PolarCode(K, E, polar::ga_info_pos(K, E)), codes::POLAR_LIST);
    return dec;
}

Mat<double> rows_of(const Mat<double>& En, const std::vector<int>& rows) {
    Mat<double> out(rows.size(), En.cols);
    for (std::size_t i = 0; i < rows.size(); ++i)
        std::copy_n(En[static_cast<std::size_t>(rows[i])], En.cols, out[i]);
    return out;
}

struct Cand {
    std::int64_t start;
    double cfo;
};

// Top (start, cfo) candidates by the sync blocks' tone shares.
std::vector<Cand> search(const cpm::Grid& g, std::span<const double> y, std::int64_t nominal) {
    const int T = g.T;
    std::vector<std::tuple<double, std::int64_t, double>> cands;
    const double step = g.rate / 4;
    for (int i = 0;; ++i) {
        const double cfo = -SEARCH_HZ + i * step;  // np.arange's start + i * step
        if (cfo >= SEARCH_HZ + 1e-9) break;
        for (std::int64_t start = nominal - SEARCH_SAMPLES; start < nominal + SEARCH_SAMPLES + 1; start += T / 8) {
            double s = 0.0;
            for (int r0 : SYNC_AT) {
                const auto Sh = cpm::shares(cpm::energies(g, y, start + static_cast<std::int64_t>(r0) * T, N_SYNC, cfo));
                double part = 0.0;
                for (int j = 0; j < N_SYNC; ++j) part += Sh[static_cast<std::size_t>(j)][tables::HAND_SYNC[static_cast<std::size_t>(j)]];
                s += part;
            }
            cands.emplace_back(s, start, cfo);
        }
    }
    std::sort(cands.begin(), cands.end(), std::greater<>());  // Python's sort(reverse=True) on the tuples
    std::vector<Cand> out;
    for (const auto& [s, start, cfo] : cands) {  // distinct peaks: half a symbol or a quarter tone apart
        const bool distinct = std::all_of(out.begin(), out.end(), [&](const Cand& o) {
            return std::abs(start - o.start) > T / 2 || std::abs(cfo - o.cfo) > g.rate / 4 + 1e-9;
        });
        if (distinct) out.push_back({start, cfo});
        if (static_cast<int>(out.size()) == TOP) break;
    }
    return out;
}

// A candidate nailed in on T/16 x rate/16 steps -> (start, data-row energies).
std::pair<std::int64_t, Mat<double>> refine(const cpm::Grid& g, std::span<const double> y, std::int64_t start, double cfo) {
    const int T = g.T;
    std::pair<std::int64_t, Mat<double>> best{start, {}};
    double best_v = -std::numeric_limits<double>::infinity();
    for (int dt = -(T / 2); dt <= T / 2; dt += T / 16)
        for (double df : {-g.rate / 16, 0.0, g.rate / 16}) {
            auto En = rows_of(cpm::energies(g, y, start + dt, N_SYM, cfo + df), data_rows());
            const auto Sh = cpm::shares(En);
            std::vector<double> top(Sh.rows);
            for (std::size_t r = 0; r < Sh.rows; ++r) top[r] = *std::max_element(Sh[r], Sh[r] + Sh.cols);
            const double v = dsp::pairwise_sum(top);
            if (v > best_v) {
                best_v = v;
                best = {start + dt, std::move(En)};
            }
        }
    return best;
}

}  // namespace

double channel_freq(int c) { return CENTER + (c - (N_CHANNELS - 1) / 2.0) * PITCH; }

std::vector<std::uint8_t> payload_bits(std::uint64_t p) {
    std::vector<std::uint8_t> b(PAYLOAD);
    for (int i = 0; i < PAYLOAD; ++i) b[static_cast<std::size_t>(i)] = (p >> (PAYLOAD - 1 - i)) & 1;
    return b;
}

std::uint64_t payload_int(std::span<const std::uint8_t> bits) {
    std::uint64_t p = 0;
    for (int i = 0; i < PAYLOAD; ++i) p = (p << 1) | (bits[static_cast<std::size_t>(i)] & 1);
    return p;
}

std::vector<std::uint8_t> crc16(std::span<const std::uint8_t> bits, unsigned init) {
    unsigned c = init;
    for (auto b : bits) {
        const unsigned fb = ((c >> 15) & 1) ^ (b & 1);
        c = ((c << 1) & 0xFFFF) ^ (fb ? 0x1021 : 0);
    }
    std::vector<std::uint8_t> out(16);
    for (int i = 0; i < 16; ++i) out[static_cast<std::size_t>(i)] = (c >> (15 - i)) & 1;
    return out;
}

std::vector<double> modulate(int tag, std::span<const std::uint8_t> payload, double freq) {
    cpm::Grid g = grid(0);
    g.center = freq;
    g.f0 = std::nearbyint((freq - (M - 1) * RATE / 2) / RATE) * RATE;
    Mat<std::uint8_t> u(1, K);
    std::copy(payload.begin(), payload.end(), u[0]);
    const auto crc = crc16(payload, static_cast<unsigned>(tag) << 8);
    std::copy(crc.begin(), crc.end(), u[0] + PAYLOAD);
    const auto coded = decoder().code().encode(u);
    std::vector<std::uint8_t> sent(E);
    for (int j = 0; j < E; ++j) sent[static_cast<std::size_t>(j)] = coded[0][tables::HAND_INTERLEAVER[static_cast<std::size_t>(j)]];
    const auto data = cpm::to_tones(g, sent);
    std::vector<int> sym(N_SYM);
    for (int r0 : SYNC_AT)
        for (int j = 0; j < N_SYNC; ++j) sym[static_cast<std::size_t>(r0 + j)] = tables::HAND_SYNC[static_cast<std::size_t>(j)];
    const auto& dr = data_rows();
    for (std::size_t i = 0; i < dr.size(); ++i) sym[static_cast<std::size_t>(dr[i])] = data[i];
    return cpm::tones(g, sym);
}

std::vector<double> tx_audio(int tag, std::uint64_t payload, int channel, double cfo) {
    auto x = modulate(tag, payload_bits(payload), channel_freq(channel));
    if (cfo != 0.0) {  // hfchannel.freq_shift, the phase wrapped to one turn first
        const auto z = dsp::hilbert(x);
        for (std::size_t n = 0; n < x.size(); ++n) {
            const double c = cfo * static_cast<double>(n) / config::FS;
            x[n] = (z[n] * std::exp(dsp::cdouble(0.0, 2 * std::numbers::pi * (c - std::floor(c))))).real();
        }
    }
    double peak = 0.0;
    for (const auto& z : dsp::hilbert(x)) peak = std::max(peak, std::abs(z));
    for (double& v : x) v /= peak;
    return x;
}

std::vector<Decoded> decode_window(std::span<const double> y, int tag, std::span<const int> channels, std::int64_t nominal) {
    // energies() zero-pads past the end, so hand.py's explicit padding isn't needed
    const auto& dec = decoder();
    const unsigned init = static_cast<unsigned>(tag) << 8;
    std::vector<Decoded> out;
    for (int c : channels) {
        const auto& g = grid(c);
        const auto cands = search(g, y, nominal);
        if (cands.empty()) continue;
        Mat<float> llr(cands.size(), E);
        for (std::size_t i = 0; i < cands.size(); ++i) {
            const auto En = rows_of(cpm::energies(g, y, cands[i].start, N_SYM, cands[i].cfo), data_rows());
            const auto l = cpm::llrs(g, En);
            for (int j = 0; j < E; ++j)
                llr[i][tables::HAND_INTERLEAVER[static_cast<std::size_t>(j)]] = static_cast<float>(l[static_cast<std::size_t>(j)]);
        }
        const auto res = dec.decode(llr);
        bool done = false;
        for (std::size_t i = 0; i < cands.size() && !done; ++i)
            for (int l = 0; l < dec.list_size() && !done; ++l) {
                const std::span<const std::uint8_t> u(res.paths[i] + static_cast<std::size_t>(l) * K, K);
                const auto crc = crc16(u.first(PAYLOAD), init);
                if (!std::equal(crc.begin(), crc.end(), u.begin() + PAYLOAD)) continue;
                const auto [start, En] = refine(g, y, cands[i].start, cands[i].cfo);
                out.push_back({c, payload_int(u), cpm::measure(g, En, N_DATA).snr_est,
                               1000.0 * static_cast<double>(start - nominal) / config::FS});
                done = true;
            }
    }
    return out;
}

}  // namespace data2g::hand
