#include "arq/onod.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <utility>

#include "config/config.hpp"
#include "modem/modem.hpp"
#include "waveform/dsp.hpp"
#include "waveform/ofdm.hpp"
#include "waveform/sync.hpp"

namespace data2g::arq::onod {

using cd = std::complex<double>;

namespace {
struct Variant {
    std::string_view band;
    int k;
    double thr;
    bool sent;  // K's
};
// onod.THR and K
constexpr Variant VARIANTS[] = {{"w", 4, 0.42, false}, {"w", 9, 0.29, true}, {"n10", 10, 0.42, true}};

std::uint64_t splitmix64(std::uint64_t& state) {
    std::uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
}  // namespace

int k_of(std::string_view band) {
    for (const auto& v : VARIANTS)
        if (v.sent && v.band == band) return v.k;
    return 0;
}

std::optional<double> thr(std::string_view band, int k) {
    for (const auto& v : VARIANTS)
        if (v.band == band && v.k == k) return v.thr;
    return std::nullopt;
}

std::string_view band_of(std::string_view submode) {
    const auto* s = config::submode(submode);
    if (!s || submode.find("ack") == std::string_view::npos || !k_of(s->sync_band)) return {};
    return s->sync_band;
}

std::vector<std::uint8_t> layout(int k) {
    const int d = config::DATA_SYMS_PER_FRAME;
    const int first = k - d * ((k - 1) / d);
    std::vector<std::uint8_t> out(static_cast<std::size_t>(first), 0);
    out.push_back(1);
    for (int i = 0; i < (k - first) / d; ++i) {
        out.insert(out.end(), static_cast<std::size_t>(d), 0);
        out.push_back(1);
    }
    return out;
}

std::int64_t samples(std::string_view band, int k) {
    return waveform::band(band).preamble_samples() + static_cast<std::int64_t>(layout(k).size()) * config::NSYM;
}

double seconds(std::string_view band, int k) {
    return static_cast<double>(config::LEADIN_SAMPLES + samples(band, k) + config::LEADOUT_SAMPLES) / config::FS;
}

Family family(std::string_view band, int k, const nod::FamilyKey& key) {
    const std::size_t n = 2 * static_cast<std::size_t>(waveform::band(band).nc()) * static_cast<std::size_t>(k);
    const std::size_t nh = modem::header_cols(band).size();
    std::uint64_t state = 0x0F0DULL << 48 | (static_cast<std::uint64_t>(key[0]) & 0xFFFFFFFFULL) << 16 |
                          static_cast<std::uint64_t>(key[1] & 1) << 8 | static_cast<std::uint64_t>(key[2] & 7) << 4 |
                          static_cast<std::uint64_t>(key[3] & 7);
    Family out(M, n);
    std::vector<float> bits((n + 63) / 64 * 64);
    std::vector<double> head(nh);
    for (int m = 0; m < M; ++m) {
        for (;;) {
            for (std::size_t i = 0; i < bits.size(); i += 64) {
                const std::uint64_t z = splitmix64(state);
                for (int j = 0; j < 64; ++j) bits[i + static_cast<std::size_t>(j)] = 1.0f - 2.0f * static_cast<float>((z >> j) & 1);
            }
            if (n < nh) break;
            std::copy(bits.begin(), bits.begin() + static_cast<std::ptrdiff_t>(nh), head.begin());
            const auto corr = modem::header_corr(head, band);
            if (static_cast<double>(*std::max_element(corr.begin(), corr.end())) / static_cast<double>(nh) <= SCREEN) break;
        }
        std::copy(bits.begin(), bits.begin() + static_cast<std::ptrdiff_t>(n), out[static_cast<std::size_t>(m)]);
    }
    return out;
}

std::vector<double> waveform(std::string_view band, int k, std::span<const float> seq) {
    std::vector<std::uint8_t> bit(seq.size());
    for (std::size_t i = 0; i < seq.size(); ++i) bit[i] = seq[i] < 0;
    return modem::keyed_waveform(band, layout(k), bit);
}

std::optional<Scores> scores(std::span<const double> x, std::int64_t lo, std::int64_t hi, double cfo, std::string_view band,
                             int k, const Family& fam) {
    const auto z0 = waveform::freq_correct(waveform::to_baseband(x), cfo);
    const auto len = static_cast<std::int64_t>(z0.size());
    const auto& b = waveform::band(band);
    const std::int64_t need = samples(band, k);
    const auto lay = layout(k);
    const auto [S, freqs] = waveform::detection_stat(z0, b, modem::PRIME_REACH_HZ);
    const std::int64_t a = std::max<std::int64_t>(0, lo), e = std::min<std::int64_t>(static_cast<std::int64_t>(S.cols), hi + 1);
    if (e <= a) return std::nullopt;
    std::vector<double> D(static_cast<std::size_t>(e - a));
    for (std::size_t i = 0; i < D.size(); ++i) {
        double m = S[0][static_cast<std::size_t>(a) + i];
        for (std::size_t r = 1; r < S.rows; ++r) m = std::max(m, S[r][static_cast<std::size_t>(a) + i]);
        D[i] = m;
    }
    Scores out;
    out.score.fill(-INFINITY);
    out.start.fill(0);
    out.cfo.fill(0.0);
    for (int peak = 0; peak < modem::PRIME_PEAKS; ++peak) {
        std::int64_t n = std::max_element(D.begin(), D.end()) - D.begin();  // np.argmax: the first maximum
        if (D[static_cast<std::size_t>(n)] <= 0) break;
        std::fill(D.begin() + std::max<std::int64_t>(0, n - config::M),
                  D.begin() + std::min<std::int64_t>(static_cast<std::int64_t>(D.size()), n + config::M), -1.0);
        n += a;
        std::size_t fi = 0;
        for (std::size_t r = 1; r < S.rows; ++r)
            if (S[r][static_cast<std::size_t>(n)] > S[fi][static_cast<std::size_t>(n)]) fi = r;
        const auto [start, f] = waveform::refine(z0, b, n, freqs[fi]);
        const auto zb = waveform::freq_correct(z0, f);
        for (int al : {0, -1, 1, -2, 2}) {
            const std::int64_t st = start + static_cast<std::int64_t>(al) * config::M;
            if (!(0 <= st && st <= len - need)) continue;
            const auto soft_d = modem::keyed_read(zb, st, band, lay);
            std::vector<float> soft(soft_d.begin(), soft_d.end());  // float32, as Python
            double sq = 0.0;
            for (float v : soft) sq += static_cast<double>(v) * static_cast<double>(v);
            const auto norm = static_cast<float>(std::sqrt(static_cast<double>(static_cast<float>(sq)) * static_cast<double>(soft.size())) + 1e-12);
            for (int m = 0; m < M; ++m) {
                double c = 0.0;
                const float* s = fam[static_cast<std::size_t>(m)];
                for (std::size_t i = 0; i < soft.size(); ++i) c += static_cast<double>(s[i]) * static_cast<double>(soft[i]);
                const double sc = static_cast<double>(static_cast<float>(c) / norm);
                if (sc > out.score[static_cast<std::size_t>(m)]) {
                    out.score[static_cast<std::size_t>(m)] = sc;
                    out.start[static_cast<std::size_t>(m)] = st;
                    out.cfo[static_cast<std::size_t>(m)] = cfo + f;
                }
            }
        }
    }
    return out;
}

std::optional<Detection> detect(std::span<const double> x, std::int64_t lo, std::int64_t hi, double cfo, std::string_view band,
                                int k, const Family& fam, std::optional<double> t_in) {
    const auto got = scores(x, lo, hi, cfo, band, k, fam);
    if (!got) return std::nullopt;
    int b0 = 0, b1 = -1;  // the best and the runner-up
    for (int m = 1; m < M; ++m) {
        if (got->score[static_cast<std::size_t>(m)] > got->score[static_cast<std::size_t>(b0)]) {
            b1 = b0;
            b0 = m;
        } else if (b1 < 0 || got->score[static_cast<std::size_t>(m)] > got->score[static_cast<std::size_t>(b1)]) {
            b1 = m;
        }
    }
    const double t = t_in ? *t_in : thr(band, k).value();
    if (got->score[static_cast<std::size_t>(b0)] < t || got->score[static_cast<std::size_t>(b1)] >= t || b0 >= N_MEANINGS)
        return std::nullopt;
    return Detection{b0, got->start[static_cast<std::size_t>(b0)], got->score[static_cast<std::size_t>(b0)],
                     got->cfo[static_cast<std::size_t>(b0)]};
}

}  // namespace data2g::arq::onod
