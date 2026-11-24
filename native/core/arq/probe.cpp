#include "arq/probe.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <numbers>
#include <numeric>
#include <stdexcept>

#include "config/config.hpp"
#include "dsp/fft.hpp"
#include "generated/config.hpp"
#include "waveform/dsp.hpp"
#include "waveform/ofdm.hpp"

namespace data2g::arq::probe {

namespace {

using cd = std::complex<double>;

// probe.py _hash: murmur3's 32-bit finaliser of (k + salt), to [0, 1)
double hash01(int k, std::uint32_t salt) {
    std::uint32_t h = (static_cast<std::uint32_t>(k) + salt) * 0x9E3779B1u;
    h ^= h >> 16;
    h *= 0x85EBCA6Bu;
    h ^= h >> 13;
    h *= 0xC2B2AE35u;
    h ^= h >> 16;
    return static_cast<double>(h) / 4294967296.0;
}

std::pair<double, double> band_hz(const std::string& name) {
    const auto& f = waveform::band(name).freqs;
    return {static_cast<double>(f.front()), static_cast<double>(f.back())};
}

// probe.py _db: each floored, so a stretch of exact zeros is 0 dB, not 0/0
double db(double s, double n = 1.0) { return 10.0 * std::log10(std::max(s, 1e-30) / std::max(n, 1e-30)); }

std::string f1(double v) {
    char b[32];
    std::snprintf(b, sizeof b, "%.1f", v);
    return b;
}
std::string f0(double v) {
    char b[32];
    std::snprintf(b, sizeof b, "%.0f", v);
    return b;
}

struct Lists {
    std::vector<int> tones, holes, stair_tones, stair_holes;
};

const Lists& lists() {
    static const Lists L = [] {
        Lists l;
        const double bin = bin_hz();
        const int lo = static_cast<int>(std::ceil(TONE_LO_HZ / bin)), hi = static_cast<int>(TONE_HI_HZ / bin);
        for (int k = lo; k <= hi; ++k)
            if (k % 2 == 1) (hash01(k, 7919) < 0.25 ? l.holes : l.tones).push_back(k);
        const auto [a, b] = band_hz(STAIR_BAND);
        for (int k : l.tones)
            if (k * bin >= a && k * bin <= b) l.stair_tones.push_back(k);
        for (int k : l.holes)
            if (k * bin >= a && k * bin <= b) l.stair_holes.push_back(k);
        return l;
    }();
    return L;
}

std::vector<double> tilt_gain(const std::vector<int>& bins) {
    const auto [lo, hi] = band_hz("f108");
    std::vector<double> g;
    for (int k : bins)
        g.push_back(std::pow(std::clamp(k * bin_hz(), lo, hi) / std::sqrt(lo * hi),
                             config::FM_TX_TILT_DB_OCT / (20.0 * std::log10(2.0))));
    return g;
}

double rms(std::span<const double> x) {
    double s = 0;
    for (double v : x) s += v * v;
    return std::sqrt(s / static_cast<double>(x.size()));
}

// probe.py _period: one period of these tones (hash phases), unit rms; `tilt`: the flat path's
std::vector<double> one_period(const std::vector<int>& tn, bool tilt) {
    const int P = period();
    std::vector<cd> X(static_cast<std::size_t>(P), cd{});
    const auto g = tilt ? tilt_gain(tn) : std::vector<double>(tn.size(), 1.0);
    for (std::size_t i = 0; i < tn.size(); ++i) {
        const cd v = std::polar(g[i], 2.0 * std::numbers::pi * hash01(tn[i], 104729));
        X[static_cast<std::size_t>(tn[i])] = v;
        X[static_cast<std::size_t>(P - tn[i])] = std::conj(v);  // Hermitian: np.fft.irfft
    }
    const auto z = dsp::fft(X, false);
    std::vector<double> x(static_cast<std::size_t>(P));
    for (int n = 0; n < P; ++n) x[static_cast<std::size_t>(n)] = z[static_cast<std::size_t>(n)].real();
    const double r = rms(x);
    for (double& v : x) v /= r;
    return x;
}

std::vector<double> stair_period(std::string_view tx_path) {
    const int P = period();
    std::vector<double> p;
    if (tx_path == "flat") {
        const double c = config::FM_TX_PEAK * std::pow(10.0, -*std::max_element(STEPS_DB.begin(), STEPS_DB.end()) / 20.0);
        p = one_period(lists().stair_tones, true);
        for (double& v : p) v = std::clamp(v, -c, c);
    } else {
        const auto base = one_period(lists().stair_tones, false);
        std::vector<double> t;
        for (int r = 0; r < 8; ++r) t.insert(t.end(), base.begin(), base.end());
        const double head = config::submode(STAIR_BAND + "-256l-r88")->headroom;
        const auto x = waveform::tx_condition(t, head, config::CLIP_OVERSHOOT, 0, std::numeric_limits<std::size_t>::max(),
                                              waveform::band(STAIR_BAND).tx_bandpass());
        p.assign(x.begin() + 4 * P, x.begin() + 5 * P);  // steady state: periodic again
    }
    const double r = rms(p);
    for (double& v : p) v /= r;
    return p;
}

struct Stretch {
    std::vector<double> sig, noise, hole;
};

// probe.py _periods: every period's spectrum from the probe's start, the sample clocks' offset taken out
// (tau fitted on the sounding, each period turned back by its index)
std::vector<std::vector<cd>> periods(std::span<const double> y) {
    const int P = period();
    const int n = static_cast<int>(STEPS_DB.size()) * (SETTLE + MEAS) + SETTLE + SOUND_MEAS;
    const auto off = static_cast<std::size_t>(std::lround(GAP_S * config::FS));
    std::vector<std::vector<cd>> F;
    for (int r = 0; r < n; ++r) {
        std::vector<cd> z(static_cast<std::size_t>(P));
        for (int i = 0; i < P; ++i) {
            const std::size_t j = off + static_cast<std::size_t>(r * P + i);
            z[static_cast<std::size_t>(i)] = j < y.size() ? y[j] : 0.0;
        }
        F.push_back(dsp::fft(z, true));
    }
    double num = 0, den = 0;
    for (int k : lists().tones) {
        cd c{};
        for (int r = SETTLE - GUARD + 1; r < SETTLE - GUARD + SOUND_MEAS; ++r)
            c += F[static_cast<std::size_t>(r)][static_cast<std::size_t>(k)] *
                 std::conj(F[static_cast<std::size_t>(r - 1)][static_cast<std::size_t>(k)]);
        const double w = std::abs(c);
        num += w * k * std::arg(c);
        den += w * static_cast<double>(k) * k;
    }
    const double tau = -num / std::max(2.0 * std::numbers::pi / P * den, 1e-30);
    for (int r = 0; r < n; ++r)
        for (int k = 0; k <= P / 2; ++k)
            F[static_cast<std::size_t>(r)][static_cast<std::size_t>(k)] *= std::polar(1.0, 2.0 * std::numbers::pi * k * tau * r / P);
    return F;
}

Stretch stretch(const std::vector<std::vector<cd>>& F, int r0, int n, const std::vector<int>& tn, const std::vector<int>& hl) {
    Stretch s;
    for (int k : tn) {
        cd m{};
        for (int r = r0; r < r0 + n; ++r) m += F[static_cast<std::size_t>(r)][static_cast<std::size_t>(k)];
        m /= static_cast<double>(n);
        double v = 0;
        for (int r = r0; r < r0 + n; ++r) v += std::norm(F[static_cast<std::size_t>(r)][static_cast<std::size_t>(k)] - m);
        v /= static_cast<double>(n - 1);
        s.noise.push_back(v);
        s.sig.push_back(std::max(std::norm(m) - v / n, 0.0));
    }
    for (int k : hl) {
        double v = 0;
        for (int r = r0; r < r0 + n; ++r) v += std::norm(F[static_cast<std::size_t>(r)][static_cast<std::size_t>(k)]);
        s.hole.push_back(v / n);
    }
    return s;
}

double sum(const std::vector<double>& v) { return std::accumulate(v.begin(), v.end(), 0.0); }
double mean(const std::vector<double>& v) { return sum(v) / static_cast<double>(v.size()); }

double sinad_db(const Stretch& st) {
    const double s = sum(st.sig), n = sum(st.noise);
    const double d = std::max(mean(st.hole) - mean(st.noise), 0.0) * static_cast<double>(st.sig.size());
    return 10.0 * std::log10(std::max(s, 1e-30) / std::max(n + d, 1e-30));
}

int q(double v, double step, int lo, int hi) {
    return std::clamp(static_cast<int>(std::nearbyint(v / step)), lo, hi) & 0xFF;  // Python's round: half to even
}

}  // namespace

int period() { return config::FS / 50; }
double bin_hz() { return static_cast<double>(config::FS) / period(); }
const std::vector<int>& tones() { return lists().tones; }
const std::vector<int>& holes() { return lists().holes; }

std::size_t n_samples() {
    return static_cast<std::size_t>(std::lround(GAP_S * config::FS)) +
           static_cast<std::size_t>((SETTLE + SOUND_MEAS + static_cast<int>(STEPS_DB.size()) * (SETTLE + MEAS)) * period());
}

std::vector<double> waveform(std::string_view tx_path, double level) {
    const auto snd = one_period(lists().tones, tx_path == "flat");
    const auto st = stair_period(tx_path);
    std::vector<double> x(static_cast<std::size_t>(std::lround(GAP_S * config::FS)), 0.0);
    const double gs = std::pow(10.0, SOUND_DB / 20.0);
    for (int r = 0; r < SETTLE + SOUND_MEAS; ++r)
        for (double v : snd) x.push_back(v * gs * level);
    for (double g : STEPS_DB) {
        const double a = std::pow(10.0, g / 20.0);
        for (int r = 0; r < SETTLE + MEAS; ++r)
            for (double v : st) x.push_back(v * a * level);
    }
    return x;
}

Report measure(std::span<const double> y, const ping::Level& lvl) {
    const auto& L = lists();
    const auto F = periods(y);
    const Stretch sd = stretch(F, SETTLE - GUARD, SOUND_MEAS, L.tones, L.holes);
    Report m;
    int r = SETTLE + SOUND_MEAS;
    for (std::size_t i = 0; i < STEPS_DB.size(); ++i) {
        const Stretch s = stretch(F, r + SETTLE - GUARD, MEAS, L.stair_tones, L.stair_holes);
        m.snr[i] = db(sum(s.sig), sum(s.noise));
        m.sinad[i] = sinad_db(s);
        r += SETTLE + MEAS;
    }
    const std::size_t nt = L.tones.size();
    std::vector<double> f(nt), snr(nt);
    for (std::size_t i = 0; i < nt; ++i) {
        f[i] = L.tones[i] * bin_hz();
        snr[i] = db(sd.sig[i], sd.noise[i]);
    }
    const auto [lo, hi] = band_hz(STAIR_BAND);
    std::vector<std::size_t> nb;
    for (std::size_t i = 0; i < nt; ++i)
        if (f[i] >= lo && f[i] <= hi) nb.push_back(i);
    std::vector<double> s_nb;
    for (auto i : nb) s_nb.push_back(snr[i]);
    std::sort(s_nb.begin(), s_nb.end());
    const std::size_t h = s_nb.size() / 2;
    const double med = s_nb.size() % 2 ? s_nb[h] : 0.5 * (s_nb[h - 1] + s_nb[h]);
    const double thr = med - EDGE_DB;
    std::vector<bool> ok(nt);
    for (std::size_t i = 0; i < nt; ++i) {
        int c = 0;
        for (std::size_t j = i == 0 ? 0 : i - 1; j <= std::min(nt - 1, i + 1); ++j) c += snr[j] >= thr;
        ok[i] = c >= 2;
    }
    std::size_t i0 = 0;
    for (std::size_t i = 1; i < nt; ++i)
        if (std::abs(f[i] - 1000.0) < std::abs(f[i0] - 1000.0)) i0 = i;
    std::size_t a_ = i0, b_ = i0;
    while (a_ > 0 && ok[a_ - 1]) --a_;
    while (b_ < nt - 1 && ok[b_ + 1]) ++b_;
    m.lo_hz = ok[i0] ? f[a_] : 0.0;
    m.hi_hz = ok[i0] ? f[b_] : 0.0;
    const auto [w_lo, w_hi] = band_hz("f108");
    double ts = 0, tn = 0, ns = 0, nn = 0;
    for (std::size_t i = 0; i < nt; ++i)
        if (f[i] >= w_hi - (w_hi - w_lo) / 3 && f[i] <= w_hi) ts += sd.sig[i], tn += sd.noise[i];
    for (auto i : nb) ns += sd.sig[i], nn += sd.noise[i];
    m.snr_nb = db(ns, nn);
    m.snr_top = db(ts, tn);
    // np.array_split(nb, 3): the first len % 3 parts one longer
    const std::size_t n3 = nb.size() / 3, r3 = nb.size() % 3;
    const std::size_t len0 = n3 + (r3 > 0), len2 = n3, start2 = nb.size() - len2;
    double s0 = 0, n0 = 0, s2 = 0, n2 = 0;
    for (std::size_t k = 0; k < len0; ++k) s0 += sd.sig[nb[k]], n0 += sd.noise[nb[k]];
    for (std::size_t k = start2; k < nb.size(); ++k) s2 += sd.sig[nb[k]], n2 += sd.noise[nb[k]];
    m.snr_tilt = db(s0, n0) - db(s2, n2);
    m.noise_tilt = db(n2 / static_cast<double>(len2), n0 / static_cast<double>(len0));
    m.rms_dbfs = lvl.rms_dbfs;
    m.peak_dbfs = lvl.peak_dbfs;
    m.clip_permille = lvl.clip_permille;
    return m;
}

Bytes pack_report(const Report& m) {
    Bytes b;
    for (int v : {q(m.lo_hz, bin_hz(), 0, 255), q(m.hi_hz, bin_hz(), 0, 255), q(m.snr_nb, 0.5, -128, 127),
                  q(m.snr_top, 0.5, -128, 127), q(m.snr_tilt, 0.5, -128, 127), q(m.noise_tilt, 0.5, -128, 127)})
        b.push_back(static_cast<std::uint8_t>(v));
    for (double v : m.snr) b.push_back(static_cast<std::uint8_t>(q(v, 0.5, -128, 127)));
    for (double v : m.sinad) b.push_back(static_cast<std::uint8_t>(q(v, 0.5, -128, 127)));
    b.push_back(static_cast<std::uint8_t>(q(m.rms_dbfs, 0.5, -128, 0)));
    b.push_back(static_cast<std::uint8_t>(q(m.peak_dbfs, 0.5, -128, 0)));
    b.push_back(static_cast<std::uint8_t>(q(m.clip_permille, 1, 0, 255)));
    return b;
}

Report unpack_report(ByteView b) {
    if (b.size() < static_cast<std::size_t>(REPORT_BYTES)) throw std::invalid_argument("probe: short report");
    const auto s = [&](std::size_t i) { return static_cast<double>(static_cast<std::int8_t>(b[i])); };
    Report m;
    const std::size_t n = STEPS_DB.size();
    m.lo_hz = b[0] * bin_hz();
    m.hi_hz = b[1] * bin_hz();
    m.snr_nb = s(2) * 0.5;
    m.snr_top = s(3) * 0.5;
    m.snr_tilt = s(4) * 0.5;
    m.noise_tilt = s(5) * 0.5;
    for (std::size_t i = 0; i < n; ++i) m.snr[i] = s(6 + i) * 0.5, m.sinad[i] = s(6 + n + i) * 0.5;
    m.rms_dbfs = s(6 + 2 * n) * 0.5;
    m.peak_dbfs = s(7 + 2 * n) * 0.5;
    m.clip_permille = b[8 + 2 * n];
    return m;
}

std::string fmt(const Report& m) {
    std::string st, sn, si;
    for (std::size_t i = 0; i < STEPS_DB.size(); ++i) {
        char b[16];
        std::snprintf(b, sizeof b, "%+.0f", STEPS_DB[i]);
        st += (i ? "/" : "") + std::string(b);
        sn += (i ? "/" : "") + f1(m.snr[i]);
        si += (i ? "/" : "") + f1(m.sinad[i]);
    }
    return "band=" + f0(m.lo_hz) + "-" + f0(m.hi_hz) + " snr=" + f1(m.snr_nb) + " top=" + f1(m.snr_top) +
           " tilt=" + f1(m.snr_tilt) + " noise_tilt=" + f1(m.noise_tilt) + " steps=" + st + " step_snr=" + sn +
           " sinad=" + si + " rms=" + f1(m.rms_dbfs) + " peak=" + f1(m.peak_dbfs) + " clip=" + std::to_string(m.clip_permille);
}

double best_step(const Report& m) {
    const std::size_t n = STEPS_DB.size() - 1;  // the repeat is not a level of its own
    std::vector<double> cap(n);
    for (std::size_t i = 0; i < n; ++i) cap[i] = std::min(m.sinad[i], CAP_DB);
    const double top = *std::max_element(cap.begin(), cap.end());
    if (cap[REF] >= top - TIE_DB) return 0.0;
    double best = -1e9;
    for (std::size_t i = 0; i < n; ++i)
        if (cap[i] >= top - TIE_DB) best = std::max(best, STEPS_DB[i]);
    return best;
}

std::vector<std::string> verdict(const Report& m, std::string_view tx_path) {
    std::vector<std::string> out;
    if (m.clip_permille || m.peak_dbfs >= -0.1) out.push_back("the far end's sound card clipped: it should lower its input level");
    if (m.hi_hz <= 0) {
        out.push_back("the probe was not heard: check the level and the path");
        return out;
    }
    const bool wide = m.hi_hz >= WIDE_HZ;
    out.push_back("passband " + f0(m.lo_hz) + "-" + f0(m.hi_hz) + " Hz: " + (wide ? "WIDE works" : "NARROW only"));
    if (tx_path == "mic" && m.snr_tilt >= 1.0 && m.noise_tilt >= 1.5)
        out.push_back("top of the data band " + f0(m.snr_tilt) +
                      " dB under the bottom (a flat 9600 path's noise): set tx_path flat if this is a data port");
    if (std::abs(std::min(m.sinad[REF], CAP_DB) - std::min(m.sinad.back(), CAP_DB)) >= VARIED_DB) {
        out.push_back("S/N varied during the probe: ping again before changing the level");
        return out;
    }
    const double d = best_step(m);
    if (d < 0)
        out.push_back("your audio drives the radio's limiter: lower the TX level about " + f0(-d) + " dB" +
                      (tx_path == "flat" ? ", or set tx_path mic on a mic/speaker radio" : ""));
    else if (d > 0)
        out.push_back("more level helps: raise the TX level about " + f0(d) + " dB if the radio's data input has headroom");
    else
        out.push_back("level OK (SINAD " + f0(m.sinad[REF]) + " dB)");
    return out;
}

}  // namespace data2g::arq::probe
