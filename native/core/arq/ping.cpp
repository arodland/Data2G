#include "arq/ping.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <stdexcept>

#include "arq/burst.hpp"
#include "arq/phy.hpp"
#include "generated/config.hpp"

namespace data2g::arq::ping {

namespace {

using cd = std::complex<double>;

int q(double v, double step, int lo, int hi) { return std::clamp(static_cast<int>(std::lround(v / step)), lo, hi); }

double db(double p) { return 10.0 * std::log10(std::max(p, 1e-7)); }

std::string f1(double v) {
    char b[32];
    std::snprintf(b, sizeof b, "%.1f", v);
    return b;
}

// ping.py _dither: tiny deterministic noise (the receiver needs a noise estimate), a hash of the index
double dither(std::uint64_t k) {
    return 1e-4 * (static_cast<double>((k * 2654435761ULL) % (1ULL << 32)) / 2147483648.0 - 1.0);
}

double mean_db(std::span<const double> v) {
    double s = 0;
    for (double x : v) s += x;
    return 10.0 * std::log10(s / static_cast<double>(v.size()));
}

}  // namespace

double tx_level(std::string_view tx_path) {
    return tx_path == "mic" ? *std::min_element(config::FM_MIC_GAIN.begin(), config::FM_MIC_GAIN.end()) : 1.0;
}

std::string fmt(const Report& m) {
    return "snr=" + f1(m.snr_db) + " rms=" + f1(m.rms_dbfs) + " peak=" + f1(m.peak_dbfs) +
           " clip=" + std::to_string(m.clip_permille) + " thirds=" + f1(m.thirds_db[0]) + "/" + f1(m.thirds_db[1]) + "/" +
           f1(m.thirds_db[2]) + " dist=" + f1(m.dist_db) + " noise_tilt=" + f1(m.noise_tilt_db);
}

Bytes ping_body(const std::string& frm, const std::string& to, const std::optional<Bytes>& report, int payload_bytes) {
    Bytes head = pack_call(frm);
    const Bytes t = pack_call(to);
    head.insert(head.end(), t.begin(), t.end());
    head.push_back(report ? 1 : 0);
    if (report) head.insert(head.end(), report->begin(), report->end());
    const long fill = static_cast<long>(MAX_CTL) * payload_bytes - 4 - 2 - static_cast<long>(head.size());
    head.resize(head.size() + static_cast<std::size_t>(std::min<long>(255 - static_cast<long>(head.size()), std::max(0L, fill))), 0);
    return head;
}

std::tuple<std::string, std::string, std::optional<Bytes>> unpack_ping(ByteView body) {
    if (body.size() < 17) throw std::invalid_argument("ping: short");
    std::string frm = unpack_call(body.subspan(0, 8)), to = unpack_call(body.subspan(8, 8));
    if (body[16] & 1) {
        if (body.size() < 17 + static_cast<std::size_t>(REPORT_BYTES)) throw std::invalid_argument("ping: short report");
        return {frm, to, Bytes(body.begin() + 17, body.begin() + 17 + REPORT_BYTES)};
    }
    return {frm, to, std::nullopt};
}

Bytes pack_report(const Report& m) {
    const int s[REPORT_BYTES] = {q(m.snr_db, 0.25, -128, 127), q(m.rms_dbfs, 0.5, -128, 0), q(m.peak_dbfs, 0.5, -128, 0),
                                 q(m.clip_permille, 1, 0, 255), q(m.thirds_db[0], 0.5, -128, 127),
                                 q(m.thirds_db[1], 0.5, -128, 127), q(m.thirds_db[2], 0.5, -128, 127),
                                 q(m.dist_db, 0.1, 0, 255), q(m.noise_tilt_db, 0.5, -128, 127)};
    Bytes out;
    for (int v : s) out.push_back(static_cast<std::uint8_t>(v & 0xFF));
    return out;
}

Report unpack_report(ByteView b) {
    const auto s = [&](int i) { return b[i] > 127 ? static_cast<int>(b[i]) - 256 : static_cast<int>(b[i]); };
    Report m;
    m.snr_db = s(0) * 0.25;
    m.rms_dbfs = s(1) * 0.5;
    m.peak_dbfs = s(2) * 0.5;
    m.clip_permille = b[3];
    for (int i = 0; i < 3; ++i) m.thirds_db[i] = s(4 + i) * 0.5;
    m.dist_db = b[7] * 0.1;
    m.noise_tilt_db = s(8) * 0.5;
    return m;
}

Report level(std::span<const double> x) {
    Report m;
    if (x.empty()) return m;
    double ss = 0, pk = 0;
    std::size_t clip = 0;
    const double lim = std::pow(10.0, CLIP_DBFS / 20.0);
    for (double v : x) {
        ss += v * v;
        pk = std::max(pk, v * v);
        clip += std::abs(v) >= lim;
    }
    m.rms_dbfs = db(ss / static_cast<double>(x.size()));
    m.peak_dbfs = db(pk);
    m.clip_permille = static_cast<int>(std::lround(1000.0 * static_cast<double>(clip) / static_cast<double>(x.size())));
    return m;
}

Report measure(const modem::Received& r, const std::vector<Bytes>& payloads, const Report& lvl) {
    TxBurst sent;
    sent.submode = r.spec->name;
    for (std::size_t i = 0; i < payloads.size(); ++i) sent.slots.push_back({ctl_mask(0, static_cast<int>(i), 0), 0, payloads[i]});
    const auto x0 = tx_audio(sent, "mic", 1.0);
    std::vector<double> clean(x0.size() + config::FS, 0.0);
    std::copy(x0.begin(), x0.end(), clean.begin() + config::FS / 2);
    for (std::size_t k = 0; k < clean.size(); ++k) clean[k] += dither(k);
    const auto rc = modem::receive(clean);  // acquired as any burst (ping.py measure)
    const auto& est = r.est;
    const std::size_t rows = est.h.rows, nc = est.h.cols;
    // each side over its own pilot-based channel (the clean burst is read at its own timing)
    std::vector<cd> x(rows * nc);
    cd num = 0;
    double den = 0;
    for (std::size_t i = 0; i < rows; ++i) {
        const cd* yr = modem::data_row(r.raw, i);
        const cd* xr = modem::data_row(rc.raw, i);
        for (std::size_t k = 0; k < nc; ++k) {
            x[i * nc + k] = xr[k] / rc.est.h[i][k];
            const cd hx = est.h[i][k] * x[i * nc + k];
            num += std::conj(hx) * yr[k];
            den += std::norm(hx);
        }
    }
    const cd a = num / den;  // the one gain the estimate may miss
    const std::size_t per = static_cast<std::size_t>(config::DATA_SYMS_PER_FRAME);
    std::vector<double> per_frame;
    std::vector<double> n0_c(nc, 0.0), p_c(nc, 0.0);
    double x2 = 0;
    for (const cd& v : x) x2 += std::norm(v);
    x2 /= static_cast<double>(x.size());
    for (std::size_t f = 0; f < rows / per; ++f) {
        double e = 0, v = 0;
        for (std::size_t i = f * per; i < (f + 1) * per; ++i) {
            const cd* yr = modem::data_row(r.raw, i);
            for (std::size_t k = 0; k < nc; ++k) {
                const double n0 = est.n0_at(i, k);
                e += std::norm(yr[k] - a * est.h[i][k] * x[i * nc + k]);
                v += n0 + est.mse[i][k];
                n0_c[k] += n0;
                p_c[k] += std::norm(est.h[i][k]) * x2;
            }
        }
        per_frame.push_back(e / v);
    }
    std::sort(per_frame.begin(), per_frame.end());
    const std::size_t n = per_frame.size();
    const double med = n % 2 ? per_frame[n / 2] : 0.5 * (per_frame[n / 2 - 1] + per_frame[n / 2]);
    // np.array_split's thirds: the first nc % 3 get one more
    std::vector<double> snr_k(nc);
    for (std::size_t k = 0; k < nc; ++k) snr_k[k] = p_c[k] / n0_c[k];
    std::array<double, 3> thirds{}, nt{};
    std::size_t at = 0;
    for (std::size_t j = 0; j < 3; ++j) {
        const std::size_t len = nc / 3 + (j < nc % 3);
        thirds[j] = mean_db(std::span<const double>(snr_k).subspan(at, len));
        std::vector<double> nn(n0_c.begin() + static_cast<long>(at), n0_c.begin() + static_cast<long>(at + len));
        for (double& v : nn) v /= static_cast<double>(rows);
        nt[j] = mean_db(nn);
        at += len;
    }
    Report m = lvl;
    Heard h;
    h.ofdm = std::make_shared<const modem::Received>(r);
    m.snr_db = measure(h).snr_est;
    m.thirds_db = thirds;
    m.noise_tilt_db = nt[2] - nt[0];
    m.dist_db = std::max(0.0, 10.0 * std::log10(med));
    return m;
}

std::vector<std::string> verdict(const Report& far, std::string_view tx_path, const std::string& band) {
    std::vector<std::string> out;
    const double tilt = far.thirds_db[0] - far.thirds_db[2];
    if (far.snr_db < MIN_SNR_DB) {
        out.push_back("S/N " + std::to_string(static_cast<int>(std::lround(far.snr_db))) + " dB: too low to judge distortion");
    } else if (far.dist_db > DIST_DB.at(band)) {
        out.push_back(std::string("your audio drives the radio's limiter: lower the TX level") +
                      (tx_path == "flat" ? ", or set tx_path mic on a mic/speaker radio" : ""));
    }
    if (tx_path == "mic" && tilt >= TILT_DB && far.noise_tilt_db >= NOISE_TILT_DB)
        out.push_back("top of band " + std::to_string(static_cast<int>(std::lround(tilt))) +
                      " dB under the bottom (a flat 9600 path's noise): set tx_path flat if this is a data port");
    if (far.clip_permille || far.peak_dbfs >= CLIP_DBFS) out.push_back("the far end's sound card clipped: it should lower its input level");
    if (out.empty()) out.push_back("level OK");
    return out;
}

}  // namespace data2g::arq::ping
