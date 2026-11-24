#include "arq/ping.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "arq/probe.hpp"
#include "generated/config.hpp"

namespace data2g::arq::ping {

double tx_level(std::string_view tx_path) {
    return tx_path == "mic" ? *std::min_element(config::FM_MIC_GAIN.begin(), config::FM_MIC_GAIN.end()) : 1.0;
}

Bytes ping_body(const std::string& frm, const std::string& to, const std::optional<Bytes>& report) {
    Bytes head = pack_call(frm);
    const Bytes t = pack_call(to);
    head.insert(head.end(), t.begin(), t.end());
    head.push_back(report ? 1 : 0);
    if (report) head.insert(head.end(), report->begin(), report->end());
    return head;
}

std::tuple<std::string, std::string, std::optional<Bytes>> unpack_ping(ByteView body) {
    if (body.size() < 17) throw std::invalid_argument("ping: short");
    std::string frm = unpack_call(body.subspan(0, 8)), to = unpack_call(body.subspan(8, 8));
    if (body[16] & 1) {
        if (body.size() < 17 + static_cast<std::size_t>(probe::REPORT_BYTES)) throw std::invalid_argument("ping: short report");
        return {frm, to, Bytes(body.begin() + 17, body.begin() + 17 + probe::REPORT_BYTES)};
    }
    return {frm, to, std::nullopt};
}

Level level(std::span<const double> x) {
    Level m;
    if (x.empty()) return m;
    double ss = 0, pk = 0;
    std::size_t clip = 0;
    const double lim = std::pow(10.0, CLIP_DBFS / 20.0);
    for (double v : x) {
        ss += v * v;
        pk = std::max(pk, v * v);
        clip += std::abs(v) >= lim;
    }
    const auto db = [](double p) { return 10.0 * std::log10(std::max(p, 1e-7)); };
    m.rms_dbfs = db(ss / static_cast<double>(x.size()));
    m.peak_dbfs = db(pk);
    m.clip_permille = static_cast<int>(std::lround(1000.0 * static_cast<double>(clip) / static_cast<double>(x.size())));
    return m;
}

}  // namespace data2g::arq::ping
