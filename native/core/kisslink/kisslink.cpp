#include "kisslink/kisslink.hpp"

#include <algorithm>
#include <cctype>
#include <stdexcept>

#include "codes/codes.hpp"
#include "tnc/tnc.hpp"

namespace data2g::kisslink {

namespace {

constexpr const char* LOG = "data2g.kiss";

std::string call_of(const std::uint8_t* b) {
    std::string call;
    for (int i = 0; i < 6; ++i) call += static_cast<char>(b[i] >> 1);
    const auto lo = call.find_first_not_of(" \t\n\r\f\v");
    call = lo == std::string::npos ? "" : call.substr(lo, call.find_last_not_of(" \t\n\r\f\v") - lo + 1);
    const int ssid = (b[6] >> 1) & 0x0F;
    return ssid ? call + "-" + std::to_string(ssid) : call;
}

int be16(const std::uint8_t* p) { return p[0] << 8 | p[1]; }

void put_hb(Bytes& out, int h, int b) {
    out.push_back(static_cast<std::uint8_t>(h >> 8));
    out.push_back(static_cast<std::uint8_t>(h & 255));
    out.push_back(static_cast<std::uint8_t>(b));
}

int ceil_div(std::size_t a, std::size_t b) { return static_cast<int>((a + b - 1) / b); }

}  // namespace

std::string_view broadcast_mode(int cap) {
    if (cap == 0) return "n10-qpsk-r1/5";
    if (cap == 2) return "qpsk-r1/5";
    throw std::out_of_range("no broadcast mode for cap " + std::to_string(cap));
}

std::optional<Ax25> parse_ax25(ByteView frame) {
    std::vector<const std::uint8_t*> addrs;
    std::size_t i = 0;
    bool last = false;
    while (i + 7 <= frame.size()) {
        addrs.push_back(frame.data() + i);
        i += 7;
        if (addrs.back()[6] & 1) {  // address extension bit: the last address
            last = true;
            break;
        }
    }
    if (!last || addrs.size() < 2 || addrs.size() > 10 || i >= frame.size()) return std::nullopt;
    for (const auto* a : addrs)  // a shifted ASCII callsign on every address
        for (int k = 0; k < 6; ++k)
            if (a[k] < 0x40 || a[k] > 0xB4 || (a[k] & 1)) return std::nullopt;
    const int ctrl = frame[i];
    const bool ui = (ctrl & 0xEF) == 0x03;  // UI, with the P/F bit either way
    const std::uint8_t* repeated = nullptr;  // H bit: has repeated
    const std::uint8_t* pending = nullptr;
    for (std::size_t k = 2; k < addrs.size(); ++k) {
        if (addrs[k][6] & 0x80)
            repeated = addrs[k];
        else if (!pending)
            pending = addrs[k];
    }
    return Ax25{call_of(addrs[0]), call_of(addrs[1]), call_of(pending ? pending : addrs[0]),
                call_of(repeated ? repeated : addrs[1]), !ui};
}

int station_hash(std::string_view call) {
    std::uint32_t h = 0x811C9DC5;
    for (char c : call) h = (h ^ static_cast<std::uint8_t>(std::toupper(static_cast<unsigned char>(c)))) * 0x01000193u;
    const int v = static_cast<int>(h & 0xFFFF);
    return v ? v : 1;
}

KissLink::KissLink(int cap_, std::string broadcast_, arq::Clock clock_)
    : cap(cap_), clock(std::move(clock_)), broadcast(broadcast_.empty() ? std::string(broadcast_mode(cap_)) : broadcast_) {
    const arq::Mode* m = arq::mode(broadcast);
    const auto ok = arq::allowed(cap);
    if (!m || std::find(ok.begin(), ok.end(), m) == ok.end())
        throw std::invalid_argument("broadcast mode '" + broadcast + "': not a mode within the bandwidth cap");
}

void KissLink::command(int cmd, ByteView payload) {
    if (payload.empty() || (cmd != 2 && cmd != 3)) {
        if (arq::log_enabled(LOG, 10)) arq::log_write(LOG, 10, arq::format("KISS command %d ignored", cmd));
        return;
    }
    if (cmd == 2)
        persist = payload[0];
    else
        slot_s = std::max(SLOT_S, payload[0] / 100.0);
    if (arq::log_enabled(LOG, 20))
        arq::log_write(LOG, 20, arq::format("KISS %s = %d", cmd == 2 ? "P" : "SLOTTIME", payload[0]));
}

Peer* KissLink::peer(int h) {
    for (auto& [k, p] : peers)
        if (k == h) return &p;
    return nullptr;
}

std::pair<std::string, int> KissLink::route(ByteView frame) const {
    const auto ax = parse_ax25(frame);
    if (ax && ax->connected) {
        const int h = station_hash(ax->next_hop);
        for (const auto& [k, p] : peers)
            if (k == h && p.report && clock() - p.report->second <= REPORT_MAX_S) {
                const arq::Mode* m = arq::decode(p.report->first >> 2);
                const auto ok = arq::allowed(cap);
                if (m && std::find(ok.begin(), ok.end(), m) != ok.end()) return {std::string(m->name), p.report->first & 3};
            }
    }
    return {broadcast, static_cast<int>(arq::SIZE_S.size()) - 1};
}

// Control codeword payloads: header and as many fresh reports as fit.
std::vector<Bytes> KissLink::control(const arq::Mode& m, int sender) {
    const std::size_t pb = static_cast<std::size_t>(arq::ctl_payload_bytes(m));
    const int n_max = arq::max_ctl(m);
    const double now = clock();
    std::vector<std::pair<int, Peer*>> fresh;
    for (auto& [h, p] : peers)
        if (now - p.heard <= HEARD_MAX_S && p.shifter.measured) fresh.emplace_back(h, &p);
    std::stable_sort(fresh.begin(), fresh.end(), [](const auto& a, const auto& b) { return a.second->heard > b.second->heard; });
    const int room = (n_max * static_cast<int>(pb) - 4) / 3;
    Bytes body;
    arq::StationView stub;
    stub.cap = cap;
    for (std::size_t i = 0; i < fresh.size() && static_cast<int>(i) < std::max(0, room); ++i) {
        const auto r = fresh[i].second->shifter.recommend(stub);
        put_hb(body, fresh[i].first, (r.data << 2) | r.hint);
    }
    const int n_ctl = std::max(1, ceil_div(4 + body.size(), pb));
    Bytes stream{static_cast<std::uint8_t>(VERSION << 4 | (n_ctl - 1))};
    put_hb(stream, sender, static_cast<int>(body.size() / 3));
    stream.insert(stream.end(), body.begin(), body.end());
    stream.resize(static_cast<std::size_t>(n_ctl) * pb, 0);
    std::vector<Bytes> out;
    for (int i = 0; i < n_ctl; ++i)
        out.emplace_back(stream.begin() + static_cast<std::ptrdiff_t>(i * pb), stream.begin() + static_cast<std::ptrdiff_t>((i + 1) * pb));
    return out;
}

arq::TxBurstPtr KissLink::next_burst() {
    while (!queue.empty()) {
        const auto [mode, hint] = route(queue[0]);
        const arq::Mode& m = arq::mode_at(mode);
        const auto first = parse_ax25(queue[0]);
        const int sender = first ? station_hash(first->sender) : 0;
        if (sender) me.insert(sender);
        const auto ctl = control(m, sender);
        const double seconds = mode != broadcast ? arq::SIZE_S.at(static_cast<std::size_t>(hint)) : BROADCAST_S;
        const std::size_t pb = static_cast<std::size_t>(arq::payload_bytes(m));
        // the size class is a preference: a burst grows to carry its first
        // frame, up to what its header can say
        const int limit = m.is_cpm() ? 1 + 1 + tables::CPM.max_data : config::MAX_CODEWORDS;
        const int need = static_cast<int>(ctl.size()) + ceil_div(2 + queue[0].size(), pb);
        const int n_max = std::min(limit, std::max(arq::slots_for(m, seconds), need));
        const std::size_t room = static_cast<std::size_t>(n_max - static_cast<int>(ctl.size())) * pb;
        std::vector<Bytes> taken, rest;
        std::size_t size = 0;
        for (auto& f : queue) {
            if (route(f).first == mode && size + 2 + f.size() <= room) {
                size += 2 + f.size();
                taken.push_back(std::move(f));
            } else {
                rest.push_back(std::move(f));
            }
        }
        queue = std::move(rest);
        if (taken.empty()) {  // the first frame alone doesn't fit: it never will in this mode
            if (arq::log_enabled(LOG, 40))
                arq::log_write(LOG, 40, arq::format("%zu-byte frame dropped: a %s burst carries at most %zu",
                                                    queue[0].size(), mode.c_str(), room - 2));
            queue.erase(queue.begin());
            continue;
        }
        Bytes stream;
        for (const auto& f : taken) {
            stream.push_back(static_cast<std::uint8_t>(f.size() >> 8));
            stream.push_back(static_cast<std::uint8_t>(f.size() & 255));
            stream.insert(stream.end(), f.begin(), f.end());
        }
        stream.resize(stream.size() + (pb - stream.size() % pb) % pb, 0);
        auto b = std::make_shared<arq::TxBurst>();
        b->submode = mode;
        for (std::size_t i = 0; i < ctl.size(); ++i) b->slots.push_back({arq::ctl_mask(0, static_cast<int>(i), KISS_KEY), 0, ctl[i]});
        for (std::size_t j = 0; j * pb < stream.size(); ++j)
            b->slots.push_back({arq::data_mask(0, static_cast<std::int64_t>(ctl.size() + j), KISS_KEY), 0,
                                Bytes(stream.begin() + static_cast<std::ptrdiff_t>(j * pb),
                                      stream.begin() + static_cast<std::ptrdiff_t>((j + 1) * pb))});
        b->burst_seq = ++n_sent;
        return b;
    }
    return nullptr;
}

std::optional<std::vector<Bytes>> KissLink::on_burst(const arq::Heard& r, std::shared_ptr<const arq::SlotSoft> soft) {
    // no DD budget, as Python's ModemRx(r, {})
    arq::ModemRx rx(r, nullptr, std::nullopt, std::move(soft));
    const std::string mode = rx.submode();
    const int n = rx.n_cw();
    const double now = clock();
    const auto c0 = rx.decode_plain(0, arq::ctl_mask(0, 0, KISS_KEY));
    int sender = 0;
    std::vector<std::pair<int, int>> reports;
    std::optional<int> start;
    if (c0 && (*c0)[0] >> 4 == VERSION) {
        const int n_ctl = ((*c0)[0] & 3) + 1;
        std::vector<std::optional<Bytes>> ctl{c0};
        for (int i = 1; i < n_ctl; ++i) ctl.push_back(rx.decode_plain(i, arq::ctl_mask(0, i, KISS_KEY)));
        start = n_ctl;
        sender = be16(&(*c0)[1]);
        const int k = (*c0)[3];
        if (std::all_of(ctl.begin(), ctl.end(), [](const auto& c) { return c.has_value(); })) {
            Bytes stream;
            for (const auto& c : ctl) stream.insert(stream.end(), c->begin(), c->end());
            stream.erase(stream.begin(), stream.begin() + 4);
            for (int j = 0; j < std::min(k, static_cast<int>(stream.size() / 3)); ++j)
                reports.emplace_back(be16(&stream[3 * j]), stream[3 * j + 2]);
        }
    }
    if (!start) {
        // control lost (or not KISS): slot 1 decoding as KISS data says it's
        // a KISS burst with a one-codeword control. Otherwise it isn't ours.
        if (n < 2 || !rx.decode_plain(1, arq::data_mask(0, 1, KISS_KEY))) return std::nullopt;
        start = 1;
    }
    std::vector<Bytes> payloads;
    std::vector<bool> ok;
    const std::size_t pb = static_cast<std::size_t>(arq::payload_bytes(r.mode()));
    for (int i = *start; i < n; ++i) {
        auto p = rx.decode_plain(i, arq::data_mask(0, i, KISS_KEY));
        ok.push_back(p.has_value());
        payloads.push_back(p ? std::move(*p) : Bytes(pb, 0));
    }
    auto frames = payloads.empty() ? std::vector<Bytes>{} : tnc::unpack(payloads, ok).first;
    if (!sender && !frames.empty()) {
        const auto ax = parse_ax25(frames[0]);
        sender = ax ? station_hash(ax->sender) : 0;
    }
    if (sender && !me.count(sender)) {
        Peer* p = peer(sender);
        if (!p) {
            Peer fresh;
            fresh.shifter.use_cpm = true;
            fresh.shifter.min_success = MIN_SUCCESS;
            fresh.heard = now;
            peers.emplace_back(sender, std::move(fresh));
            p = &peers.back().second;
        }
        p->heard = now;
        p->shifter.outcome(mode, static_cast<int>(std::count(ok.begin(), ok.end(), true)), static_cast<int>(ok.size()));
        p->shifter.observe(arq::measure(r), mode, now);
        for (const auto& [h, rec] : reports)
            if (me.count(h)) p->report = {{rec, now}};
    }
    return frames;
}

}  // namespace data2g::kisslink
