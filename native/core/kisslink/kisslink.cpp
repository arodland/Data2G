#include "kisslink/kisslink.hpp"

#include <algorithm>
#include <cctype>
#include <stdexcept>

#include "arq/frames.hpp"
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

// --- groups and control ------------------------------------------------------------

std::string group_name(const std::string& group) { return arq::unpack_call(arq::pack_call(group)); }

int group_key(const std::string& group) {
    std::uint32_t h = 0x811C9DC5;
    for (auto b : arq::pack_call(group)) h = (h ^ b) * 0x01000193u;
    const int v = static_cast<int>(h & 0xFFFF);
    return v ? v : 1;
}

namespace {

std::uint64_t be64(ByteView b) {  // a packed callsign's 60 bits
    std::uint64_t v = 0;
    for (auto x : b) v = v << 8 | x;
    return v;
}

Bytes be8(std::uint64_t v) {
    Bytes out(8);
    for (int i = 0; i < 8; ++i) out[static_cast<std::size_t>(7 - i)] = static_cast<std::uint8_t>(v >> (8 * i));
    return out;
}
Bytes tlv(int t, const Bytes& v) {
    Bytes out{static_cast<std::uint8_t>(t), static_cast<std::uint8_t>(v.size())};
    out.insert(out.end(), v.begin(), v.end());
    return out;
}

Bytes group_tlv(int n, const Port& p) {
    if (p.from_call) return tlv(T_GROUP_FROM, pack_pair(p.group, *p.from_call));
    return n == 0 ? Bytes{} : tlv(T_GROUP, arq::pack_call(p.group));
}

}  // namespace

// 120 bits, MSB first: the group's 60, then the call's (MSVC has no 128-bit integers)
Bytes pack_pair(const std::string& group, const std::string& call) {
    const std::uint64_t g = be64(arq::pack_call(group)), c = be64(arq::pack_call(call));
    Bytes out(15, 0);
    for (int b = 0; b < 120; ++b) {
        const auto bit = b < 60 ? (g >> (59 - b)) & 1 : (c >> (119 - b)) & 1;
        out[static_cast<std::size_t>(b / 8)] |= static_cast<std::uint8_t>(bit << (7 - b % 8));
    }
    return out;
}

std::pair<std::string, std::string> unpack_pair(ByteView b) {
    if (b.size() != 15) throw std::invalid_argument("group+from: not 15 bytes");
    std::uint64_t g = 0, c = 0;
    for (int i = 0; i < 120; ++i) {
        const std::uint64_t bit = (b[static_cast<std::size_t>(i / 8)] >> (7 - i % 8)) & 1;
        if (i < 60) g = g << 1 | bit;
        else c = c << 1 | bit;
    }
    return {arq::unpack_call(be8(g)), arq::unpack_call(be8(c))};
}

std::map<int, Bytes> parse_tlvs(ByteView b) {
    std::map<int, Bytes> out;
    std::size_t i = 0;
    while (i + 2 <= b.size() && b[i]) {
        const int t = b[i];
        const std::size_t n = b[i + 1];
        if (i + 2 + n > b.size()) throw std::invalid_argument("truncated TLV");
        out[t] = Bytes(b.begin() + static_cast<std::ptrdiff_t>(i + 2), b.begin() + static_cast<std::ptrdiff_t>(i + 2 + n));
        i += 2 + n;
    }
    return out;
}

ControlRead read_control(const std::map<int, Bytes>& tlvs) {
    ControlRead c{std::string(PORT0_GROUP), std::nullopt, std::nullopt};
    if (auto it = tlvs.find(T_GROUP_FROM); it != tlvs.end()) {
        if (it->second.size() != 15) throw std::invalid_argument("T_GROUP_FROM length");
        auto [g, call] = unpack_pair(it->second);
        c.group = g;
        c.call = call;
    } else if (auto it2 = tlvs.find(T_GROUP); it2 != tlvs.end()) {
        if (it2->second.size() != 8) throw std::invalid_argument("T_GROUP length");
        c.group = arq::unpack_call(it2->second);
    }
    if (auto it = tlvs.find(T_REPORTS); it != tlvs.end()) {
        if (it->second.size() < 2 || (it->second.size() - 2) % 3) throw std::invalid_argument("T_REPORTS length");
        c.reports = it->second;
    }
    return c;
}

// --- the link ------------------------------------------------------------------------

KissLink::KissLink(int cap_, std::string broadcast_, arq::Clock clock_)
    : cap(cap_), clock(std::move(clock_)), broadcast(broadcast_.empty() ? std::string(broadcast_mode(cap_)) : broadcast_) {
    check_mode(broadcast);
    ports.emplace(0, Port{std::string(PORT0_GROUP), broadcast, false, std::nullopt});
}

void KissLink::check_mode(const std::string& mode) const {
    const arq::Mode* m = arq::mode(mode);
    const auto ok = arq::allowed(cap);
    if (!m || std::find(ok.begin(), ok.end(), m) == ok.end())
        throw std::invalid_argument("mode '" + mode + "': not a mode within the bandwidth cap");
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

// The control a port needs (its group, and room for a report's sender) must
// fit the mode's control codewords.
void KissLink::check_fits(int n, const Port& p, const std::string& mode, bool shift) const {
    const arq::Mode& m = arq::mode_at(mode);
    const std::size_t need = 1 + group_tlv(n, p).size() + (shift ? 4 : 0);
    if (need > static_cast<std::size_t>(std::min(4, arq::max_ctl(m)) * arq::ctl_payload_bytes(m)))
        throw std::invalid_argument("mode " + mode + ": a " + std::to_string(need) + "-byte control doesn't fit");
}

int KissLink::open(const std::string& group, const std::optional<std::string>& from_call) {
    Port p{group_name(group), broadcast, false, from_call ? std::optional(group_name(*from_call)) : std::nullopt};
    check_fits(1, p, p.mode, false);
    for (int n = 1; n < N_PORTS; ++n)
        if (!ports.count(n)) {
            if (arq::log_enabled(LOG, 20))
                arq::log_write(LOG, 20, arq::format("broadcast port %d open: %s%s", n, p.group.c_str(),
                                                    p.from_call ? (" from " + *p.from_call).c_str() : ""));
            ports.emplace(n, std::move(p));
            return n;
        }
    throw std::invalid_argument("every port is open");
}

void KissLink::close(int n) {
    if (n < 1 || n >= N_PORTS || !ports.count(n)) throw std::invalid_argument("port " + std::to_string(n) + " is not open");
    ports.erase(n);
    drop_where(n, [n](const Queued& q) { return q.port == n; });
}

void KissLink::set_mode(int n, const std::string& mode, bool shift) {
    auto it = ports.find(n);
    if (it == ports.end()) throw std::invalid_argument("port " + std::to_string(n) + " is not open");
    check_mode(mode);
    check_fits(n, it->second, mode, shift);
    it->second.mode = mode;
    it->second.shift = shift;
}

void KissLink::drop_where(int port, const std::function<bool(const Queued&)>& which) {
    const auto k = std::count_if(queue.begin(), queue.end(), which);
    if (!k) return;
    std::erase_if(queue, which);
    events.push_back("BCAST " + std::to_string(port) + " DROPPED " + std::to_string(k));
}

void KissLink::enqueue(Bytes frame, int port, std::optional<std::int64_t> ack) {
    if (!ports.count(port)) {
        events.push_back("BCAST " + std::to_string(port) + " DROPPED 1");
        return;
    }
    queue.push_back({port, std::move(frame), ack});
}

namespace {
bool same_burst(const arq::TxBurst& a, const arq::TxBurst& b) {  // TxBurst's == in Python (a dataclass)
    if (a.submode != b.submode || a.burst_seq != b.burst_seq || a.slots.size() != b.slots.size()) return false;
    for (std::size_t i = 0; i < a.slots.size(); ++i) {
        const auto &x = a.slots[i], &y = b.slots[i];
        if (x.mask_id.key != y.mask_id.key || x.mask_id.direction != y.mask_id.direction || x.mask_id.seq != y.mask_id.seq ||
            x.rv != y.rv || x.payload != y.payload)
            return false;
    }
    return true;
}
}  // namespace

std::vector<std::string> KissLink::take_events() { return std::exchange(events, {}); }

std::vector<std::pair<int, std::int64_t>> KissLink::take_acks() { return std::exchange(acks, {}); }

void KissLink::on_sent(const arq::TxBurstPtr& burst) {
    if (inflight_ && burst && (burst == inflight_ || same_burst(*burst, *inflight_))) {
        acks.insert(acks.end(), inflight_acks_.begin(), inflight_acks_.end());
        inflight_.reset();
        inflight_acks_.clear();
    }
}

void KissLink::missed(const std::string& submode, int n_cw) {
    events.push_back("BCAST * MISSED " + submode + " " + std::to_string(n_cw));
}

Peer* KissLink::peer(int h) {
    for (auto& [k, p] : peers)
        if (k == h) return &p;
    return nullptr;
}

std::pair<std::string, int> KissLink::route(const Port& port, ByteView frame) const {
    if (port.shift) {
        const auto ax = parse_ax25(frame);
        if (ax && ax->connected) {
            const int h = station_hash(ax->next_hop);
            for (const auto& [k, p] : peers)
                if (k == h && p.report && clock() - p.report->second <= REPORT_MAX_S) {
                    const arq::Mode* m = arq::decode(p.report->first >> 2);
                    const auto ok = arq::allowed(cap);
                    if (m && std::find(ok.begin(), ok.end(), m) != ok.end())
                        return {std::string(m->name), p.report->first & 3};
                }
        }
    }
    return {port.mode, static_cast<int>(arq::SIZE_S.size()) - 1};
}

// Control codeword payloads: header, the group, and on a shifting port as
// many fresh reports as fit.
std::vector<Bytes> KissLink::control(const arq::Mode& m, int n, const Port& port, int sender) {
    const std::size_t pb = static_cast<std::size_t>(arq::ctl_payload_bytes(m));
    const int n_max = std::min(4, arq::max_ctl(m));
    Bytes body = group_tlv(n, port);
    if (port.shift) {
        const double now = clock();
        std::vector<std::pair<int, Peer*>> fresh;
        for (auto& [h, p] : peers)
            if (now - p.heard <= HEARD_MAX_S && p.shifter.measured) fresh.emplace_back(h, &p);
        std::stable_sort(fresh.begin(), fresh.end(),
                         [](const auto& a, const auto& b) { return a.second->heard > b.second->heard; });
        const int room = std::min((n_max * static_cast<int>(pb) - 1 - static_cast<int>(body.size()) - 4) / 3, (255 - 2) / 3);
        Bytes rep{static_cast<std::uint8_t>(sender >> 8), static_cast<std::uint8_t>(sender & 255)};
        arq::StationView stub;
        stub.cap = cap;
        for (std::size_t i = 0; i < fresh.size() && static_cast<int>(i) < std::max(0, room); ++i) {
            const auto r = fresh[i].second->shifter.recommend(stub);
            put_hb(rep, fresh[i].first, (r.data << 2) | r.hint);
        }
        const Bytes t = tlv(T_REPORTS, rep);
        body.insert(body.end(), t.begin(), t.end());
    }
    const int n_ctl = std::max(1, ceil_div(1 + body.size(), pb));
    Bytes stream{static_cast<std::uint8_t>(VERSION << 4 | (n_ctl - 1))};
    stream.insert(stream.end(), body.begin(), body.end());
    stream.resize(static_cast<std::size_t>(n_ctl) * pb, 0);
    std::vector<Bytes> out;
    for (int i = 0; i < n_ctl; ++i)
        out.emplace_back(stream.begin() + static_cast<std::ptrdiff_t>(i * pb),
                         stream.begin() + static_cast<std::ptrdiff_t>((i + 1) * pb));
    return out;
}

arq::TxBurstPtr KissLink::next_burst() {
    while (!queue.empty()) {
        const int n = queue[0].port;
        const Port& port = ports.at(n);
        const auto [mode, hint] = route(port, queue[0].frame);
        const arq::Mode& m = arq::mode_at(mode);
        int sender = 0;
        if (port.shift)
            if (const auto ax = parse_ax25(queue[0].frame)) {
                sender = station_hash(ax->sender);
                me.insert(sender);
            }
        const auto ctl = control(m, n, port, sender);
        const double seconds = mode != port.mode ? arq::SIZE_S.at(static_cast<std::size_t>(hint)) : BROADCAST_S;
        const std::size_t pb = static_cast<std::size_t>(arq::payload_bytes(m));
        // the size class is a preference: a burst grows to carry its first
        // frame, up to what its header can say
        const int limit = m.is_cpm() ? 1 + 1 + tables::CPM.max_data : config::max_codewords(m.ofdm->sync_band);
        const int need = static_cast<int>(ctl.size()) + ceil_div(2 + queue[0].frame.size(), pb);
        const int n_max = std::min(limit, std::max(arq::slots_for(m, seconds), need));
        const std::size_t room = static_cast<std::size_t>(n_max - static_cast<int>(ctl.size())) * pb;
        std::vector<Queued> taken, rest;
        std::size_t size = 0;
        for (auto& q : queue) {
            if (q.port == n && route(port, q.frame).first == mode && size + 2 + q.frame.size() <= room) {
                size += 2 + q.frame.size();
                taken.push_back(std::move(q));
            } else {
                rest.push_back(std::move(q));
            }
        }
        if (taken.empty()) {  // the first frame alone doesn't fit: it never will in this mode
            if (arq::log_enabled(LOG, 40))
                arq::log_write(LOG, 40, arq::format("%zu-byte frame dropped: a %s burst carries at most %zu",
                                                    rest[0].frame.size(), mode.c_str(), room - 2));
            rest.erase(rest.begin());
            queue = std::move(rest);
            events.push_back("BCAST " + std::to_string(n) + " DROPPED 1");
            continue;
        }
        queue = std::move(rest);
        Bytes stream;
        for (const auto& q : taken) {
            stream.push_back(static_cast<std::uint8_t>(q.frame.size() >> 8));
            stream.push_back(static_cast<std::uint8_t>(q.frame.size() & 255));
            stream.insert(stream.end(), q.frame.begin(), q.frame.end());
        }
        stream.resize(stream.size() + (pb - stream.size() % pb) % pb, 0);
        const int key = port.key();
        auto b = std::make_shared<arq::TxBurst>();
        b->submode = mode;
        for (std::size_t i = 0; i < ctl.size(); ++i) b->slots.push_back({arq::ctl_mask(0, static_cast<int>(i), key), 0, ctl[i]});
        for (std::size_t j = 0; j * pb < stream.size(); ++j)
            b->slots.push_back({arq::data_mask(0, static_cast<std::int64_t>(ctl.size() + j), key), 0,
                                Bytes(stream.begin() + static_cast<std::ptrdiff_t>(j * pb),
                                      stream.begin() + static_cast<std::ptrdiff_t>((j + 1) * pb))});
        b->burst_seq = ++n_sent;
        inflight_ = b;
        inflight_acks_.clear();
        for (const auto& q : taken)
            if (q.ack) inflight_acks_.emplace_back(n, *q.ack);
        return b;
    }
    return nullptr;
}

// The burst's control, checked under the key of the group it names.
std::optional<std::tuple<ControlRead, int>> KissLink::read_burst_control(arq::ModemRx& rx, int n_cw) {
    for (const auto& c0 : rx.raw(0)) {
        const int n_ctl = (c0[0] & 3) + 1;
        if (c0[0] >> 4 != VERSION || n_ctl > n_cw) continue;
        Bytes stream = c0;
        for (int i = 1; i < n_ctl; ++i) {  // the best guess, to read
            const auto r = rx.raw(i);
            const Bytes& p = r.empty() ? Bytes(c0.size(), 0) : r[0];
            stream.insert(stream.end(), p.begin(), p.end());
        }
        int key;
        try {
            key = group_key(read_control(parse_tlvs(ByteView(stream).subspan(1))).group);
        } catch (const std::invalid_argument&) {
            continue;
        }
        Bytes ctl;
        bool all = true;
        for (int i = 0; i < n_ctl && all; ++i) {
            const auto p = rx.decode_plain(i, arq::ctl_mask(0, i, key));
            if (!p || (i == 0 && *p != c0)) all = false;
            else ctl.insert(ctl.end(), p->begin(), p->end());
        }
        if (!all) continue;
        try {
            return std::tuple{read_control(parse_tlvs(ByteView(ctl).subspan(1))), n_ctl};
        } catch (const std::invalid_argument& e) {
            arq::log_write(LOG, 30, arq::format("RX malformed broadcast control (%s): dropped", e.what()));
            return std::nullopt;
        }
    }
    return std::nullopt;
}

std::optional<std::vector<std::pair<int, Bytes>>> KissLink::on_burst(const arq::Heard& r,
                                                                     std::shared_ptr<const arq::SlotSoft> soft,
                                                                     std::optional<double> dd_budget) {
    arq::ModemRx rx(r, nullptr, dd_budget, std::move(soft));
    return on_burst(r, rx);
}

std::optional<std::vector<std::pair<int, Bytes>>> KissLink::on_burst(const arq::Heard& r, arq::ModemRx& rx) {
    const int n = rx.n_cw();
    const double now = clock();
    const std::size_t pb = static_cast<std::size_t>(arq::payload_bytes(r.mode()));
    auto data = [&](int start, int key) {  // data slots from `start` under `key` -> (frames, per-slot ok, frames lost)
        std::vector<Bytes> payloads;
        std::vector<bool> ok;
        for (int i = start; i < n; ++i) {
            auto p = rx.decode_plain(i, arq::data_mask(0, i, key));
            ok.push_back(p.has_value());
            payloads.push_back(p ? std::move(*p) : Bytes(pb, 0));
        }
        auto [frames, lost] = payloads.empty() ? std::pair<std::vector<Bytes>, int>{} : tnc::unpack(payloads, ok);
        return std::tuple{std::move(frames), std::move(ok), lost};
    };
    auto ports_where = [&](auto pred) {
        std::vector<int> out;
        for (const auto& [i, p] : ports)
            if (pred(p)) out.push_back(i);
        return out;
    };
    std::vector<std::pair<int, Bytes>> out;
    if (auto ctl = read_burst_control(rx, n)) {
        auto& [c, n_ctl] = *ctl;
        auto [frames, ok, lost] = data(n_ctl, group_key(c.group));
        const auto mine = ports_where([&](const Port& p) { return p.group == c.group; });
        if (c.reports && std::any_of(mine.begin(), mine.end(), [&](int i) { return ports.at(i).shift; }))
            learn(*c.reports, frames, r, ok, now);
        for (int i : mine) {
            events.push_back("BCAST " + std::to_string(i) + " HEARD" + (c.call ? " " + *c.call : ""));
            if (lost) events.push_back("BCAST " + std::to_string(i) + " LOST " + std::to_string(lost));
            for (const auto& f : frames) out.emplace_back(i, f);
        }
        return out;
    }
    // control lost: slot 1 passing as data under an open port's key says the
    // burst is that group's, with a one-codeword control (docs/broadcast.md §2)
    if (n < 2) return std::nullopt;
    std::set<int> keys;
    for (const auto& [i, p] : ports) keys.insert(p.key());
    std::optional<int> key;
    for (int k : keys)
        if (rx.decode_plain(1, arq::data_mask(0, 1, k))) {
            key = k;
            break;
        }
    if (!key) return std::nullopt;
    auto [frames, ok, lost] = data(1, *key);
    const auto mine = ports_where([&](const Port& p) { return p.key() == *key; });
    std::set<std::string> groups;
    for (int i : mine) groups.insert(ports.at(i).group);
    if (groups.size() > 1) {  // two groups, one key: whose it is can't be told
        arq::log_write(LOG, 30, arq::format("broadcast burst under a key %zu open groups share: dropped", mine.size()));
        for (int i : mine)
            events.push_back("BCAST " + std::to_string(i) + " LOST " + std::to_string(frames.size() + static_cast<std::size_t>(lost)));
        return out;
    }
    for (int i : mine) {
        events.push_back("BCAST " + std::to_string(i) + " HEARD");
        if (lost) events.push_back("BCAST " + std::to_string(i) + " LOST " + std::to_string(lost));
        for (const auto& f : frames) out.emplace_back(i, f);
    }
    return out;
}

// A shifting port heard `rep` (T_REPORTS): what we know of its sender.
void KissLink::learn(const Bytes& rep, const std::vector<Bytes>& frames, const arq::Heard& r, const std::vector<bool>& ok,
                     double now) {
    int sender = be16(rep.data());
    if (!sender && !frames.empty())
        if (const auto ax = parse_ax25(frames[0])) sender = station_hash(ax->sender);
    if (!sender || me.count(sender)) return;
    const std::string mode(r.mode().name);
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
    for (std::size_t j = 2; j + 3 <= rep.size(); j += 3)
        if (me.count(be16(&rep[j]))) p->report = {{rep[j + 2], now}};
}

}  // namespace data2g::kisslink
