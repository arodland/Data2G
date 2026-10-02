#include "arq/link.hpp"

#include <algorithm>
#include <cstdarg>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace data2g::arq {

// --- logging ------------------------------------------------------------------------

namespace {
LogSink g_sink;
constexpr const char* LOG = "data2g.link";
constexpr int INFO = 20, WARNING = 30;

std::string join(const std::vector<std::string>& parts, const char* sep) {
    std::string out;
    for (std::size_t i = 0; i < parts.size(); ++i) out += (i ? sep : "") + parts[i];
    return out;
}

template <typename T>
std::string str(T v) {
    return std::to_string(v);
}
}  // namespace

void set_log_sink(LogSink sink) { g_sink = std::move(sink); }
bool log_enabled(const char* logger, int level) { return g_sink.enabled && g_sink.write && g_sink.enabled(logger, level); }
void log_write(const char* logger, int level, const std::string& msg) {
    if (log_enabled(logger, level)) g_sink.write(logger, level, msg);
}

std::string format(const char* fmt, ...) {
    va_list ap, ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    const int n = std::vsnprintf(nullptr, 0, fmt, ap);
    va_end(ap);
    std::string out(static_cast<std::size_t>(std::max(n, 0)), '\0');
    std::vsnprintf(out.data(), out.size() + 1, fmt, ap2);
    va_end(ap2);
    return out;
}

bool dup_ctl(const TxBurst& b) {
    return std::any_of(b.slots.begin(), b.slots.end(), [](const Slot& s) { return s.rv && s.mask_id.seq >= SEQ_MOD; });
}

// --- the two directions -------------------------------------------------------------

void TxSide::write(ByteView data) {
    const Bytes r = to_records(data);
    buf.insert(buf.end(), r.begin(), r.end());
}

bool TxSide::pending() const { return !cws.empty() || stream_end < buf_off + static_cast<std::int64_t>(buf.size()); }

bool TxSide::on_ack(std::int64_t cum, const std::set<std::int64_t>& received) {
    if (!(base <= cum && cum <= next))
        throw ProtocolError("peer cumulative " + str(cum) + " outside [" + str(base) + ", " + str(next) + "]");
    const bool advanced = cum > base;
    for (std::int64_t s = base; s < cum; ++s) {
        auto it = cws.find(s);
        if (it == cws.end()) continue;
        const Codeword& c = it->second;
        acked += c.length;
        acked_wire += static_cast<std::int64_t>(c.payload.size());
        acked_plain += c.comp ? c.length : static_cast<std::int64_t>(c.payload.size());
        cws.erase(it);
    }
    base = cum;
    std::set<std::int64_t> held;
    for (auto s : received)
        if (cum < s && s < next) held.insert(s);
    ack = std::make_pair(cum, std::move(held));
    std::int64_t keep = stream_end;
    if (!cws.empty()) {
        keep = cws.begin()->second.start;
        for (const auto& [_, c] : cws) keep = std::min(keep, c.start);
    }
    buf.erase(buf.begin(), buf.begin() + std::clamp<std::int64_t>(keep - buf_off, 0, static_cast<std::int64_t>(buf.size())));
    buf_off = keep;
    keep = hist_off + static_cast<std::int64_t>(hist.size());
    if (!cws.empty()) {
        keep = cws.begin()->second.cstart;
        for (const auto& [_, c] : cws) keep = std::min(keep, c.cstart);
    }
    keep -= HIST;
    if (keep > hist_off) {
        hist.erase(hist.begin(), hist.begin() + std::min<std::int64_t>(keep - hist_off, static_cast<std::int64_t>(hist.size())));
        hist_off = keep;
    }
    return advanced;
}

Codeword& TxSide::new_codeword(const std::string& submode, int pb, bool compress, std::int64_t bn) {
    const auto i = static_cast<std::size_t>(stream_end - buf_off);
    auto n = static_cast<std::int64_t>(std::min<std::size_t>(static_cast<std::size_t>(pb), buf.size() - i));
    Bytes payload(buf.begin() + static_cast<std::ptrdiff_t>(i), buf.begin() + static_cast<std::ptrdiff_t>(i + n));
    payload.resize(static_cast<std::size_t>(pb), 0);
    bool comp = false;
    if (compress) {
        const std::size_t h0 = hist.size() > HIST ? hist.size() - HIST : 0;
        const std::size_t end = std::min(buf.size(), i + 16 * static_cast<std::size_t>(pb));
        auto fit = deflate_fit(ByteView(hist).subspan(h0), ByteView(buf).subspan(i, end - i), pb);
        if (fit) {
            n = fit->first;
            payload = std::move(fit->second);
            payload.resize(static_cast<std::size_t>(pb), 0);
            comp = true;
        }
    }
    Codeword c;
    c.seq = next;
    c.start = stream_end;
    c.length = n;
    c.submode = submode;
    c.payload = payload;
    c.comp = comp;
    c.cstart = hist_off + static_cast<std::int64_t>(hist.size());
    c.first_bn = bn;
    if (comp)
        hist.insert(hist.end(), buf.begin() + static_cast<std::ptrdiff_t>(i), buf.begin() + static_cast<std::ptrdiff_t>(i + n));
    else
        hist.insert(hist.end(), payload.begin(), payload.end());
    auto& ref = cws[c.seq] = std::move(c);
    next += 1;
    stream_end += n;
    return ref;
}

std::vector<std::int64_t> TxSide::missing() const {
    std::vector<std::int64_t> out;
    if (!ack) return out;
    for (std::int64_t s = ack->first; s < next; ++s)
        if (!ack->second.count(s)) out.push_back(s);
    return out;
}

std::int64_t TxSide::abandon() {
    const std::int64_t a = base;
    if (auto it = cws.find(a); it != cws.end()) {
        stream_end = it->second.start;
        hist.resize(static_cast<std::size_t>(std::clamp<std::int64_t>(it->second.cstart - hist_off, 0,
                                                                      static_cast<std::int64_t>(hist.size()))));
    }
    cws.clear();
    next = a;
    ack.reset();
    return a;
}

bool RxSide::accept(std::int64_t seq, Bytes payload, bool comp) {
    if (seq < cum || buf.count(seq) || seq >= cum + WINDOW) return false;
    buf[seq] = {std::move(payload), comp};
    bool delivered = false;
    while (true) {
        auto it = buf.find(cum);
        if (it == buf.end()) break;
        auto [p, z] = std::move(it->second);
        buf.erase(it);
        comp_seqs.erase(cum);
        wire += static_cast<std::int64_t>(p.size());
        if (z) {
            try {
                p = inflate(hist, p);
            } catch (const std::invalid_argument& e) {
                throw ProtocolError("seq " + str(cum) + ": " + e.what());
            }
        }
        plain += static_cast<std::int64_t>(p.size());
        hist.insert(hist.end(), p.begin(), p.end());
        if (hist.size() > HIST) hist.erase(hist.begin(), hist.end() - HIST);
        const Bytes got = reader.feed(p);
        out.insert(out.end(), got.begin(), got.end());
        cum += 1;
        delivered = true;
    }
    return delivered;
}

std::vector<std::int64_t> RxSide::abandon(std::int64_t a) {
    const std::int64_t lo = std::max(a, cum);
    std::vector<std::int64_t> gone;
    for (auto it = buf.lower_bound(lo); it != buf.end();) {
        gone.push_back(it->first);
        it = buf.erase(it);
    }
    comp_seqs.erase(comp_seqs.lower_bound(lo), comp_seqs.end());
    return gone;
}

// --- one station ------------------------------------------------------------------------

namespace {
bool env_compress() {
    const char* v = std::getenv("DATA2G_COMPRESS");
    return !(v && std::strcmp(v, "0") == 0);
}

constexpr int OPTIONAL_TLVS[] = {T_BUFFER, T_CHAT, T_REPLY, T_DUPCTL};

// Control bytes with T_RV, T_NEW and a T_COMP covering the first zk data
// slots, as they will be packed.
std::int64_t ctl_bytes(Ext e, int k, bool has_new, int zk) {
    if (k) e[T_RV] = Bytes(static_cast<std::size_t>(ceil_div(2 * k, 8)));
    if (has_new) e[T_NEW] = Bytes(1);
    if (zk) e[T_COMP] = Bytes(static_cast<std::size_t>(ceil_div(zk, 8)));
    std::int64_t n = 4;
    for (const auto& [_, v] : e) n += 2 + static_cast<std::int64_t>(v.size());
    return n;
}

int ctl_size(const Ext& e, int pb, int k, bool has_new, int zk = 0) {
    return static_cast<int>(std::max<std::int64_t>(1, ceil_div(ctl_bytes(e, k, has_new, zk), pb)));
}
}  // namespace

Station::Station(int direction_, std::shared_ptr<Policy> policy_, bool master_, int key_, std::optional<int> max_misses_,
                 int cap_, bool chat_)
    : direction(direction_), policy(std::move(policy_)), master(master_), key(key_), max_misses(max_misses_), cap(cap_),
      chat(chat_), compress(env_compress()) {}

Bytes Station::read() {
    Bytes out = std::move(rx.out);
    rx.out.clear();
    return out;
}

std::string Station::mode(std::optional<int> rec) { return rec ? policy->mode_name(*rec) : "-"; }

std::string Station::burst_desc(const std::string& submode, int n_cw, bool dup) {
    std::string out = submode + " x" + str(n_cw);
    if (policy->has_airtime()) out += format(" %.1fs", policy->airtime(submode, n_cw, dup));
    return out;
}

std::string Station::snr() {
    auto s = policy->snr_est();
    return s ? format(" | snr %.1f dB", *s) : "";
}

bool Station::comp_bit(std::int64_t seq) const {
    const Codeword& c = tx.cws.at(seq);
    return c.comp && !c.comp_known;
}

TxBurstPtr Station::build(bool fresh) {
    if (latest - confirmed >= BURST_MOD - 1 && last_sent) {
        // 7 of my bursts unconfirmed: repeat the latest instead
        log_write(LOG, INFO, format("TX b%d repeat: 7 bursts unconfirmed", static_cast<int>(pmod(latest, BURST_MOD))));
        auto it = sent_seqs.find(latest);
        stats["cw_resend"] += it == sent_seqs.end() ? 0 : static_cast<std::int64_t>(it->second.size());
        return last_sent;
    }
    const std::int64_t bn = latest + 1;
    const int escalation = std::min(std::max(misses, reply_escalation), 3);
    auto [submode, max_cw] = policy->choose(*this, escalation);
    const int pb = policy->payload_bytes(submode);
    const int cpb = policy->ctl_payload_bytes(submode);
    const int max_ctl = policy->max_ctl(submode);
    Ext ext;
    std::optional<int> reset;
    if (!fresh) {
        max_cw = 0;  // control only
    } else if (resync_due) {
        reset = 1;
        resync_due = false;
    } else if (std::any_of(tx.cws.begin(), tx.cws.end(), [&](const auto& kv) { return kv.second.submode != submode; })) {
        reset = 0;  // a resend must keep its submode (§4)
    }
    if (reset) {
        const std::int64_t a = tx.abandon();
        abandon_epoch = (abandon_epoch + 1) % 128;
        abandon_tlv = Bytes{static_cast<std::uint8_t>(pmod(a, SEQ_MOD)), static_cast<std::uint8_t>(*reset | abandon_epoch << 1)};
        abandon_bursts.clear();
    }
    if (abandon_tlv) {
        ext[T_ABANDON] = *abandon_tlv;
        abandon_bursts.insert(bn);
        if ((*abandon_tlv)[1] & 1) ext[T_RESYNC] = {};
    }
    const auto r = policy->recommend(*this);
    const int rec = r ? r->rec : 0, hint = r ? r->hint : 1;
    const std::optional<int> reply = r ? r->reply : std::nullopt;
    if (reply) ext[T_REPLY] = Bytes{static_cast<std::uint8_t>(*reply)};
    if (chat) ext[T_CHAT] = {};
    if (policy->want_dup()) ext[T_DUPCTL] = {};
    if (chat || peer_chat) {
        const std::int64_t queued = tx.buf_off + static_cast<std::int64_t>(tx.buf.size()) - tx.stream_end;
        if (queued > CHAT_LINE_BYTES) {
            const auto q = std::min<std::int64_t>(queued, 65535);
            ext[T_BUFFER] = Bytes{static_cast<std::uint8_t>(q >> 8), static_cast<std::uint8_t>(q)};
        }
    }
    Core core;
    core.ftype = fresh ? ARQ : PROBE;
    core.burst_seq = static_cast<int>(pmod(bn, BURST_MOD));
    core.acted_on = acted_on_;
    core.cum = static_cast<int>(pmod(rx.cum, SEQ_MOD));
    core.reply_lost = reply_lost;
    core.recommend = rec;
    core.size_hint = hint;
    reply_lost = false;

    // what this burst's ACK conveys, exactly (the snapshot must match it)
    std::set<int> held7;
    for (const auto& [s, _] : rx.buf) held7.insert(static_cast<int>(pmod(s, SEQ_MOD)));
    Bytes bitmap = pack_bitmap(held7, core.cum);
    const std::vector<std::int64_t> missing = fresh ? tx.missing() : std::vector<std::int64_t>{};
    bool has_new = fresh && new_available() > 0;
    int k = static_cast<int>(std::min<std::int64_t>(static_cast<std::int64_t>(missing.size()), std::max(0, max_cw - 1)));
    int n_ctl = 0, dup = 1;
    while (true) {
        Ext e = ext;
        if (!bitmap.empty()) e[T_BITMAP] = bitmap;
        int zk = 0;
        for (int j = 0; j < k; ++j)
            if (comp_bit(missing[static_cast<std::size_t>(j)])) zk = j + 1;
        n_ctl = ctl_size(e, cpb, k, has_new, zk);
        dup = (fresh && peer_wants_dup && (k || has_new)) ? 2 : 1;
        if (n_ctl <= max_ctl && dup * n_ctl + k <= std::max(max_cw, dup * n_ctl) && (fresh || !has_new)) break;
        const int* opt = std::find_if(std::begin(OPTIONAL_TLVS), std::end(OPTIONAL_TLVS), [&](int t) { return ext.count(t) > 0; });
        if (k) {
            k -= 1;
        } else if (opt != std::end(OPTIONAL_TLVS)) {
            ext.erase(*opt);  // hints and requests: a later burst carries them
        } else if (!bitmap.empty()) {
            bitmap.clear();  // costs extra resends, never wrong ones
        } else if (has_new) {
            has_new = false;
        } else {
            throw std::invalid_argument("control does not fit " + submode);
        }
    }
    if (!bitmap.empty()) ext[T_BITMAP] = bitmap;
    std::set<std::int64_t> conveyed;
    if (!bitmap.empty())
        for (int x : unpack_bitmap(bitmap, core.cum)) conveyed.insert(unwrap(x, rx.cum));

    const std::vector<std::int64_t> resend(missing.begin(), missing.begin() + k);
    const int cycle = policy->rv_cycle(submode);
    std::vector<int> rvs;
    for (auto x : resend) rvs.push_back(static_cast<int>(pmod(tx.cws.at(x).heard, cycle)));
    std::vector<const Codeword*> fresh_cws;
    const int room = std::max(0, max_cw - dup * n_ctl - static_cast<int>(resend.size()));
    // a new codeword is compressed only when its T_COMP bit fits the
    // control's padding: compression never costs a control codeword
    const std::int64_t spare = static_cast<std::int64_t>(n_ctl) * cpb - ctl_bytes(ext, k, has_new, 0);
    while (has_new && static_cast<int>(fresh_cws.size()) < room && tx.next - tx.base < WINDOW) {
        if (new_available() <= 0) break;
        const bool fits = compress && 2 + ceil_div(k + static_cast<std::int64_t>(fresh_cws.size()) + 1, 8) <= spare;
        fresh_cws.push_back(&tx.new_codeword(submode, pb, fits, bn));
    }
    core.k = static_cast<int>(resend.size());
    if (!resend.empty()) ext[T_RV] = pack_rv(rvs);
    if (!fresh_cws.empty()) ext[T_NEW] = Bytes{static_cast<std::uint8_t>(pmod(fresh_cws[0]->seq, SEQ_MOD))};
    std::vector<bool> flags;
    for (auto x : resend) flags.push_back(comp_bit(x));
    for (auto* c : fresh_cws) flags.push_back(c->comp);
    if (Bytes comp = pack_flags(flags); !comp.empty()) ext[T_COMP] = comp;
    if (dup == 2 && resend.empty() && fresh_cws.empty()) dup = 1;  // nothing but control after all
    if (dup == 2) core.ftype = ARQ_DUP;
    Control control{core, ext};
    const auto ctl = control.pack(cpb);
    core.n_ctl = control.core.n_ctl;
    if (core.n_ctl > n_ctl) throw std::logic_error("T_COMP outgrew the control's padding");
    auto burst = std::make_shared<TxBurst>();
    burst->submode = submode;
    for (std::size_t i = 0; i < ctl.size(); ++i)
        for (int rv = 0; rv < dup; ++rv) burst->slots.push_back({ctl_mask(direction, static_cast<int>(i), key), rv, ctl[i]});
    for (std::size_t j = 0; j < resend.size(); ++j)
        burst->slots.push_back({data_mask(direction, resend[j], key), rvs[j], tx.cws.at(resend[j]).payload});
    for (auto* c : fresh_cws) burst->slots.push_back({data_mask(direction, c->seq, key), 0, c->payload});
    snapshots[bn] = {rx.cum, conveyed};
    auto& sent = sent_seqs[bn] = resend;
    for (auto* c : fresh_cws) sent.push_back(c->seq);
    latest = bn;
    for (auto it = snapshots.begin(); it != snapshots.end() && it->first < confirmed;) {
        sent_seqs.erase(it->first);
        it = snapshots.erase(it);
    }
    burst->burst_seq = bursts_sent;
    bursts_sent += 1;
    last_sent = burst;
    std::int64_t n_comp = 0, n_bytes = 0;
    for (auto* c : fresh_cws) {
        n_comp += c->comp;
        n_bytes += c->length;
    }
    stats["cw_new"] += static_cast<std::int64_t>(fresh_cws.size());
    stats["cw_comp"] += n_comp;
    stats["cw_resend"] += static_cast<std::int64_t>(resend.size());
    if (log_enabled(LOG, INFO)) {
        const char* kind = (!resend.empty() || !fresh_cws.empty()) ? "data" : (fresh ? "ack" : "poll");
        std::vector<std::string> parts{std::string(kind) + " " + burst_desc(submode, static_cast<int>(burst->slots.size()), dup == 2)};
        if (!resend.empty()) {
            std::vector<std::string> rs;
            for (std::size_t j = 0; j < resend.size(); ++j) rs.push_back(str(resend[j]) + "/rv" + str(rvs[j]));
            parts.push_back("resend " + join(rs, " "));
        }
        if (!fresh_cws.empty()) {
            std::string s = "new " + str(fresh_cws.front()->seq);
            if (fresh_cws.size() > 1) s += "-" + str(fresh_cws.back()->seq);
            s += " " + str(n_bytes) + " B";
            if (n_comp) s += " (" + str(n_comp) + " compressed)";
            parts.push_back(s);
        }
        if (tx.pending()) parts.push_back("unacked " + str(tx.next - tx.base) + ", queued " + str(new_available()) + " B");
        parts.push_back("ack cum " + str(rx.cum) + (rx.buf.empty() ? "" : " +" + str(rx.buf.size()) + " held"));
        parts.push_back("ask data " + mode(rec) + " size " + str(hint) + ", reply " + mode(reply));
        if (escalation) parts.push_back("escalated " + str(escalation));
        if (misses) parts.push_back("timeout " + str(misses));
        if (ext.count(T_ABANDON)) parts.push_back(std::string(ext.count(T_RESYNC) ? "resync" : "abandon") + " at " + str(tx.base));
        if (dup == 2) parts.push_back("dup ctl");
        if (core.reply_lost) parts.push_back("reply lost");
        log_write(LOG, INFO, format("TX b%d %s", static_cast<int>(pmod(bn, BURST_MOD)), join(parts, " | ").c_str()));
    }
    return burst;
}

TxBurstPtr Station::on_timeout(bool allow_repeat) {
    if (!master) throw std::logic_error("on_timeout: not the master");
    misses += 1;
    stats["timeouts"] += 1;
    if (max_misses && misses > *max_misses) {
        fail("link lost");
        return nullptr;
    }
    if (allow_repeat && misses <= REPEATS_BEFORE_SHRINK && last_sent) {
        log_write(LOG, INFO, format("TX b%d repeat: timeout %d", static_cast<int>(pmod(latest, BURST_MOD)), misses));
        auto it = sent_seqs.find(latest);
        stats["cw_resend"] += it == sent_seqs.end() ? 0 : static_cast<std::int64_t>(it->second.size());
        return last_sent;  // identical, same burst seq (§6 step 1)
    }
    return build(false);
}

bool Station::handle(RxBurst& rx_) {
    const bool ok = handle_inner(rx_);
    stats[ok ? "rx_ok" : "rx_lost"] += 1;
    if (!ok && log_enabled(LOG, INFO))
        log_write(LOG, INFO, "RX " + burst_desc(rx_.submode(), rx_.n_cw()) + snr() + " | control lost, discarded");
    return ok;
}

bool Station::handle_inner(RxBurst& rxb) {
    std::optional<Bytes> first = rxb.decode(0, ctl_mask(peer(), 0, key), 0, nullptr);
    bool paired = false;
    if (!first && rxb.n_cw() >= 2) {
        // an ARQ_DUP burst's control pair, combined
        first = ctl_pair(rxb, 0, 0);
        paired = first.has_value();
    }
    const bool outcome = policy->has_outcome();
    if (!first) {
        if (outcome) policy->outcome(rxb.submode(), 0, 0, false);
        return false;
    }
    Core core0 = Core::unpack(*first);
    const int dup = core0.ftype == ARQ_DUP ? 2 : 1;
    if (paired && dup == 1) return false;  // combined as a pair, but not sent as one
    std::vector<Bytes> payloads{*first};
    for (int i = 1; i < core0.n_ctl; ++i) {
        auto p = rxb.decode(dup * i, ctl_mask(peer(), i, key), 0, nullptr);
        if (!p && dup == 2) p = ctl_pair(rxb, 2 * i, i);
        if (!p) return false;
        payloads.push_back(std::move(*p));
    }
    Control ctl;
    try {
        ctl = Control::unpack(payloads);
    } catch (const std::invalid_argument&) {
        return false;
    }
    const Core& core = ctl.core;
    const Ext& ext = ctl.ext;
    misses = 0;
    const auto was = std::make_pair(peer_recommend, peer_size_hint);
    peer_recommend = core.recommend;
    peer_size_hint = core.size_hint;
    auto e_reply = ext.find(T_REPLY);
    peer_reply_recommend = (e_reply != ext.end() && !e_reply->second.empty()) ? std::optional<int>(e_reply->second[0]) : std::nullopt;
    peer_chat = ext.count(T_CHAT) > 0;
    peer_wants_dup = ext.count(T_DUPCTL) > 0;
    auto e_buf = ext.find(T_BUFFER);
    peer_queued = (e_buf != ext.end() && e_buf->second.size() == 2) ? (e_buf->second[0] << 8 | e_buf->second[1]) : 0;
    const bool seen = peer_burst && core.burst_seq == *peer_burst;
    const bool repeat = seen && answered_;
    // a repeat or a poll means my last reply did not get through
    reply_escalation = (repeat || core.ftype == PROBE) ? reply_escalation + 1 : 0;
    bool progress = false;
    const std::int64_t base0 = tx.base, next0 = tx.next, cum0 = rx.cum;

    // which of my bursts the peer acted on: its 3-bit seq, resolved in [confirmed, latest]
    std::vector<std::int64_t> acted_list;
    for (std::int64_t k = confirmed; k <= latest; ++k)
        if (pmod(k, BURST_MOD) == core.acted_on) acted_list.push_back(k);
    if (acted_list.size() != 1) {
        fail("protocol: acted-on " + str(core.acted_on) + " outside my bursts " + str(confirmed) + ".." + str(latest));
        return true;
    }
    const std::int64_t acted = acted_list[0];
    confirmed = acted;
    if (abandon_tlv && abandon_bursts.count(acted)) abandon_tlv.reset();  // the peer has applied it
    if (auto it = sent_seqs.find(acted); it != sent_seqs.end()) {
        for (auto x : it->second) {
            if (auto c = tx.cws.find(x); c != tx.cws.end()) {
                c->second.heard += 1;
                c->second.comp_known = c->second.comp_known || c->second.first_bn == acted;
            }
        }
        sent_seqs.erase(it);
    }
    try {
        const std::int64_t cum = unwrap(core.cum, tx.base);
        std::set<std::int64_t> received;
        if (auto b = ext.find(T_BITMAP); b != ext.end())
            for (int x : unpack_bitmap(b->second, core.cum)) received.insert(unwrap(x, cum));
        progress |= tx.on_ack(cum, received);
        acted_on_ = core.burst_seq;
    } catch (const ProtocolError& e) {
        fail(std::string("protocol: ") + e.what());  // never guess: a blind resync can corrupt
        return true;
    }

    // their data
    auto e_ab = ext.find(T_ABANDON);
    const bool abandoned = e_ab != ext.end() && at(e_ab->second, 1) >> 1 != peer_epoch;
    if (abandoned) {
        const std::int64_t a = unwrap(at(e_ab->second, 0), rx.cum);
        if (a != rx.cum) {
            fail("protocol: abandon at " + str(a) + ", cumulative " + str(rx.cum));
            return true;
        }
        peer_epoch = e_ab->second[1] >> 1;
        rx.abandon(a);
        forget_all(rxb);
    }
    const int n_ctl_slots = dup * core.n_ctl;
    const auto slots = map_slots(core, ext, rxb.n_cw() - n_ctl_slots + core.n_ctl, acted);
    auto e_comp = ext.find(T_COMP);
    const auto bits = unpack_flags(e_comp == ext.end() ? ByteView{} : ByteView(e_comp->second), static_cast<int>(slots.size()));
    for (std::size_t j = static_cast<std::size_t>(core.k); j < slots.size(); ++j)
        if (bits[j] && slots[j].first && *slots[j].first >= rx.cum) rx.comp_seqs.insert(*slots[j].first);
    std::vector<bool> comp;
    for (std::size_t j = 0; j < slots.size(); ++j) comp.push_back(bits[j] || (slots[j].first && rx.comp_seqs.count(*slots[j].first)));
    last_rx_data = !slots.empty();
    int n_ok = 0, n_new = 0, n_dec = 0, n_old = 0;
    for (std::size_t j = 0; j < slots.size(); ++j) {
        const int i = n_ctl_slots + static_cast<int>(j);
        const auto& [seq, rv] = slots[j];
        if (!seq || *seq < rx.cum) {
            n_old += 1;
            continue;
        }
        const SoftKey skey{false, peer(), *seq, 0};
        auto p = rxb.decode(i, data_mask(peer(), *seq, key), rv, &skey);
        if (i >= n_ctl_slots + core.k) {
            n_new += 1;
            n_ok += p.has_value();
        }
        if (p) {
            n_dec += 1;
            rxb.forget(skey);
            try {
                progress |= rx.accept(*seq, std::move(*p), comp[j]);
            } catch (const ProtocolError& e) {
                fail(std::string("protocol: ") + e.what());
                return true;
            }
        }
    }
    if (outcome) policy->outcome(rxb.submode(), n_ok, n_new, true);

    if (log_enabled(LOG, INFO)) {
        const char* kind = !slots.empty() ? "data" : (core.ftype == PROBE ? "poll" : "ack");
        std::vector<std::string> parts{std::string(kind) + " " + burst_desc(rxb.submode(), rxb.n_cw(), dup == 2) + snr()};
        if (!slots.empty())
            parts.push_back("resend " + str(core.k) + " + new " + str(static_cast<int>(slots.size()) - core.k) + ", decoded " +
                            str(n_dec) + "/" + str(static_cast<int>(slots.size()) - n_old) +
                            (n_old ? ", " + str(n_old) + " already had" : "") + ", cum " + str(cum0) + "->" + str(rx.cum) +
                            (rx.buf.empty() ? "" : " +" + str(rx.buf.size()) + " held"));
        if (tx.base != base0)
            parts.push_back("acked " + str(base0) + "->" + str(tx.base));
        else if (next0 > base0)
            parts.push_back("acked none of " + str(base0) + "-" + str(next0 - 1));
        const auto now = std::make_pair(peer_recommend, peer_size_hint);
        std::string ask = "wants data " + mode(now.first) + " size " + str(now.second);
        if (was.first && was != now) ask += " (was " + mode(was.first) + " size " + str(was.second) + ")";
        parts.push_back(ask + ", reply " + mode(peer_reply_recommend));
        if (repeat) parts.push_back("repeat (my reply lost)");
        if (abandoned) parts.push_back("abandon");
        if (dup == 2) parts.push_back("dup ctl");
        if (peer_wants_dup) parts.push_back("wants dup ctl");
        log_write(LOG, INFO, format("RX b%d %s", core.burst_seq, join(parts, " | ").c_str()));
    }
    reply_lost = repeat;
    peer_burst = core.burst_seq;
    answered_ = false;
    watchdog(progress);
    return true;
}

std::optional<Bytes> Station::ctl_pair(RxBurst& rxb, int slot, int i) {
    const SoftKey k{true, 0, 0, i};
    const MaskId mask = ctl_mask(peer(), i, key);
    rxb.decode(slot, mask, 0, &k);
    auto p = rxb.decode(slot + 1, mask, 1, &k);
    rxb.forget(k);
    return p;
}

std::vector<std::pair<std::optional<std::int64_t>, int>> Station::map_slots(const Core& core, const Ext& ext, int n_cw,
                                                                           std::int64_t acted) {
    std::vector<std::pair<std::optional<std::int64_t>, int>> out;
    auto snap = snapshots.find(acted);
    auto e_rv = ext.find(T_RV);
    const auto rvs = unpack_rv(e_rv == ext.end() ? ByteView{} : ByteView(e_rv->second), core.k);
    if (core.k) {
        if (snap == snapshots.end()) {
            out.insert(out.end(), static_cast<std::size_t>(core.k), {std::nullopt, 0});
        } else {
            const auto& [cum, received] = snap->second;
            std::int64_t s = cum;
            for (int j = 0; j < core.k; ++s)
                if (!received.count(s)) out.emplace_back(s, rvs[static_cast<std::size_t>(j++)]);
        }
    }
    const int n_new = n_cw - core.n_ctl - core.k;
    if (n_new > 0) {
        auto e_new = ext.find(T_NEW);
        if (e_new == ext.end()) {
            out.insert(out.end(), static_cast<std::size_t>(n_new), {std::nullopt, 0});
        } else {
            const std::int64_t start = unwrap(at(e_new->second, 0), rx.cum);
            for (int j = 0; j < n_new; ++j) out.emplace_back(start + j, 0);
        }
    }
    return out;
}

void Station::forget_all(RxBurst& rxb) {
    for (std::int64_t s = rx.cum; s < rx.cum + WINDOW; ++s) rxb.forget(SoftKey{false, peer(), s, 0});
}

void Station::fail(const std::string& why) {
    log_write(LOG, WARNING, "link failed: " + why);
    state = LinkState::FAILED;
    fail_reason = why;
}

void Station::watchdog(bool progress) {
    if (progress) {
        no_progress = 0;
        resyncs = 0;
        return;
    }
    if (!tx.pending()) return;
    no_progress += 1;
    if (no_progress >= NO_PROGRESS_TURNS) {
        no_progress = 0;
        resyncs += 1;
        if (resyncs > RESYNCS_BEFORE_FAIL) {
            fail("no progress");
        } else {
            resync_due = true;
            log_write(LOG, INFO, format("watchdog: %d turns without progress, resync %d of %d", NO_PROGRESS_TURNS, resyncs,
                                        RESYNCS_BEFORE_FAIL));
        }
    }
}

}  // namespace data2g::arq
