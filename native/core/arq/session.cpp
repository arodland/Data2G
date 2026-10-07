#include "arq/session.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace data2g::arq {

namespace {
constexpr const char* LOG = "data2g.session";
constexpr int INFO = 20, WARNING = 30;
constexpr double INF = std::numeric_limits<double>::infinity();

bool live(SessionState s) { return s == SessionState::CONNECTED || s == SessionState::DISCONNECTING; }

// int.from_bytes(b[lo:hi], "big"), slicing as Python does
std::int64_t be(ByteView b, std::size_t lo, std::size_t hi) {
    std::int64_t v = 0;
    for (std::size_t i = lo; i < std::min(hi, b.size()); ++i) v = v << 8 | b[i];
    return v;
}

Bytes cat(std::initializer_list<ByteView> parts) {
    Bytes out;
    for (auto p : parts) out.insert(out.end(), p.begin(), p.end());
    return out;
}

Bytes u16(std::int64_t v) { return {static_cast<std::uint8_t>(v >> 8), static_cast<std::uint8_t>(v)}; }

std::uint8_t byte(std::int64_t v) {
    if (v < 0 || v > 255) throw std::invalid_argument("bytes must be in range(0, 256)");
    return static_cast<std::uint8_t>(v);
}

std::uint8_t tenths(double t_turn) { return byte(static_cast<std::int64_t>(std::nearbyint(t_turn * 10))); }  // round(): half to even
// What is wrong with a CRC-valid session frame body, or nullopt. A bad one
// is dropped as if it had not decoded (the peer retries).
std::optional<std::string> check_frame(ByteView body) {
    const int sub = body[0];
    const std::size_t need = sub == CONNECT ? 22 : sub == CONNECT_ACK ? 5 : sub == CONNECT_NAK ? 4 : 1;
    if (body.size() < need) return frame_desc(body.first(1)) + " of " + std::to_string(body.size()) + " B";
    try {
        if (sub == CONNECT) unpack_call(body.subspan(2, 8)), unpack_call(body.subspan(10, 8));
    } catch (const std::invalid_argument& e) {
        return frame_desc(body.first(1)) + ": " + e.what();
    }
    if (sub == CONNECT_ACK && body[3] > 2) return "CONNECT_ACK: cap code " + std::to_string(body[3]);
    return std::nullopt;
}
}  // namespace

const char* state_name(SessionState s) {
    static const char* names[] = {"idle", "listen", "connecting", "connected", "disconnecting", "closed"};
    return names[static_cast<int>(s)];
}

int session_key(const std::string& caller, const std::string& callee, std::int64_t nonce) {
    std::uint32_t h = 0x811C9DC5;
    for (unsigned char b : caller + "|" + callee + "|" + std::to_string(nonce)) h = (h ^ b) * 0x01000193u;
    return (h & 0xFFFF) ? static_cast<int>(h & 0xFFFF) : 1;
}

std::string frame_desc(ByteView body) {
    const int sub = at(body, 0);
    static const char* names[] = {nullptr, "CONNECT", "CONNECT_ACK", "CONNECT_NAK", "DISC", "DISC_ACK"};
    std::string out = (sub >= CONNECT && sub <= DISC_ACK) ? names[sub] : "session frame " + std::to_string(sub);
    if (sub == CONNECT && body.size() >= 18) out += " " + unpack_call(body.subspan(2, 8)) + ">" + unpack_call(body.subspan(10, 8));
    return out;
}

Session::Session(std::string call_, std::shared_ptr<Policy> policy_, double t_turn_, std::shared_ptr<Rng> rng_,
                 std::vector<std::string> aliases_, double stats_interval_s_)
    : call(std::move(call_)), policy(std::move(policy_)), t_turn(t_turn_),
      rng(rng_ ? std::move(rng_) : std::make_shared<DefaultRng>()), aliases(std::move(aliases_)),
      stats_interval_s(stats_interval_s_) {}

std::shared_ptr<Station> Session::make_station(int direction, bool master_, int key) {
    return std::make_shared<Station>(direction, policy, master_, key, std::nullopt, cap, chat);
}

// --- host side ------------------------------------------------------------------------

void Session::set_chat(bool on) {
    chat = on;
    if (station) station->chat = on;
}

void Session::connect(const std::string& peer_, int cap_, double now) {
    peer = peer_;
    for (auto& c : peer)
        if (c >= 'a' && c <= 'z') c = static_cast<char>(c - 'a' + 'A');
    cap = cap_;
    state = SessionState::CONNECTING;
    nonce = rng->randrange(1 << 16);
    tries = 0;
    queue(connect_burst(), now);
}

void Session::write(ByteView data) {
    if (station) {
        station->write(data);
        if (build_at) {
            build_at = -INF;  // an idle poll due: send it now, with the data
            idle_wait = 0.0;
        }
    } else {
        pending_write.insert(pending_write.end(), data.begin(), data.end());
    }
}

Bytes Session::read() { return station ? station->read() : Bytes{}; }

// --- clock side ---------------------------------------------------------------------

TxBurstPtr Session::poll(double now) {
    now_ = now;
    if (stats_interval_s && live(state) && now >= stats_since + stats_interval_s) {
        log_stats("stats", now, stats_since, stats_prev);
        stats_since = now;
        stats_prev = stats();
    }
    if (state == SessionState::CLOSED && !out) return nullptr;
    if (live(state)) {
        if (now >= last_heard + tune.link_lost_s) {
            close(state == SessionState::CONNECTED ? "link lost" : "disconnected (unconfirmed)");
            return nullptr;
        }
        if (state == SessionState::CONNECTED && now >= last_data + tune.idle_close_s) want_disc = true;
    }
    if (deadline && now >= *deadline) {
        deadline.reset();
        on_timeout(now);
    }
    if (build_at && now >= *build_at && state == SessionState::CONNECTED) {
        build_at.reset();
        if (!disc(now)) {
            queue(station->build(), now);
            station->answered();
        }
    }
    if (auto w = wake_time(); w && now >= *w) {
        wakes += 1;
        wake_wait = INF;  // until on_tx_end arms the next
        if (disc(now)) {
            log_write(LOG, INFO, "wake: breaking idle with DISC");
        } else if (wakes == 1) {
            log_write(LOG, INFO, format("wake: breaking idle with %lld B queued", static_cast<long long>(station->new_available())));
            queue(station->build(), now);
        } else {
            log_write(LOG, INFO, format("wake %d: no answer, repeating", wakes));
            queue(station->last_sent, now);
        }
    }
    if (out && due && now >= *due) {
        TxBurstPtr b = std::move(out);
        out.reset();
        due.reset();
        return b;
    }
    return nullptr;
}

std::optional<double> Session::next_event() {
    std::optional<double> best;
    auto take = [&](std::optional<double> t) {
        if (t && (!best || *t < *best)) best = t;
    };
    take(due);
    take(deadline);
    take(build_at);
    take(wake_time());
    if (live(state)) take(last_heard + tune.link_lost_s);
    return best;
}

void Session::on_tx_end(const TxBurstPtr& burst, double now) {
    const bool m = master();
    // only the caller retries (§6), and a callee's DISC
    if ((m && (state == SessionState::CONNECTING || state == SessionState::CONNECTED)) || state == SessionState::DISCONNECTING) {
        double wait = tune.reply_start_s;
        // a reply whose header I miss is still on air: don't poll over it
        if (state == SessionState::CONNECTED && station && burst)
            if (auto hold = policy->reply_hold(*station, *burst)) wait = std::max(wait, *hold);
        deadline = now + t_turn + wait;
    } else if (!m) {
        quiet_from = now;
        wake_wait = t_turn + tune.reply_start_s + tune.wake_guard_s + rng->uniform(0, tune.wake_jitter_s * std::pow(2.0, wakes));
    }
}

void Session::on_header(const std::string& submode, int n_cw, double now) {
    const double end = now + policy->airtime(submode, n_cw, false);
    if (deadline) {
        // a peer's ID burst (one codeword in the connect mode) runs straight
        // into its reply, whose header must still be found (engine ID_INTERVAL_S)
        const bool is_id = n_cw == 1 && submode == policy->connect_mode(cap, 0);
        deadline = std::max(*deadline, end + t_turn + (is_id ? tune.reply_start_s : 0.0));
    }
    if (!master()) quiet_from = std::max(quiet_from, end);
}

void Session::on_rx(RxBurst& rx, double now) {
    if (auto f = session_frame(rx)) {
        on_session_frame(*f, now);
        return;
    }
    Station* st = station.get();
    if (!st || !live(state)) return;
    if (!st->handle(rx)) return;  // not ours, or unreadable: stay silent (§3)
    heard(now);
    if (st->state == LinkState::FAILED) {
        close("link failed: " + st->fail_reason);
        return;
    }
    if (!st->rx.out.empty() || st->last_rx_data || st->tx.pending()) last_data = now;
    if (disc(now)) return;
    if (!master()) {
        queue(st->build(), now);
        st->answered();
        return;
    }
    // the caller: go on now, or after an idle backoff when neither side has data
    const bool woke = build_at.has_value();
    build_at.reset();
    const bool busy = st->tx.pending() || st->last_rx_data || !pending_write.empty() || woke;
    if (busy) {
        idle_wait = 0.0;
        queue(st->build(), now);
        st->answered();
    } else {
        const bool c = st->chat || st->peer_chat;
        const double lo = c ? tune.chat_keepalive_lo_s : tune.keepalive_lo_s, hi = c ? tune.chat_keepalive_hi_s : tune.keepalive_hi_s;
        if (tune.keepalive_doubling)
            idle_wait = idle_wait == 0.0 ? lo : std::min(hi, 2 * idle_wait);
        else
            idle_wait = rng->uniform(lo, hi);
        build_at = now + idle_wait;
    }
}

// --- internals ------------------------------------------------------------------------

bool Session::master() const { return station ? station->master : state == SessionState::CONNECTING; }

void Session::queue(TxBurstPtr burst, double when) {
    out = std::move(burst);
    due = when;
}

void Session::heard(double now) {
    last_heard = now;
    deadline.reset();
    quiet_from = now;
    wakes = 0;
}

void Session::disconnect() {
    want_disc = true;
    if (build_at) build_at = -INF;  // the caller's idle poll: a DISC now instead
}

bool Session::disc(double now) {
    if (!want_disc || station->tx.pending()) return false;
    tries = 0;
    queue(disc_burst(), now);
    state = SessionState::DISCONNECTING;
    return true;
}

std::optional<double> Session::wake_time() {
    Station* st = station.get();
    if (!st || master() || state != SessionState::CONNECTED || out ||
        wakes >= ((st->chat || st->peer_chat) ? tune.chat_wake_tries : tune.wake_tries) || !st->last_sent)
        return std::nullopt;
    if (!wakes && !(want_disc && !st->tx.pending())) {
        auto it = st->sent_seqs.find(st->latest);
        if ((it != st->sent_seqs.end() && !it->second.empty()) || !st->tx.pending()) return std::nullopt;
    }
    return quiet_from + wake_wait;
}

void Session::close(const std::string& why, TxBurstPtr final_burst, double now) {
    build_at.reset();
    if (state != SessionState::CLOSED) {
        events.push_back("DISCONNECTED " + why);
        if (station) log_stats("session", std::max(now, now_), connected_at, {});
    }
    state = SessionState::CLOSED;
    close_reason = why;
    if (final_burst) {
        out = std::move(final_burst);
        due = now;
    } else {
        out.reset();
        due.reset();
    }
    deadline.reset();
}

void Session::connected(double now) {
    state = SessionState::CONNECTED;
    heard(now);
    last_data = connected_at = stats_since = now;
    stats_prev.clear();
}

std::map<std::string, std::int64_t> Session::stats() {
    auto d = station->stats;
    d["tx_bytes"] = station->tx.acked;
    d["rx_bytes"] = station->rx.reader.delivered;
    d["tx_plain"] = station->tx.acked_plain;
    d["tx_wire"] = station->tx.acked_wire;
    d["rx_plain"] = station->rx.plain;
    d["rx_wire"] = station->rx.wire;
    return d;
}

void Session::log_stats(const char* label, double now, double since, const std::map<std::string, std::int64_t>& prev) {
    if (!log_enabled(LOG, INFO)) return;
    std::map<std::string, std::int64_t> d;
    for (const auto& [k, v] : stats()) {
        auto p = prev.find(k);
        d[k] = v - (p == prev.end() ? 0 : p->second);
    }
    const double dt = std::max(now - since, 1e-9);
    const std::int64_t cw = d["cw_new"] + d["cw_resend"];
    auto ratio = [&](const std::string& k) {
        const auto wire = d[k + "_wire"];
        return wire ? format("%.2f", static_cast<double>(d[k + "_plain"]) / static_cast<double>(wire)) : std::string("-");
    };
    log_write(LOG, INFO,
              format("%s %.0f s: tx %lld B acked (%.0f bps, compression %s), rx %lld B (%.0f bps, compression %s)"
                     " | data cw sent %lld, %.0f%% resends | bursts heard %lld, %lld control lost | timeouts %lld",
                     label, dt, static_cast<long long>(d["tx_bytes"]), 8.0 * static_cast<double>(d["tx_bytes"]) / dt,
                     ratio("tx").c_str(), static_cast<long long>(d["rx_bytes"]), 8.0 * static_cast<double>(d["rx_bytes"]) / dt,
                     ratio("rx").c_str(), static_cast<long long>(cw),
                     100.0 * static_cast<double>(d["cw_resend"]) / static_cast<double>(std::max<std::int64_t>(cw, 1)),
                     static_cast<long long>(d["rx_ok"] + d["rx_lost"]), static_cast<long long>(d["rx_lost"]),
                     static_cast<long long>(d["timeouts"])));
}

void Session::on_timeout(double now) {
    if (state == SessionState::CONNECTING) {
        tries += 1;
        log_write(LOG, INFO, format("no answer to CONNECT %s, try %d of %d", peer.c_str(), tries, tune.connect_tries));
        if (tries >= tune.connect_tries) {
            close("no answer");
        } else {
            auto b = connect_burst();
            queue(std::move(b), now + rng->uniform(3.0, 5.0));
        }
    } else if (state == SessionState::DISCONNECTING) {
        tries += 1;
        log_write(LOG, INFO, format("no answer to DISC, try %d of %d", tries, tune.disc_tries));
        if (tries >= tune.disc_tries)
            close("disconnected (unconfirmed)");
        else
            queue(disc_burst(), now);
    } else if (state == SessionState::CONNECTED) {
        // an identical repeat only after a short burst
        const auto& last = station->last_sent;
        const bool short_ = !last || policy->airtime(last->submode, static_cast<int>(last->slots.size()), dup_ctl(*last)) <= tune.repeat_max_s;
        auto b = station->on_timeout(short_);
        if (!b)
            close("link failed: " + station->fail_reason);
        else
            queue(std::move(b), now);
    }
}

// session frames: control-only bursts with a T_SESS extension. Connect
// frames use key 0 (no session yet); DISC / DISC_ACK the session key.

TxBurstPtr Session::session_burst(int direction, int key, const Bytes& body, std::optional<std::string> mode) {
    const std::string m = (mode && !mode->empty()) ? *mode : policy->connect_mode(cap, tries);
    const int cpb = policy->ctl_payload_bytes(m);
    auto b = std::make_shared<TxBurst>();
    b->submode = m;
    b->cap = cap;
    if (body[0] == CONNECT && compact(m)) {
        Bytes p = pack_connect(body);
        p.resize(static_cast<std::size_t>(cpb), 0);
        b->slots.push_back({COMPACT_CONNECT, 0, std::move(p)});
    } else {
        Core core;
        core.ftype = SESSION;
        Control control{core, {{T_SESS, body}}};
        const auto ctl = control.pack(cpb);
        for (std::size_t i = 0; i < ctl.size(); ++i) b->slots.push_back({ctl_mask(direction, static_cast<int>(i), key), 0, ctl[i]});
    }
    const std::string retry = (body[0] == CONNECT || body[0] == DISC) ? " try " + std::to_string(tries + 1) : "";
    const std::string desc = frame_desc(body);
    log_write(LOG, INFO, format("TX %s %s x%d%s", desc.c_str(), m.c_str(), static_cast<int>(b->slots.size()), retry.c_str()));
    return b;
}

bool Session::compact(const std::string& mode) {
    // in its Control envelope it needs more codewords than the mode's control may have (CPM: one of 20 B)
    const int cpb = policy->ctl_payload_bytes(mode);
    return (CONNECT_CTL_BYTES + cpb - 1) / cpb > policy->max_ctl(mode);
}

TxBurstPtr Session::connect_burst() {
    const Bytes me = pack_call(call), them = pack_call(peer), n = u16(nonce);
    const Bytes tail{byte(cap), tenths(t_turn)};
    const Bytes head{CONNECT, VERSION};
    return session_burst(0, 0, cat({head, me, them, n, tail}));
}

TxBurstPtr Session::disc_burst() { return session_burst(station->direction, station->key, Bytes{DISC}); }

TxBurstPtr Session::accept_burst() {
    const Bytes head{CONNECT_ACK}, n = u16(nonce), tail{byte(cap), tenths(t_turn)};
    return session_burst(1, 0, cat({head, n, tail}), answer_mode);
}

std::optional<Session::Frame> Session::session_frame(RxBurst& rx) {
    std::vector<std::pair<int, int>> keys;  // (direction, key)
    if (station) {
        keys.emplace_back(station->peer(), station->key);
        if (!station->master) keys.emplace_back(0, 0);
    } else if (state == SessionState::LISTEN) {
        keys.emplace_back(0, 0);
    } else if (state == SessionState::CONNECTING) {
        keys.emplace_back(1, 0);
    }
    if (std::find(keys.begin(), keys.end(), std::pair{0, 0}) != keys.end() && compact(rx.submode())) {
        if (auto p = rx.decode(0, COMPACT_CONNECT, 0, nullptr)) {
            Bytes body = unpack_connect(*p);
            if (auto why = check_frame(body)) {
                log_write(LOG, WARNING, "RX malformed " + *why + ": dropped");
                return std::nullopt;
            }
            return Frame{0, std::move(body), rx.submode()};
        }
    }
    for (auto [direction, key] : keys) {
        auto first = rx.decode(0, ctl_mask(direction, 0, key), 0, nullptr);
        if (!first) continue;
        const Core core = Core::unpack(*first);
        if (core.ftype != SESSION) return std::nullopt;
        if (core.n_ctl > rx.n_cw()) {
            log_write(LOG, WARNING, format("RX malformed session frame (%d control codewords in a burst of %d): dropped",
                                           core.n_ctl, rx.n_cw()));
            return std::nullopt;
        }
        std::vector<Bytes> payloads{*first};
        for (int i = 1; i < core.n_ctl; ++i) {
            auto p = rx.decode(i, ctl_mask(direction, i, key), 0, nullptr);
            if (!p) return std::nullopt;
            payloads.push_back(std::move(*p));
        }
        Bytes body;
        try {
            const auto c = Control::unpack(payloads);
            if (auto it = c.ext.find(T_SESS); it != c.ext.end()) body = it->second;
        } catch (const std::invalid_argument& e) {
            log_write(LOG, WARNING, std::string("RX malformed session frame (") + e.what() + "): dropped");
            return std::nullopt;
        }
        if (body.empty()) return std::nullopt;
        if (auto why = check_frame(body)) {
            log_write(LOG, WARNING, "RX malformed " + *why + ": dropped");
            return std::nullopt;
        }
        return Frame{key, std::move(body), rx.submode()};
    }
    return std::nullopt;
}

void Session::on_session_frame(const Frame& f, double now) {
    const Bytes& body = f.body;
    const int sub = body[0];
    log_write(LOG, INFO, "RX " + frame_desc(body) + " " + f.mode);
    if (sub == CONNECT && body.size() >= 22 && f.key == 0) {
        const std::string caller = unpack_call(ByteView(body).subspan(2, 8)), callee = unpack_call(ByteView(body).subspan(10, 8));
        const std::int64_t n = be(body, 18, 20);
        std::string me = call;
        for (auto& c : me)
            if (c >= 'a' && c <= 'z') c = static_cast<char>(c - 'a' + 'A');
        if (callee != me && std::find(aliases.begin(), aliases.end(), callee) == aliases.end()) return;
        if (station && n == nonce) {
            answer_mode = f.mode;
            queue(accept_burst(), now);  // our ACK was lost: say it again
            return;
        }
        if (station && !master() && caller == peer && live(state)) {
            // my caller dialed again: it gave up on our session, so take the new one (session.py)
            close("peer reconnected", nullptr, now);
            state = SessionState::LISTEN;
            want_disc = sent_disc_ack = false;
            tries = 0;
        }
        if (state != SessionState::LISTEN) return;
        if (body[1] != VERSION) {
            const Bytes nak{CONNECT_NAK, body[18], body[19], 1};
            queue(session_burst(1, 0, nak, f.mode), now);
            return;
        }
        call = callee;  // the call it dialed (the session key derives from it)
        peer = caller;
        nonce = n;
        cap = std::min<int>(body[20], cap);
        station = make_station(1, false, session_key(caller, call, n));
        flush_writes();
        connected(now);
        events.push_back("CONNECTED " + caller);
        answer_mode = f.mode;
        queue(accept_burst(), now);
    } else if (sub == CONNECT_ACK && state == SessionState::CONNECTING && f.key == 0) {
        if (be(body, 1, 3) != nonce) return;
        cap = at(body, 3);
        station = make_station(0, true, session_key(call, peer, nonce));
        flush_writes();
        connected(now);
        events.push_back("CONNECTED " + peer);
        queue(station->build(), now);
    } else if (sub == CONNECT_NAK && state == SessionState::CONNECTING && f.key == 0) {
        if (be(body, 1, 3) == nonce) close("refused");
    } else if (sub == DISC && station && f.key == station->key) {
        // also when already closed: our DISC_ACK may have been lost
        heard(now);
        auto final_burst = session_burst(station->direction, f.key, Bytes{DISC_ACK}, f.mode);
        close("disconnected by peer", std::move(final_burst), now);
    } else if (sub == DISC_ACK && state == SessionState::DISCONNECTING && f.key == station->key) {
        close("disconnected");
    }
}

void Session::flush_writes() {
    if (!pending_write.empty()) {
        station->write(pending_write);
        pending_write.clear();
    }
}

}  // namespace data2g::arq
