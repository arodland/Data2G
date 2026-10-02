#include "arq/engine.hpp"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>

#include "codes/codes.hpp"

namespace data2g::arq {

namespace {

constexpr const char* LOG = "data2g.engine";
constexpr int INFO = 20, WARNING = 30;

std::string upper(std::string s) {
    for (auto& c : s) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    return s;
}

std::string spec_name(const tnc::Pending& p) {
    return std::string(p.is_cpm() ? p.cpm().spec->name : p.ofdm().spec->name);
}

Heard heard_of(tnc::Rx&& rx) {
    Heard h;
    if (auto* o = std::get_if<modem::Received>(&rx)) h.ofdm = std::make_shared<const modem::Received>(std::move(*o));
    else h.cpm = std::make_shared<const cpm::Received>(std::move(std::get<cpm::Received>(rx)));
    return h;
}

std::vector<std::string_view> all_grids() {
    std::vector<std::string_view> out;
    for (const auto& g : tables::CPM_GRIDS) out.push_back(g.name);
    return out;
}

void put16(std::vector<std::uint8_t>& b, unsigned v) {
    b.push_back(static_cast<std::uint8_t>(v));
    b.push_back(static_cast<std::uint8_t>(v >> 8));
}
void put32(std::vector<std::uint8_t>& b, std::uint32_t v) {
    put16(b, v & 0xFFFF);
    put16(b, v >> 16);
}

std::string json_obj(const std::vector<std::pair<std::string, std::string>>& fields) {
    std::string s = "{";
    for (std::size_t i = 0; i < fields.size(); ++i) {
        if (i) s += ", ";
        s += json_str(fields[i].first) + ": " + fields[i].second;
    }
    return s + "}";
}

std::string hex(const Bytes& b) {
    static const char* d = "0123456789abcdef";
    std::string s;
    for (auto c : b) {
        s += d[c >> 4];
        s += d[c & 15];
    }
    return s;
}

}  // namespace

// --- recording -------------------------------------------------------------------------

std::string json_str(std::string_view s) {
    std::string out = "\"";
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') {
            out += '\\';
            out += static_cast<char>(c);
        } else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c == '\t') out += "\\t";
        else if (c < 0x20 || c >= 0x7F) {  // json.dumps' ensure_ascii (callsigns and modes are ASCII)
            char buf[8];
            std::snprintf(buf, sizeof buf, "\\u%04x", c);
            out += buf;
        } else out += static_cast<char>(c);
    }
    return out + "\"";
}

std::string json_num(double v) {
    if (std::isnan(v)) return "NaN";
    if (std::isinf(v)) return v > 0 ? "Infinity" : "-Infinity";
    char buf[64];
    const double a = std::fabs(v);
    // float.__repr__: shortest round trip, positional for 1e-4 <= |v| < 1e16
    const bool fixed = a == 0 || (a >= 1e-4 && a < 1e16);
    auto r = std::to_chars(buf, buf + sizeof buf, v, fixed ? std::chars_format::fixed : std::chars_format::scientific);
    std::string s(buf, r.ptr);
    if (fixed && s.find('.') == std::string::npos) s += ".0";
    return s;
}

std::string json_measured(const Measured& m) {
    std::vector<std::pair<std::string, std::string>> f = {
        {"snr_est", json_num(m.snr_est)}, {"spread_est", json_num(m.spread_est)},
        {"delay_est_ms", json_num(m.delay_est_ms)}, {"headroom", json_num(m.headroom)}, {"frames", json_num(m.frames)}};
    for (std::size_t i = 0; i < CONSTS.size(); ++i) f.emplace_back("mi_" + std::string(CONSTS[i]), json_num(m.mi[i]));
    return json_obj(f);
}

std::uint16_t to_half(double x) {
    const std::uint16_t sign = std::signbit(x) ? 0x8000 : 0;
    const double a = std::fabs(x);
    if (std::isnan(a)) return sign | 0x7E00;
    if (a >= 65520.0) return sign | 0x7C00;  // rounds to inf
    if (a < 6.103515625e-05)                 // subnormal: units of 2^-24 (1024 rounds up to the least normal)
        return static_cast<std::uint16_t>(sign | static_cast<int>(std::nearbyint(a * 16777216.0)));
    int e;
    std::frexp(a, &e);  // a = f 2^e, f in [0.5, 1)
    int m = static_cast<int>(std::nearbyint(std::ldexp(a, 11 - e)));  // [1024, 2048], half to even
    int E = e - 1 + 15;
    if (m == 2048) {
        m = 1024;
        ++E;
    }
    if (E >= 31) return sign | 0x7C00;
    return static_cast<std::uint16_t>(sign | E << 10 | (m - 1024));
}

std::vector<std::uint8_t> npz_f32(const std::string& name, std::span<const float> a) {
    std::string hdr = "{'descr': '<f4', 'fortran_order': False, 'shape': (" + std::to_string(a.size()) + ",), }";
    hdr.append(64 - (10 + hdr.size() + 1) % 64, ' ');
    hdr += '\n';
    std::vector<std::uint8_t> npy = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0};
    put16(npy, static_cast<unsigned>(hdr.size()));
    npy.insert(npy.end(), hdr.begin(), hdr.end());
    for (float v : a) {
        std::uint32_t u;
        std::memcpy(&u, &v, 4);
        put32(npy, u);
    }
    // a zip of one stored member, as np.load reads it
    const std::string member = name + ".npy";
    const std::uint32_t crc = codes::crc32(npy);
    const auto size = static_cast<std::uint32_t>(npy.size());
    auto header = [&](std::vector<std::uint8_t>& b, bool central) {
        put32(b, central ? 0x02014B50 : 0x04034B50);
        if (central) put16(b, 20);  // made by
        put16(b, 20);               // needed
        put16(b, 0);                // flags
        put16(b, 0);                // stored
        put16(b, 0);                // time
        put16(b, 0x21);             // 1980-01-01
        put32(b, crc);
        put32(b, size);
        put32(b, size);
        put16(b, static_cast<unsigned>(member.size()));
        put16(b, 0);  // extra
        if (central) {
            put16(b, 0);  // comment
            put16(b, 0);  // disk
            put16(b, 0);  // internal attributes
            put32(b, 0);  // external
            put32(b, 0);  // the local header's offset
        }
        b.insert(b.end(), member.begin(), member.end());
    };
    std::vector<std::uint8_t> out;
    header(out, false);
    out.insert(out.end(), npy.begin(), npy.end());
    const auto cd = static_cast<std::uint32_t>(out.size());
    header(out, true);
    const auto cd_size = static_cast<std::uint32_t>(out.size()) - cd;
    put32(out, 0x06054B50);
    put16(out, 0);
    put16(out, 0);
    put16(out, 1);
    put16(out, 1);
    put32(out, cd_size);
    put32(out, cd);
    put16(out, 0);
    return out;
}

Recorder::Recorder(const std::string& dir, const std::string& call) : dir_(dir) {
    std::filesystem::create_directories(dir_);
    log_.open(dir_ + "/events.jsonl", std::ios::app);
    audio_.open(dir_ + "/audio_in.f16", std::ios::app | std::ios::binary);
    if (!log_ || !audio_) throw std::runtime_error("cannot record to " + dir_);
    const double wall = std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
    event("start", {{"call", json_str(call)}, {"wall", json_num(wall)}, {"fs", std::to_string(config::FS)}});
}

void Recorder::audio(std::span<const double> x) {
    std::vector<std::uint16_t> h(x.size());
    std::transform(x.begin(), x.end(), h.begin(), to_half);  // little-endian hosts, as numpy writes it
    audio_.write(reinterpret_cast<const char*>(h.data()), static_cast<std::streamsize>(h.size() * 2));
    audio_.flush();  // whole on disk at every block: readable while live, nothing lost on a crash
}

void Recorder::event(std::string_view kind, const std::vector<std::pair<std::string, std::string>>& fields) {
    std::vector<std::pair<std::string, std::string>> f = {{"kind", json_str(kind)}};
    f.insert(f.end(), fields.begin(), fields.end());
    log_ << json_obj(f) << '\n' << std::flush;  // line-buffered, as Python opens it
}

std::string Recorder::rx(double t, std::span<const double> audio, const tnc::Pending& header, bool lost,
                         const std::optional<Measured>& meas) {
    char name[32];
    std::snprintf(name, sizeof name, "rx_%05d.npz", n_++);
    const std::vector<float> a(audio.begin(), audio.end());
    const auto bytes = npz_f32("audio", a);
    std::ofstream(dir_ + "/" + name, std::ios::binary).write(reinterpret_cast<const char*>(bytes.data()),
                                                              static_cast<std::streamsize>(bytes.size()));
    event("rx", {{"t", json_num(t)},
                 {"file", json_str(name)},
                 {"submode", json_str(spec_name(header))},
                 {"n_cw", std::to_string(header.n_cw())},
                 {"score", json_num(header.score())},
                 {"lost", lost ? "true" : "false"},
                 {"meas", meas ? json_measured(*meas) : "null"}});
    return name;
}

// --- engine ----------------------------------------------------------------------------

Engine::Engine(std::string call, EngineConfig cfg, EngineHooks hooks)
    : call_(upper(std::move(call))),
      cfg_(std::move(cfg)),
      hooks_(std::move(hooks)),
      rng_(hooks_.rng ? hooks_.rng
                      : std::make_shared<DefaultRng>(cfg_.seed ? *cfg_.seed : std::random_device{}())),
      accept_(modem::Accept::of({}, MAX_BURST_S, cfg_.min_header_score)),
      receiver_(accept_, all_grids()),
      ptt_delay_(static_cast<std::int64_t>(cfg_.ptt_delay_s * config::FS)) {
    if (!cfg_.record_dir.empty()) rec_.emplace(cfg_.record_dir, call_);
    new_session();
    if (cfg_.worker) worker_ = std::thread(&Engine::work, this);
}

Engine::~Engine() { stop(); }

void Engine::stop() {
    {
        std::lock_guard lock(mu_);
        stop_ = true;
    }
    wake_.notify_all();
    done_cv_.notify_all();
    if (worker_.joinable()) worker_.join();
}

bool Engine::idle() const {
    const auto s = session_->state;
    return s == SessionState::IDLE || s == SessionState::LISTEN || s == SessionState::CLOSED;
}

void Engine::new_session() {
    auto policy = hooks_.policy ? hooks_.policy() : std::make_shared<GearPolicy>();
    const double seed = rng_->uniform(0.0, 1.0);  // random.Random(self.rng.random())
    std::shared_ptr<Rng> rng;
    if (hooks_.session_rng) rng = hooks_.session_rng(seed);
    else {
        std::uint64_t bits;
        std::memcpy(&bits, &seed, sizeof bits);
        rng = std::make_shared<DefaultRng>(bits);
    }
    session_ = hooks_.session ? hooks_.session(call_, std::move(policy), std::move(rng), aliases_, cfg_.stats_interval_s)
                              : std::make_shared<Session>(call_, std::move(policy), 1.0, std::move(rng), aliases_,
                                                          cfg_.stats_interval_s);
    session_->set_chat(chat_);
    store_.clear();
}

void Engine::listen(bool on) {
    if (idle()) {
        new_session();
        if (on) session_->listen();
    }
}

void Engine::set_call(const std::string& call, const std::vector<std::string>& aliases) {
    call_ = upper(call);
    aliases_.clear();
    for (const auto& a : aliases) aliases_.push_back(upper(a));
    if (idle()) listen(session_->state == SessionState::LISTEN);
}

void Engine::abort() {
    tx_.reset();
    transmitting_ = false;
    request_reset();
    new_session();
}

void Engine::connect(const std::string& peer, int cap) {
    if (!idle()) throw std::runtime_error(std::string("session ") + state_name(session_->state));
    new_session();
    session_->connect(peer, cap, now());
}

void Engine::set_chat(bool on) {
    chat_ = on;
    session_->set_chat(on);
}

std::vector<std::string> Engine::events() {
    std::vector<std::string> ev = std::move(events_);
    events_.clear();
    ev.insert(ev.end(), session_->events.begin(), session_->events.end());
    session_->events.clear();
    return ev;
}

void Engine::send_cq(const std::string& call, int cap) {
    if (!idle()) throw std::runtime_error(std::string("session ") + state_name(session_->state));
    const std::string mode = session_->policy->connect_mode(cap, 0);
    Bytes body = pack_call(call);
    body.push_back(static_cast<std::uint8_t>(cap));
    Control ctl{Core{.ftype = SESSION}, {{T_CQ, body}}};
    const auto payloads = ctl.pack(session_->policy->payload_bytes(mode));
    auto b = std::make_shared<TxBurst>();
    b->submode = mode;
    for (std::size_t i = 0; i < payloads.size(); ++i) b->slots.push_back({ctl_mask(0, static_cast<int>(i), 0), 0, payloads[i]});
    extra_.push_back(std::move(b));
}

void Engine::post(std::function<void()> f) {
    {
        std::lock_guard lock(mu_);
        posted_.push_back(std::move(f));
    }
    wake_.notify_one();
}

void Engine::run_posted() {
    std::deque<std::function<void()>> todo;
    {
        std::lock_guard lock(mu_);
        todo.swap(posted_);
    }
    for (auto& f : todo) f();
}

// --- the receiver stage ----------------------------------------------------------------

void Engine::apply_reset() {
    receiver_reset();
    gen_ = want_gen_;
    busy_now_ = receiver_busy();
    channel_busy_now_ = receiver_channel_busy();
}

void Engine::request_reset() {
    ++want_gen_;
    if (!cfg_.worker) apply_reset();  // one thread: at once, as Python does
}

Engine::Out Engine::step(std::span<const double> x) {
    if (gen_ != want_gen_) apply_reset();
    Block b;
    b.x.assign(x.begin(), x.end());
    b.gen = gen_;
    b.seq = seq_++;
    if (!transmitting_) b.items = receiver_feed(x);  // our own transmission is not heard
    b.busy = receiver_busy();
    busy_now_ = b.busy;
    channel_busy_now_ = receiver_channel_busy();
    if (!cfg_.worker) {
        run_posted();
        Done d = process(b);
        if (after_block_) after_block_(d.out.ptt);
        return std::move(d.out);
    }
    const bool slow = std::any_of(b.items.begin(), b.items.end(),
                                  [](const auto& it) { return !std::holds_alternative<tnc::HeaderEvent>(it); });
    const std::uint64_t seq = b.seq;
    std::unique_lock lock(mu_);
    const bool lagging = slow_ > 0;  // a burst ahead is being received: don't wait for it
    if (slow) ++slow_;
    in_.push_back(std::move(b));
    wake_.notify_one();
    if (!lagging && !slow) done_cv_.wait(lock, [&] { return n_done_ > seq || stop_; });
    // catching up: silence the session stage made while it lagged is dropped
    while (out_.size() > 1 && !out_.front().sound) out_.pop_front();
    if (out_.empty()) return {std::vector<double>(x.size(), 0.0), false};
    Out o = std::move(out_.front().out);
    out_.pop_front();
    return o;
}

void Engine::work() {
    std::unique_lock lock(mu_);
    while (true) {
        wake_.wait(lock, [&] { return stop_ || !in_.empty() || !posted_.empty(); });
        if (stop_) return;
        if (in_.empty()) {
            lock.unlock();
            run_posted();
            lock.lock();
            continue;
        }
        Block b = std::move(in_.front());
        in_.pop_front();
        const bool slow = std::any_of(b.items.begin(), b.items.end(),
                                      [](const auto& it) { return !std::holds_alternative<tnc::HeaderEvent>(it); });
        lock.unlock();
        run_posted();
        Done d = process(b);
        if (after_block_) after_block_(d.out.ptt);
        lock.lock();
        if (slow) --slow_;
        out_.push_back(std::move(d));
        ++n_done_;
        done_cv_.notify_all();
    }
}

// --- the session stage -----------------------------------------------------------------

Engine::Done Engine::process(Block& b) {
    const auto k = static_cast<std::int64_t>(b.x.size());
    const double t = now() + static_cast<double>(k) / config::FS;  // the block's end: when anything in it is known
    Done d;
    d.out.audio.assign(static_cast<std::size_t>(k), 0.0);
    if (rec_) rec_->audio(tx_ ? d.out.audio : b.x);
    if (!tx_) {
        const bool fresh = b.gen == want_gen_;  // else fed before a reset this stage asked for
        if (fresh) hear(b.items, t);
        const bool busy = fresh && b.busy;
        TxBurstPtr burst;
        if (!busy) {  // a burst still arriving holds any reply (half duplex)
            burst = session_->poll(t);
            if (!burst && !extra_.empty()) {
                burst = extra_.front();
                extra_.pop_front();
            }
        }
        if (!burst && cfg_.kiss && idle()) burst = kiss_burst(k, busy);  // KISS only between ARQ sessions
        else kiss_busy_ = 0;
        if (burst) start_tx(burst, t);
    }
    if (tx_) {
        const auto n = std::min<std::size_t>(static_cast<std::size_t>(k), tx_->audio.size() - tx_->pos);
        std::copy_n(tx_->audio.begin() + static_cast<std::ptrdiff_t>(tx_->pos), n, d.out.audio.begin());
        tx_->pos += n;
        d.sound = true;
        if (tx_->pos >= tx_->audio.size()) {
            tx_.reset();
            request_reset();  // our own transmission was not heard
            session_->on_tx_end(now() + static_cast<double>(n) / config::FS);
        }
    }
    n_ += k;
    d.out.ptt = tx_.has_value();
    transmitting_ = d.out.ptt;
    return d;
}

void Engine::hear(std::vector<tnc::Receiver::Item>& items, double t) {
    for (auto& it : items) {
        if (auto* h = std::get_if<tnc::HeaderEvent>(&it)) {
            session_->on_header(spec_name(h->header), h->header.n_cw(), t);
            continue;
        }
        tnc::BurstEvent ev = std::holds_alternative<tnc::DecodeRequest>(it)
                                 ? tnc::decode(std::move(std::get<tnc::DecodeRequest>(it)), accept_)
                                 : std::move(std::get<tnc::BurstEvent>(it));
        hear_burst(ev, t);
    }
}

void Engine::hear_burst(tnc::BurstEvent& ev, double t) {
    std::optional<Heard> r;
    std::optional<Measured> meas;
    if (ev.rx) {
        r = heard_of(std::move(*ev.rx));
        meas = measure(*r);
    }
    if (rec_) rec_->rx(t, ev.audio, ev.header, !r, meas);
    if (on_burst_)
        on_burst_({t, spec_name(ev.header), ev.header.n_cw(), !r, meas ? std::optional(meas->snr_est) : std::nullopt});
    if (!r) {
        if (log_enabled(LOG, INFO))
            log_write(LOG, INFO, format("RX %s x%d: header heard (score %.2f), burst lost", spec_name(ev.header).c_str(),
                                        ev.header.n_cw(), ev.header.score()));
        return;
    }
    auto soft = soft_bits(*r);
    ModemRx rx(*r, &store_, cfg_.dd_budget_s, soft, cfg_.dd);
    // in a session, its peer's bursts are the likely ones: a control
    // codeword under the session's key claims the burst before KISS tries
    // its keys on it
    const auto st = session_->station;
    const bool ours = st && (session_->state == SessionState::CONNECTED || session_->state == SessionState::DISCONNECTING) &&
                      rx.decode(0, ctl_mask(st->peer(), 0, st->key), 0, nullptr).has_value();
    if (cfg_.kiss && !ours) {
        if (auto frames = cfg_.kiss->on_burst(*r, soft, cfg_.dd_budget_s)) {  // a KISS burst: not the session's
            kiss_rx_.insert(kiss_rx_.end(), std::make_move_iterator(frames->begin()), std::make_move_iterator(frames->end()));
            return;
        }
    }
    if (cq(rx)) return;
    session_->policy->observe(*meas, rx.submode(), t);
    session_->on_rx(rx, t);
}

// The next KISS burst if it may go now: after waiting on BUSY, p-persistent
// slots; on a free channel at once; BUSY holds it, but not past
// busy_limit_s of it unbroken.
TxBurstPtr Engine::kiss_burst(std::int64_t k, bool busy) {
    auto& link = *cfg_.kiss;
    if (link.queue.empty()) {
        kiss_busy_ = 0;
        kiss_deferred_ = false;
        return nullptr;
    }
    if (busy) {
        kiss_busy_ += k;
        kiss_deferred_ = true;
        if (static_cast<double>(kiss_busy_) < link.busy_limit_s * config::FS) return nullptr;
        log_write(LOG, WARNING, format("KISS: BUSY for %.0f s, sending anyway", static_cast<double>(kiss_busy_) / config::FS));
    } else {
        kiss_busy_ = 0;
        if (kiss_deferred_) {
            if (n_ < kiss_slot_) return nullptr;
            if (rng_->uniform(0.0, 1.0) >= (link.persist + 1) / 256.0) {
                kiss_slot_ = n_ + static_cast<std::int64_t>(link.slot_s * config::FS);
                return nullptr;
            }
        }
    }
    kiss_busy_ = 0;
    kiss_deferred_ = false;
    return link.next_burst();
}

bool Engine::cq(ModemRx& rx) {
    const auto first = rx.decode(0, ctl_mask(0, 0, 0), 0, nullptr);
    if (!first) return false;
    const Core core = Core::unpack(*first);
    if (core.ftype != SESSION) return false;
    std::vector<Bytes> payloads = {*first};
    for (int i = 1; i < core.n_ctl; ++i) {
        auto p = rx.decode(i, ctl_mask(0, i, 0), 0, nullptr);
        if (!p) return false;
        payloads.push_back(std::move(*p));
    }
    Bytes body;
    try {
        const auto ext = Control::unpack(payloads).ext;
        const auto it = ext.find(T_CQ);
        if (it != ext.end()) body = it->second;
    } catch (const std::invalid_argument&) {
        return false;
    }
    if (body.size() < 9) return false;
    events_.push_back("CQFRAME " + unpack_call(ByteView(body).first(8)) + " " + std::to_string(body[8]));
    return true;
}

void Engine::start_tx(const TxBurstPtr& burst, double t) {
    const auto x = tx_audio(*burst);
    // peak at full scale: the modem's unit-RMS audio peaks at 2-3, and a sound card clips at 1
    double peak = 0.0;
    for (double v : x) peak = std::max(peak, std::fabs(v));
    Tx tx{burst, std::vector<double>(static_cast<std::size_t>(ptt_delay_), 0.0), 0};
    for (double v : x) tx.audio.push_back(v / peak);
    if (rec_) {
        std::string slots = "[";
        for (std::size_t i = 0; i < burst->slots.size(); ++i) {
            const auto& s = burst->slots[i];
            if (i) slots += ", ";
            slots += json_obj({{"mask", "[" + std::to_string(s.mask_id.key) + ", " + std::to_string(s.mask_id.direction) + ", " +
                                            std::to_string(s.mask_id.seq) + "]"},
                               {"rv", std::to_string(s.rv)},
                               {"payload", json_str(hex(s.payload))}});
        }
        rec_->event("tx", {{"t", json_num(t)},
                           {"submode", json_str(burst->submode)},
                           {"burst_seq", std::to_string(burst->burst_seq)},
                           {"slots", slots + "]"},
                           {"seconds", json_num(static_cast<double>(tx.audio.size()) / config::FS)}});
    }
    tx_ = std::move(tx);
}

}  // namespace data2g::arq
