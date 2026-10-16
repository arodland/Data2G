#include "station.hpp"

#include <QHostAddress>
#include <QHostInfo>
#include <QTcpSocket>

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <ctime>
#include <functional>
#include <utility>

#include "arq/engine.hpp"
#ifdef __linux__
#include <pthread.h>
#endif
#include "app/realtime.hpp"
#include "app/tcp_servers.hpp"
#include "app/transmitter.hpp"
#include "audio/audio.hpp"
#include "audio/fifo.hpp"
#include "audio/pipe.hpp"
#include "audio/thread.hpp"
#include "generated/config.hpp"
#include "host/host.hpp"
#include "kisslink/kisslink.hpp"
#include "monitor/monitor.hpp"
#include "rig/controller.hpp"
#include "rig/keyer.hpp"

#ifdef DATA2G_HAVE_AUDIO
#include "audio/card.hpp"
#endif
#ifdef DATA2G_HAVE_RIG
#include "rig/hamlib/hamlib.hpp"
#endif

namespace data2g::app {

#ifdef DATA2G_HAVE_AUDIO
struct Station::SoundCard {
    std::unique_ptr<audio::card::Capture> cap;
    std::unique_ptr<audio::card::Playback> play;
};
#else
struct Station::SoundCard {};
#endif

namespace {

QHostAddress resolve(const std::string& name) {
    QHostAddress a;
    if (a.setAddress(QString::fromStdString(name))) return a;
    const QHostInfo info = QHostInfo::fromName(QString::fromStdString(name));
    for (const QHostAddress& x : info.addresses())
        if (x.protocol() == QAbstractSocket::IPv4Protocol) return x;
    if (!info.addresses().isEmpty()) return info.addresses().first();
    throw std::runtime_error("can't resolve " + name);
}

constexpr std::size_t MAX_BURSTS = 256;  // kept for a front end that polls slowly (or never: data2g-host)
constexpr std::size_t TAP_SAMPLES = 4096;

}  // namespace

// The monitor window's decoder: bursts from the session stage, dumped on a
// thread of its own so its decodes never delay the session's.
struct Station::MonitorThread {
    monitor::Monitor m;
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::pair<arq::BurstHeard, std::string>> in;
    std::deque<monitor::Dump> out;
    bool stop = false;
    std::thread thread{[this] { run(); }};  // last: everything above is ready

    ~MonitorThread() {
        {
            std::lock_guard lock(mu);
            stop = true;
        }
        cv.notify_one();
        thread.join();
    }
    void push(const arq::BurstHeard& b) {
        const auto now = std::chrono::system_clock::now();
        const std::time_t t = std::chrono::system_clock::to_time_t(now);
        const std::tm tm = local_tm(t);
        char when[16];
        std::strftime(when, sizeof when, "%H:%M:%S", &tm);
        std::lock_guard lock(mu);
        if (in.size() >= MAX_BURSTS) return;  // ponytail: a decoder this far behind loses bursts (shown as gaps)
        in.emplace_back(b, when);
        cv.notify_one();
    }
    void run() {
        std::unique_lock lock(mu);
        while (true) {
            cv.wait(lock, [this] { return stop || !in.empty(); });
            if (stop) return;
            auto [b, when] = std::move(in.front());
            in.pop_front();
            lock.unlock();
            monitor::Dump d = m.burst(b, when);
            lock.lock();
            out.push_back(std::move(d));
            if (out.size() > MAX_BURSTS) out.pop_front();
        }
    }
};

// What the session stage hands the owner after a command or a block.
struct Station::Outbox {
    std::vector<std::pair<std::string, std::uint64_t>> cmd;  // (line, client it is for; 0: all)
    arq::Bytes data;
    std::vector<std::pair<int, arq::Bytes>> kiss;  // (port, frame) heard
    std::vector<std::pair<int, std::int64_t>> acks;  // (port, ACKMODE id) gone out
    unsigned dial = 0;     // data goes to ACCEPT's connection of this dial generation (0: the data port)
    unsigned hang_up = 0;  // then that connection closes: its session is over
};

// ACCEPT: the session stage waits for the owner's thread to connect, at most
// host::DIAL_TIMEOUT_S; a connection made after it gave up is closed.
struct Station::Dial {
    std::mutex mu;
    std::condition_variable cv;
    std::optional<bool> up;
    bool abandoned = false;
};

// --- the station -------------------------------------------------------------------------

Station::Station(Args a) : a_(std::move(a)), rate_(config::FS), tap_(TAP_SAMPLES, 0.0) {}

Station::~Station() { stop(); }

// `asker`: the client whose command produced what the host has queued. That is its reply and goes to it alone,
// except DISCONNECTED (ABORT's), which is news for every client. 0: everything is for all.
void Station::flush(std::uint64_t asker) {
    Outbox o;
    for (auto& line : host_->out_cmd) {
        const auto to = line == "DISCONNECTED" ? 0 : asker;
        o.cmd.emplace_back(std::move(line), to);
    }
    host_->out_cmd.clear();
    o.data.swap(host_->out_data);
    o.kiss.swap(engine_->kiss_rx());
    o.acks.swap(link_->acks);
    o.dial = dial_live_;
    if (const auto st = engine_->session().state;
        dial_live_ && st != arq::SessionState::CONNECTED && st != arq::SessionState::DISCONNECTING)
        o.hang_up = std::exchange(dial_live_, 0u);
    if (o.cmd.empty() && o.data.empty() && o.kiss.empty() && o.acks.empty() && !o.hang_up) return;
    QMetaObject::invokeMethod(&ctx_, [this, o = std::move(o)] { deliver(o); }, Qt::QueuedConnection);
}

void Station::note_link() {
    auto& s = engine_->session();
    LinkStatus l;
    l.state = arq::state_name(s.state);
    l.peer = s.peer;
    l.cap = s.cap;
    if (s.station) {
        const auto st = s.stats();
        l.tx_bytes = st.at("tx_bytes");
        l.rx_bytes = st.at("rx_bytes");
    }
    if (const auto& tx = engine_->tx()) mode_ = tx->burst->submode;
    l.mode = mode_;
    std::lock_guard lock(status_mu_);
    link_status_ = std::move(l);
}

void Station::deliver(const Outbox& o) {
    if (cmd_)
        for (const auto& [line, to] : o.cmd) cmd_->send(QByteArray::fromStdString(line + "\r"), to);
    const QByteArray data(reinterpret_cast<const char*>(o.data.data()), static_cast<qsizetype>(o.data.size()));
    if (o.dial) {
        if (dialed_ && dialed_gen_ == o.dial && !data.isEmpty()) dialed_->write(data);
    } else if (data_ && !data.isEmpty()) {
        data_->send(data);
    }
    if (o.hang_up && dialed_gen_ == o.hang_up) hang_up();
    if (kiss_) {
        for (const auto& [port, f] : o.kiss) kiss_->broadcast(f, port);
        for (const auto& [port, id] : o.acks) kiss_->send_ack(id, port);
    }
}

void Station::hang_up() {
    if (!dialed_) return;
    dialed_->disconnect(&ctx_);
    dialed_->disconnectFromHost();  // after what is written
    dialed_->deleteLater();
    dialed_ = nullptr;
}

void Station::dial(const std::string& host, int port, unsigned gen, std::shared_ptr<Dial> d) {
    hang_up();  // a session before (a peer that reconnected)
    auto* s = new QTcpSocket(&ctx_);
    dialed_ = s;
    dialed_gen_ = gen;
    auto settle = [this, s, d](bool up) {
        std::unique_lock lock(d->mu);
        if (d->up) return;
        if (d->abandoned) {
            lock.unlock();
            if (up && dialed_ == s) hang_up();
            return;
        }
        d->up = up;
        d->cv.notify_all();
    };
    QObject::connect(s, &QTcpSocket::connected, &ctx_, [settle] { settle(true); });
    QObject::connect(s, &QTcpSocket::errorOccurred, &ctx_, [this, s, settle](QAbstractSocket::SocketError) {
        logf(WARNING, "ACCEPT: %s", s->errorString().toStdString().c_str());
        settle(false);
    });
    QObject::connect(s, &QTcpSocket::readyRead, &ctx_, [this, s, gen] {
        const QByteArray d = s->readAll();
        engine_->post([this, gen, b = arq::Bytes(d.begin(), d.end())] {
            if (dial_live_ == gen) host_->data_in(b);
            flush();
        });
    });
    QObject::connect(s, &QTcpSocket::disconnected, &ctx_, [this, s, gen] {
        if (dialed_ == s) hang_up();
        engine_->post([this, gen] {
            if (dial_live_ != gen) return;
            logf(INFO, "ACCEPT: the far end closed: disconnecting");
            engine_->session().disconnect();
            flush();
        });
    });
    s->connectToHost(QString::fromStdString(host), static_cast<quint16>(port));
}

void Station::watch_counters() {
    if (const auto n = cap_->overflows(); n != overflows_) {
        overflows_ = n;
        logf(WARNING, "RX audio overflow (%llu)", static_cast<unsigned long long>(n));
    }
    if (const auto n = play_->underruns(); n != underruns_) {
        underruns_ = n;
        logf(WARNING, "TX audio underrun (%llu)", static_cast<unsigned long long>(n));
    }
    if (const auto n = cap_->late_events(); n != late_) {
        late_ = n;
        logf(WARNING, "RX audio %.1f s behind the card", cap_->backlog_s());
    }
    if (const auto n = cap_->dropped(); n != dropped_) {
        logf(WARNING, "RX audio: %llu samples dropped, the capture FIFO full", static_cast<unsigned long long>(n - dropped_));
        dropped_ = n;
    }
}

void Station::tap(std::span<const double> x, bool tx) {
    std::lock_guard lock(tap_mu_);
    for (double v : x) tap_[tapped_++ % tap_.size()] = v;
    tap_tx_ = tx;
}

std::vector<double> Station::input_tail(std::size_t n, std::uint64_t* total, bool* tx) const {
    std::lock_guard lock(tap_mu_);
    n = std::min({n, tap_.size(), static_cast<std::size_t>(tapped_)});
    std::vector<double> out(n);
    for (std::size_t i = 0; i < n; ++i) out[i] = tap_[(tapped_ - n + i) % tap_.size()];
    if (total) *total = tapped_;
    if (tx) *tx = tap_tx_;
    return out;
}

LinkStatus Station::link() const {
    std::lock_guard lock(status_mu_);
    return link_status_;
}

std::vector<monitor::Dump> Station::take_dumps() {
    if (!monitor_) return {};
    std::lock_guard lock(monitor_->mu);
    std::vector<monitor::Dump> out(std::make_move_iterator(monitor_->out.begin()), std::make_move_iterator(monitor_->out.end()));
    monitor_->out.clear();
    return out;
}

std::vector<BurstLogEntry> Station::take_bursts() {
    std::lock_guard lock(status_mu_);
    std::vector<BurstLogEntry> out(std::make_move_iterator(bursts_.begin()), std::make_move_iterator(bursts_.end()));
    bursts_.clear();
    return out;
}


void Station::log_burst() {
    bursts_.push_back(reading_);
    if (bursts_.size() > MAX_BURSTS) bursts_.pop_front();
}

std::optional<RxConstellation> Station::take_constellation() {
    if (!live_asked_ && engine_ && a_.live_constellation) {
        live_asked_ = true;
        engine_->set_preview(true);  // only a front end that shows it pays for it
    }
    std::lock_guard lock(status_mu_);
    return std::exchange(constellation_, std::nullopt);
}

bool Station::busy() const { return engine_ && engine_->channel_busy(); }

AudioCounters Station::counters() const {
    if (!cap_ || !play_) return {};
    return {cap_->overflows(), play_->underruns(), cap_->late_events(), cap_->dropped(), cap_->backlog_s(),
            engine_ ? engine_->decode_dropped() : 0};
}

// The receive thread: capture block in, engine step, burst out to the transmit thread. It runs the engine's
// acquisition (and, with no decode worker, the decode) and so can be busy for a long stretch; nothing it does
// is on the transmitter's path.
void Station::engine_loop() {
#ifdef __linux__
    pthread_setname_np(pthread_self(), "d2g-engine");  // visible in top -H and tools/sample_threads.py
#endif
    const std::size_t block = config::FS / 10;
    int slow = 0;
    // Every 10 s at DEBUG: where the loop spends its time. A backlog that grows while "read" is ~0 and "step" is
    // small means the time goes between the two (tap, hand-over); a large "step" with low CPU means it waits.
    using clk = std::chrono::steady_clock;
    const auto secs = [](clk::duration d) { return std::chrono::duration<double>(d).count(); };
    clk::duration read_t{}, step_t{}, step_max{};
    int blocks = 0;
    auto window = clk::now();
    try {
        while (!stop_) {
            const auto r0 = clk::now();
            auto x = cap_->read(block);
            read_t += clk::now() - r0;
            if (!x) break;
            const std::vector<double> heard = *x;
            if (tx_->active()) std::fill(x->begin(), x->end(), 0.0);  // the radio's receive audio is muted while it transmits
            const auto t0 = std::chrono::steady_clock::now();
            auto out = engine_->step(*x);
            // the waterfall shows what we send while we send it (the radio's
            // RX audio is muted then), what we hear otherwise
            if (out.ptt && out.audio.size() == heard.size()) tap(out.audio, true);
            else tap(heard, false);
            const double took = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            if (took > static_cast<double>(block) / config::FS) {
                ++slow;
                logf(DEBUG, "step took %.2f s (%d slow)", took, slow);
            }
            step_t += std::chrono::duration_cast<clk::duration>(std::chrono::duration<double>(took));
            step_max = std::max(step_max, std::chrono::duration_cast<clk::duration>(std::chrono::duration<double>(took)));
            ++blocks;
            if (const auto now = clk::now(); now - window >= std::chrono::seconds(10)) {
                logf(DEBUG, "engine loop: %d blocks in %.1f s, read wait %.2f s, step %.2f s (max %.2f), backlog %.2f s", blocks,
                     secs(now - window), secs(read_t), secs(step_t), secs(step_max), cap_->backlog_s());
                read_t = step_t = step_max = {};
                blocks = 0;
                window = now;
            }
            if (out.tx_abort) tx_->abort();
            if (out.tx_burst) tx_->submit(std::move(out.tx_burst));
            watch_counters();
            if (a_.rx_stall_ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(a_.rx_stall_ms));
        }
    } catch (const std::exception& e) {
        logf(CRITICAL, "engine thread: %s", e.what());
        failed_ = true;
        tx_->abort();  // never leave the radio transmitting
    }
}

// PTT ON/OFF to the command clients from the transmit thread, at the moment the rig is keyed or released: a client
// that controls the rig itself follows these, so they cannot wait for (or come from) the engine's block step.
void Station::announce_ptt(bool on) {
    Outbox o;
    o.cmd.emplace_back(on ? "PTT ON" : "PTT OFF", 0);
    QMetaObject::invokeMethod(&ctx_, [this, o = std::move(o)] { deliver(o); }, Qt::QueuedConnection);
}

bool Station::ptt() const { return tx_ && tx_->ptt(); }

std::optional<double> Station::rig_frequency() const { return rig_ ? rig_->frequency_hz() : std::nullopt; }

void Station::start() {
    if (running_) return;
    try {
        stop_ = false;
        failed_ = false;
        link_ = std::make_unique<kisslink::KissLink>(kiss_cap(a_.kiss_bw), a_.broadcast_mode.value_or(""));
        link_->busy_limit_s = a_.kiss_busy_limit;
        monitor_ = std::make_unique<MonitorThread>();
        arq::EngineConfig cfg;
        cfg.ptt_delay_s = a_.ptt_on_delay_ms / 1000.0;
        cfg.record_dir = a_.record_dir;
        cfg.record_async = true;  // a slow disk must not delay a decode or an ACK
        cfg.min_header_score = a_.min_header_score;
        cfg.kiss = link_.get();
        cfg.stats_interval_s = a_.stats_interval;
        cfg.noise_rule = a_.noise_rule;
        cfg.dd_budget_s = a_.dd_budget;
        cfg.worker = a_.decode_worker;
        engine_ = std::make_unique<arq::Engine>(a_.mycall.value_or("NOCALL"), cfg);
        host_ = std::make_unique<host::Host>(*engine_, a_.buffer_credit < 0 ? std::nullopt : std::optional<int>(a_.buffer_credit));
        host_->announce_ptt = false;  // the Transmitter tells the clients, as it keys and unkeys the rig
        host_->dial = [this](const host::Addr& addr) {
            const unsigned gen = ++dial_gen_;
            auto d = std::make_shared<Dial>();
            QMetaObject::invokeMethod(&ctx_, [this, addr, gen, d] { dial(addr.host, addr.port, gen, d); }, Qt::QueuedConnection);
            std::unique_lock lock(d->mu);
            // ponytail: holds the session stage up to DIAL_TIMEOUT_S; fine for a local server
            if (!d->cv.wait_for(lock, std::chrono::duration<double>(host::DIAL_TIMEOUT_S), [&] { return d->up.has_value(); })) {
                d->abandoned = true;
                logf(WARNING, "ACCEPT: no connection to %s:%d within %.1f s", addr.host.c_str(), addr.port, host::DIAL_TIMEOUT_S);
                return false;
            }
            if (*d->up) dial_live_ = gen;
            return *d->up;
        };
        engine_->set_after_block([this](bool ptt) {
            host_->after_step(ptt);
            note_link();
            flush();
        });
        engine_->set_on_preview([this](const arq::BurstPreview& p) {
            auto plot = rx_constellation(p);
            if (!plot_visible(plot)) return;  // nothing to show yet: keep the last plot
            std::lock_guard lock(status_mu_);
            constellation_ = plot;
            live_ = std::move(plot);
        });
        engine_->set_on_burst([this](const arq::BurstHeard& b) {
            mode_ = b.submode;
            auto plot = rx_constellation(b);
            std::lock_guard lock(status_mu_);
            if (plot && plot_visible(*plot)) constellation_ = std::move(plot);
            else if (live_) {  // lost, or nothing on the plot: the live view stays (lost ones marked)
                constellation_ = std::move(live_);
                constellation_->live = false;
                constellation_->lost = b.lost;
            }
            live_.reset();
            reading_ = {std::chrono::system_clock::now(), b.submode, b.n_cw, b.lost, b.snr_db};
            if (b.lost) log_burst();  // else once read (on_decoded)
            if (monitor_on_) monitor_->push(b);
        });
        engine_->set_on_decoded([this](int decoded, int failed) {
            reading_.decoded = decoded;
            reading_.failed = failed;
            std::lock_guard lock(status_mu_);
            log_burst();
        });
        auto post = [this](std::function<void()> f) {
            engine_->post([this, f = std::move(f)] {
                f();
                flush();  // replies at once, not after the next block (clients time out at ~2 s)
            });
        };

        {
            const QHostAddress addr = resolve(a_.host);
            cmd_ = std::make_unique<Port>(
                addr, a_.command_port, true,
                [this, post](const QByteArray& line, std::uint64_t client) {
                    std::string text;  // ascii, as host.py decodes it (others replaced)
                    for (unsigned char c : line) text += c < 128 ? static_cast<char>(c) : '?';
                    logf(INFO, "command: %s", text.c_str());
                    post([this, text, client, peer = cmd_->peer(client)] {
                        flush();  // what was already pending goes first, to everyone
                        host_->command(text, peer);
                        flush(client);  // the reply goes to the client that asked
                    });
                },
                [post, this] { post([this] { host_->client_gone(); }); }, true);
            data_ = std::make_unique<Port>(addr, a_.command_port + 1, false, [this, post](const QByteArray& d, std::uint64_t) {
                arq::Bytes b(d.begin(), d.end());
                post([this, b = std::move(b)] { host_->data_in(b); });
            });
        }
        kiss_ = std::make_unique<KissServer>(
            resolve(a_.kiss_address), a_.kiss_port,
            [this, post](int port, tnc::Bytes f, std::optional<std::int64_t> ack) {
                post([this, port, ack, f = std::move(f)]() mutable { link_->enqueue(std::move(f), port, ack); });
            },
            [this, post](int cmd, tnc::Bytes p) { post([this, cmd, p = std::move(p)] { link_->command(cmd, p); }); });

        // PTT
        rig::Keyer::Ptt ptt;
        rig::Keyer::MustRelease must_release;
        if (rig_enabled(a_)) {
#ifdef DATA2G_HAVE_RIG
            if (a_.rig_debug) rig::set_debug_sink([](const std::string& line) { log_line(INFO, "hamlib: " + line); });
            const bool polling = a_.rig_poll_interval > 0;
            rig_ = std::make_unique<rig::RigController>(nullptr, [polling](const std::string& text, bool error) {
                // a healthy poll's readout ("Rig: 14.1000 MHz") is shown, not logged
                if (error || !polling || text.find(" MHz") == std::string::npos) log_line(error ? ERROR : INFO, "rig: " + text);
            });
            const rig::HamlibConfig hc = hamlib_config(a_);
            rig::RigConfig rc;
            rc.poll_interval_s = a_.rig_poll_interval;  // 0: key only, as tnc.Rigctld
            rig_->start(rig::make_hamlib_backend(hc), rc);
            rig::RigController* r = rig_.get();
            if (hc.ptt_method != rig::PttMethod::Vox) {
                ptt = [set = r->ptt_function()](bool on) {  // no polls while keyed
                    logf(DEBUG, "PTT %s", on ? "on" : "off");
                    set(on);
                };
                must_release = [r] { return r->keyed_since_open(); };
            }
#else
            log_line(WARNING, "built without Hamlib: no PTT (rig settings ignored)");
#endif
        }

        // audio
        if (a_.realtime) {
            audio::set_thread_init([this](const char* role) {
                const auto r = request_realtime(20);
                if (r.ok) logf(INFO, "real-time %s thread: %s", role, r.how.c_str());
                else if (!rt_warned_.exchange(true)) logf(WARNING, "no real-time scheduling for the audio threads: %s", r.how.c_str());
            });
        }
        const double lead_s = a_.tx_lead_ms / 1000.0;
        cap_ = std::make_unique<audio::CaptureFifo>(config::FS);
        const auto report = [](const std::string& s) { log_line(ERROR, s); };
        if (!a_.audio_io.empty()) {
            const auto comma = a_.audio_io.find(',');
            if (!a_.audio_io.starts_with("pipe:") || comma == std::string::npos) throw UsageError("--audio-io: want pipe:IN,OUT");
            play_ = std::make_unique<audio::PlaybackFifo>(config::FS, lead_s);
            const std::string in = a_.audio_io.substr(5, comma - 5), out = a_.audio_io.substr(comma + 1);
            pipe_ = std::make_unique<audio::PipeIo>(in, out, *cap_, *play_, report);
            logf(INFO, "audio: in %s, out %s (float32 at %d Hz)", in.c_str(), out.c_str(), config::FS);
        } else {
#ifdef DATA2G_HAVE_AUDIO
            rate_ = a_.sample_rate;
            if (rate_ <= 0 || rate_ % config::FS)
                throw UsageError("--sample-rate must be a multiple of " + std::to_string(config::FS));
            play_ = std::make_unique<audio::PlaybackFifo>(rate_, lead_s);
            const auto in = audio::select_device(audio::card::input_devices(), a_.input_device.value_or(""), "input");
            const auto out = audio::select_device(audio::card::output_devices(), a_.output_device.value_or(""), "output");
            card_ = std::make_unique<SoundCard>();
            card_->cap = std::make_unique<audio::card::Capture>(in, rate_, *cap_, report);
            card_->play = std::make_unique<audio::card::Playback>(out, rate_, *play_, report);
            logf(INFO, "audio: in %s (%d ch), out %s (%d ch) at %d Hz", card_->cap->device_name().c_str(), card_->cap->channels(),
                 card_->play->device_name().c_str(), card_->play->channels(), rate_);
#else
            throw UsageError("built without sound card audio: only --audio-io pipe:IN,OUT");
#endif
        }

        logf(INFO, "commands on %s:%d, data on %d", a_.host.c_str(), a_.command_port, a_.command_port + 1);
        logf(INFO, "KISS on %s:%d: %d Hz cap, broadcasts in %s", a_.kiss_address.c_str(), a_.kiss_port, a_.kiss_bw,
             link_->broadcast.c_str());
        logf(INFO, "recording to %s", a_.record_dir.empty() ? "(off)" : a_.record_dir.c_str());
        logf(INFO, "decode worker %s", a_.decode_worker ? "on" : "off");

        running_ = true;
        tx_ = std::make_unique<Transmitter>(*play_, rate_, std::pow(10.0, a_.output_volume / 20), std::move(ptt), std::move(must_release),
                                            a_.ptt_off_delay_ms / 1000.0, [this](bool on) { announce_ptt(on); });
        engine_thread_ = std::thread(&Station::engine_loop, this);
    } catch (...) {
        running_ = true;  // so stop() undoes whatever got started
        stop();
        throw;
    }
}

void Station::stop() {
    if (!running_) return;
    running_ = false;
    stop_ = true;
    // the transmitter first: PTT drops now, not after the engine's step (a long decode, a dial wait) ends
    if (tx_) tx_->stop();  // PTT is off when it returns
    if (cap_) cap_->close();
    if (engine_thread_.joinable()) engine_thread_.join();
    if (engine_) engine_->stop();
    monitor_.reset();  // after the engine: nothing pushes to it now
    tx_.reset();  // PTT off, if the rig was ever keyed (~Keyer)
    if (rig_) {
        rig_->stop();
        rig_->wait_for_shutdown(2.0);
#ifdef DATA2G_HAVE_RIG
        if (a_.rig_debug) rig::set_debug_sink({});
#endif
    }
    if (kiss_) kiss_->close_clients();
    if (pipe_) pipe_->stop();
#ifdef DATA2G_HAVE_AUDIO
    if (card_) {
        if (card_->cap) card_->cap->stop();
        if (card_->play) card_->play->stop();
    }
#endif
    // released in reverse order of use; the ports close here, so a restart can listen again
    cmd_.reset();
    data_.reset();
    kiss_.reset();
    card_.reset();
    pipe_.reset();
    rig_.reset();
    host_.reset();
    engine_.reset();
    link_.reset();
    play_.reset();
    cap_.reset();
    audio::set_thread_init({});
    rt_warned_ = false;
    rate_ = config::FS;
    overflows_ = underruns_ = late_ = dropped_ = 0;
}

}  // namespace data2g::app
