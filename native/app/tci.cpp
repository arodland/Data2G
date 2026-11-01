#include "app/tci.hpp"

#include <QElapsedTimer>
#include <QTimer>
#include <QUrl>
#include <QWebSocket>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

#include "app/log.hpp"
#include "audio/audio.hpp"
#include "audio/fifo.hpp"
#include "audio/thread.hpp"
#include "generated/config.hpp"

namespace data2g::app {

namespace {

static_assert(std::endian::native == std::endian::little, "TCI's samples are little-endian; copied as they are");

// The Stream header: 16 little-endian uint32s ahead of the samples (receiver, sample_rate, format, codec, crc,
// length, type, channels, count, reserved).
constexpr qsizetype HEADER = 64;
enum Field { RECEIVER = 0, SAMPLE_RATE = 1, FORMAT = 2, LENGTH = 5, TYPE = 6, CHANNELS = 7, COUNT = 8 };
enum Type : std::uint32_t { RX_AUDIO = 1, TX_AUDIO = 2, TX_CHRONO = 3 };
constexpr std::uint32_t FLOAT32 = 3;

constexpr int TICK_MS = 20;
constexpr qint64 RX_GAP_MS = 250;      // no RX audio this long: silence fed at real time
constexpr qint64 CHRONO_GAP_MS = 500;  // keyed with no TX_CHRONO this long: fed on our clock
constexpr int RETRY_MS = 2000;
// TX_STREAM_AUDIO_BUFFERING's default (50 ms) and a TX_CHRONO period (2048 samples at 48 kHz); the PTT off delay
// covers a server that holds more
constexpr double OUTPUT_LATENCY_S = 0.1;

// `f` has a header (HEADER bytes or more).
std::array<std::uint32_t, 16> header(const QByteArray& f) {
    std::array<std::uint32_t, 16> h;
    std::memcpy(h.data(), f.constData(), sizeof h);
    return h;
}

void put(QByteArray& f, int i, std::uint32_t v) { std::memcpy(f.data() + 4 * i, &v, 4); }

// TCI's sample types; int24 (1) is one CapturePipeline lacks.
std::optional<audio::SampleFormat> sample_format(std::uint32_t f) {
    switch (f) {
        case 0: return audio::SampleFormat::Int16;
        case 2: return audio::SampleFormat::Int32;
        case 3: return audio::SampleFormat::Float;
        default: return std::nullopt;
    }
}

}  // namespace

struct Tci::Worker : QObject {
    Worker(std::string host, int port, int rate, std::string modulation, audio::CaptureFifo& cap, audio::PlaybackFifo& play)
        : url(QStringLiteral("ws://%1:%2").arg(QString::fromStdString(host)).arg(port)),
          rate(rate),
          modulation(std::move(modulation)),
          cap(cap),
          play(play) {}

    const QUrl url;
    const int rate;
    const std::string modulation;
    audio::CaptureFifo& cap;
    audio::PlaybackFifo& play;

    std::atomic<bool> sent_key{false};  // read by keyed() on the transmit thread
    mutable std::mutex mu;
    std::optional<double> dial;  // mu

    // the rest on the worker's thread only
    QWebSocket* ws = nullptr;
    QTimer* tick = nullptr;
    QElapsedTimer clock;
    bool ready = false, ptt_on = false, closing = false, retry_pending = false;
    QString last_error;
    std::unique_ptr<audio::CapturePipeline> pipe;
    std::uint32_t pipe_format = 0, pipe_channels = 0, pipe_rate = 0;
    bool format_warned = false;
    qint64 last_rx = 0, fill_from = 0;  // ms on `clock`
    qint64 last_chrono = -1, keyed_at = 0, push_from = 0;
    std::uint32_t tx_count = 0;
    std::vector<float> pulled;
    std::vector<double> silence;

    void start() {
        audio::thread_init("tci");
        clock.start();
        play.set_output_latency(OUTPUT_LATENCY_S);
        ws = new QWebSocket(QString(), QWebSocketProtocol::VersionLatest, this);
        connect(ws, &QWebSocket::connected, this, [this] {
            logf(INFO, "TCI: connected to %s", url.toString().toStdString().c_str());
            last_error.clear();
        });
        connect(ws, &QWebSocket::disconnected, this, [this] { lost(); });
        connect(ws, &QWebSocket::errorOccurred, this, [this](QAbstractSocket::SocketError) {
            if (ws->errorString() != last_error && !closing) {
                last_error = ws->errorString();
                logf(WARNING, "TCI %s: %s (retrying every %d s)", url.toString().toStdString().c_str(),
                     last_error.toStdString().c_str(), RETRY_MS / 1000);
            }
            lost();
        });
        connect(ws, &QWebSocket::textMessageReceived, this, [this](const QString& m) { text(m); });
        connect(ws, &QWebSocket::binaryMessageReceived, this, [this](const QByteArray& f) { binary(f); });
        tick = new QTimer(this);
        tick->setTimerType(Qt::PreciseTimer);
        connect(tick, &QTimer::timeout, this, [this] { on_tick(); });
        tick->start(TICK_MS);
        open();
    }

    void open() {
        retry_pending = false;
        if (!closing) ws->open(url);
    }

    void lost() {
        if (ready) logf(WARNING, "TCI: connection lost (retrying every %d s)", RETRY_MS / 1000);
        ready = ptt_on = false;
        sent_key = false;
        pipe.reset();
        if (closing || retry_pending) return;
        retry_pending = true;
        QTimer::singleShot(RETRY_MS, this, [this] { open(); });
    }

    // On the worker's thread, before it quits: what lives on it is deleted here.
    void close() {
        closing = true;
        delete tick;
        tick = nullptr;
        if (ws) ws->close();
        delete ws;
        ws = nullptr;
    }

    void send(const QString& cmd) {
        logf(DEBUG, "TCI > %s", cmd.toStdString().c_str());
        ws->sendTextMessage(cmd);
    }

    bool set_ptt(bool on) {
        if (!ready) return false;
        send(on ? QStringLiteral("trx:0,true,tci;") : QStringLiteral("trx:0,false;"));
        ptt_on = on;
        if (on) {
            keyed_at = push_from = clock.elapsed();
            tx_count = 0;
            sent_key = true;
        }
        return true;
    }

    void text(const QString& message) {
        for (const QString& part : message.split(QLatin1Char(';'), Qt::SkipEmptyParts)) {
            const QString cmd = part.trimmed();
            const qsizetype colon = cmd.indexOf(QLatin1Char(':'));
            const QString name = cmd.left(colon).toLower();
            const QStringList args = colon < 0 ? QStringList() : cmd.mid(colon + 1).split(QLatin1Char(','));
            if (name != QLatin1String("vfo")) logf(DEBUG, "TCI < %s", cmd.toStdString().c_str());
            if (name == QLatin1String("ready")) {
                ready = true;
                // one command a message; the server answers each with its value (a rate it lacks: the one it keeps)
                send(QStringLiteral("audio_samplerate:%1;").arg(rate));
                send(QStringLiteral("audio_stream_sample_type:float32;"));
                send(QStringLiteral("audio_stream_channels:1;"));
                if (!modulation.empty()) send(QStringLiteral("modulation:0,%1;").arg(QString::fromStdString(modulation)));
                send(QStringLiteral("audio_start:0;"));
            } else if (name == QLatin1String("vfo") && args.size() == 3 && args[0] == QLatin1String("0") &&
                       args[1] == QLatin1String("0")) {
                bool ok = false;
                const double hz = args[2].toDouble(&ok);
                std::lock_guard lock(mu);
                if (ok) dial = hz;
            } else if (name == QLatin1String("audio_samplerate") && args.size() == 1 && args[0].toInt() != rate) {
                // our TX frames say the rate they are at; whether the server converts them is its business
                logf(WARNING, "TCI: the server keeps its audio at %s Hz, not %d", args[0].toStdString().c_str(), rate);
            } else if (name == QLatin1String("protocol") || name == QLatin1String("device")) {
                logf(INFO, "TCI %s: %s", name.toStdString().c_str(), args.join(QLatin1Char(' ')).toStdString().c_str());
            }
        }
    }

    void binary(const QByteArray& f) {
        if (f.size() < HEADER) return;
        const auto h = header(f);
        if (h[RECEIVER] != 0) return;
        const qint64 now = clock.elapsed();
        if (h[TYPE] == RX_AUDIO) {
            const auto fmt = sample_format(h[FORMAT]);
            const std::uint32_t r = h[SAMPLE_RATE];
            // 0 before TCI 1.9, which was stereo
            const std::uint32_t ch = h[CHANNELS] ? h[CHANNELS] : 2;
            if (!fmt || r < config::FS || r % config::FS) {
                if (!format_warned)
                    logf(ERROR, "TCI: RX audio as type %u at %u Hz: want int16/int32/float32 at 8, 24 or 48 kHz",
                         h[FORMAT], r);
                format_warned = true;
                return;
            }
            if (!pipe || h[FORMAT] != pipe_format || ch != pipe_channels || r != pipe_rate) {
                pipe = std::make_unique<audio::CapturePipeline>(*fmt, static_cast<int>(ch), static_cast<int>(r), cap);
                pipe_format = h[FORMAT];
                pipe_channels = ch;
                pipe_rate = r;
            }
            // the samples by the frame's size, not `length`: its meaning (per channel or not) changed in 1.9
            (*pipe)({reinterpret_cast<const std::byte*>(f.constData() + HEADER), static_cast<std::size_t>(f.size() - HEADER)});
            last_rx = fill_from = now;
        } else if (h[TYPE] == TX_CHRONO) {
            last_chrono = now;
            if (!ptt_on) return;
            // the reply echoes length and channels; before 1.9 (channels 0) length was per channel, in stereo
            const std::uint32_t length = h[LENGTH], ch = h[CHANNELS];
            send_tx(ch ? length / ch : length, ch ? ch : 2, length, ch);
        }
    }

    // `frames` from the playback FIFO, on every one of `channels`, as one TX_AUDIO frame.
    void send_tx(std::size_t frames, std::uint32_t channels, std::uint32_t length, std::uint32_t channels_field) {
        pulled.resize(frames);
        play.pull(pulled);
        QByteArray out(HEADER + static_cast<qsizetype>(frames * channels * sizeof(float)), '\0');
        put(out, SAMPLE_RATE, static_cast<std::uint32_t>(rate));
        put(out, FORMAT, FLOAT32);
        put(out, LENGTH, length);
        put(out, TYPE, TX_AUDIO);
        put(out, CHANNELS, channels_field);
        put(out, COUNT, tx_count++);
        char* d = out.data() + HEADER;
        for (std::size_t i = 0; i < frames; ++i)
            for (std::uint32_t c = 0; c < channels; ++c, d += sizeof(float)) std::memcpy(d, &pulled[i], sizeof(float));
        ws->sendBinaryMessage(out);
    }

    void on_tick() {
        const qint64 now = clock.elapsed();
        if (now - last_rx > RX_GAP_MS) {
            silence.assign(static_cast<std::size_t>((now - fill_from) * config::FS / 1000), 0.0);
            cap.write(silence);
            fill_from = now;
        }
        if (!ptt_on) return;
        // A server that never sends TX_CHRONO is fed at once; one that has, only once it misses this over's
        if (last_chrono < 0 || now - std::max(last_chrono, keyed_at) > CHRONO_GAP_MS) {
            const auto frames = static_cast<std::size_t>((now - push_from) * rate / 1000);
            if (frames) send_tx(frames, 1, static_cast<std::uint32_t>(frames), 1);
        }
        push_from = now;
    }
};

Tci::Tci(std::string host, int port, int rate, std::string modulation, audio::CaptureFifo& capture, audio::PlaybackFifo& playback)
    : w_(new Worker(std::move(host), port, rate, std::move(modulation), capture, playback)) {
    w_->moveToThread(&thread_);
    QObject::connect(&thread_, &QThread::started, w_, [w = w_] { w->start(); });
    thread_.start();
}

Tci::~Tci() {
    QMetaObject::invokeMethod(w_, [w = w_] { w->close(); }, Qt::BlockingQueuedConnection);
    thread_.quit();
    thread_.wait();
    delete w_;
}

void Tci::set_ptt(bool on) {
    bool ok = false;
    QMetaObject::invokeMethod(w_, [this, on, &ok] { ok = w_->set_ptt(on); }, Qt::BlockingQueuedConnection);
    if (!ok) throw std::runtime_error("TCI: not connected");
}

bool Tci::keyed() const { return w_->sent_key; }

std::optional<double> Tci::frequency_hz() const {
    std::lock_guard lock(w_->mu);
    return w_->dial;
}

}  // namespace data2g::app
