#include "audio/qt/qtaudio.hpp"

#include <QAudioDevice>
#include <QAudioFormat>
#include <QAudioSink>
#include <QAudioSource>
#include <QIODevice>
#include <QMediaDevices>
#include <QObject>
#include <QThread>
#include <QTimer>
#include <QVersionNumber>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <thread>
#include <utility>

namespace data2g::audio::qt {

namespace {

using Clock = std::chrono::steady_clock;

// How much the capture device may hold before our thread drains it.
constexpr double CAPTURE_BUFFER_S = 2.0;

// Qt <= 6.8's PulseAudio source makes the buffer size the fragment size:
// a 2 s buffer delivered audio in 2 s lumps, and the engine, clocked by
// capture, then stalled 2 s at a time (TX underruns, late replies). There
// it gets host.py's PortAudio period instead; 6.9+ treats it as a ring.
double capture_buffer_s(int rate) {
    if (QVersionNumber::fromString(QString::fromLatin1(qVersion())) >= QVersionNumber(6, 9)) return CAPTURE_BUFFER_S;
    return 256.0 * std::max(1, rate / 6000) / rate;
}

std::vector<DeviceInfo> infos(const QList<QAudioDevice>& devices) {
    std::vector<DeviceInfo> out;
    for (const QAudioDevice& d : devices) out.push_back({d.description().toStdString(), d.maximumChannelCount()});
    return out;
}

QAudioDevice pick(const QList<QAudioDevice>& devices, std::optional<std::size_t> index, const QAudioDevice& fallback) {
    if (!index) return fallback;
    if (*index >= static_cast<std::size_t>(devices.size())) throw std::invalid_argument("no such audio device");
    return devices.at(static_cast<qsizetype>(*index));
}

struct Format {
    QAudioFormat qt;
    SampleFormat ours;
};

// At the requested rate and channel count (host.py opens float32 at
// --sample-rate); float first, then the integer formats a USB interface
// may insist on. Never the backend's resampler.
Format choose_format(const QAudioDevice& device, int rate) {
    const int channels = open_channels({"", device.maximumChannelCount()});
    const std::pair<QAudioFormat::SampleFormat, SampleFormat> formats[] = {
        {QAudioFormat::Float, SampleFormat::Float},
        {QAudioFormat::Int16, SampleFormat::Int16},
        {QAudioFormat::Int32, SampleFormat::Int32}};
    for (const auto& [qf, ours] : formats) {
        QAudioFormat fmt;
        fmt.setSampleRate(rate);
        fmt.setChannelCount(channels);
        fmt.setSampleFormat(qf);
        if (device.isFormatSupported(fmt)) return {fmt, ours};
    }
    throw std::runtime_error("\"" + device.description().toStdString() + "\" can't do " + std::to_string(rate) +
                             " Hz, " + std::to_string(channels) + " channel(s)");
}

std::string error_text(QAudio::Error e) {
    switch (e) {
        case QAudio::OpenError: return "could not open the audio device (in use, or gone?)";
        case QAudio::IOError: return "audio device I/O error; the device may have been unplugged";
        case QAudio::FatalError: return "the audio device stopped working";
#if QT_VERSION < QT_VERSION_CHECK(6, 9, 0)  // later Qt never reports it
        case QAudio::UnderrunError: return "the sound card ran out of audio (underrun)";
#endif
        default: return "audio error";
    }
}

// One stream, opened and closed on its own thread.
class Worker : public QObject {
public:
    Worker(QAudioDevice device, int rate, Report report)
        : device_(std::move(device)), rate_(rate), report_(std::move(report)) {}

    void start() {
        try {
            open();
            started_.store(true, std::memory_order_release);
        } catch (const std::exception& e) {
            error_ = e.what();
            failed_.store(true, std::memory_order_release);
        }
    }
    virtual void close() = 0;

    bool started() const { return started_.load(std::memory_order_acquire); }
    bool failed() const { return failed_.load(std::memory_order_acquire); }
    const std::string& error() const { return error_; }
    std::string name() const { return device_.description().toStdString(); }
    int channels() const { return channels_.load(std::memory_order_relaxed); }

protected:
    virtual void open() = 0;

    void report(const char* dir, QAudio::Error e) {
        if (e == last_error_) return;
        last_error_ = e;
        if (e != QAudio::NoError && report_) report_(std::string("[audio ") + dir + "] " + error_text(e));
    }

    QAudioDevice device_;
    int rate_;
    Report report_;
    std::atomic<int> channels_{0};

private:
    std::atomic<bool> started_{false}, failed_{false};
    std::string error_;
    QAudio::Error last_error_ = QAudio::NoError;
};

// The thread a Worker lives on, with SSTVAE's start/stop discipline.
struct Thread {
    QThread thread;
    std::unique_ptr<Worker> worker;

    void run(std::unique_ptr<Worker> w) {
        worker = std::move(w);
        worker->moveToThread(&thread);
        QObject::connect(&thread, &QThread::started, worker.get(), [w = worker.get()] { w->start(); });
        thread.start();
        // Bounded, so a wedged backend fails the constructor instead of hanging it.
        const Clock::time_point t0 = Clock::now();
        while (!worker->started() && !worker->failed()) {
            if (Clock::now() - t0 > std::chrono::seconds(5)) {
                stop();
                throw std::runtime_error("timed out opening \"" + worker->name() + "\"");
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        if (worker->failed()) {
            const std::string why = worker->error();
            stop();
            throw std::runtime_error(why);
        }
    }

    // Tear the device down on its own thread: QAudioSource/Sink are thread-affine.
    void stop() {
        if (!thread.isRunning()) return;
        Worker* w = worker.get();
        QMetaObject::invokeMethod(w, [w] { w->close(); }, Qt::BlockingQueuedConnection);
        thread.quit();
        thread.wait();
    }
};

class CaptureWorker final : public Worker {
public:
    CaptureWorker(QAudioDevice device, int rate, CaptureFifo& fifo, Report report)
        : Worker(std::move(device), rate, std::move(report)), fifo_(fifo) {}

    void close() override {
        if (io_ != nullptr) disconnect(io_, nullptr, this, nullptr);
        io_ = nullptr;
        if (source_) source_->stop();
        source_.reset();
    }

    std::uint64_t frames_in() const { return frames_in_.load(std::memory_order_relaxed); }

protected:
    void open() override {
        const Format f = choose_format(device_, rate_);
        channels_ = f.qt.channelCount();
        pipeline_ = std::make_unique<CapturePipeline>(f.ours, f.qt.channelCount(), rate_, fifo_);
        source_ = std::make_unique<QAudioSource>(device_, f.qt);
        source_->setBufferSize(f.qt.bytesForDuration(static_cast<qint64>(capture_buffer_s(rate_) * 1e6)));
        connect(source_.get(), &QAudioSource::stateChanged, this, [this] { report("in", source_->error()); });
        io_ = source_->start();
        if (io_ == nullptr) throw std::runtime_error("could not start capture on \"" + name() + "\"");
        connect(io_, &QIODevice::readyRead, this, [this] { drain(); });
    }

private:
    void drain() {
        const QByteArray raw = io_->readAll();
        if (raw.isEmpty()) return;
        (*pipeline_)({reinterpret_cast<const std::byte*>(raw.constData()), static_cast<std::size_t>(raw.size())});
        frames_in_.store(pipeline_->frames_in(), std::memory_order_relaxed);
    }

    CaptureFifo& fifo_;
    std::unique_ptr<CapturePipeline> pipeline_;
    std::unique_ptr<QAudioSource> source_;
    QIODevice* io_ = nullptr;
    std::atomic<std::uint64_t> frames_in_{0};
};

// What the sink pulls from: the PlaybackFifo, converted to device bytes.
// Always a full read (silence past the FIFO's end), so the stream never
// idles and keying needs no restart.
class FifoDevice final : public QIODevice {
public:
    FifoDevice(PlaybackFifo& fifo, SampleFormat fmt, int channels) : fifo_(fifo), fmt_(fmt), channels_(channels) {}
    bool isSequential() const override { return true; }
    // Keyed: what is queued (Qt 6.9+ reads no more than this). Idle: endless silence.
    qint64 bytesAvailable() const override {
        return fifo_.keyed() ? static_cast<qint64>(fifo_.queued()) * bytes_per_sample(fmt_) * channels_
                             : std::numeric_limits<int>::max();
    }
    // Bytes handed to the sink since start: what it has, played or not.
    qint64 pulled() const { return pulled_.load(std::memory_order_relaxed); }

protected:
    qint64 readData(char* data, qint64 maxlen) override {
        const qint64 frame = bytes_per_sample(fmt_) * channels_;
        const auto frames = static_cast<std::size_t>(maxlen / frame);
        mono_.resize(frames);
        // Keyed, a shortfall is a short read, not zeros in the burst: the
        // sink asks to top up its buffer, which usually still holds plenty.
        mono_.resize(fifo_.pull_some(mono_));
        // Silence (lead, tail, idle, an underrun's padding) goes out as
        // -90 dBFS noise, +-1 LSB at 16 bits: nothing downstream sees
        // digital zero and decides the stream has stopped.
        for (float& v : mono_)
            if (v == 0.0f) v = (rng_() & 1) ? FLOOR : -FLOOR;
        to_device(mono_, fmt_, channels_, bytes_);
        std::memcpy(data, bytes_.data(), bytes_.size());
        pulled_.fetch_add(static_cast<qint64>(bytes_.size()), std::memory_order_relaxed);
        return static_cast<qint64>(bytes_.size());
    }
    qint64 writeData(const char*, qint64) override { return -1; }

private:
    PlaybackFifo& fifo_;
    SampleFormat fmt_;
    int channels_;
    std::vector<float> mono_;
    std::vector<std::byte> bytes_;
    std::atomic<qint64> pulled_{0};
    static constexpr float FLOOR = 3.2e-5f;  // -90 dBFS
    std::minstd_rand rng_;  // the sink may read off its own thread (Qt 6.9+)
};

class PlaybackWorker final : public Worker {
public:
    PlaybackWorker(QAudioDevice device, int rate, PlaybackFifo& fifo, Report report, double buffer_s)
        : Worker(std::move(device), rate, std::move(report)), fifo_(fifo), buffer_s_(buffer_s) {}

    void close() override {
        fifo_.set_on_write({});
        if (timer_) timer_->stop();
        if (sink_) sink_->stop();
        sink_.reset();
        if (source_) source_->close();
    }

protected:
    void open() override {
        const Format f = choose_format(device_, rate_);
        channels_ = f.qt.channelCount();
        source_ = std::make_unique<FifoDevice>(fifo_, f.ours, f.qt.channelCount());
        source_->open(QIODevice::ReadOnly | QIODevice::Unbuffered);  // no read-ahead hidden in Qt
        // After a short read Qt 6.8 stops pulling until readyRead; 6.9+ pulls at once on it.
        fifo_.set_on_write([d = source_.get()] {
            QMetaObject::invokeMethod(d, [d] { Q_EMIT d->readyRead(); }, Qt::QueuedConnection);
        });
        sink_ = std::make_unique<QAudioSink>(device_, f.qt);
        // host.py's period: 256 frames per 6 kHz (2048 at 48 kHz), two of them.
        const int period = 256 * std::max(1, rate_ / 6000);
        sink_->setBufferSize(buffer_s_ > 0 ? f.qt.bytesForDuration(static_cast<qint64>(buffer_s_ * 1e6))
                                           : f.qt.bytesForFrames(2 * period));
        connect(sink_.get(), &QAudioSink::stateChanged, this, [this] { report("out", sink_->error()); });
        sink_->start(source_.get());
        if (sink_->error() != QAudio::NoError) throw std::runtime_error("could not start playback on \"" + name() + "\"");
        // bufferSize() is only the stream's target; the server's own
        // buffering comes on top (Qt 6.8 on PipeWire: 63 ms reported, 120 ms
        // measured). What the sink has minus what it has played is the
        // real latency; where processedUSecs() counts pulls it reads ~0.
        const double buffered = static_cast<double>(f.qt.durationForBytes(sink_->bufferSize())) / 1e6;
        fifo_.set_output_latency(buffered);
        timer_ = new QTimer(this);
        connect(timer_, &QTimer::timeout, this, [this, f, buffered] {
            const auto ahead = static_cast<double>(f.qt.durationForBytes(source_->pulled()) - sink_->processedUSecs()) / 1e6;
            fifo_.set_output_latency(std::clamp(ahead, buffered, 5.0));
        });
        timer_->start(100);
    }

private:
    PlaybackFifo& fifo_;
    double buffer_s_;
    std::unique_ptr<FifoDevice> source_;
    std::unique_ptr<QAudioSink> sink_;
    QTimer* timer_ = nullptr;
};

}  // namespace

std::vector<DeviceInfo> input_devices() { return infos(QMediaDevices::audioInputs()); }
std::vector<DeviceInfo> output_devices() { return infos(QMediaDevices::audioOutputs()); }

struct Capture::Impl {
    Thread t;
    CaptureWorker* worker = nullptr;
};

Capture::Capture(std::optional<std::size_t> device, int rate, CaptureFifo& fifo, Report on_error)
    : impl_(std::make_unique<Impl>()) {
    auto w = std::make_unique<CaptureWorker>(
        pick(QMediaDevices::audioInputs(), device, QMediaDevices::defaultAudioInput()), rate, fifo, std::move(on_error));
    impl_->worker = w.get();
    impl_->t.run(std::move(w));
}

Capture::~Capture() { stop(); }
void Capture::stop() { impl_->t.stop(); }
std::string Capture::device_name() const { return impl_->worker->name(); }
int Capture::channels() const { return impl_->worker->channels(); }
std::uint64_t Capture::frames_in() const { return impl_->worker->frames_in(); }

struct Playback::Impl {
    Thread t;
};

Playback::Playback(std::optional<std::size_t> device, int rate, PlaybackFifo& fifo, Report on_error, double buffer_s)
    : impl_(std::make_unique<Impl>()) {
    impl_->t.run(std::make_unique<PlaybackWorker>(
        pick(QMediaDevices::audioOutputs(), device, QMediaDevices::defaultAudioOutput()), rate, fifo,
        std::move(on_error), buffer_s));
}

Playback::~Playback() { stop(); }
void Playback::stop() { impl_->t.stop(); }
std::string Playback::device_name() const { return impl_->t.worker->name(); }
int Playback::channels() const { return impl_->t.worker->channels(); }

}  // namespace data2g::audio::qt
