#include "audio/qt/qtaudio.hpp"

#include "audio/thread.hpp"

#include <QAudioDevice>
#include <QAudioFormat>
#include <QAudioSink>
#include <QAudioSource>
#include <QIODevice>
#include <QMediaDevices>
#include <QObject>
#include <QThread>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <thread>
#include <utility>

namespace data2g::audio::qt {

namespace {

using Clock = std::chrono::steady_clock;

// The capture device's buffer. Some backends (PulseAudio on a Raspberry Pi) deliver a whole buffer at a time, so
// this is also the latency of the received audio: keep it short. DATA2G_CAPTURE_BUFFER_MS overrides it.
double capture_buffer_s() {
    const char* e = std::getenv("DATA2G_CAPTURE_BUFFER_MS");
    const double ms = e ? std::atof(e) : 0;
    return ms > 0 ? ms / 1000 : 0.5;
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
            audio::thread_init(role());  // on this stream's own thread
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
    virtual const char* role() const = 0;

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
    const char* role() const override { return "capture"; }

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
        source_->setBufferSize(f.qt.bytesForDuration(static_cast<qint64>(capture_buffer_s() * 1e6)));
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
    qint64 bytesAvailable() const override { return std::numeric_limits<int>::max(); }

protected:
    qint64 readData(char* data, qint64 maxlen) override {
        const qint64 frame = bytes_per_sample(fmt_) * channels_;
        const auto frames = static_cast<std::size_t>(maxlen / frame);
        mono_.resize(frames);
        fifo_.pull(mono_);
        to_device(mono_, fmt_, channels_, bytes_);
        std::memcpy(data, bytes_.data(), bytes_.size());
        return static_cast<qint64>(bytes_.size());
    }
    qint64 writeData(const char*, qint64) override { return -1; }

private:
    PlaybackFifo& fifo_;
    SampleFormat fmt_;
    int channels_;
    std::vector<float> mono_;
    std::vector<std::byte> bytes_;
};

class PlaybackWorker final : public Worker {
public:
    PlaybackWorker(QAudioDevice device, int rate, PlaybackFifo& fifo, Report report)
        : Worker(std::move(device), rate, std::move(report)), fifo_(fifo) {}
    const char* role() const override { return "playback"; }

    void close() override {
        if (sink_) sink_->stop();
        sink_.reset();
        if (source_) source_->close();
    }

protected:
    void open() override {
        const Format f = choose_format(device_, rate_);
        channels_ = f.qt.channelCount();
        source_ = std::make_unique<FifoDevice>(fifo_, f.ours, f.qt.channelCount());
        source_->open(QIODevice::ReadOnly);
        sink_ = std::make_unique<QAudioSink>(device_, f.qt);
        // host.py's period: 256 frames per 6 kHz (2048 at 48 kHz), two of them.
        const int period = 256 * std::max(1, rate_ / 6000);
        sink_->setBufferSize(f.qt.bytesForFrames(2 * period));
        connect(sink_.get(), &QAudioSink::stateChanged, this, [this] { report("out", sink_->error()); });
        sink_->start(source_.get());
        if (sink_->error() != QAudio::NoError) throw std::runtime_error("could not start playback on \"" + name() + "\"");
        fifo_.set_output_latency(static_cast<double>(f.qt.durationForBytes(sink_->bufferSize())) / 1e6);
    }

private:
    PlaybackFifo& fifo_;
    std::unique_ptr<FifoDevice> source_;
    std::unique_ptr<QAudioSink> sink_;
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

Playback::Playback(std::optional<std::size_t> device, int rate, PlaybackFifo& fifo, Report on_error)
    : impl_(std::make_unique<Impl>()) {
    impl_->t.run(std::make_unique<PlaybackWorker>(
        pick(QMediaDevices::audioOutputs(), device, QMediaDevices::defaultAudioOutput()), rate, fifo,
        std::move(on_error)));
}

Playback::~Playback() { stop(); }
void Playback::stop() { impl_->t.stop(); }
std::string Playback::device_name() const { return impl_->t.worker->name(); }
int Playback::channels() const { return impl_->t.worker->channels(); }

}  // namespace data2g::audio::qt
