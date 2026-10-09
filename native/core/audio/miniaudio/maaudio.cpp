#include "audio/miniaudio/maaudio.hpp"

#include "audio/thread.hpp"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdlib>
#include <stdexcept>
#include <string_view>
#include <utility>

// Device I/O only: none of miniaudio's decoders, engine or node graph.
#define MA_NO_DECODING
#define MA_NO_ENCODING
#define MA_NO_GENERATION
#define MA_NO_RESOURCE_MANAGER
#define MA_NO_NODE_GRAPH
#define MA_NO_ENGINE
#define NOMINMAX  // miniaudio includes <windows.h>, whose max() macro breaks std::max
#define MINIAUDIO_IMPLEMENTATION
#include <miniaudio.h>

namespace data2g::audio::ma {

namespace {

void check(ma_result r, const std::string& what) {
    if (r != MA_SUCCESS) throw std::runtime_error(what + ": " + ma_result_description(r));
}

std::string lowered(std::string s) {
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

// One context for the process, so the lists and the devices opened from
// them come from the same backend.
struct Context {
    ma_context ctx;

    Context() {
        ma_backend enabled[ma_backend_null + 1];
        size_t n = 0;
        check(ma_get_enabled_backends(enabled, ma_backend_null + 1, &n), "audio backends");
        // What the sound server shows (pavucontrol, qpwgraph), not "miniaudio".
        ma_context_config c = ma_context_config_init();
        c.pulse.pApplicationName = "Data2G";
        c.jack.pClientName = "Data2G";
        const char* want = std::getenv("DATA2G_AUDIO_BACKEND");
        if (want == nullptr || *want == '\0') {
            check(ma_context_init(nullptr, 0, &c, &ctx), "no working audio backend");
            return;
        }
        std::string names;
        for (size_t i = 0; i < n; ++i) {
            if (lowered(ma_get_backend_name(enabled[i])) == lowered(want)) {
                check(ma_context_init(&enabled[i], 1, &c, &ctx), std::string("audio backend ") + want);
                return;
            }
            names += std::string(i ? ", " : "") + ma_get_backend_name(enabled[i]);
        }
        throw std::runtime_error(std::string("DATA2G_AUDIO_BACKEND=") + want + ": not one of " + names);
    }
    ~Context() { ma_context_uninit(&ctx); }
};

ma_context& context() {
    static Context c;
    return c.ctx;
}

using Listed = std::vector<std::pair<DeviceInfo, ma_device_id>>;

// ma_context_enumerate_devices holds the context's lock through the
// callbacks; ma_context_get_devices' buffer would not survive a second
// caller on another thread (the GUI lists while the station opens).
Listed list(ma_device_type type) {
    struct Want {
        ma_device_type type;
        Listed out, monitors;
    } l{type, {}, {}};
    check(ma_context_enumerate_devices(
              &context(),
              [](ma_context* ctx, ma_device_type t, const ma_device_info* d, void* user) -> ma_bool32 {
                  auto& l = *static_cast<Want*>(user);
                  if (t != l.type) return MA_TRUE;
                  // Enumeration fills the formats on some backends only; the
                  // stream opens at the device's own count either way.
                  int ch = 0;
                  for (ma_uint32 i = 0; i < d->nativeDataFormatCount; ++i)
                      ch = std::max(ch, static_cast<int>(d->nativeDataFormats[i].channels));
                  // PulseAudio's monitor sources go last, so "PCM2903C" matches
                  // the codec rather than its monitor (Qt doesn't list them).
                  const bool monitor = ctx->backend == ma_backend_pulseaudio &&
                                       std::string_view(d->id.pulse).ends_with(".monitor");
                  (monitor ? l.monitors : l.out).push_back({{d->name, ch > 0 ? ch : 2}, d->id});
                  return MA_TRUE;
              },
              &l),
          "listing audio devices");
    l.out.insert(l.out.end(), l.monitors.begin(), l.monitors.end());
    return std::move(l.out);
}

std::vector<DeviceInfo> infos(ma_device_type type) {
    std::vector<DeviceInfo> out;
    for (auto& [info, id] : list(type)) out.push_back(info);
    return out;
}

// One open device; the subclass moves the samples. Not movable: miniaudio
// holds its address.
class Stream {
public:
    Stream(ma_device_type type, const char* dir, Report report) : type_(type), dir_(dir), report_(std::move(report)) {}
    virtual ~Stream() { close(); }

    void open(std::optional<std::size_t> index, int rate) {
        ma_device_id id;
        if (index) {
            const auto devs = list(type_);
            if (*index >= devs.size()) throw std::invalid_argument("no such audio device");
            id = devs[*index].second;
        }
        ma_device_config c = ma_device_config_init(type_);
        const auto side = [&](auto& s) {  // c.capture and c.playback are different types
            s.pDeviceID = index ? &id : nullptr;
            s.format = ma_format_f32;
            s.channels = 0;  // the device's own
        };
        type_ == ma_device_type_capture ? side(c.capture) : side(c.playback);
        c.sampleRate = static_cast<ma_uint32>(rate);
        // host.py's period: 256 frames per 6 kHz (2048 at 48 kHz), two of them.
        c.periodSizeInFrames = static_cast<ma_uint32>(256 * std::max(1, rate / 6000));
        c.periods = 2;
        c.resampling.linear.lpfOrder = MA_MAX_FILTER_ORDER;
        c.dataCallback = [](ma_device* d, void* out, const void* in, ma_uint32 frames) {
            // On miniaudio's device thread, once per thread (a reroute may bring a new one).
            thread_local bool initialised = false;
            if (!initialised) {
                initialised = true;
                audio::thread_init(d->type == ma_device_type_capture ? "capture" : "playback");
            }
            static_cast<Stream*>(d->pUserData)->data(out, in, frames);
        };
        c.notificationCallback = [](const ma_device_notification* n) {
            static_cast<Stream*>(n->pDevice->pUserData)->notify(n->type);
        };
        c.pUserData = this;
        c.pulse.pStreamNameCapture = "Receive";
        c.pulse.pStreamNamePlayback = "Transmit";
        check(ma_device_init(&context(), &c, &dev_), std::string("could not open audio ") + dir_);
        inited_ = true;
        prepare();
        check(ma_device_start(&dev_), "could not start audio " + std::string(dir_) + " on \"" + name() + "\"");
    }

    void close() {
        if (!inited_) return;
        stopping_.store(true, std::memory_order_relaxed);
        ma_device_uninit(&dev_);  // stops first; no callback runs after this returns
        inited_ = false;
    }

    std::string name() {
        char buf[MA_MAX_DEVICE_NAME_LENGTH + 1] = "";
        if (inited_) ma_device_get_name(&dev_, type_, buf, sizeof buf, nullptr);
        return buf;
    }

    std::string description() {
        if (!inited_) return "";
        const auto describe = [&](const auto& s) {
            std::string d = name() + " [" + ma_get_backend_name(context().backend) + ": " +
                            ma_get_format_name(s.internalFormat) + " " + std::to_string(s.internalChannels) + " ch " +
                            std::to_string(s.internalSampleRate) + " Hz, period " +
                            std::to_string(s.internalPeriodSizeInFrames) + " x " + std::to_string(s.internalPeriods);
            if (s.internalSampleRate != dev_.sampleRate) d += ", resampled to " + std::to_string(dev_.sampleRate);
            return d + "]";
        };
        return type_ == ma_device_type_capture ? describe(dev_.capture) : describe(dev_.playback);
    }

    int channels() const {
        return inited_ ? static_cast<int>(type_ == ma_device_type_capture ? dev_.capture.channels : dev_.playback.channels)
                       : 0;
    }

protected:
    virtual void prepare() = 0;  // after init, before start: the format is known
    virtual void data(void* out, const void* in, ma_uint32 frames) = 0;

    ma_device dev_{};

private:
    void notify(ma_device_notification_type t) {
        if (!report_) return;
        const char* what = nullptr;
        if (t == ma_device_notification_type_stopped && !stopping_.load(std::memory_order_relaxed))
            what = "the audio device stopped; it may have been unplugged";
        else if (t == ma_device_notification_type_rerouted)
            what = "the system moved the stream to another device";
        else if (t == ma_device_notification_type_interruption_began)
            what = "the system interrupted the stream";
        if (what) report_(std::string("[audio ") + dir_ + "] " + what);
    }

    ma_device_type type_;
    const char* dir_;
    Report report_;
    bool inited_ = false;
    std::atomic<bool> stopping_{false};
};

class CaptureStream final : public Stream {
public:
    CaptureStream(int rate, CaptureFifo& fifo, Report report)
        : Stream(ma_device_type_capture, "in", std::move(report)), rate_(rate), fifo_(fifo) {}
    ~CaptureStream() override { close(); }  // before pipeline_ goes

    std::uint64_t frames_in() const { return pipeline_ ? pipeline_->frames_in() : 0; }

protected:
    void prepare() override {
        pipeline_ = std::make_unique<CapturePipeline>(SampleFormat::Float, channels(), rate_, fifo_);
    }
    void data(void*, const void* in, ma_uint32 frames) override {
        const std::size_t bytes = std::size_t{frames} * static_cast<std::size_t>(channels()) * sizeof(float);
        (*pipeline_)({static_cast<const std::byte*>(in), bytes});
    }

private:
    int rate_;
    CaptureFifo& fifo_;
    std::unique_ptr<CapturePipeline> pipeline_;
};

class PlaybackStream final : public Stream {
public:
    PlaybackStream(PlaybackFifo& fifo, Report report)
        : Stream(ma_device_type_playback, "out", std::move(report)), fifo_(fifo) {}
    ~PlaybackStream() override { close(); }

protected:
    void prepare() override {
        const auto& p = dev_.playback;
        fifo_.set_output_latency(static_cast<double>(p.internalPeriodSizeInFrames) * p.internalPeriods /
                                 p.internalSampleRate);
        mono_.resize(std::size_t{p.internalPeriodSizeInFrames} * 4);
    }
    // Always a full period (silence past the FIFO's end), so keying needs no restart.
    void data(void* out, const void*, ma_uint32 frames) override {
        if (mono_.size() < frames) mono_.resize(frames);
        const std::span<float> mono(mono_.data(), frames);
        fifo_.pull(mono);
        const auto ch = static_cast<std::size_t>(channels());
        auto* y = static_cast<float*>(out);
        for (std::size_t i = 0; i < frames; ++i) std::fill_n(y + i * ch, ch, mono[i]);
    }

private:
    PlaybackFifo& fifo_;
    std::vector<float> mono_;
};

}  // namespace

std::vector<DeviceInfo> input_devices() { return infos(ma_device_type_capture); }
std::vector<DeviceInfo> output_devices() { return infos(ma_device_type_playback); }

struct Capture::Impl {
    CaptureStream s;
};

Capture::Capture(std::optional<std::size_t> device, int rate, CaptureFifo& fifo, Report on_error)
    : impl_(new Impl{{rate, fifo, std::move(on_error)}}) {
    impl_->s.open(device, rate);
}
Capture::~Capture() = default;
void Capture::stop() { impl_->s.close(); }
std::string Capture::device_name() const { return impl_->s.description(); }
int Capture::channels() const { return impl_->s.channels(); }
std::uint64_t Capture::frames_in() const { return impl_->s.frames_in(); }

struct Playback::Impl {
    PlaybackStream s;
};

Playback::Playback(std::optional<std::size_t> device, int rate, PlaybackFifo& fifo, Report on_error)
    : impl_(new Impl{{fifo, std::move(on_error)}}) {
    impl_->s.open(device, rate);
}
Playback::~Playback() = default;
void Playback::stop() { impl_->s.close(); }
std::string Playback::device_name() const { return impl_->s.description(); }
int Playback::channels() const { return impl_->s.channels(); }

}  // namespace data2g::audio::ma
