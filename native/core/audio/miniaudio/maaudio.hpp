// Sound card capture and playback through miniaudio (data2g_audio with
// -DDATA2G_AUDIO_BACKEND=miniaudio). The same interface as audio/qt/qtaudio.hpp,
// so audio/card.hpp can swap them; conversion and buffering are shared
// (core/audio/{audio,fifo}.hpp).
//
// No Qt and no event loop: miniaudio calls back on its own thread per
// device, which calls audio::thread_init() (audio/thread.hpp) on its first
// callback. DATA2G_AUDIO_BACKEND in the environment (e.g. ALSA, PulseAudio,
// JACK, WASAPI, DirectSound, WinMM, "Core Audio"; any case) forces one
// backend instead of miniaudio's first working one.
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "audio/audio.hpp"
#include "audio/fifo.hpp"

namespace data2g::audio::ma {

using Report = std::function<void(const std::string&)>;  // called from miniaudio's device thread

// Each direction's devices, in the backend's order: what select_device() indexes.
std::vector<DeviceInfo> input_devices();
std::vector<DeviceInfo> output_devices();

// Opens `device` (an index into input_devices(); nothing: the default) at
// `rate` (a multiple of FS) as float32 at the device's own channel count,
// and feeds channel 0 through a CapturePipeline into `fifo` at FS. A device
// that can't run at `rate` is resampled by miniaudio (device_name() says
// so). Throws if the device can't be opened.
class Capture {
public:
    Capture(std::optional<std::size_t> device, int rate, CaptureFifo& fifo, Report on_error = {});
    ~Capture();
    Capture(const Capture&) = delete;
    Capture& operator=(const Capture&) = delete;

    void stop();
    // The name, then what the backend actually opened: "NAME [ALSA: s16 2 ch 48000 Hz, period 2048 x 2]".
    std::string device_name() const;
    int channels() const;
    std::uint64_t frames_in() const;  // device frames seen: "is audio arriving"

    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
};

// Opens `device` (an index into output_devices()) at `rate` and plays from
// `fifo` continuously (silence when it is empty), the same signal on every
// channel. Sets the FIFO's output latency for drain().
class Playback {
public:
    Playback(std::optional<std::size_t> device, int rate, PlaybackFifo& fifo, Report on_error = {});
    ~Playback();
    Playback(const Playback&) = delete;
    Playback& operator=(const Playback&) = delete;

    void stop();
    std::string device_name() const;
    int channels() const;

    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
};

}  // namespace data2g::audio::ma
