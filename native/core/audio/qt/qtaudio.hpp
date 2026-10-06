// Sound card capture and playback through QtMultimedia (data2g_audio_qt,
// built with DATA2G_BUILD_QTAUDIO). The thin part: device enumeration and
// moving bytes. Conversion and buffering are in core/audio/{audio,fifo}.hpp,
// Qt-free and tested against a fake device.
//
// Lifted from SSTVAE's core/audio/qt/qtaudio.{hpp,cpp}. Each stream runs on
// its own QThread with its own event loop: a busy engine thread can't delay
// the drain, and WASAPI only moves bytes for a sink whose owning thread
// pumps events. Unlike SSTVAE, playback is a stream (pull mode from the
// PlaybackFifo) rather than one waveform per call, as host.py's Player is.
//
// Needs a QCoreApplication to exist (QMediaDevices).
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "audio/audio.hpp"
#include "audio/fifo.hpp"

namespace data2g::audio::qt {

using Report = std::function<void(const std::string&)>;  // called from the stream's thread

// Each direction's devices, in Qt's order: what select_device() indexes.
std::vector<DeviceInfo> input_devices();
std::vector<DeviceInfo> output_devices();

// Opens `device` (an index into input_devices(); nothing: the default) at
// `rate` (a multiple of FS), stereo where it has two channels, and feeds
// channel 0 through a CapturePipeline into `fifo` at FS. Throws if the
// device can't be opened.
class Capture {
public:
    Capture(std::optional<std::size_t> device, int rate, CaptureFifo& fifo, Report on_error = {});
    ~Capture();
    Capture(const Capture&) = delete;
    Capture& operator=(const Capture&) = delete;

    void stop();
    std::string device_name() const;
    int channels() const;
    std::uint64_t frames_in() const;  // device frames seen: "is audio arriving"

    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
};

// Opens `device` (an index into output_devices()) at `rate` and plays from
// `fifo` continuously (silence when it is empty), the same signal on every
// channel. Sets the FIFO's output latency for drain(). `buffer_s`: the
// sink's buffer as asked of Qt; 0: two host.py periods (85 ms at 48 kHz).
class Playback {
public:
    Playback(std::optional<std::size_t> device, int rate, PlaybackFifo& fifo, Report on_error = {},
             double buffer_s = 0);
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

}  // namespace data2g::audio::qt
