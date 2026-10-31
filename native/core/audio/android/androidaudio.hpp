// Sound card capture and playback on Android (data2g_audio on an Android
// build): the same surface as core/audio/qt/, so audio/card.hpp picks it and
// the Station needs no change. The Java half is
// java/org/cleverdomain/data2g/AudioBridge.java.
//
// Lifted from SSTVAE's core/audio/android/, which has the longer reasoning:
// Java AudioRecord rather than AAudio/Oboe or Qt Multimedia, because only it
// offers the UNPROCESSED source (no AGC or noise suppression on the modem's
// input) and dependable routing to a USB interface. Changed for Data2G:
// playback is a stream pulled from the PlaybackFifo, as qtaudio's is, rather
// than one waveform per call.
//
// The direction rule is SSTVAE's: on the data path Java calls C++ (each
// stream's own Java thread pushes capture and pulls playback), so nothing
// attaches per buffer; C++ calls Java only to list, open and close.
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "audio/audio.hpp"
#include "audio/fifo.hpp"

namespace data2g::audio::android {

using Report = std::function<void(const std::string&)>;  // called from the stream's Java thread

// Once, from the UI thread (it has the app's class loader; a thread we made
// does not), before anything else here. Qt owns JNI_OnLoad, hence a call.
struct JavaVM_;
void set_java_vm(JavaVM_* vm);

// Each direction's devices, AudioManager's order, "Product (Type)" names.
// Opened mono, so every one reports one channel.
std::vector<DeviceInfo> input_devices();
std::vector<DeviceInfo> output_devices();

// Opens `device` (an index into input_devices(); nothing: the platform's
// choice) at `rate` if it can, else the first multiple of FS it can, and
// feeds it through a CapturePipeline into `fifo` at FS. Throws if capture
// can't start.
class Capture {
public:
    Capture(std::optional<std::size_t> device, int rate, CaptureFifo& fifo, Report on_error = {});
    ~Capture();
    Capture(const Capture&) = delete;
    Capture& operator=(const Capture&) = delete;

    void stop();
    std::string device_name() const;  // where Android routed it, once it is running
    int channels() const { return 1; }
    std::uint64_t frames_in() const;

    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
};

// Opens `device` (an index into output_devices()) at `rate` and plays from
// `fifo` continuously, silence when it is empty. Sets the FIFO's output
// latency for drain().
class Playback {
public:
    Playback(std::optional<std::size_t> device, int rate, PlaybackFifo& fifo, Report on_error = {});
    ~Playback();
    Playback(const Playback&) = delete;
    Playback& operator=(const Playback&) = delete;

    void stop();
    std::string device_name() const;
    int channels() const { return 1; }

    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
};

}  // namespace data2g::audio::android
