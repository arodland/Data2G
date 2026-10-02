// The radio through libhamlib, linked in-process (data2g_rig, built with
// DATA2G_BUILD_RIG). Lifted from SSTVAE's core/rig/hamlib.hpp, which has
// the longer reasoning; its settings surface is WSJT-X's Radio tab. Model 2
// (NET rigctl) is a rigctld client, so `--rigctld-host/--rigctld-port` keep
// working as device "host:port".
//
// Not lifted: serial_defaults() (only SSTVAE's Android bridge needs it; it
// reads struct rig_caps the same way model_info() does here).
#pragma once

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "rig/backend.hpp"

namespace data2g::rig {

inline constexpr int MODEL_DUMMY = 1;
inline constexpr int MODEL_NET_RIGCTL = 2;

// What the model's rig_pathname names, so a settings page shows only the
// fields that mean something: a serial device and its line settings, a
// host:port, some other device path, or nothing (the dummy).
enum class PortType { None, Serial, Network, Other };

struct RigModel {
    int model = 0;
    std::string manufacturer, name, version;
    std::string status;  // Hamlib's backend status: Alpha, Beta, Stable...
    PortType port = PortType::Other;

    // "Elecraft K4 (Stable)": what a picker shows.
    std::string label() const;
};

// Every model this Hamlib knows, sorted by manufacturer then name. Read from
// struct rig_caps (no pthread members, so safe under the MSVC shim), not by
// parsing `rigctld -l`'s fixed-width columns.
std::vector<RigModel> list_models();
// One model's entry; nullopt if this Hamlib does not know it.
std::optional<RigModel> model_info(int model);

// Vox: do not key at all (the rig keys on its own audio).
enum class PttMethod { Vox, Cat, Dtr, Rts };
// Which input CAT keying selects on a rig with two (RIG_PTT_RIG_MICDATA).
enum class PttAudio { Mic, Data };
// Serial line settings. Default: leave the token unset, the backend's own.
enum class DataBits { Default, Seven, Eight };
enum class StopBits { Default, One, Two };
enum class Parity { Default, None, Odd, Even };
enum class Handshake { Default, None, XonXoff, Hardware };
// Held for the session: an interface powered from the control lines.
enum class LineState { Default, High, Low };
// Set once the rig is open; None leaves the operator's mode alone.
enum class RigMode { None, Usb, PktUsb };

struct HamlibConfig {
    int model = MODEL_NET_RIGCTL;
    // Serial device, or "host:port" for MODEL_NET_RIGCTL. Empty: Hamlib's default.
    std::string device = "localhost:4532";
    int baud = 0;  // 0: the backend's default
    DataBits data_bits = DataBits::Default;
    StopBits stop_bits = StopBits::Default;
    Parity parity = Parity::Default;
    Handshake handshake = Handshake::Default;
    LineState dtr = LineState::Default;
    LineState rts = LineState::Default;

    PttMethod ptt_method = PttMethod::Cat;
    PttAudio ptt_audio = PttAudio::Mic;
    std::string ptt_device;  // DTR/RTS keying on another port; empty: the CAT device

    RigMode mode = RigMode::None;

    // One timeout and one retry, both ours: RigController::stop() abandons a
    // worker that only exits when its call gives up.
    int timeout_ms = 1000;
    int retries = 1;
};

std::unique_ptr<RigBackend> make_hamlib_backend(const HamlibConfig& config);

// Whether HamlibConfig::ptt_audio means anything to this model.
bool supports_ptt_audio_source(int model);

std::string hamlib_version();

// Hamlib's own trace, a line at a time (newline removed), from whichever
// thread is in Hamlib. Must not block or call Hamlib. Empty: off (and
// DATA2G_HAMLIB_DEBUG's stderr trace back on, if set).
using DebugSink = std::function<void(const std::string& line)>;
void set_debug_sink(DebugSink sink);

}  // namespace data2g::rig
