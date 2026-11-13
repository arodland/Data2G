// The radio through libhamlib, linked in-process (data2g_rig, built with
// DATA2G_BUILD_RIG). Lifted from SSTVAE's core/rig/hamlib.hpp, which has
// the longer reasoning; its settings surface is WSJT-X's Radio tab. Model 2
// (NET rigctl) is a rigctld client: device "host:port", empty for
// localhost:4532.
//
// The settings themselves (HamlibConfig, its enums, RigModel) are
// rig/config.hpp, in data2g_core: only the functions here need libhamlib.
#pragma once

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "rig/backend.hpp"
#include "rig/config.hpp"
#include "rig/transport.hpp"

namespace data2g::rig {

// Every model this Hamlib knows, sorted by manufacturer then name. Read from
// struct rig_caps (no pthread members, so safe under the MSVC shim), not by
// parsing `rigctld -l`'s fixed-width columns.
std::vector<RigModel> list_models();
// One model's entry; nullopt if this Hamlib does not know it.
std::optional<RigModel> model_info(int model);

std::unique_ptr<RigBackend> make_hamlib_backend(const HamlibConfig& config);

// What the model's backend would configure a serial port with (struct
// rig_caps; baud is serial_rate_max, as rig_init takes it), for the bridged
// path (rig/bridged.hpp), where Hamlib never opens the port. 9600 8-N-1 for
// a model Hamlib does not know.
SerialDefaults serial_defaults(int model);

// Whether HamlibConfig::ptt_audio means anything to this model.
bool supports_ptt_audio_source(int model);

std::string hamlib_version();

// Hamlib's own trace, a line at a time (newline removed), from whichever
// thread is in Hamlib. Must not block or call Hamlib. Empty: off (and
// DATA2G_HAMLIB_DEBUG's stderr trace back on, if set).
using DebugSink = std::function<void(const std::string& line)>;
void set_debug_sink(DebugSink sink);

}  // namespace data2g::rig
