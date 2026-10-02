// The radio through libhamlib, linked in-process (data2g_rig, built with
// DATA2G_BUILD_RIG). Trimmed from SSTVAE's core/rig/hamlib.hpp to what the
// host uses today: keying, through any model. Model 2 (NET rigctl) is a
// rigctld client, so `--rigctld-host/--rigctld-port` keep working as
// device "host:port".
//
// ponytail: serial line settings, rig mode, mic/data PTT, list_models and
// the debug sink were left in SSTVAE; lift them with the settings dialog.
#pragma once

#include <memory>
#include <string>

#include "rig/backend.hpp"

namespace data2g::rig {

inline constexpr int MODEL_DUMMY = 1;
inline constexpr int MODEL_NET_RIGCTL = 2;

enum class PttMethod { Cat, Dtr, Rts };

struct HamlibConfig {
    int model = MODEL_NET_RIGCTL;
    // Serial device, or "host:port" for MODEL_NET_RIGCTL.
    std::string device = "localhost:4532";
    int baud = 0;  // 0: the backend's default
    PttMethod ptt_method = PttMethod::Cat;
    std::string ptt_device;  // DTR/RTS keying on another port; empty: the CAT device
    // One timeout and one retry, both ours: RigController::stop() abandons a
    // worker that only exits when its call gives up.
    int timeout_ms = 1000;
    int retries = 1;
};

std::unique_ptr<RigBackend> make_hamlib_backend(const HamlibConfig& config);

std::string hamlib_version();

}  // namespace data2g::rig
