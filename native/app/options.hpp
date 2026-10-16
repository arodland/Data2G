// The station's options: host.py main()'s argparse, plus the C++ host's
// additions and the rig (Hamlib) settings. parse() reads a command line over
// defaults or a front end's saved settings; check() is what main() checks
// once parsed; hamlib_config() is the one mapping to Hamlib's. data2g-gui's
// settings dialog edits an Args and needs nothing else of the station.
#pragma once

#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "rig/config.hpp"

namespace data2g::app {

std::string default_record_dir();  // recordings/YYYYmmdd-HHMMSS

struct Args {
    int kiss_port = 8100;
    std::string kiss_address = "127.0.0.1";
    double kiss_busy_limit = 60.0;
    int kiss_bw = 2400;
    std::optional<std::string> broadcast_mode, mycall, input_device, output_device;
    std::string host = "127.0.0.1";
    int command_port = 8300;
    bool list_audio_devices = false, list_modes = false;
    int sample_rate = 48000;
    double output_volume = 0.0;
    std::string rigctld_host = "localhost";
    int rigctld_port = 4532;
    // Hamlib (SSTVAE's rig settings). Model 2 with no rig_device is rigctld
    // at rigctld_host:rigctld_port, so the two flags above work as before.
    // Enumerated values in SSTVAE's lowercase spellings; "default": leave
    // the backend's own. See rig_choices().
    bool rig = true;  // false: no rig (so no PTT)
    int rig_model = 2;
    std::string rig_device;  // serial device or host:port; empty: model 2's rigctld, else Hamlib's
    int rig_baud = 0;        // 0: the backend's
    std::string rig_data_bits = "default", rig_stop_bits = "default", rig_parity = "default",
                rig_handshake = "default", rig_dtr = "default", rig_rts = "default";
    std::string ptt_method = "cat", ptt_device, ptt_audio = "mic";
    std::string rig_mode = "none";
    int rig_timeout_ms = 1000, rig_retries = 1;
    double rig_poll_interval = 0.0;  // s; 0: key only (frequency not read)
    bool rig_debug = false;          // Hamlib's trace into the log
    bool list_rigs = false;
    int ptt_on_delay_ms = 100, ptt_off_delay_ms = 50, tx_lead_ms = 100;
    double min_header_score = 0.0;
    int buffer_credit = -1;
    std::string record_dir = default_record_dir();
    std::string log_level = "INFO";
    double stats_interval = 60.0;
    double noise_rule = 1.0;  // the gear shifter's noise rule, its tail weight; 0: off
    double dd_budget = 1.0;   // seconds of decision-directed decoding per burst (arq::DD_BUDGET_S); 0: none
    // additions
    bool decode_worker = true;
    int rx_stall_ms = 0;   // testing: the receive loop sleeps this long after every block (a busy machine)
    bool realtime = true;  // real-time scheduling for the audio and transmit threads, where available
    bool live_constellation = true;  // fill the constellation plot in as a burst arrives, once a front end polls it
    std::string audio_io;  // "pipe:IN,OUT"; empty: the sound card

    bool operator==(const Args&) const = default;
};

// The command line over `a` (defaults, or what a front end saved). Exits as
// argparse does on -h and on a bad flag. `prog` names the binary in errors.
Args parse(int argc, char** argv, Args a = {}, const char* prog = "data2g-host");
[[noreturn]] void usage_error(const std::string& msg, const char* prog = "data2g-host");
// What main() checks once parsed; the message, or nullopt when fine.
std::optional<std::string> check(const Args& a);
int kiss_cap(int hz);
// The rig settings' allowed values, by Args field name ("rig_parity", ...).
const std::vector<std::string>& rig_choices(std::string_view field);
// Whether `a` asks for a rig at all, and the device Hamlib is given.
bool rig_enabled(const Args& a);
std::string rig_device(const Args& a);
// The one mapping from the options to Hamlib's (the GUI's test buttons use it
// too). `a` must have passed check().
rig::HamlibConfig hamlib_config(const Args& a);
// --list-rigs: Hamlib's models (number, manufacturer, model, status). False
// when built without Hamlib.
bool list_rigs();
void list_modes(int kiss_bw);

}  // namespace data2g::app
