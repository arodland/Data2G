// The station's options, the rig (Hamlib) settings among them. parse() reads a command line over
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

struct Args {
    int kiss_port = 8100;
    double kiss_busy_limit = 60.0;
    int kiss_bw = 2400;
    std::optional<std::string> broadcast_mode, mycall, input_device, output_device;
    std::string listen = "127.0.0.1";  // the VARA and KISS ports' address
    int command_port = 8300;
    bool list_audio_devices = false, list_modes = false;
    int sample_rate = 48000;
    double output_volume = 0.0;
    // Hamlib (SSTVAE's rig settings). Enumerated values in SSTVAE's
    // lowercase spellings; "default": leave the backend's own. See
    // rig_choices().
    bool rig = true;  // false: no rig (so no PTT)
    int rig_model = 2;
    std::string rig_device;  // serial device or host:port; empty: Hamlib's (model 2: localhost:4532)
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
    bool record = false;                   // log every burst heard and sent
    std::string record_dir = "recordings";  // each run in a subdirectory of its own; see record_path()
    std::string log_level = "INFO";
    double stats_interval = 60.0;
    double noise_rule = 1.0;  // the gear shifter's noise rule, its tail weight; 0: off
    double dd_budget = 1.0;   // seconds of decision-directed decoding per burst (arq::DD_BUDGET_S); 0: none
    int decoder_effort = 25;  // the LDPC decoder's iteration cap (codes::set_ldpc_iters), >= 1
    std::string cw_id = "off";  // arq::Engine::cw_id: off, add, only
    double cw_wpm = 20.0;
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
// Where a run that starts now records: record_dir/YYYYmmdd-HHMMSS, with
// ".2", ".3", ... if that exists already (a second Recorder in one would
// number its rx_NNNNN.npz from 0 again). Empty when recording is off.
std::string record_path(const Args& a);
// The one mapping from the options to Hamlib's (the GUI's test buttons use it
// too). `a` must have passed check().
rig::HamlibConfig hamlib_config(const Args& a);
// --list-rigs: Hamlib's models (number, manufacturer, model, status). False
// when built without Hamlib.
bool list_rigs();
void list_modes(int kiss_bw);

}  // namespace data2g::app
