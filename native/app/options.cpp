#include "options.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <functional>
#include <map>
#include <set>
#include <tuple>
#include <type_traits>
#include <utility>

#include "app/log.hpp"
#include "arq/modes.hpp"
#include "arq/policy.hpp"
#include "host/host.hpp"
#include "util/pool.hpp"

namespace data2g::app {

namespace {

// After "usage: PROG ", PROG padded to data2g-host's width so the columns line up.
const char* USAGE =
    "[-h] [--kiss-port KISS_PORT]\n"
    "                   [--kiss-address KISS_ADDRESS] [--kiss-busy-limit S] [--kiss-bw {2400,500}]\n"
    "                   [--broadcast-mode MODE] [--mycall MYCALL] [--host HOST] [--command-port COMMAND_PORT]\n"
    "                   [--list-audio-devices] [--input-device INPUT_DEVICE] [--output-device OUTPUT_DEVICE]\n"
    "                   [--sample-rate SAMPLE_RATE] [--output-volume OUTPUT_VOLUME] [--rigctld-host RIGCTLD_HOST]\n"
    "                   [--rigctld-port RIGCTLD_PORT] [--ptt-on-delay-ms PTT_ON_DELAY_MS]\n"
    "                   [--ptt-off-delay-ms PTT_OFF_DELAY_MS] [--tx-lead-ms TX_LEAD_MS]\n"
    "                   [--min-header-score MIN_HEADER_SCORE] [--buffer-credit BUFFER_CREDIT]\n"
    "                   [--record-dir RECORD_DIR] [--log-level LOG_LEVEL] [--stats-interval S] [--list-modes]\n"
    "                   [--noise-rule W]\n"
    "                   [--decode-worker | --no-decode-worker] [--audio-io pipe:IN,OUT] [--threads N]\n"
    "                   [--realtime | --no-realtime]\n"
    "                   [--live-constellation | --no-live-constellation]\n"
    "                   [--dd-budget S]\n"
    "                   [--rig | --no-rig] [--list-rigs] [--rig-model N] [--rig-device DEVICE] [--rig-baud BAUD]\n"
    "                   [--rig-data-bits {default,7,8}] [--rig-stop-bits {default,1,2}]\n"
    "                   [--rig-parity {default,none,odd,even}] [--rig-handshake {default,none,xonxoff,hardware}]\n"
    "                   [--rig-dtr {default,high,low}] [--rig-rts {default,high,low}]\n"
    "                   [--ptt-method {cat,dtr,rts,vox}] [--ptt-device DEVICE] [--ptt-audio {mic,data}]\n"
    "                   [--rig-mode {none,usb,pkt_usb}] [--rig-timeout-ms MS] [--rig-retries N]\n"
    "                   [--rig-poll-interval S] [--rig-debug | --no-rig-debug]\n";

const char* HELP =
    "\nData2G server: VARA-style ARQ sessions and KISS broadcast on one radio\n\n"
    "options:\n"
    "  -h, --help            show this help message and exit\n"
    "  --kiss-port KISS_PORT as VARA HF's (default 8100)\n"
    "  --kiss-address KISS_ADDRESS (default 127.0.0.1)\n"
    "  --kiss-busy-limit S   a KISS burst held this long by BUSY is sent anyway (default 60)\n"
    "  --kiss-bw {2400,500}  KISS bandwidth cap, Hz (default 2400)\n"
    "  --broadcast-mode MODE a broadcast port's transmit mode until BCAST MODE sets one\n"
    "                        (default: qpsk-r1/5, n10-qpsk-r1/5 with --kiss-bw 500)\n"
    "  --mycall MYCALL\n"
    "  --host HOST           (default 127.0.0.1)\n"
    "  --command-port COMMAND_PORT  data is on the next port (default 8300)\n"
    "  --list-audio-devices\n"
    "  --input-device INPUT_DEVICE   index or name substring (see --list-audio-devices)\n"
    "  --output-device OUTPUT_DEVICE index or name substring\n"
    "  --sample-rate SAMPLE_RATE     a multiple of 8000 (default 48000)\n"
    "  --output-volume OUTPUT_VOLUME dB; 0 puts a burst's peak at full scale (default 0)\n"
    "  --rigctld-host RIGCTLD_HOST   (default localhost)\n"
    "  --rigctld-port RIGCTLD_PORT   0: no PTT (default 4532)\n"
    "  --ptt-on-delay-ms PTT_ON_DELAY_MS   (default 100)\n"
    "  --ptt-off-delay-ms PTT_OFF_DELAY_MS (default 50)\n"
    "  --tx-lead-ms TX_LEAD_MS       TX audio queued ahead of the sound card: slack for a late audio step\n"
    "                                (default 100)\n"
    "  --min-header-score MIN_HEADER_SCORE (default 0.0)\n"
    "  --buffer-credit BUFFER_CREDIT bytes queued for the next burst that BUFFER leaves out, at most, so VARA\n"
    "                                clients that throttle on it (Pat) keep a whole burst queued; -1: the next\n"
    "                                burst's full capacity, 0: report every unacked byte (plain VARA) (default -1)\n"
    "  --record-dir RECORD_DIR       where every burst heard and sent is logged ('' turns it off)\n"
    "                                (default recordings/YYYYmmdd-HHMMSS)\n"
    "  --log-level LOG_LEVEL         (default INFO)\n"
    "  --stats-interval S            log a connection's throughput this often (0: only at disconnect)\n"
    "                                (default 60)\n"
    "  --list-modes                  modes within --kiss-bw, narrowest first\n"
    "  --noise-rule W                the gear shifter's noise rule: a mode whose band is noisier in the\n"
    "                                receiver's noise profile than the band last measured is predicted at\n"
    "                                a lower SNR; W weighs its often-loud moments (0: off) (default 1)\n"
    "\nadded in the C++ host:\n"
    "  --decode-worker, --no-decode-worker\n"
    "                        burst decode, DD and the session on their own thread, so preamble search and\n"
    "                        BUSY keep running during a decode (BUSY lag with a 1 s DD in flight: 0.02 s\n"
    "                        instead of 0.23 s). --no-decode-worker is host.py's one thread, deterministic.\n"
    "                        (default: on)\n"
    "  --audio-io pipe:IN,OUT  no sound card: raw float32 8 kHz mono read from IN and written to OUT (files\n"
    "                        or named pipes), both at real time, silence while not keyed. Two hosts cross-\n"
    "                        connect through two mkfifo pipes. --sample-rate and the devices are then unused.\n"
    "  --threads N           threads for decode and sync, the calling one included (default: min(4, cores / 2))\n"
    "  --realtime, --no-realtime\n"
    "                        real-time scheduling for the transmit and sound card threads (Linux: directly\n"
    "                        if the user's limits allow it, else through RealtimeKit), so a loaded machine\n"
    "                        doesn't leave holes in a transmission. The decode threads are left alone.\n"
    "                        (default: on)\n"
    "  --live-constellation, --no-live-constellation\n"
    "                        data2g-gui's constellation plot fills in as a burst arrives (a frame's FFTs\n"
    "                        per block on the receive thread), not only once it is decoded. Unused by\n"
    "                        data2g-host. (default: on)\n"
    "  --dd-budget S         seconds a burst's decode may spend re-estimating the channel from its own\n"
    "                        decoded codewords (DD), counted from its soft bits; the pass in progress is\n"
    "                        abandoned when it runs out. 0: none. Lower it on a slow machine (default 1)\n"
    "\nrig control (Hamlib, linked in; SSTVAE's settings):\n"
    "  --rig, --no-rig       rig control at all; --no-rig: no PTT (default: on)\n"
    "  --list-rigs           Hamlib's rig models: number, manufacturer, model, status\n"
    "  --rig-model N         Hamlib model (see --list-rigs); 2 is NET rigctl, a rigctld client (default 2)\n"
    "  --rig-device DEVICE   serial device (/dev/ttyUSB0, COM5) or host:port; for model 2, default\n"
    "                        RIGCTLD_HOST:RIGCTLD_PORT. --rigctld-host/--rigctld-port mean --rig-model 2 at\n"
    "                        that address (port 0: no rig), so neither goes with --rig-model or --rig-device\n"
    "  --rig-baud BAUD       serial speed; 0: the model's (default 0)\n"
    "  --rig-data-bits, --rig-stop-bits, --rig-parity, --rig-handshake\n"
    "                        serial line settings; default: the model's\n"
    "  --rig-dtr, --rig-rts {default,high,low}\n"
    "                        hold a control line for the session (an interface powered from it)\n"
    "  --ptt-method {cat,dtr,rts,vox}\n"
    "                        cat: a CAT command; dtr/rts: a serial control line; vox: never key (the rig\n"
    "                        keys on the audio) (default cat)\n"
    "  --ptt-device DEVICE   the port whose DTR/RTS keys, if not --rig-device's\n"
    "  --ptt-audio {mic,data}  the input CAT keying selects, on rigs with two (TS-480 and the like)\n"
    "                        (default mic)\n"
    "  --rig-mode {none,usb,pkt_usb}  set once the rig opens; none leaves it (default none)\n"
    "  --rig-timeout-ms MS   Hamlib's timeout per command (default 1000)\n"
    "  --rig-retries N       Hamlib's retries per command (default 1)\n"
    "  --rig-poll-interval S read the dial frequency this often; 0: key only (default 0)\n"
    "  --rig-debug, --no-rig-debug\n"
    "                        Hamlib's trace in the log, as 'hamlib:' lines (default: off)\n";

template <typename T>
T number(const std::string& opt, const std::string& v, const char* prog) {
    try {
        std::size_t used = 0;
        T out;
        if constexpr (std::is_same_v<T, int>) out = std::stoi(v, &used);
        else out = std::stod(v, &used);
        if (used == v.size()) return out;
    } catch (const std::exception&) {
    }
    usage_error("argument " + opt + ": invalid " + (std::is_same_v<T, int> ? "int" : "float") + " value: '" + v + "'", prog);
}

}  // namespace

void usage_error(const std::string& msg, const char* prog) {
    std::fprintf(stderr, "usage: %-11s %s%s: error: %s\n", prog, USAGE, prog, msg.c_str());
    std::exit(2);
}

std::string default_record_dir() {
    const std::time_t t = std::time(nullptr);
    const std::tm tm = local_tm(t);
    char buf[32];
    std::strftime(buf, sizeof buf, "%Y%m%d-%H%M%S", &tm);
    return std::string("recordings/") + buf;
}

Args parse(int argc, char** argv, Args a, const char* prog) {
    std::map<std::string, bool*> flags = {
        {"--list-audio-devices", &a.list_audio_devices}, {"--list-modes", &a.list_modes}, {"--list-rigs", &a.list_rigs}};
    std::map<std::string, std::pair<bool*, bool>> switches = {
        {"--decode-worker", {&a.decode_worker, true}}, {"--no-decode-worker", {&a.decode_worker, false}},
        {"--realtime", {&a.realtime, true}},   {"--no-realtime", {&a.realtime, false}},
        {"--live-constellation", {&a.live_constellation, true}},
        {"--no-live-constellation", {&a.live_constellation, false}},
        {"--rig", {&a.rig, true}},     {"--no-rig", {&a.rig, false}},
        {"--rig-debug", {&a.rig_debug, true}}, {"--no-rig-debug", {&a.rig_debug, false}}};
    const auto i_ = [prog](auto& o, auto& v) { return number<int>(o, v, prog); };
    const auto d_ = [prog](auto& o, auto& v) { return number<double>(o, v, prog); };
    std::map<std::string, std::function<void(const std::string&, const std::string&)>> valued = {
        {"--kiss-port", [&](auto& o, auto& v) { a.kiss_port = i_(o, v); }},
        {"--kiss-address", [&](auto&, auto& v) { a.kiss_address = v; }},
        {"--kiss-busy-limit", [&](auto& o, auto& v) { a.kiss_busy_limit = d_(o, v); }},
        {"--kiss-bw", [&](auto& o, auto& v) {
             a.kiss_bw = i_(o, v);
             if (a.kiss_bw != 2400 && a.kiss_bw != 500)
                 usage_error("argument --kiss-bw: invalid choice: " + v + " (choose from 2400, 500)", prog);
         }},
        {"--broadcast-mode", [&](auto&, auto& v) { a.broadcast_mode = v; }},
        {"--mycall", [&](auto&, auto& v) { a.mycall = v; }},
        {"--host", [&](auto&, auto& v) { a.host = v; }},
        {"--command-port", [&](auto& o, auto& v) { a.command_port = i_(o, v); }},
        {"--input-device", [&](auto&, auto& v) { a.input_device = v; }},
        {"--output-device", [&](auto&, auto& v) { a.output_device = v; }},
        {"--sample-rate", [&](auto& o, auto& v) { a.sample_rate = i_(o, v); }},
        {"--output-volume", [&](auto& o, auto& v) { a.output_volume = d_(o, v); }},
        {"--rigctld-host", [&](auto&, auto& v) { a.rigctld_host = v; }},
        {"--rigctld-port", [&](auto& o, auto& v) { a.rigctld_port = i_(o, v); }},
        {"--ptt-on-delay-ms", [&](auto& o, auto& v) { a.ptt_on_delay_ms = i_(o, v); }},
        {"--ptt-off-delay-ms", [&](auto& o, auto& v) { a.ptt_off_delay_ms = i_(o, v); }},
        {"--tx-lead-ms", [&](auto& o, auto& v) { a.tx_lead_ms = i_(o, v); }},
        {"--min-header-score", [&](auto& o, auto& v) { a.min_header_score = d_(o, v); }},
        {"--buffer-credit", [&](auto& o, auto& v) { a.buffer_credit = i_(o, v); }},
        {"--record-dir", [&](auto&, auto& v) { a.record_dir = v; }},
        {"--log-level", [&](auto&, auto& v) { a.log_level = v; }},
        {"--stats-interval", [&](auto& o, auto& v) { a.stats_interval = d_(o, v); }},
        {"--noise-rule", [&](auto& o, auto& v) { a.noise_rule = d_(o, v); }},
        {"--audio-io", [&](auto&, auto& v) { a.audio_io = v; }},
        {"--threads", [&](auto& o, auto& v) { pool::set_threads(i_(o, v)); }},
        {"--dd-budget", [&](auto& o, auto& v) { a.dd_budget = d_(o, v); }},
        {"--rig-model", [&](auto& o, auto& v) { a.rig_model = i_(o, v); }},
        {"--rig-device", [&](auto&, auto& v) { a.rig_device = v; }},
        {"--rig-baud", [&](auto& o, auto& v) { a.rig_baud = i_(o, v); }},
        {"--rig-data-bits", [&](auto&, auto& v) { a.rig_data_bits = v; }},
        {"--rig-stop-bits", [&](auto&, auto& v) { a.rig_stop_bits = v; }},
        {"--rig-parity", [&](auto&, auto& v) { a.rig_parity = v; }},
        {"--rig-handshake", [&](auto&, auto& v) { a.rig_handshake = v; }},
        {"--rig-dtr", [&](auto&, auto& v) { a.rig_dtr = v; }},
        {"--rig-rts", [&](auto&, auto& v) { a.rig_rts = v; }},
        {"--ptt-method", [&](auto&, auto& v) { a.ptt_method = v; }},
        {"--ptt-device", [&](auto&, auto& v) { a.ptt_device = v; }},
        {"--ptt-audio", [&](auto&, auto& v) { a.ptt_audio = v; }},
        {"--rig-mode", [&](auto&, auto& v) { a.rig_mode = v; }},
        {"--rig-timeout-ms", [&](auto& o, auto& v) { a.rig_timeout_ms = i_(o, v); }},
        {"--rig-retries", [&](auto& o, auto& v) { a.rig_retries = i_(o, v); }},
        {"--rig-poll-interval", [&](auto& o, auto& v) { a.rig_poll_interval = d_(o, v); }},
    };
    // The rig flags on this command line (value: the last one given).
    std::map<std::string, std::string> given;
    static const std::set<std::string> rig_flags = {"--rigctld-host", "--rigctld-port", "--rig-model", "--rig-device",
                                                    "--no-rig", "--ptt-device"};
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            std::printf("usage: %-11s %s%s", prog, USAGE, HELP);
            std::exit(0);
        }
        std::optional<std::string> inline_value;
        if (const auto eq = arg.find('='); arg.starts_with("--") && eq != std::string::npos) {
            inline_value = arg.substr(eq + 1);
            arg.resize(eq);
        }
        if (auto f = flags.find(arg); f != flags.end() && !inline_value) {
            *f->second = true;
        } else if (auto s = switches.find(arg); s != switches.end() && !inline_value) {
            *s->second.first = s->second.second;
        } else if (auto v = valued.find(arg); v != valued.end()) {
            if (!inline_value && i + 1 >= argc) usage_error("argument " + arg + ": expected one argument", prog);
            const std::string value = inline_value ? *inline_value : std::string(argv[++i]);
            v->second(arg, value);
            if (rig_flags.count(arg)) given[arg] = value;
            continue;
        } else {
            usage_error("unrecognized arguments: " + std::string(argv[i]), prog);
        }
        if (rig_flags.count(arg)) given[arg] = "";
        if (arg == "--rig") given.erase("--no-rig");
    }
    // The rig flags given here override saved settings as a whole: rigctld
    // means model 2 at that address, and a model or device turns the rig on.
    const bool rigctld = given.count("--rigctld-host") || given.count("--rigctld-port");
    if (rigctld && given.count("--rig-model") && a.rig_model != rig::MODEL_NET_RIGCTL)
        usage_error("--rigctld-host/--rigctld-port mean --rig-model 2 (NET rigctl), not --rig-model " + given["--rig-model"] +
                        " (give that model's address as --rig-device)",
                    prog);
    if (rigctld && given.count("--rig-device"))
        usage_error("--rigctld-host/--rigctld-port and --rig-device both give the rig's address: use one", prog);
    if (given.count("--no-rig") && (given.count("--rig-model") || given.count("--rig-device") || (rigctld && a.rigctld_port != 0)))
        usage_error("--no-rig with a rig to use (--rig-model, --rig-device or --rigctld-*)", prog);
    if (rigctld) {
        a.rig_model = rig::MODEL_NET_RIGCTL;
        a.rig_device.clear();
        a.rig = a.rigctld_port != 0;
    } else if (given.count("--rig-model") || given.count("--rig-device")) {
        a.rig = true;
    }
    if (given.count("--ptt-device") && a.ptt_method != "dtr" && a.ptt_method != "rts")
        usage_error("--ptt-device is the port whose DTR or RTS keys: it needs --ptt-method dtr or rts, not " + a.ptt_method, prog);
    return a;
}

int kiss_cap(int hz) { return hz == 500 ? 0 : 2; }

// In the order of the matching enums in rig/hamlib/hamlib.hpp: hamlib_config()
// maps a value by its index.
const std::vector<std::string>& rig_choices(std::string_view field) {
    static const std::map<std::string, std::vector<std::string>, std::less<>> choices = {
        {"rig_data_bits", {"default", "7", "8"}},
        {"rig_stop_bits", {"default", "1", "2"}},
        {"rig_parity", {"default", "none", "odd", "even"}},
        {"rig_handshake", {"default", "none", "xonxoff", "hardware"}},
        {"rig_dtr", {"default", "high", "low"}},
        {"rig_rts", {"default", "high", "low"}},
        {"ptt_method", {"vox", "cat", "dtr", "rts"}},
        {"ptt_audio", {"mic", "data"}},
        {"rig_mode", {"none", "usb", "pkt_usb"}},
    };
    return choices.find(field)->second;
}

bool rig_enabled(const Args& a) {
    return a.rig && !(a.rig_model == rig::MODEL_NET_RIGCTL && a.rig_device.empty() && a.rigctld_port == 0);
}

std::string rig_device(const Args& a) {
    if (a.rig_model == rig::MODEL_NET_RIGCTL && a.rig_device.empty()) return a.rigctld_host + ":" + std::to_string(a.rigctld_port);
    return a.rig_device;
}

namespace {

int choice_index(const std::string& field, const std::string& value) {
    const auto& c = rig_choices(field);
    return static_cast<int>(std::find(c.begin(), c.end(), value) - c.begin());
}

}  // namespace

rig::HamlibConfig hamlib_config(const Args& a) {
    rig::HamlibConfig h;
    h.model = a.rig_model;
    h.device = rig_device(a);
    h.baud = a.rig_baud;
    h.data_bits = static_cast<rig::DataBits>(choice_index("rig_data_bits", a.rig_data_bits));
    h.stop_bits = static_cast<rig::StopBits>(choice_index("rig_stop_bits", a.rig_stop_bits));
    h.parity = static_cast<rig::Parity>(choice_index("rig_parity", a.rig_parity));
    h.handshake = static_cast<rig::Handshake>(choice_index("rig_handshake", a.rig_handshake));
    h.dtr = static_cast<rig::LineState>(choice_index("rig_dtr", a.rig_dtr));
    h.rts = static_cast<rig::LineState>(choice_index("rig_rts", a.rig_rts));
    h.ptt_method = static_cast<rig::PttMethod>(choice_index("ptt_method", a.ptt_method));
    h.ptt_device = a.ptt_device;
    h.ptt_audio = static_cast<rig::PttAudio>(choice_index("ptt_audio", a.ptt_audio));
    h.mode = static_cast<rig::RigMode>(choice_index("rig_mode", a.rig_mode));
    h.timeout_ms = a.rig_timeout_ms;
    h.retries = a.rig_retries;
    return h;
}

bool list_rigs() {
#ifdef DATA2G_HAVE_RIG
    for (const auto& m : rig::list_models())
        std::printf("%6d  %-22s %-28s %s\n", m.model, m.manufacturer.c_str(), m.name.c_str(), m.status.c_str());
    return true;
#else
    return false;
#endif
}

std::optional<std::string> check(const Args& a) {
    // Saved settings come through here too, so every value is checked, not only flags.
    for (const auto& [flag, field, value] :
         {std::tuple{"--rig-data-bits", "rig_data_bits", &a.rig_data_bits}, {"--rig-stop-bits", "rig_stop_bits", &a.rig_stop_bits},
          {"--rig-parity", "rig_parity", &a.rig_parity}, {"--rig-handshake", "rig_handshake", &a.rig_handshake},
          {"--rig-dtr", "rig_dtr", &a.rig_dtr}, {"--rig-rts", "rig_rts", &a.rig_rts}, {"--ptt-method", "ptt_method", &a.ptt_method},
          {"--ptt-audio", "ptt_audio", &a.ptt_audio}, {"--rig-mode", "rig_mode", &a.rig_mode}}) {
        const auto& c = rig_choices(field);
        if (std::find(c.begin(), c.end(), *value) == c.end()) {
            std::string all;
            for (const auto& x : c) all += (all.empty() ? "" : ", ") + x;
            return std::string(flag) + ": invalid choice: '" + *value + "' (choose from " + all + ")";
        }
    }
    if (a.rig_baud < 0) return "--rig-baud: must be 0 (the model's) or more";
    if (a.rig_timeout_ms <= 0) return "--rig-timeout-ms: must be positive";
    if (a.rig_retries < 0) return "--rig-retries: must be 0 or more";
    if (a.rig_poll_interval < 0) return "--rig-poll-interval: must be 0 (key only) or more";
    if (!(a.noise_rule >= 0)) return "--noise-rule: must be 0 (off) or more";
    if (!(a.dd_budget >= 0)) return "--dd-budget: must be 0 (no DD) or more";
#ifdef DATA2G_HAVE_RIG
    if (rig_enabled(a) && !rig::model_info(a.rig_model))
        return "--rig-model " + std::to_string(a.rig_model) + ": not a model this Hamlib knows (see --list-rigs)";
#endif
    if (a.broadcast_mode) {
        const auto ok = arq::allowed(kiss_cap(a.kiss_bw));
        const auto* m = arq::mode(*a.broadcast_mode);
        if (!m || std::find(ok.begin(), ok.end(), m) == ok.end())
            return "--broadcast-mode " + *a.broadcast_mode + ": not a mode within " + std::to_string(a.kiss_bw) + " Hz";
    }
    return std::nullopt;
}

void list_modes(int kiss_bw) {
    for (const auto& l : host::mode_lines(kiss_cap(kiss_bw))) std::printf("%s\n", l.c_str());
}

}  // namespace data2g::app
