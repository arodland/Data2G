#include "station.hpp"

#include <QHostAddress>
#include <QHostInfo>
#include <QPointer>
#include <QTcpServer>
#include <QTcpSocket>

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstdarg>
#include <cstdio>
#include <ctime>
#include <functional>
#include <map>
#include <set>
#include <tuple>
#include <cstdlib>

#include "arq/engine.hpp"
#include "arq/modes.hpp"
#include "arq/policy.hpp"
#include "audio/audio.hpp"
#include "audio/fifo.hpp"
#include "audio/filters.hpp"
#include "audio/pipe.hpp"
#include "util/pool.hpp"
#include "generated/config.hpp"
#include "host/host.hpp"
#include "kisslink/kisslink.hpp"
#include "monitor/monitor.hpp"
#include "rig/controller.hpp"
#include "rig/ptt.hpp"
#include "tnc/tnc.hpp"

#ifdef DATA2G_HAVE_QTAUDIO
#include "audio/qt/qtaudio.hpp"
#endif
// Always: its types and constants need no libhamlib (only its functions,
// called under DATA2G_HAVE_RIG, do).
#include "rig/hamlib/hamlib.hpp"

namespace data2g::app {

// --- logging -----------------------------------------------------------------------------

std::atomic<int> g_level{INFO};

namespace {

std::mutex g_log_mu;

// localtime_r is POSIX; MSVC has localtime_s with the arguments swapped.
std::tm local_tm(std::time_t t) {
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    return tm;
}

const char* level_name(int level) {
    switch (level) {
        case DEBUG: return "DEBUG";
        case INFO: return "INFO";
        case WARNING: return "WARNING";
        case ERROR: return "ERROR";
        case CRITICAL: return "CRITICAL";
        default: return "Level";
    }
}

}  // namespace

void log_line(int level, const std::string& msg) {
    if (level < g_level) return;
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() % 1000;
    const std::tm tm = local_tm(t);
    char stamp[32];
    std::strftime(stamp, sizeof stamp, "%Y-%m-%d %H:%M:%S", &tm);
    std::lock_guard lock(g_log_mu);
    std::fprintf(stderr, "%s,%03d %s %s\n", stamp, static_cast<int>(ms), level_name(level), msg.c_str());
}

void logf(int level, const char* fmt, ...) {
    if (level < g_level) return;
    char buf[2048];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    log_line(level, buf);
}

std::optional<int> parse_level(const std::string& s) {
    static const std::map<std::string, int> names = {
        {"DEBUG", DEBUG}, {"INFO", INFO}, {"WARNING", WARNING}, {"WARN", WARNING}, {"ERROR", ERROR}, {"CRITICAL", CRITICAL}, {"FATAL", CRITICAL}};
    std::string u = s;
    for (auto& c : u) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    if (auto it = names.find(u); it != names.end()) return it->second;
    try {
        std::size_t used = 0;
        const int v = std::stoi(s, &used);
        if (used == s.size()) return v;
    } catch (const std::exception&) {
    }
    return std::nullopt;
}

void install_arq_log() {
    arq::set_log_sink({[](const char*, int level) { return level >= g_level; },
                       [](const char*, int level, const std::string& msg) { log_line(level, msg); }});
}

// --- the options -------------------------------------------------------------------------

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

// --- the servers -------------------------------------------------------------------------

// One TCP listener (host.py's _Port). With `multi` any number of clients may be connected: output goes to all of
// them and on_close fires when the last one leaves. Otherwise a new client replaces the old, whose close then
// isn't a loss.
struct Station::Port : QObject {
    Port(const QHostAddress& addr, quint16 port, bool lines, std::function<void(const QByteArray&, std::uint64_t)> on_input,
         std::function<void()> on_close = {}, bool multi = false)
        : lines_(lines), multi_(multi), on_input_(std::move(on_input)), on_close_(std::move(on_close)) {
        if (!srv_.listen(addr, port))
            throw std::runtime_error("can't listen on " + addr.toString().toStdString() + ":" + std::to_string(port) + ": " +
                                     srv_.errorString().toStdString());
        connect(&srv_, &QTcpServer::newConnection, this, [this] { accept(); });
    }
    // Its sockets are srv_'s children, closed after the members their
    // handlers use are gone: unhook them first.
    ~Port() override {
        for (auto* c : srv_.findChildren<QTcpSocket*>()) c->disconnect(this);
    }

    // To every client, or to one by id (nowhere if it has gone).
    void send(const QByteArray& data, std::uint64_t to = 0) {
        if (data.isEmpty()) return;
        for (auto& [c, cl] : clients_)
            if (to == 0 || cl.id == to) c->write(data);
    }

private:
    void accept() {
        while (QTcpSocket* c = srv_.nextPendingConnection()) {
            logf(INFO, "client %s:%d on port %d", c->peerAddress().toString().toStdString().c_str(), c->peerPort(), srv_.serverPort());
            if (!multi_) {
                // the new client first: the old one's close isn't a loss
                auto old = std::move(clients_);
                clients_.clear();
                for (auto& [o, _] : old) {
                    o->disconnect(this);
                    o->close();
                    o->deleteLater();
                }
            }
            clients_[c].id = ++last_id_;
            connect(c, &QTcpSocket::readyRead, this, [this, c] { read(c); });
            connect(c, &QTcpSocket::disconnected, this, [this, c] {
                c->deleteLater();
                if (clients_.erase(c) && clients_.empty() && on_close_) on_close_();
            });
        }
    }

    void read(QTcpSocket* c) {
        const QByteArray d = c->readAll();
        Client& cl = clients_[c];
        if (!lines_) {
            on_input_(d, cl.id);
            return;
        }
        QByteArray& buf = cl.buf;
        buf += d;
        while (true) {
            const qsizetype r = buf.indexOf('\r'), n = buf.indexOf('\n');
            const qsizetype i = r < 0 ? n : (n < 0 ? r : std::min(r, n));
            if (i < 0) break;
            const QByteArray line = buf.left(i);
            buf.remove(0, i + 1);
            if (!line.trimmed().isEmpty()) on_input_(line, cl.id);
        }
    }

    QTcpServer srv_;
    struct Client {
        std::uint64_t id = 0;
        QByteArray buf;  // its partial line
    };
    std::map<QTcpSocket*, Client> clients_;
    std::uint64_t last_id_ = 0;
    bool lines_, multi_;
    std::function<void(const QByteArray&, std::uint64_t)> on_input_;
    std::function<void()> on_close_;
};

// tnc.py's KissServer: any number of clients; data and ACKMODE frames to
// on_packet (port, frame, ack id), other KISS commands to on_command; frames
// heard go to every client, an ack to the client that asked for it.
struct Station::KissServer : QObject {
    KissServer(const QHostAddress& addr, quint16 port, std::function<void(int, tnc::Bytes, std::optional<std::int64_t>)> on_packet,
               std::function<void(int, tnc::Bytes)> on_command)
        : on_packet_(std::move(on_packet)), on_command_(std::move(on_command)) {
        if (!srv_.listen(addr, port))
            throw std::runtime_error("can't listen on " + addr.toString().toStdString() + ":" + std::to_string(port) + ": " +
                                     srv_.errorString().toStdString());
        connect(&srv_, &QTcpServer::newConnection, this, [this] {
            while (QTcpSocket* c = srv_.nextPendingConnection()) {
                const std::string peer = c->peerAddress().toString().toStdString() + ":" + std::to_string(c->peerPort());
                clients_[c] = {};
                logf(INFO, "KISS client %s connected", peer.c_str());
                connect(c, &QTcpSocket::readyRead, this, [this, c] {
                    const QByteArray d = c->readAll();
                    auto it = clients_.find(c);
                    if (it == clients_.end()) return;
                    const auto* p = reinterpret_cast<const std::uint8_t*>(d.constData());
                    for (auto& [cmd, payload] : it->second.feed({p, static_cast<std::size_t>(d.size())})) {
                        if ((cmd & 0x0F) == tnc::KISS_DATA) {
                            on_packet_(cmd >> 4, std::move(payload), std::nullopt);
                        } else if ((cmd & 0x0F) == tnc::KISS_ACKMODE && payload.size() >= 2) {  // [tag, 2][frame]
                            const std::int64_t id = ++n_acks_;
                            acks_[id] = {c, tnc::Bytes(payload.begin(), payload.begin() + 2)};
                            on_packet_(cmd >> 4, tnc::Bytes(payload.begin() + 2, payload.end()), id);
                        } else {
                            on_command_(cmd & 0x0F, std::move(payload));
                        }
                    }
                });
                connect(c, &QTcpSocket::disconnected, this, [this, c, peer] {
                    clients_.erase(c);
                    std::erase_if(acks_, [c](const auto& kv) { return kv.second.first == c; });
                    c->deleteLater();
                    logf(INFO, "KISS client %s disconnected", peer.c_str());
                });
            }
        });
    }

    ~KissServer() override {  // as ~Port
        for (auto* c : srv_.findChildren<QTcpSocket*>()) c->disconnect(this);
    }

    void broadcast(const tnc::Bytes& data, int port) {
        const auto f = tnc::kiss_encode(data, port);
        const QByteArray b(reinterpret_cast<const char*>(f.data()), static_cast<qsizetype>(f.size()));
        for (auto& [c, _] : clients_) c->write(b);
    }
    // An ACKMODE frame went out: its tag back to the client that sent it.
    void send_ack(std::int64_t id, int port) {
        const auto it = acks_.find(id);
        if (it == acks_.end()) return;  // its client has gone
        const auto f = tnc::kiss_encode(it->second.second, port, tnc::KISS_ACKMODE);
        if (clients_.count(it->second.first))
            it->second.first->write(QByteArray(reinterpret_cast<const char*>(f.data()), static_cast<qsizetype>(f.size())));
        acks_.erase(it);
    }
    void close_clients() {
        // disconnectFromHost() can emit disconnected synchronously, which erases from clients_
        std::vector<QTcpSocket*> cs;
        for (auto& [c, _] : clients_) cs.push_back(c);
        for (auto* c : cs) c->disconnectFromHost();
    }

private:
    QTcpServer srv_;
    std::map<QTcpSocket*, tnc::KissDecoder> clients_;
    std::map<std::int64_t, std::pair<QTcpSocket*, tnc::Bytes>> acks_;  // ACKMODE id -> (client, tag)
    std::int64_t n_acks_ = 0;
    std::function<void(int, tnc::Bytes, std::optional<std::int64_t>)> on_packet_;
    std::function<void(int, tnc::Bytes)> on_command_;
};

#ifdef DATA2G_HAVE_QTAUDIO
struct Station::SoundCard {
    std::unique_ptr<audio::qt::Capture> cap;
    std::unique_ptr<audio::qt::Playback> play;
};
#else
struct Station::SoundCard {};
#endif

namespace {

QHostAddress resolve(const std::string& name) {
    QHostAddress a;
    if (a.setAddress(QString::fromStdString(name))) return a;
    const QHostInfo info = QHostInfo::fromName(QString::fromStdString(name));
    for (const QHostAddress& x : info.addresses())
        if (x.protocol() == QAbstractSocket::IPv4Protocol) return x;
    if (!info.addresses().isEmpty()) return info.addresses().first();
    throw std::runtime_error("can't resolve " + name);
}

constexpr std::size_t MAX_BURSTS = 256;  // kept for a front end that polls slowly (or never: data2g-host)
constexpr std::size_t TAP_SAMPLES = 4096;

}  // namespace

// The monitor window's decoder: bursts from the session stage, dumped on a
// thread of its own so its decodes never delay the session's.
struct Station::MonitorThread {
    monitor::Monitor m;
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::pair<arq::BurstHeard, std::string>> in;
    std::deque<monitor::Dump> out;
    bool stop = false;
    std::thread thread{[this] { run(); }};  // last: everything above is ready

    ~MonitorThread() {
        {
            std::lock_guard lock(mu);
            stop = true;
        }
        cv.notify_one();
        thread.join();
    }
    void push(const arq::BurstHeard& b) {
        const auto now = std::chrono::system_clock::now();
        const std::time_t t = std::chrono::system_clock::to_time_t(now);
        const std::tm tm = local_tm(t);
        char when[16];
        std::strftime(when, sizeof when, "%H:%M:%S", &tm);
        std::lock_guard lock(mu);
        if (in.size() >= MAX_BURSTS) return;  // ponytail: a decoder this far behind loses bursts (shown as gaps)
        in.emplace_back(b, when);
        cv.notify_one();
    }
    void run() {
        std::unique_lock lock(mu);
        while (true) {
            cv.wait(lock, [this] { return stop || !in.empty(); });
            if (stop) return;
            auto [b, when] = std::move(in.front());
            in.pop_front();
            lock.unlock();
            monitor::Dump d = m.burst(b, when);
            lock.lock();
            out.push_back(std::move(d));
            if (out.size() > MAX_BURSTS) out.pop_front();
        }
    }
};

// What the session stage hands the owner after a command or a block.
struct Station::Outbox {
    std::vector<std::pair<std::string, std::uint64_t>> cmd;  // (line, client it is for; 0: all)
    arq::Bytes data;
    std::vector<std::pair<int, arq::Bytes>> kiss;  // (port, frame) heard
    std::vector<std::pair<int, std::int64_t>> acks;  // (port, ACKMODE id) gone out
};

// --- the station -------------------------------------------------------------------------

Station::Station(Args a) : a_(std::move(a)), rate_(config::FS), tap_(TAP_SAMPLES, 0.0) {}

Station::~Station() { stop(); }

// `asker`: the client whose command produced what the host has queued. That is its reply and goes to it alone,
// except DISCONNECTED (ABORT's), which is news for every client. 0: everything is for all.
void Station::flush(std::uint64_t asker) {
    Outbox o;
    for (auto& line : host_->out_cmd) {
        const auto to = line == "DISCONNECTED" ? 0 : asker;
        o.cmd.emplace_back(std::move(line), to);
    }
    host_->out_cmd.clear();
    o.data.swap(host_->out_data);
    o.kiss.swap(engine_->kiss_rx());
    o.acks.swap(link_->acks);
    if (o.cmd.empty() && o.data.empty() && o.kiss.empty() && o.acks.empty()) return;
    QMetaObject::invokeMethod(&ctx_, [this, o = std::move(o)] { deliver(o); }, Qt::QueuedConnection);
}

void Station::note_link() {
    auto& s = engine_->session();
    LinkStatus l;
    l.state = arq::state_name(s.state);
    l.peer = s.peer;
    l.cap = s.cap;
    if (s.station) {
        const auto st = s.stats();
        l.tx_bytes = st.at("tx_bytes");
        l.rx_bytes = st.at("rx_bytes");
    }
    if (const auto& tx = engine_->tx()) mode_ = tx->burst->submode;
    l.mode = mode_;
    std::lock_guard lock(status_mu_);
    link_status_ = std::move(l);
}

void Station::deliver(const Outbox& o) {
    if (cmd_)
        for (const auto& [line, to] : o.cmd) cmd_->send(QByteArray::fromStdString(line + "\r"), to);
    if (data_ && !o.data.empty())
        data_->send(QByteArray(reinterpret_cast<const char*>(o.data.data()), static_cast<qsizetype>(o.data.size())));
    if (kiss_) {
        for (const auto& [port, f] : o.kiss) kiss_->broadcast(f, port);
        for (const auto& [port, id] : o.acks) kiss_->send_ack(id, port);
    }
}

void Station::watch_counters() {
    if (const auto n = cap_->overflows(); n != overflows_) {
        overflows_ = n;
        logf(WARNING, "RX audio overflow (%llu)", static_cast<unsigned long long>(n));
    }
    if (const auto n = play_->underruns(); n != underruns_) {
        underruns_ = n;
        logf(WARNING, "TX audio underrun (%llu)", static_cast<unsigned long long>(n));
    }
    if (const auto n = cap_->late_events(); n != late_) {
        late_ = n;
        logf(WARNING, "RX audio %.1f s behind the card", cap_->backlog_s());
    }
    if (const auto n = cap_->dropped(); n != dropped_) {
        logf(WARNING, "RX audio: %llu samples dropped, the capture FIFO full", static_cast<unsigned long long>(n - dropped_));
        dropped_ = n;
    }
}

void Station::tap(std::span<const double> x, bool tx) {
    std::lock_guard lock(tap_mu_);
    for (double v : x) tap_[tapped_++ % tap_.size()] = v;
    tap_tx_ = tx;
}

std::vector<double> Station::input_tail(std::size_t n, std::uint64_t* total, bool* tx) const {
    std::lock_guard lock(tap_mu_);
    n = std::min({n, tap_.size(), static_cast<std::size_t>(tapped_)});
    std::vector<double> out(n);
    for (std::size_t i = 0; i < n; ++i) out[i] = tap_[(tapped_ - n + i) % tap_.size()];
    if (total) *total = tapped_;
    if (tx) *tx = tap_tx_;
    return out;
}

LinkStatus Station::link() const {
    std::lock_guard lock(status_mu_);
    return link_status_;
}

std::vector<monitor::Dump> Station::take_dumps() {
    if (!monitor_) return {};
    std::lock_guard lock(monitor_->mu);
    std::vector<monitor::Dump> out(std::make_move_iterator(monitor_->out.begin()), std::make_move_iterator(monitor_->out.end()));
    monitor_->out.clear();
    return out;
}

std::vector<BurstLogEntry> Station::take_bursts() {
    std::lock_guard lock(status_mu_);
    std::vector<BurstLogEntry> out(std::make_move_iterator(bursts_.begin()), std::make_move_iterator(bursts_.end()));
    bursts_.clear();
    return out;
}

bool Station::busy() const { return engine_ && engine_->channel_busy(); }

AudioCounters Station::counters() const {
    if (!cap_ || !play_) return {};
    return {cap_->overflows(), play_->underruns(), cap_->late_events(), cap_->dropped(), cap_->backlog_s(),
            engine_ ? engine_->decode_dropped() : 0};
}

void Station::engine_loop() {
    const std::size_t block = config::FS / 10;
    audio::Interpolator interp(rate_);
    const double gain = std::pow(10.0, a_.output_volume / 20);
    int slow = 0;
    try {
        while (!stop_) {
            auto x = cap_->read(block);
            if (!x) break;
            const std::vector<double> heard = *x;
            if (keyer_->keyed()) std::fill(x->begin(), x->end(), 0.0);
            const auto t0 = std::chrono::steady_clock::now();
            auto out = engine_->step(*x);
            // the waterfall shows what we send while we send it (the radio's
            // RX audio is muted then), what we hear otherwise
            if (out.ptt && out.audio.size() == heard.size()) tap(out.audio, true);
            else tap(heard, false);
            const double took = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            if (took > static_cast<double>(block) / config::FS) {
                ++slow;
                logf(DEBUG, "step took %.2f s (%d slow)", took, slow);
            }
            if (out.ptt && !keyer_->keyed()) keyer_->key();
            ptt_ = keyer_->keyed();
            if (keyer_->keyed()) {
                auto y = interp(out.audio);
                for (auto& v : y) v = std::clamp(v * gain, -1.0, 1.0);
                play_->write(y);
            }
            if (keyer_->keyed() && !out.ptt) keyer_->unkey();
            ptt_ = keyer_->keyed();
            watch_counters();
        }
    } catch (const std::exception& e) {
        logf(CRITICAL, "engine thread: %s", e.what());
        failed_ = true;
        try {
            if (keyer_) keyer_->unkey();  // never leave the radio transmitting
        } catch (...) {
        }
    }
}

std::optional<double> Station::rig_frequency() const { return rig_ ? rig_->frequency_hz() : std::nullopt; }

void Station::start() {
    if (running_) return;
    try {
        stop_ = false;
        failed_ = false;
        link_ = std::make_unique<kisslink::KissLink>(kiss_cap(a_.kiss_bw), a_.broadcast_mode.value_or(""));
        link_->busy_limit_s = a_.kiss_busy_limit;
        monitor_ = std::make_unique<MonitorThread>();
        arq::EngineConfig cfg;
        cfg.ptt_delay_s = a_.ptt_on_delay_ms / 1000.0;
        cfg.record_dir = a_.record_dir;
        cfg.min_header_score = a_.min_header_score;
        cfg.kiss = link_.get();
        cfg.stats_interval_s = a_.stats_interval;
        cfg.noise_rule = a_.noise_rule;
        cfg.worker = a_.decode_worker;
        engine_ = std::make_unique<arq::Engine>(a_.mycall.value_or("NOCALL"), cfg);
        host_ = std::make_unique<host::Host>(*engine_, a_.buffer_credit < 0 ? std::nullopt : std::optional<int>(a_.buffer_credit));
        engine_->set_after_block([this](bool ptt) {
            host_->after_step(ptt);
            note_link();
            flush();
        });
        engine_->set_on_burst([this](const arq::BurstHeard& b) {
            mode_ = b.submode;
            std::lock_guard lock(status_mu_);
            bursts_.push_back({std::chrono::system_clock::now(), b.submode, b.n_cw, b.lost, b.snr_db});
            if (bursts_.size() > MAX_BURSTS) bursts_.pop_front();
            if (monitor_on_) monitor_->push(b);
        });
        auto post = [this](std::function<void()> f) {
            engine_->post([this, f = std::move(f)] {
                f();
                flush();  // replies at once, not after the next block (clients time out at ~2 s)
            });
        };

        {
            const QHostAddress addr = resolve(a_.host);
            cmd_ = std::make_unique<Port>(
                addr, a_.command_port, true,
                [this, post](const QByteArray& line, std::uint64_t client) {
                    std::string text;  // ascii, as host.py decodes it (others replaced)
                    for (unsigned char c : line) text += c < 128 ? static_cast<char>(c) : '?';
                    logf(INFO, "command: %s", text.c_str());
                    post([this, text, client] {
                        flush();  // what was already pending goes first, to everyone
                        host_->command(text);
                        flush(client);  // the reply goes to the client that asked
                    });
                },
                [post, this] { post([this] { host_->client_gone(); }); }, true);
            data_ = std::make_unique<Port>(addr, a_.command_port + 1, false, [this, post](const QByteArray& d, std::uint64_t) {
                arq::Bytes b(d.begin(), d.end());
                post([this, b = std::move(b)] { host_->data_in(b); });
            });
        }
        kiss_ = std::make_unique<KissServer>(
            resolve(a_.kiss_address), a_.kiss_port,
            [this, post](int port, tnc::Bytes f, std::optional<std::int64_t> ack) {
                post([this, port, ack, f = std::move(f)]() mutable { link_->enqueue(std::move(f), port, ack); });
            },
            [this, post](int cmd, tnc::Bytes p) { post([this, cmd, p = std::move(p)] { link_->command(cmd, p); }); });

        // PTT
        rig::Keyer::Ptt ptt;
        rig::Keyer::MustRelease must_release;
        if (rig_enabled(a_)) {
#ifdef DATA2G_HAVE_RIG
            if (a_.rig_debug) rig::set_debug_sink([](const std::string& line) { log_line(INFO, "hamlib: " + line); });
            const bool polling = a_.rig_poll_interval > 0;
            rig_ = std::make_unique<rig::RigController>(nullptr, [polling](const std::string& text, bool error) {
                // a healthy poll's readout ("Rig: 14.1000 MHz") is shown, not logged
                if (error || !polling || text.find(" MHz") == std::string::npos) log_line(error ? ERROR : INFO, "rig: " + text);
            });
            const rig::HamlibConfig hc = hamlib_config(a_);
            rig::RigConfig rc;
            rc.poll_interval_s = a_.rig_poll_interval;  // 0: key only, as tnc.Rigctld
            rig_->start(rig::make_hamlib_backend(hc), rc);
            rig::RigController* r = rig_.get();
            if (hc.ptt_method != rig::PttMethod::Vox) {
                // no polls while keyed; resumed after the off, so none delays it
                ptt = [r](bool on) {
                    if (on) {
                        r->pause_polling();
                        r->set_ptt(true);
                        return;
                    }
                    try {
                        r->set_ptt(false);
                    } catch (...) {
                        r->resume_polling();
                        throw;
                    }
                    r->resume_polling();
                };
                must_release = [r] { return r->keyed_since_open(); };
            }
#else
            log_line(WARNING, "built without Hamlib: no PTT (rig settings ignored)");
#endif
        }

        // audio
        const double lead_s = a_.tx_lead_ms / 1000.0;
        cap_ = std::make_unique<audio::CaptureFifo>(config::FS);
        const auto report = [](const std::string& s) { log_line(ERROR, s); };
        if (!a_.audio_io.empty()) {
            const auto comma = a_.audio_io.find(',');
            if (!a_.audio_io.starts_with("pipe:") || comma == std::string::npos) throw UsageError("--audio-io: want pipe:IN,OUT");
            play_ = std::make_unique<audio::PlaybackFifo>(config::FS, lead_s);
            const std::string in = a_.audio_io.substr(5, comma - 5), out = a_.audio_io.substr(comma + 1);
            pipe_ = std::make_unique<audio::PipeIo>(in, out, *cap_, *play_, report);
            logf(INFO, "audio: in %s, out %s (float32 at %d Hz)", in.c_str(), out.c_str(), config::FS);
        } else {
#ifdef DATA2G_HAVE_QTAUDIO
            rate_ = a_.sample_rate;
            if (rate_ <= 0 || rate_ % config::FS)
                throw UsageError("--sample-rate must be a multiple of " + std::to_string(config::FS));
            play_ = std::make_unique<audio::PlaybackFifo>(rate_, lead_s);
            const auto in = audio::select_device(audio::qt::input_devices(), a_.input_device.value_or(""), "input");
            const auto out = audio::select_device(audio::qt::output_devices(), a_.output_device.value_or(""), "output");
            card_ = std::make_unique<SoundCard>();
            card_->cap = std::make_unique<audio::qt::Capture>(in, rate_, *cap_, report);
            card_->play = std::make_unique<audio::qt::Playback>(out, rate_, *play_, report);
            logf(INFO, "audio: in %s (%d ch), out %s (%d ch) at %d Hz", card_->cap->device_name().c_str(), card_->cap->channels(),
                 card_->play->device_name().c_str(), card_->play->channels(), rate_);
#else
            throw UsageError("built without Qt Multimedia: only --audio-io pipe:IN,OUT");
#endif
        }
        keyer_ = std::make_unique<rig::Keyer>(ptt, *play_, a_.ptt_off_delay_ms / 1000.0,
                                              [](const std::string& s) { log_line(ERROR, s); }, must_release);

        logf(INFO, "commands on %s:%d, data on %d", a_.host.c_str(), a_.command_port, a_.command_port + 1);
        logf(INFO, "KISS on %s:%d: %d Hz cap, broadcasts in %s", a_.kiss_address.c_str(), a_.kiss_port, a_.kiss_bw,
             link_->broadcast.c_str());
        logf(INFO, "recording to %s", a_.record_dir.empty() ? "(off)" : a_.record_dir.c_str());
        logf(INFO, "decode worker %s", a_.decode_worker ? "on" : "off");

        running_ = true;
        engine_thread_ = std::thread(&Station::engine_loop, this);
    } catch (...) {
        running_ = true;  // so stop() undoes whatever got started
        stop();
        throw;
    }
}

void Station::stop() {
    if (!running_) return;
    running_ = false;
    stop_ = true;
    if (cap_) cap_->close();
    if (engine_thread_.joinable()) engine_thread_.join();
    if (engine_) engine_->stop();
    monitor_.reset();  // after the engine: nothing pushes to it now
    keyer_.reset();  // PTT off, if the rig was ever keyed
    ptt_ = false;
    if (rig_) {
        rig_->stop();
        rig_->wait_for_shutdown(2.0);
#ifdef DATA2G_HAVE_RIG
        if (a_.rig_debug) rig::set_debug_sink({});
#endif
    }
    if (kiss_) kiss_->close_clients();
    if (pipe_) pipe_->stop();
#ifdef DATA2G_HAVE_QTAUDIO
    if (card_) {
        if (card_->cap) card_->cap->stop();
        if (card_->play) card_->play->stop();
    }
#endif
    // released in reverse order of use; the ports close here, so a restart can listen again
    cmd_.reset();
    data_.reset();
    kiss_.reset();
    card_.reset();
    pipe_.reset();
    rig_.reset();
    host_.reset();
    engine_.reset();
    link_.reset();
    play_.reset();
    cap_.reset();
    rate_ = config::FS;
    overflows_ = underruns_ = late_ = dropped_ = 0;
}

}  // namespace data2g::app
