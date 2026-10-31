#include "options.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <functional>
#include <map>
#include <set>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>

#include "app/log.hpp"
#ifdef DATA2G_HAVE_RIG
#include "rig/hamlib/hamlib.hpp"  // list_models, model_info
#endif
#include "arq/modes.hpp"
#include "host/host.hpp"
#include "kisslink/kisslink.hpp"
#include "util/pool.hpp"

namespace data2g::app {

namespace {

// After "usage: PROG ", PROG padded to data2g-host's width so the columns line up.
const char* USAGE =
    "[-h] [--mycall MYCALL] [--listen ADDRESS] [--command-port COMMAND_PORT] [--kiss-port KISS_PORT]\n"
    "                   [--kiss-busy-limit S] [--kiss-bw {2400,500}] [--broadcast-mode MODE]\n"
    "                   [--list-audio-devices] [--input-device INPUT_DEVICE] [--output-device OUTPUT_DEVICE]\n"
    "                   [--sample-rate SAMPLE_RATE] [--output-volume OUTPUT_VOLUME] [--ptt-on-delay-ms PTT_ON_DELAY_MS]\n"
    "                   [--ptt-off-delay-ms PTT_OFF_DELAY_MS] [--tx-lead-ms TX_LEAD_MS]\n"
    "                   [--min-header-score MIN_HEADER_SCORE] [--buffer-credit BUFFER_CREDIT]\n"
    "                   [--record | --no-record] [--record-dir RECORD_DIR] [--log-level LOG_LEVEL]\n"
    "                   [--stats-interval S] [--list-modes] [--noise-rule W]\n"
    "                   [--decode-worker | --no-decode-worker] [--audio-io pipe:IN,OUT]\n"
    "                   [--tci | --no-tci] [--tci-host HOST[:PORT]] [--threads N] [--realtime | --no-realtime]\n"
    "                   [--live-constellation | --no-live-constellation]\n"
    "                   [--dd-budget S] [--decoder-effort N] [--cw-id {off,add,only}] [--cw-wpm WPM]\n"
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
    "  --mycall MYCALL\n"
    "  --listen ADDRESS      the VARA and KISS ports' address (default 127.0.0.1)\n"
    "  --command-port COMMAND_PORT  data is on the next port (default 8300)\n"
    "  --kiss-port KISS_PORT KISS over TCP, as VARA HF's (default 8100)\n"
    "  --kiss-busy-limit S   a KISS burst held this long by BUSY is sent anyway (default 60)\n"
    "  --kiss-bw {2400,500}  KISS bandwidth cap, Hz (default 2400)\n"
    "  --broadcast-mode MODE a broadcast port's transmit mode until BCAST MODE sets one\n"
    "                        (default: qpsk-r1/5, n10-qpsk-r1/5 with --kiss-bw 500)\n"
    "  --list-audio-devices\n"
    "  --input-device INPUT_DEVICE   index or name substring (see --list-audio-devices)\n"
    "  --output-device OUTPUT_DEVICE index or name substring\n"
    "  --sample-rate SAMPLE_RATE     a multiple of 8000 (default 48000)\n"
    "  --output-volume OUTPUT_VOLUME dB; 0 puts a burst's peak at full scale (default 0)\n"
    "  --ptt-on-delay-ms PTT_ON_DELAY_MS   (default 100)\n"
    "  --ptt-off-delay-ms PTT_OFF_DELAY_MS (default 50)\n"
    "  --tx-lead-ms TX_LEAD_MS       TX audio queued ahead of the sound card: slack for a late audio step\n"
    "                                (default 100)\n"
    "  --min-header-score MIN_HEADER_SCORE (default 0.0)\n"
    "  --buffer-credit BUFFER_CREDIT bytes queued for the next burst that BUFFER leaves out, at most, so VARA\n"
    "                                clients that throttle on it (Pat) keep a whole burst queued; -1: the next\n"
    "                                burst's full capacity, 0: report every unacked byte (plain VARA) (default -1)\n"
    "  --record, --no-record         log every burst heard and sent (default: off)\n"
    "  --record-dir RECORD_DIR       where: each run in its own RECORD_DIR/YYYYmmdd-HHMMSS; implies --record\n"
    "                                (default recordings)\n"
    "  --log-level LOG_LEVEL         (default INFO)\n"
    "  --stats-interval S            log a connection's throughput this often (0: only at disconnect)\n"
    "                                (default 60)\n"
    "  --list-modes                  modes within --kiss-bw, narrowest first\n"
    "  --noise-rule W                the gear shifter's noise rule: a mode whose band is noisier in the\n"
    "                                receiver's noise profile than the band last measured is predicted at\n"
    "                                a lower SNR; W weighs its often-loud moments (0: off) (default 1)\n"
    "  --decode-worker, --no-decode-worker\n"
    "                        burst decode, DD and the session on their own thread, so preamble search and\n"
    "                        BUSY keep running during a decode (BUSY lag with a 1 s DD in flight: 0.02 s\n"
    "                        instead of 0.23 s). --no-decode-worker is host.py's one thread, deterministic.\n"
    "                        (default: on)\n"
    "  --audio-io pipe:IN,OUT  no sound card: raw float32 8 kHz mono read from IN and written to OUT (files\n"
    "                        or named pipes), both at real time, silence while not keyed. Two hosts cross-\n"
    "                        connect through two mkfifo pipes. --sample-rate and the devices are then unused.\n"
    "  --tci, --no-tci       the radio over TCI (ExpertSDR, Thetis, AetherSDR...): audio, PTT and the dial,\n"
    "                        in place of the sound card and Hamlib. --sample-rate 8000, 24000 or 48000;\n"
    "                        --rig-mode usb or pkt_usb sets USB or DIGU on connect (default: off)\n"
    "  --tci-host HOST[:PORT]  the TCI server (default localhost:50001)\n"
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
    "  --decoder-effort N    the LDPC decoder's iteration cap (data2g-gui offers 15-40). More decodes a little\n"
    "                        further into a fade (about 0.1 dB from 20 to 25) and spends longer on a burst\n"
    "                        that fails. Lower it on a slow machine (default 25)\n"
    "\nrig control (Hamlib, linked in):\n"
    "  --rig, --no-rig       rig control at all; --no-rig: no PTT (default: on)\n"
    "  --list-rigs           Hamlib's rig models: number, manufacturer, model, status\n"
    "  --rig-model N         Hamlib model (see --list-rigs); 2 is NET rigctl, a rigctld client (default 2)\n"
    "  --rig-device DEVICE   serial device (/dev/ttyUSB0, COM5) or host:port; default: Hamlib's\n"
    "                        (for model 2, rigctld at localhost:4532)\n"
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

std::string record_path(const Args& a) {
    if (!a.record) return "";
    const std::tm tm = local_tm(std::time(nullptr));
    char buf[32];
    std::strftime(buf, sizeof buf, "%Y%m%d-%H%M%S", &tm);
    const std::filesystem::path base = std::filesystem::path(a.record_dir) / buf;
    std::filesystem::path p = base;
    for (int n = 2; std::filesystem::exists(p); ++n) p = base.string() + "." + std::to_string(n);
    return p.string();
}

Args parse(int argc, char** argv, Args a, const char* prog) {
    std::map<std::string, bool*> flags = {
        {"--list-audio-devices", &a.list_audio_devices}, {"--list-modes", &a.list_modes}, {"--list-rigs", &a.list_rigs}};
    std::map<std::string, std::pair<bool*, bool>> switches = {
        {"--decode-worker", {&a.decode_worker, true}}, {"--no-decode-worker", {&a.decode_worker, false}},
        {"--realtime", {&a.realtime, true}},   {"--no-realtime", {&a.realtime, false}},
        {"--live-constellation", {&a.live_constellation, true}},
        {"--no-live-constellation", {&a.live_constellation, false}},
        {"--tci", {&a.tci, true}},     {"--no-tci", {&a.tci, false}},
        {"--rig", {&a.rig, true}},     {"--no-rig", {&a.rig, false}},
        {"--record", {&a.record, true}}, {"--no-record", {&a.record, false}},
        {"--rig-debug", {&a.rig_debug, true}}, {"--no-rig-debug", {&a.rig_debug, false}}};
    const auto i_ = [prog](auto& o, auto& v) { return number<int>(o, v, prog); };
    const auto d_ = [prog](auto& o, auto& v) { return number<double>(o, v, prog); };
    std::map<std::string, std::function<void(const std::string&, const std::string&)>> valued = {
        {"--kiss-port", [&](auto& o, auto& v) { a.kiss_port = i_(o, v); }},
        {"--kiss-busy-limit", [&](auto& o, auto& v) { a.kiss_busy_limit = d_(o, v); }},
        {"--kiss-bw", [&](auto& o, auto& v) {
             a.kiss_bw = i_(o, v);
             if (a.kiss_bw != 2400 && a.kiss_bw != 500)
                 usage_error("argument --kiss-bw: invalid choice: " + v + " (choose from 2400, 500)", prog);
         }},
        {"--broadcast-mode", [&](auto&, auto& v) { a.broadcast_mode = v; }},
        {"--mycall", [&](auto&, auto& v) { a.mycall = v; }},
        {"--listen", [&](auto&, auto& v) { a.listen = v; }},
        {"--command-port", [&](auto& o, auto& v) { a.command_port = i_(o, v); }},
        {"--input-device", [&](auto&, auto& v) { a.input_device = v; }},
        {"--output-device", [&](auto&, auto& v) { a.output_device = v; }},
        {"--sample-rate", [&](auto& o, auto& v) { a.sample_rate = i_(o, v); }},
        {"--output-volume", [&](auto& o, auto& v) { a.output_volume = d_(o, v); }},
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
        {"--tci-host", [&](auto&, auto& v) { a.tci_host = v; }},
        {"--threads", [&](auto& o, auto& v) { pool::set_threads(i_(o, v)); }},
        {"--dd-budget", [&](auto& o, auto& v) { a.dd_budget = d_(o, v); }},
        {"--decoder-effort", [&](auto& o, auto& v) { a.decoder_effort = i_(o, v); }},
        {"--cw-id", [&](auto&, auto& v) { a.cw_id = v; }},
        {"--cw-wpm", [&](auto& o, auto& v) { a.cw_wpm = d_(o, v); }},
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
    // The flags on this command line that others depend on (value: the last one given).
    std::map<std::string, std::string> given;
    static const std::set<std::string> tracked = {"--rig-model", "--rig-device", "--no-rig", "--ptt-device",
                                                    "--record-dir", "--no-record"};
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
            if (tracked.count(arg)) given[arg] = value;
            continue;
        } else {
            usage_error("unrecognized arguments: " + std::string(argv[i]), prog);
        }
        if (tracked.count(arg)) given[arg] = "";
        if (arg == "--rig") given.erase("--no-rig");
        if (arg == "--record") given.erase("--no-record");
    }
    // A model or device given here turns a saved 'no rig' on, as a directory
    // to record to turns recording on.
    if (given.count("--no-rig") && (given.count("--rig-model") || given.count("--rig-device")))
        usage_error("--no-rig with a rig to use (--rig-model or --rig-device)", prog);
    if (given.count("--rig-model") || given.count("--rig-device")) a.rig = true;
    if (given.count("--no-record") && given.count("--record-dir"))
        usage_error("--no-record with a directory to record to (--record-dir)", prog);
    if (given.count("--record-dir")) a.record = true;
    if (given.count("--ptt-device") && a.ptt_method != "dtr" && a.ptt_method != "rts")
        usage_error("--ptt-device is the port whose DTR or RTS keys: it needs --ptt-method dtr or rts, not " + a.ptt_method, prog);
    return a;
}

int kiss_cap(int hz) { return hz == 500 ? 0 : 2; }

std::optional<std::pair<std::string, int>> tci_address(const std::string& s) {
    const auto colon = s.rfind(':');
    const std::string host = s.substr(0, colon);
    if (host.empty()) return std::nullopt;
    if (colon == std::string::npos) return std::pair{host, 50001};
    const std::string port = s.substr(colon + 1);
    if (port.empty() || port.size() > 5 || port.find_first_not_of("0123456789") != std::string::npos) return std::nullopt;
    const int p = std::stoi(port);
    if (p < 1 || p > 65535) return std::nullopt;
    return std::pair{host, p};
}

namespace {

template <typename E, std::size_t N>
std::vector<std::string> names(const std::array<rig::Named<E>, N>& table) {
    std::vector<std::string> out;
    for (const auto& n : table) out.emplace_back(n.name);
    return out;
}

// `a` passed check(): every value is one of its table's names.
template <typename E, std::size_t N>
E named(const std::array<rig::Named<E>, N>& table, const std::string& value) {
    return rig::from_name(table, value).value();
}

}  // namespace

// The spellings are rig/config.hpp's name tables, in their order.
const std::vector<std::string>& rig_choices(std::string_view field) {
    static const std::map<std::string, std::vector<std::string>, std::less<>> choices = {
        {"rig_data_bits", names(rig::DATA_BITS_NAMES)},
        {"rig_stop_bits", names(rig::STOP_BITS_NAMES)},
        {"rig_parity", names(rig::PARITY_NAMES)},
        {"rig_handshake", names(rig::HANDSHAKE_NAMES)},
        {"rig_dtr", names(rig::LINE_STATE_NAMES)},
        {"rig_rts", names(rig::LINE_STATE_NAMES)},
        {"ptt_method", names(rig::PTT_METHOD_NAMES)},
        {"ptt_audio", names(rig::PTT_AUDIO_NAMES)},
        {"rig_mode", names(rig::RIG_MODE_NAMES)},
    };
    return choices.find(field)->second;
}

rig::HamlibConfig hamlib_config(const Args& a) {
    rig::HamlibConfig h;
    h.model = a.rig_model;
    h.device = a.rig_device;
    h.baud = a.rig_baud;
    h.data_bits = named(rig::DATA_BITS_NAMES, a.rig_data_bits);
    h.stop_bits = named(rig::STOP_BITS_NAMES, a.rig_stop_bits);
    h.parity = named(rig::PARITY_NAMES, a.rig_parity);
    h.handshake = named(rig::HANDSHAKE_NAMES, a.rig_handshake);
    h.dtr = named(rig::LINE_STATE_NAMES, a.rig_dtr);
    h.rts = named(rig::LINE_STATE_NAMES, a.rig_rts);
    h.ptt_method = named(rig::PTT_METHOD_NAMES, a.ptt_method);
    h.ptt_device = a.ptt_device;
    h.ptt_audio = named(rig::PTT_AUDIO_NAMES, a.ptt_audio);
    h.mode = named(rig::RIG_MODE_NAMES, a.rig_mode);
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
    if (a.decoder_effort < 1) return "--decoder-effort: must be 1 or more";
    if (!tci_address(a.tci_host)) return "--tci-host " + a.tci_host + ": want HOST or HOST:PORT";
    if (a.tci) {
        if (!a.audio_io.empty()) return "--tci and --audio-io: one or the other";
        if (a.sample_rate != 8000 && a.sample_rate != 24000 && a.sample_rate != 48000)
            return "--tci: --sample-rate must be 8000, 24000 or 48000 (TCI's rates that are multiples of 8000)";
    }
    if (a.cw_id != "off" && a.cw_id != "add" && a.cw_id != "only")
        return "--cw-id: invalid choice: '" + a.cw_id + "' (choose from off, add, only)";
    if (!(a.cw_wpm >= 15 && a.cw_wpm <= 40)) return "--cw-wpm: 15-40";
#ifdef DATA2G_HAVE_RIG
    if (a.rig && !rig::model_info(a.rig_model))
        return "--rig-model " + std::to_string(a.rig_model) + ": not a model this Hamlib knows (see --list-rigs)";
#endif
    if (a.broadcast_mode) {
        try {
            kisslink::check_mode(*a.broadcast_mode, kiss_cap(a.kiss_bw));
        } catch (const std::invalid_argument&) {
            return "--broadcast-mode " + *a.broadcast_mode + ": not a mode within " + std::to_string(a.kiss_bw) + " Hz";
        }
    }
    return std::nullopt;
}

void list_modes(int kiss_bw) {
    for (const auto& l : host::mode_lines(kiss_cap(kiss_bw))) std::printf("%s\n", l.c_str());
}

}  // namespace data2g::app
