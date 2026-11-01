// data2g-host's options (app/options.*): what the flags set, the mapping
// to HamlibConfig, and check()'s messages. The usage errors (which exit)
// are tested on the binary, in tests/test_native_host_e2e.py.

#include <filesystem>
#include <string>
#include <vector>

#include "app/options.hpp"
#include "codes/codes.hpp"
#include "check.hpp"

using namespace data2g;

namespace {

app::Args parse(std::vector<std::string> args, app::Args base = {}) {
    args.insert(args.begin(), "data2g-host");
    std::vector<char*> argv;
    for (auto& s : args) argv.push_back(s.data());
    return app::parse(static_cast<int>(argv.size()), argv.data(), base);
}

void test_rig_flags() {
    check::current_step = "rig";
    const app::Args d = parse({});
    check::is_true(d.rig && d.rig_model == rig::MODEL_NET_RIGCTL && d.rig_device.empty(), "default: model 2, Hamlib's device");
    check::is_true(!parse({"--no-rig"}).rig, "--no-rig: no rig");

    // a model or device turns a saved 'off' on
    app::Args saved;
    saved.rig = false;
    check::is_true(parse({"--rig-model", "1"}, saved).rig, "--rig-model turns a saved 'off' on");
    check::is_true(parse({"--rig-device", "shack:4532"}, saved).rig, "--rig-device turns a saved 'off' on");
    check::is_true(!parse({}, saved).rig, "...and nothing else does");
    const app::Args dev = parse({"--rig-model", "2", "--rig-device", "shack:4532"});
    check::equal(app::hamlib_config(dev).device, std::string("shack:4532"), "model 2 with --rig-device");
}

void test_listen() {
    check::current_step = "listen";
    check::equal(parse({}).listen, std::string("127.0.0.1"), "default 127.0.0.1");
    check::equal(parse({"--listen", "0.0.0.0"}).listen, std::string("0.0.0.0"), "--listen");
}

void test_tci() {
    check::current_step = "tci";
    check::is_true(app::tci_address("sdr.lan") == std::pair<std::string, int>{"sdr.lan", 50001}, "HOST: port 50001");
    check::is_true(app::tci_address("10.0.0.5:50001") == std::pair<std::string, int>{"10.0.0.5", 50001}, "HOST:PORT");
    for (const char* bad : {":40001", "sdr:", "sdr:0", "sdr:65536", "sdr:4x"})
        check::is_true(!app::tci_address(bad), std::string("not an address: ") + bad);
    const auto d = parse({});
    check::is_true(!d.tci && d.tci_host == "localhost:50001", "default: off, localhost:50001");
    const auto t = parse({"--tci", "--tci-host", "sdr.lan"});
    check::is_true(t.tci && t.tci_host == "sdr.lan", "--tci, --tci-host");
    check::is_true(!parse({"--no-tci"}, t).tci, "--no-tci over a saved 'on'");
    check::is_true(!app::check(t), "--tci at 48 kHz: fine");
    check::is_true(app::check(parse({"--tci", "--sample-rate", "16000"})).has_value(), "--tci at 16 kHz: not a TCI rate");
    check::is_true(!app::check(parse({"--sample-rate", "16000"})), "16 kHz without --tci: fine");
    check::is_true(app::check(parse({"--tci", "--audio-io", "pipe:a,b"})).has_value(), "--tci with --audio-io");
    check::is_true(app::check(parse({"--tci-host", "sdr:x"})).has_value(), "--tci-host with a bad port");
}

void test_record() {
    check::current_step = "record";
    const app::Args d = parse({});
    check::is_true(!d.record && app::record_path(d).empty(), "off by default");
    const app::Args on = parse({"--record"});
    check::is_true(on.record && std::filesystem::path(app::record_path(on)).parent_path() == "recordings", "--record: under recordings/");
    const app::Args dir = parse({"--record-dir", "elsewhere"});
    check::is_true(dir.record && std::filesystem::path(app::record_path(dir)).parent_path() == "elsewhere", "--record-dir implies --record");
    app::Args saved;
    saved.record = true;
    check::is_true(!parse({"--no-record"}, saved).record, "--no-record over a saved 'on'");
    check::is_true(parse({"--no-record", "--record"}).record, "the last of --record/--no-record wins");
}

void test_every_flag() {
    check::current_step = "flags";
    const app::Args a = parse({"--rig-model", "3073", "--rig-device", "COM5", "--rig-baud", "19200", "--rig-data-bits", "8",
                               "--rig-stop-bits", "2", "--rig-parity", "even", "--rig-handshake", "hardware", "--rig-dtr", "high",
                               "--rig-rts", "low", "--ptt-method", "rts", "--ptt-device", "COM6", "--ptt-audio", "data",
                               "--rig-mode", "pkt_usb", "--rig-timeout-ms", "750", "--rig-retries", "3", "--rig-poll-interval", "2.5",
                               "--rig-debug"});
    check::is_true(!app::check(a), "all valid: " + app::check(a).value_or(""));
    check::is_true(a.rig_debug && a.rig_poll_interval == 2.5, "--rig-debug, --rig-poll-interval");
    const rig::HamlibConfig h = app::hamlib_config(a);
    check::is_true(h.model == 3073 && h.device == "COM5" && h.baud == 19200, "model, device, baud");
    check::is_true(h.data_bits == rig::DataBits::Eight && h.stop_bits == rig::StopBits::Two && h.parity == rig::Parity::Even &&
                       h.handshake == rig::Handshake::Hardware,
                   "serial line settings");
    check::is_true(h.dtr == rig::LineState::High && h.rts == rig::LineState::Low, "control lines");
    check::is_true(h.ptt_method == rig::PttMethod::Rts && h.ptt_device == "COM6" && h.ptt_audio == rig::PttAudio::Data,
                   "PTT method, device, audio");
    check::is_true(h.mode == rig::RigMode::PktUsb && h.timeout_ms == 750 && h.retries == 3, "mode, timeout, retries");
    const rig::HamlibConfig d = app::hamlib_config(parse({}));
    check::is_true(d.model == 2 && d.device.empty() && d.ptt_method == rig::PttMethod::Cat &&
                       d.parity == rig::Parity::Default && d.mode == rig::RigMode::None,
                   "defaults map to the backend's own");
    check::is_true(app::hamlib_config(parse({"--ptt-method", "vox"})).ptt_method == rig::PttMethod::Vox, "vox");
}

void test_check() {
    check::current_step = "check";
    app::Args a;
    a.rig_parity = "mark";  // e.g. a hand-edited settings file
    check::equal(app::check(a).value_or(""), std::string("--rig-parity: invalid choice: 'mark' (choose from default, none, odd, even)"),
                 "a bad choice is named");
    a = {};
    a.rig_timeout_ms = 0;
    check::is_true(app::check(a).has_value(), "timeout 0 refused");
#ifdef DATA2G_HAVE_RIG
    a = parse({"--rig-model", "999999"});
    check::is_true(app::check(a).value_or("").find("not a model this Hamlib knows") != std::string::npos, "unknown model refused");
    a.rig = false;
    check::is_true(!app::check(a).has_value(), "...but not when the rig is off");
#endif
}

void test_noise_rule() {
    check::current_step = "noise rule";
    check::is_true(parse({}).noise_rule == 1.0, "default 1");
    check::is_true(parse({"--noise-rule", "0.4"}).noise_rule == 0.4, "--noise-rule 0.4");
    check::is_true(parse({"--noise-rule=0"}).noise_rule == 0.0 && !app::check(parse({"--noise-rule=0"})), "0: off, valid");
    check::equal(app::check(parse({"--noise-rule", "-1"})).value_or(""), std::string("--noise-rule: must be 0 (off) or more"),
                 "negative refused");
}

void test_decoder_effort() {
    check::current_step = "decoder effort";
    check::is_true(parse({}).decoder_effort == codes::DEFAULT_ITERS, "default: codes::DEFAULT_ITERS");
    check::is_true(parse({"--decoder-effort", "17"}).decoder_effort == 17 && !app::check(parse({"--decoder-effort=17"})),
                   "--decoder-effort 17: any cap, not only the GUI's steps");
    check::equal(app::check(parse({"--decoder-effort", "0"})).value_or(""), std::string("--decoder-effort: must be 1 or more"),
                 "0 refused");
}

void test_realtime_switch() {
    check::current_step = "realtime";
    check::is_true(parse({}).realtime, "on by default");
    check::is_true(!parse({"--no-realtime"}).realtime, "--no-realtime");
    check::is_true(parse({"--no-realtime", "--realtime"}).realtime, "--realtime");
}

void test_dd_budget() {
    check::current_step = "dd budget";
    check::is_true(parse({}).dd_budget == 1.0, "default 1");
    check::is_true(parse({"--dd-budget", "0.3"}).dd_budget == 0.3, "--dd-budget 0.3");
    check::is_true(parse({"--dd-budget=0"}).dd_budget == 0.0 && !app::check(parse({"--dd-budget=0"})), "0: none, valid");
    check::equal(app::check(parse({"--dd-budget", "-1"})).value_or(""), std::string("--dd-budget: must be 0 (no DD) or more"),
                 "negative refused");
}

void test_cw_id() {
    check::current_step = "cw id";
    check::is_true(parse({}).cw_id == "off" && parse({}).cw_wpm == 20.0, "default off, 20 wpm");
    const app::Args a = parse({"--cw-id", "only", "--cw-wpm", "25"});
    check::is_true(a.cw_id == "only" && a.cw_wpm == 25.0 && !app::check(a), "--cw-id only --cw-wpm 25");
    check::equal(app::check(parse({"--cw-id", "on"})).value_or(""),
                 std::string("--cw-id: invalid choice: 'on' (choose from off, add, only)"), "a bad choice is named");
    check::equal(app::check(parse({"--cw-wpm", "50"})).value_or(""), std::string("--cw-wpm: 15-40"), "too fast");
}

}  // namespace

int main() {
    test_rig_flags();
    test_listen();
    test_tci();
    test_record();
    test_every_flag();
    test_check();
    test_noise_rule();
    test_decoder_effort();
    test_dd_budget();
    test_cw_id();
    test_realtime_switch();
    return check::report("args");
}
