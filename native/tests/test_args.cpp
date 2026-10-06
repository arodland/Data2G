// data2g-host's rig options (app/station.*): what the flags set, how the
// legacy --rigctld-host/--rigctld-port map onto Hamlib model 2, the mapping
// to HamlibConfig, and check()'s messages. The usage errors (which exit)
// are tested on the binary, in tests/test_native_host_e2e.py.

#include <string>
#include <vector>

#include "app/station.hpp"
#include "check.hpp"

using namespace data2g;

namespace {

app::Args parse(std::vector<std::string> args, app::Args base = {}) {
    args.insert(args.begin(), "data2g-host");
    std::vector<char*> argv;
    for (auto& s : args) argv.push_back(s.data());
    return app::parse(static_cast<int>(argv.size()), argv.data(), base);
}

void test_rigctld_is_model_2() {
    check::current_step = "rigctld";
    const app::Args d = parse({});
    check::is_true(app::rig_enabled(d) && d.rig_model == rig::MODEL_NET_RIGCTL, "default: a rig, model 2");
    check::equal(app::rig_device(d), std::string("localhost:4532"), "default: rigctld on localhost:4532");
    check::is_true(!app::rig_enabled(parse({"--rigctld-port", "0"})), "--rigctld-port 0: no rig");
    check::is_true(!app::rig_enabled(parse({"--no-rig"})), "--no-rig: no rig");

    // flags override saved settings as a whole
    app::Args saved;
    saved.rig_model = 3073;
    saved.rig_device = "/dev/ttyUSB0";
    saved.rig = false;
    const app::Args r = parse({"--rigctld-host", "radio.lan", "--rigctld-port=4600"}, saved);
    check::is_true(app::rig_enabled(r) && r.rig_model == rig::MODEL_NET_RIGCTL && r.rig_device.empty(),
                   "--rigctld-*: model 2 over a saved serial rig, turned on");
    check::equal(app::rig_device(r), std::string("radio.lan:4600"), "--rigctld-*: that address");
    check::is_true(parse({"--rig-model", "1"}, saved).rig, "--rig-model turns a saved 'off' on");
    check::is_true(parse({"--rigctld-port", "4532", "--rig-model", "2"}).rig_model == 2, "--rig-model 2 with --rigctld-*: fine");
    const app::Args dev = parse({"--rig-model", "2", "--rig-device", "shack:4532"});
    check::equal(app::rig_device(dev), std::string("shack:4532"), "model 2 with --rig-device");
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
    check::is_true(d.model == 2 && d.device == "localhost:4532" && d.ptt_method == rig::PttMethod::Cat &&
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

}  // namespace

int main() {
    test_rigctld_is_model_2();
    test_every_flag();
    test_check();
    test_noise_rule();
    return check::report("args");
}
