// Lifted from SSTVAE's core/rig/hamlib.cpp; see that file for the longer
// reasoning behind each choice (it is the one that met the Windows and
// Hamlib-version traps).
#include "rig/hamlib/hamlib.hpp"

#include <hamlib/rig.h>
#include <hamlib/riglist.h>

#include <algorithm>
#include <cstdarg>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace data2g::rig {

namespace {

// Quiet unless DATA2G_HAMLIB_DEBUG is set: a CAT failure is reported
// through our own status text. Also the level a removed sink returns to.
enum rig_debug_level_e env_debug_level() {
    const char* debug = std::getenv("DATA2G_HAMLIB_DEBUG");
    const bool verbose = debug != nullptr && *debug != '\0' && std::strcmp(debug, "0") != 0;
    return verbose ? RIG_DEBUG_TRACE : RIG_DEBUG_NONE;
}

// Hamlib's backend registry is global and loading it is not reentrant.
void load_backends_once() {
    static std::once_flag once;
    std::call_once(once, [] {
        rig_set_debug(env_debug_level());
        rig_load_all_backends();
    });
}

std::string hamlib_error(const char* what, int code) {
    const char* text = rigerror(code);
    return std::string(what) + ": " + (text != nullptr ? text : "unknown error") + " (" + std::to_string(code) + ")";
}

// --- the trace sink: Hamlib hands us fmt + va_list; a line may span calls --

std::mutex& debug_mu() {
    static std::mutex m;
    return m;
}
DebugSink& debug_sink() {
    static DebugSink sink;
    return sink;
}
std::string& debug_partial() {
    static std::string partial;
    return partial;
}
constexpr std::size_t kMaxPartialBytes = 8192;  // a format with no newline can't grow it forever

std::string format_debug(const char* fmt, va_list ap) {
    va_list sizing;
    va_copy(sizing, ap);
    char stack[512];
    const int n = std::vsnprintf(stack, sizeof(stack), fmt, sizing);
    va_end(sizing);
    if (n < 0) return {};
    if (static_cast<std::size_t>(n) < sizeof(stack)) return std::string(stack, static_cast<std::size_t>(n));
    std::string big(static_cast<std::size_t>(n), '\0');
    va_list again;
    va_copy(again, ap);
    std::vsnprintf(&big[0], static_cast<std::size_t>(n) + 1, fmt, again);
    va_end(again);
    return big;
}

// A C callback under Hamlib's debug mutex: nothing may escape, and the sink
// runs outside our lock.
int debug_trampoline(enum rig_debug_level_e, rig_ptr_t, const char* fmt, va_list ap) {
    try {
        if (fmt == nullptr) return 0;
        const std::string text = format_debug(fmt, ap);
        if (text.empty()) return 0;
        std::vector<std::string> lines;
        DebugSink sink;
        {
            std::lock_guard<std::mutex> lock(debug_mu());
            sink = debug_sink();
            if (!sink) return 0;
            std::string& partial = debug_partial();
            partial += text;
            std::size_t start = 0;
            for (std::size_t nl; (nl = partial.find('\n', start)) != std::string::npos; start = nl + 1)
                lines.emplace_back(partial, start, nl - start);
            partial.erase(0, start);
            if (partial.size() > kMaxPartialBytes) {
                lines.push_back(partial);
                partial.clear();
            }
        }
        for (std::string& line : lines) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            sink(line);
        }
    } catch (...) {
    }
    return 0;
}

PortType port_type(int t) {
    switch (t) {
        case RIG_PORT_NONE: return PortType::None;
        case RIG_PORT_SERIAL: return PortType::Serial;
        case RIG_PORT_NETWORK:
        case RIG_PORT_UDP_NETWORK: return PortType::Network;
        default: return PortType::Other;
    }
}

RigModel from_caps(const struct rig_caps* caps) {
    RigModel m;
    m.model = static_cast<int>(caps->rig_model);
    m.manufacturer = caps->mfg_name != nullptr ? caps->mfg_name : "";
    m.name = caps->model_name != nullptr ? caps->model_name : "";
    m.version = caps->version != nullptr ? caps->version : "";
    const char* status = rig_strstatus(caps->status);
    m.status = status != nullptr ? status : "";
    m.port = port_type(caps->port_type);
    return m;
}

class HamlibBackend final : public RigBackend {
public:
    explicit HamlibBackend(HamlibConfig config) : config_(std::move(config)) {}
    ~HamlibBackend() override { close(); }

    void open() override {
        load_backends_once();
        if (rig_ != nullptr) return;
        rig_ = rig_init(static_cast<rig_model_t>(config_.model));
        if (rig_ == nullptr) throw RigError("Hamlib does not know rig model " + std::to_string(config_.model));
        // Tokens, not the port structs: those are Hamlib internals (IN_HAMLIB).
        if (!config_.device.empty()) set_conf("rig_pathname", config_.device);
        if (config_.baud > 0) set_conf("serial_speed", std::to_string(config_.baud));
        apply_serial_settings();
        apply_ptt_settings();
        set_conf("timeout", std::to_string(config_.timeout_ms));
        set_conf("retry", std::to_string(config_.retries));
        // Hamlib's own polling thread off: one command in flight, ours.
        set_conf("poll_interval", "0");
        const int rc = rig_open(rig_);
        if (rc != RIG_OK) {
            const std::string msg = hamlib_error("could not open the rig", rc);
            rig_cleanup(rig_);
            rig_ = nullptr;
            throw RigError(msg);
        }
        open_ = true;
        apply_mode();
    }

    void close() noexcept override {
        if (rig_ == nullptr) return;
        if (open_) rig_close(rig_);
        open_ = false;
        rig_cleanup(rig_);
        rig_ = nullptr;
    }

    void set_ptt(bool on) override {
        require_open();
        // RIG_PTT_ON, not ON_MIC: on a MICDATA rig they differ (TX vs TX0).
        const ptt_t key = config_.ptt_audio == PttAudio::Data && config_.ptt_method == PttMethod::Cat ? RIG_PTT_ON_DATA : RIG_PTT_ON;
        const int rc = rig_set_ptt(rig_, RIG_VFO_CURR, on ? key : RIG_PTT_OFF);
        if (rc != RIG_OK) throw RigError(hamlib_error(on ? "PTT on failed" : "PTT off failed", rc));
    }

    double frequency_hz() override {
        require_open();
        freq_t freq = 0;
        const int rc = rig_get_freq(rig_, RIG_VFO_CURR, &freq);
        if (rc != RIG_OK) throw RigError(hamlib_error("could not read frequency", rc));
        return static_cast<double>(freq);
    }

    // Through rig_get_caps_cptr: nothing here dereferences a RIG* (its
    // pthread members differ in size under the MSVC shim).
    std::string description() const override {
        const auto model = static_cast<rig_model_t>(config_.model);
        const char* mfg = rig_get_caps_cptr(model, RIG_CAPS_MFG_NAME_CPTR);
        const char* name = rig_get_caps_cptr(model, RIG_CAPS_MODEL_NAME_CPTR);
        std::string out = mfg && name ? std::string(mfg) + " " + name : "model " + std::to_string(config_.model);
        if (!config_.device.empty()) out += " at " + config_.device;
        return out;
    }

private:
    void require_open() const {
        if (rig_ == nullptr || !open_) throw RigError("the rig is not open");
    }

    // Token names and values from Hamlib 4.7.2's src/serial_cfg_params.h and
    // src/conf.c: a misspelled token is silently ignored, not an error.
    void apply_serial_settings() {
        switch (config_.data_bits) {
            case DataBits::Seven: set_conf("data_bits", "7"); break;
            case DataBits::Eight: set_conf("data_bits", "8"); break;
            case DataBits::Default: break;
        }
        switch (config_.stop_bits) {
            case StopBits::One: set_conf("stop_bits", "1"); break;
            case StopBits::Two: set_conf("stop_bits", "2"); break;
            case StopBits::Default: break;
        }
        switch (config_.parity) {
            case Parity::None: set_conf("serial_parity", "None"); break;
            case Parity::Odd: set_conf("serial_parity", "Odd"); break;
            case Parity::Even: set_conf("serial_parity", "Even"); break;
            case Parity::Default: break;
        }
        switch (config_.handshake) {
            case Handshake::None: set_conf("serial_handshake", "None"); break;
            case Handshake::XonXoff: set_conf("serial_handshake", "XONXOFF"); break;
            case Handshake::Hardware: set_conf("serial_handshake", "Hardware"); break;
            case Handshake::Default: break;
        }
        for (const auto& [state, token] : {std::pair{config_.dtr, "dtr_state"}, std::pair{config_.rts, "rts_state"}}) {
            if (state == LineState::High) set_conf(token, "ON");
            if (state == LineState::Low) set_conf(token, "OFF");
        }
    }

    void apply_ptt_settings() {
        switch (config_.ptt_method) {
            case PttMethod::Vox: set_conf("ptt_type", "None"); break;
            // "RIG" would downgrade a MICDATA rig to plain mic keying.
            case PttMethod::Cat: set_conf("ptt_type", config_.ptt_audio == PttAudio::Data ? "RIGMICDATA" : "RIG"); break;
            case PttMethod::Dtr: set_conf("ptt_type", "DTR"); break;
            case PttMethod::Rts: set_conf("ptt_type", "RTS"); break;
        }
        if (!config_.ptt_device.empty() && (config_.ptt_method == PttMethod::Dtr || config_.ptt_method == PttMethod::Rts))
            set_conf("ptt_pathname", config_.ptt_device);
    }

    // A rig command, after open; a refusal is not fatal (it still keys).
    void apply_mode() {
        rmode_t mode = RIG_MODE_NONE;
        switch (config_.mode) {
            case RigMode::Usb: mode = RIG_MODE_USB; break;
            case RigMode::PktUsb: mode = RIG_MODE_PKTUSB; break;
            case RigMode::None: return;
        }
        rig_set_mode(rig_, RIG_VFO_CURR, mode, RIG_PASSBAND_NOCHANGE);
    }

    // A token the backend lacks is skipped (netrigctl has no serial retry).
    void set_conf(const char* name, const std::string& value) {
        const hamlib_token_t token = rig_token_lookup(rig_, name);
        if (token == RIG_CONF_END) return;
        rig_set_conf(rig_, token, value.c_str());
    }

    HamlibConfig config_;
    RIG* rig_ = nullptr;
    bool open_ = false;
};

}  // namespace

std::string RigModel::label() const {
    std::string out = manufacturer;
    if (!out.empty() && !name.empty()) out += " ";
    out += name;
    if (!status.empty()) out += " (" + status + ")";
    return out;
}

std::vector<RigModel> list_models() {
    load_backends_once();
    std::vector<RigModel> models;
    rig_list_foreach(
        [](const struct rig_caps* caps, rig_ptr_t data) -> int {
            static_cast<std::vector<RigModel>*>(data)->push_back(from_caps(caps));
            return 1;  // keep going
        },
        &models);
    std::sort(models.begin(), models.end(), [](const RigModel& a, const RigModel& b) {
        if (a.manufacturer != b.manufacturer) return a.manufacturer < b.manufacturer;
        if (a.name != b.name) return a.name < b.name;
        return a.model < b.model;
    });
    return models;
}

std::optional<RigModel> model_info(int model) {
    load_backends_once();
    struct Search {
        int model;
        std::optional<RigModel> out;
    } search{model, std::nullopt};
    rig_list_foreach(
        [](const struct rig_caps* caps, rig_ptr_t data) -> int {
            auto* s = static_cast<Search*>(data);
            if (static_cast<int>(caps->rig_model) != s->model) return 1;
            s->out = from_caps(caps);
            return 0;  // stop
        },
        &search);
    return search.out;
}

bool supports_ptt_audio_source(int model) {
    load_backends_once();
    return rig_get_caps_int(static_cast<rig_model_t>(model), RIG_CAPS_PTT_TYPE) == RIG_PTT_RIG_MICDATA;
}

std::unique_ptr<RigBackend> make_hamlib_backend(const HamlibConfig& config) {
    return std::make_unique<HamlibBackend>(config);
}

std::string hamlib_version() {
    const char* v = rig_version();  // a function, not the data symbol: MSVC can't import data from a DLL
    return v != nullptr ? v : "";
}

void set_debug_sink(DebugSink sink) {
    load_backends_once();  // first, or its rig_set_debug would undo ours
    const bool active = static_cast<bool>(sink);
    {
        std::lock_guard<std::mutex> lock(debug_mu());
        debug_sink() = std::move(sink);
        debug_partial().clear();
    }
    // Registered only while a sink exists: rig_debug writes to stderr *or*
    // the callback, so a permanent one would swallow DATA2G_HAMLIB_DEBUG.
    rig_set_debug_callback(active ? &debug_trampoline : nullptr, nullptr);
    rig_set_debug(active ? RIG_DEBUG_TRACE : env_debug_level());
}

}  // namespace data2g::rig
