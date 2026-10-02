// Lifted from SSTVAE's core/rig/hamlib.cpp; see that file for the longer
// reasoning behind each choice (it is the one that met the Windows and
// Hamlib-version traps).
#include "rig/hamlib/hamlib.hpp"

#include <hamlib/rig.h>

#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <utility>

namespace data2g::rig {

namespace {

// Hamlib's backend registry is global and loading it is not reentrant.
// Quiet unless DATA2G_HAMLIB_DEBUG is set: a CAT failure is reported
// through our own status text.
void load_backends_once() {
    static std::once_flag once;
    std::call_once(once, [] {
        const char* debug = std::getenv("DATA2G_HAMLIB_DEBUG");
        const bool verbose = debug != nullptr && *debug != '\0' && std::strcmp(debug, "0") != 0;
        rig_set_debug(verbose ? RIG_DEBUG_TRACE : RIG_DEBUG_NONE);
        rig_load_all_backends();
    });
}

std::string hamlib_error(const char* what, int code) {
    const char* text = rigerror(code);
    return std::string(what) + ": " + (text != nullptr ? text : "unknown error") + " (" + std::to_string(code) + ")";
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
        switch (config_.ptt_method) {
            case PttMethod::Cat: set_conf("ptt_type", "RIG"); break;
            case PttMethod::Dtr: set_conf("ptt_type", "DTR"); break;
            case PttMethod::Rts: set_conf("ptt_type", "RTS"); break;
        }
        if (!config_.ptt_device.empty() && config_.ptt_method != PttMethod::Cat)
            set_conf("ptt_pathname", config_.ptt_device);
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
        const int rc = rig_set_ptt(rig_, RIG_VFO_CURR, on ? RIG_PTT_ON : RIG_PTT_OFF);
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
        if (config_.model == MODEL_NET_RIGCTL) out += " at " + config_.device;
        return out;
    }

private:
    void require_open() const {
        if (rig_ == nullptr || !open_) throw RigError("the rig is not open");
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

std::unique_ptr<RigBackend> make_hamlib_backend(const HamlibConfig& config) {
    return std::make_unique<HamlibBackend>(config);
}

std::string hamlib_version() {
    const char* v = rig_version();  // a function, not the data symbol: MSVC can't import data from a DLL
    return v != nullptr ? v : "";
}

}  // namespace data2g::rig
