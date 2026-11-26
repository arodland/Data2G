#include "config/config.hpp"

#include <cstdlib>
#include <stdexcept>
#include <string>

namespace data2g::config {

const Band& band(std::string_view name) {
    for (const auto& b : BANDS)
        if (b.name == name) return b;
    throw std::out_of_range("no band " + std::string(name));
}

const Submode* submode(std::string_view name) {
    for (const auto& s : SUBMODES)
        if (s.name == name) return &s;
    return nullptr;
}

std::size_t row(const Submode& spec) {
    if (&spec < SUBMODES.data() || &spec >= SUBMODES.data() + SUBMODES.size())
        throw std::invalid_argument("spec is not an element of config::SUBMODES");
    return static_cast<std::size_t>(&spec - SUBMODES.data());
}

const Submode* by_index(std::string_view sync_band, int index) {
    const Submode* out = nullptr;  // as Python's dict, the last wins
    for (const auto& s : SUBMODES)
        if (s.sync_band == sync_band && s.index == index) out = &s;
    return out;
}

std::string_view sc_layout() {
    static const std::string layout = [] {
        // unset: this branch's default (fast-fading-integration), fk3d2, where the scattered pilots have a path
        // (config.py _SC_DEFAULT: the HF profile, 6-symbol frames, none of the FM-path study switches set by hand);
        // "off" or empty: the pilot rows
        const auto set = [](const char* n) { const char* e = std::getenv(n); return e && *e; };
        const bool path = PROFILE == "hf" && SYMS_PER_FRAME == 6 && !set("DATA2G_SPREAD_WINDOW") &&
                          !set("DATA2G_EST_WINDOW") && !set("DATA2G_DECODE_EARLY") && !set("DATA2G_SYMBOL_NOISE");
        const char* v = std::getenv("DATA2G_PILOTS");
        const std::string s = !v ? (path ? "fk3d2" : "") : std::string_view(v) == "off" ? "" : v;
        const auto colon = s.find(':');
        const std::string name = s.substr(0, colon), density = colon == std::string::npos ? "" : s.substr(colon + 1);
        const bool ok = name.empty() || name == "k3d2" || name == "gold" || name == "mix" || name == "mixs" ||
                        name == "k2d3" || name == "fk2d3" || name == "fk3d2";
        if (!ok || (density != "" && density != "1/6"))
            throw std::invalid_argument("DATA2G_PILOTS=" + s + ": k3d2, gold, mix, mixs, k2d3, fk2d3 or fk3d2, at 1/6");
        if (!name.empty() && (PROFILE != "hf" || SYMS_PER_FRAME != 6))
            throw std::invalid_argument("DATA2G_PILOTS: the HF profile and 6-symbol frames only");
        return name;
    }();
    return layout;
}

bool sc_band(std::string_view name) {
    const auto l = sc_layout();
    if (l.empty()) return false;
    if (l == "k3d2") return name == "w" || name == "w48";
    return name == "w" || name == "w48" || name == "n10" || name == "n4";
}

}  // namespace data2g::config
