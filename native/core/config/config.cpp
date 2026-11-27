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

bool sc_band(std::string_view name) {
    if (PROFILE != "hf" || SYMS_PER_FRAME != 6) return false;
    return name == "w" || name == "w48" || name == "n10" || name == "n4";
}

}  // namespace data2g::config
