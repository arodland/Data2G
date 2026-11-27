#include "config/config.hpp"

#include <cstdint>
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

int connect_hash(std::string_view call, int bits) {
    while (!call.empty() && static_cast<unsigned char>(call.front()) <= ' ') call.remove_prefix(1);
    while (!call.empty() && static_cast<unsigned char>(call.back()) <= ' ') call.remove_suffix(1);
    std::uint32_t h = 0x811C9DC5u;
    for (char c : call) {
        const auto u = static_cast<unsigned char>(c);
        const auto b = (u >= 'a' && u <= 'z') ? static_cast<std::uint32_t>(u - 'a' + 'A') : static_cast<std::uint32_t>(u);
        h = (h ^ b) * 0x01000193u;
    }
    return static_cast<int>((h ^ (h >> 16)) & ((1u << bits) - 1u));
}

}  // namespace data2g::config
