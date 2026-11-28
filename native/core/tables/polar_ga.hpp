// Polar info sets Python designs at run time, compiled in:
// generated/polar_ga.cpp.
#pragma once

#include <cstdint>
#include <span>
#include <string_view>

namespace data2g::tables {

// codes.polar_code with no frozen file (the CPM control codewords), frozen
// here by (k, e).
struct PolarDesign {
    int k, e;
    std::span<const std::uint16_t> info_pos;
};
extern const std::span<const PolarDesign> POLAR_GA;

// codes.polar_order (GA-reliability bit permutation) per polar spec name.
struct PolarOrder {
    std::string_view name;
    std::span<const std::uint16_t> order;
};
extern const std::span<const PolarOrder> POLAR_ORDERS;

}  // namespace data2g::tables
