// data2g/modem.py's scattered-pilot receive internals (_sc_full, _sc_estimate, _sc_joint, _sc_cal, _copy_rows), for
// modem.cpp and scattered.cpp only; the public part is in modem.hpp.
#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include "modem/modem.hpp"

namespace data2g::modem::sc {

// modem._sc_full: the burst's whole pilot symbols through the data's window (rows, nc of the data band) and their
// seconds from the first frame's start.
struct Full {
    Mat<cd> rows;
    std::vector<double> t;
};
Full full(std::span<const cd> z, const HeaderRead& hd, const Mat<cd>& raw, const Mat<cd>& hp, std::optional<int> kc,
          int shift, std::string_view db);

// modem._sc_estimate: raw (n_f * 6 rows, receive()'s shape), the estimate and the DD pass's info.
struct Estimated {
    Mat<cd> raw;
    DataEstimate est;
    std::shared_ptr<const ScInfo> info;
};
Estimated estimate(std::span<const cd> z, const HeaderRead& hd, const Mat<cd>& raw, const Mat<cd>& hp,
                   std::optional<int> kc, int n_f, int shift, equalizer::Support support, const Spec& spec);

}  // namespace data2g::modem::sc
