// Known answers that need no Python. Parity with data2g/codes.py is
// tests/test_native_parity.py's job.

#include <algorithm>
#include <string_view>
#include <vector>

#include "check.hpp"
#include "codes/codes.hpp"

using namespace data2g;

int main() {
    check::report_crashes_instead_of_prompting();
    const std::string_view text = "123456789";
    const std::vector<std::uint8_t> msg(text.begin(), text.end());

    // Catalogued check values (reveng CRC catalogue).
    check::equal(codes::crc16(msg), 0x29B1, "CRC-16/CCITT-FALSE check");
    check::equal(codes::crc32(msg), 0xCBF43926u, "CRC-32 check");
    // CRC-24C has no catalogue entry with this init; the Python reference value.
    check::equal(codes::crc24(msg), 0x4F9FD0u, "CRC-24C (init 0xFFFFFF)");

    const auto framed = codes::with_crc(msg, 16, 0x1234);
    check::equal(framed.size(), msg.size() + 2, "with_crc length");
    check::equal((framed[9] << 8 | framed[10]), 0x29B1 ^ 0x1234, "with_crc masks big-endian");

    // PN9 x^9 + x^5 + 1 is maximal length: period 511, 256 ones.
    const auto pn = codes::scrambler(1022);
    check::is_true(std::equal(pn.begin(), pn.begin() + 511, pn.begin() + 511), "PN9 period 511");
    check::equal(std::count(pn.begin(), pn.begin() + 511, 1), 256, "PN9 balance");
    check::is_true(std::ranges::all_of(codes::scrambler(64, 0), [](auto b) { return b == 0; }), "PLAIN seed is all zero");

    std::vector<int> seeds;
    for (int i = 0; i < 511; ++i) seeds.push_back(codes::scramble_seed(i));
    std::ranges::sort(seeds);
    check::is_true(std::ranges::adjacent_find(seeds) == seeds.end() && seeds.front() == 1 && seeds.back() == 511,
                   "seeds 1..511 distinct over positions 0..510");
    check::equal(codes::scramble_seed(codes::PLAIN), 0, "PLAIN seed");

    for (const auto& s : config::SUBMODES) {
        auto perm = codes::interleaver(s);
        std::vector<int> sorted(perm.begin(), perm.end());
        std::ranges::sort(sorted);
        bool ok = static_cast<int>(sorted.size()) == s.coded_bits;
        for (int i = 0; ok && i < s.coded_bits; ++i) ok = sorted[i] == i;
        check::is_true(ok, std::string(s.name) + " interleaver is a permutation of coded_bits");
        check::equal(codes::info_pos(s).empty(), s.code != "polar", std::string(s.name) + " info set iff polar");
        if (s.code == "polar") check::equal(static_cast<int>(codes::info_pos(s).size()), s.k, std::string(s.name) + " info set size");
    }
    check::is_true(codes::submode("w48-256l-r5/8") != nullptr && codes::submode("nope") == nullptr, "submode lookup");
    return check::report("codes");
}
