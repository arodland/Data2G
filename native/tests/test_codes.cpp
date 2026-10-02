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

    // The codec: every spec round-trips noise-free, through each decode path.
    check::is_true(codes::spec("fsk16r25-ctl") && codes::spec("fsk8r50-r1/2") && !codes::spec("nope"), "spec lookup");
    std::vector<const codes::Spec*> all;
    for (const auto& s : config::SUBMODES) all.push_back(&codes::spec(s));
    for (auto name : {"fsk16r25-r1/3", "fsk8r50-r1/2", "fsk32r62-ctl"}) all.push_back(codes::spec(name));
    for (const auto* sp : all) {
        const auto& s = *sp;
        const std::string name(s.name);
        check::equal(s.crc_bits, s.polar ? 24 : s.k >= 512 ? 32 : 16, name + " crc_bits");
        const int B = 3, n = s.coded_bits;
        std::vector<std::vector<std::uint8_t>> pls;
        Mat<float> soft(B, n);
        Mat<double> buf, soft0(B, n), soft1(B, n);
        const std::vector<std::uint32_t> masks = {0, 0xABCD1234u, 7};
        for (int b = 0; b < B; ++b) {
            std::vector<std::uint8_t> p(s.payload_bytes);
            for (std::size_t i = 0; i < p.size(); ++i) p[i] = static_cast<std::uint8_t>(i * 37 + b * 101);
            const auto c = codes::encode(s, p, 0, masks[b], b);
            const auto c1 = codes::encode(s, p, 1, masks[b], b);
            const auto f0 = codes::flip(s, b, 0), f1 = codes::flip(s, b, 1);
            for (int i = 0; i < n; ++i) {
                soft[b][i] = 4.0f * (1 - 2 * c[i]);
                soft0[b][i] = 0.5 * f0[i] * (1 - 2 * c[i]);  // unscrambled
                soft1[b][i] = 0.5 * f1[i] * (1 - 2 * c1[i]);
            }
            pls.push_back(p);
        }
        codes::combine(s, buf, soft0, std::vector<int>{0});
        codes::combine(s, buf, soft1, std::vector<int>{1});
        check::equal(static_cast<int>(buf.cols), codes::buffer_len(s), name + " buffer length");
        const auto many = codes::decode_many(s, soft, masks);
        const auto raw = codes::decode_raw(s, soft);
        const auto ir = codes::decode_buffer(s, buf, 1, masks, std::vector<int>{codes::PLAIN});
        for (int b = 0; b < B; ++b) {
            check::is_true(many[b].ok && many[b].data == pls[b], name + " decode_many round trip");
            const auto got = codes::check(s, raw.cands.row(b), raw.usable.row(b), masks[b]);
            check::is_true(got && *got == pls[b], name + " decode_raw + check");
            check::is_true(!codes::check(s, raw.cands.row(b), raw.usable.row(b), masks[b] ^ 1), name + " wrong mask fails");
            check::is_true(ir[b].ok && ir[b].data == pls[b], name + " decode_buffer of RV 0 + 1");
        }
        std::vector<int> v(static_cast<std::size_t>(4 * n));
        for (std::size_t i = 0; i < v.size(); ++i) v[i] = static_cast<int>(i);
        const auto sp4 = codes::spread(std::span<const int>(v), 4, s.bits_per_cu);
        check::is_true(codes::despread(std::span<const int>(sp4), 4, s.bits_per_cu) == v, name + " despread(spread)");
    }
    return check::report("codes");
}
