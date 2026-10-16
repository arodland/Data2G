// tnc with no Python: KISS framing round trips, escapes split across
// reads. Parity: tests/test_native_tnc.py.

#include <algorithm>
#include <string>

#include "check.hpp"
#include "tnc/tnc.hpp"

using namespace data2g;

int main() {
    check::report_crashes_instead_of_prompting();
    check::Watchdog watchdog(60, "test_tnc");

    check::current_step = "kiss";
    std::vector<tnc::Bytes> frames = {{}, {1, 2, 3}, {tnc::FEND, tnc::FESC, tnc::TFEND, tnc::TFESC, 0}};
    tnc::Bytes wire;
    for (const auto& f : frames) {
        const auto e = tnc::kiss_encode(f, 3);
        wire.insert(wire.end(), e.begin(), e.end());
    }
    tnc::KissDecoder dec;
    std::vector<std::pair<int, tnc::Bytes>> got;
    for (std::size_t i = 0; i < wire.size(); i += 2) {
        auto r = dec.feed(std::span(wire).subspan(i, std::min<std::size_t>(2, wire.size() - i)));
        got.insert(got.end(), r.begin(), r.end());
    }
    check::equal(static_cast<int>(got.size()), 3, "kiss frames");
    for (std::size_t i = 0; i < got.size(); ++i)
        check::is_true(got[i].first == 0x30 && got[i].second == frames[i], "kiss frame " + std::to_string(i));

    return check::report("test_tnc");
}
