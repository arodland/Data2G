// tnc with no Python: KISS framing round trips, pack/unpack, and the
// streaming Receiver hearing a burst in noise (in two chunkings) and
// nothing in silence. Parity: tests/test_native_tnc.py.

#include <algorithm>
#include <random>
#include <string>

#include "check.hpp"
#include "codes/codes.hpp"
#include "tnc/tnc.hpp"

using namespace data2g;

namespace {

std::vector<tnc::Event> feed_all(tnc::Receiver& rx, const std::vector<double>& y, std::size_t chunk) {
    std::vector<tnc::Event> out;
    for (std::size_t i = 0; i < y.size(); i += chunk) {
        auto ev = rx.feed(std::span(y).subspan(i, std::min(chunk, y.size() - i)));
        out.insert(out.end(), std::make_move_iterator(ev.begin()), std::make_move_iterator(ev.end()));
    }
    return out;
}

}  // namespace

int main() {
    check::report_crashes_instead_of_prompting();
    check::Watchdog watchdog(300, "test_tnc");
    std::mt19937 rng(1);

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

    check::current_step = "framing";
    const auto& spec = *codes::submode("qpsk-r1/5");
    std::vector<tnc::Bytes> packets = {tnc::Bytes(10, 7), tnc::Bytes(60, 9), tnc::Bytes(1, 1)};
    const auto pl = tnc::pack(packets, spec);
    check::equal(static_cast<int>(pl.size()), 2, "pack: codewords");
    check::is_true(tnc::unpack(pl, {true, true}) == std::pair{packets, 0}, "unpack all good");
    check::is_true(tnc::unpack(pl, {true, false}) == std::pair{std::vector<tnc::Bytes>{packets[0]}, 2}, "unpack lost");
    bool threw = false;
    try {
        tnc::pack({tnc::Bytes(tnc::capacity(spec))}, spec);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check::is_true(threw, "pack past capacity throws");

    check::current_step = "receiver";
    const auto& s = *codes::submode("qpsk-r1/2");
    std::vector<std::vector<std::uint8_t>> sent(3);
    for (auto& p : sent)
        for (int i = 0; i < s.payload_bytes; ++i) p.push_back(static_cast<std::uint8_t>(rng() & 0xFF));
    const auto x = modem::modulate(sent, s);
    std::vector<double> y(3 * config::FS, 0.0);
    const std::size_t at = 2 * config::FS + 123;
    y.resize(at + x.size() + 2 * config::FS, 0.0);
    std::normal_distribution<double> n(0.0, 0.1);
    for (std::size_t i = 0; i < y.size(); ++i) y[i] = (i >= at && i < at + x.size() ? 0.5 * x[i - at] : 0.0) + n(rng);
    const auto acc = modem::Accept::of({}, 16.0);
    std::vector<std::int64_t> starts;
    for (std::size_t chunk : {800, 400}) {
        tnc::Receiver rx(acc, {"c8r50"});
        const auto ev = feed_all(rx, y, chunk);
        const std::string c = " (chunk " + std::to_string(chunk) + ")";
        check::equal(static_cast<int>(ev.size()), 2, "events" + c);
        if (ev.size() != 2) continue;
        const auto* h = std::get_if<tnc::HeaderEvent>(&ev[0]);
        const auto* b = std::get_if<tnc::BurstEvent>(&ev[1]);
        check::is_true(h && !h->header.is_cpm() && h->header.ofdm().spec == &s && h->header.n_cw() == 3, "header" + c);
        check::is_true(b && b->rx && std::holds_alternative<modem::Received>(*b->rx), "burst received" + c);
        if (b && b->rx) {
            const auto d = modem::decode_received(std::get<modem::Received>(*b->rx));
            check::is_true(d.payloads == sent, "payloads" + c);
            starts.push_back(b->header.start());
        }
        check::is_true(!rx.busy(), "idle after" + c);
    }
    check::is_true(starts.size() == 2 && starts[0] == starts[1] &&
                       std::abs(starts[0] - static_cast<std::int64_t>(at + config::LEADIN_SAMPLES)) < 40,
                   "burst start, chunking independent");

    check::current_step = "silence";
    tnc::Receiver quiet(acc);
    check::is_true(feed_all(quiet, std::vector<double>(5 * config::FS, 0.0), 800).empty() && !quiet.busy(), "silence");

    check::current_step = "receive_any";
    const auto r = tnc::receive_any(std::span(y).subspan(at - config::FS / 2), config::FS);
    check::is_true(r && std::holds_alternative<modem::Received>(*r) && std::get<modem::Received>(*r).spec == &s,
                   "receive_any");
    return check::report("test_tnc");
}
