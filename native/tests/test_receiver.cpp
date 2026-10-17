// The streaming Receiver with no Python: a burst in noise (in two
// chunkings), nothing in silence, and receive_any. Parity:
// tests/test_native_receiver.py.

#include <algorithm>
#include <random>
#include <string>

#include "check.hpp"
#include "codes/codes.hpp"
#include "receiver/receiver.hpp"

using namespace data2g;

namespace {

std::vector<receiver::Event> feed_all(receiver::Receiver& rx, const std::vector<double>& y, std::size_t chunk) {
    std::vector<receiver::Event> out;
    for (std::size_t i = 0; i < y.size(); i += chunk) {
        auto ev = rx.feed(std::span(y).subspan(i, std::min(chunk, y.size() - i)));
        out.insert(out.end(), std::make_move_iterator(ev.begin()), std::make_move_iterator(ev.end()));
    }
    return out;
}

}  // namespace

int main() {
    check::report_crashes_instead_of_prompting();
    check::Watchdog watchdog(300, "test_receiver");
    std::mt19937 rng(1);

    check::current_step = "receiver";
    const auto& s = *config::submode("qpsk-r1/2");
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
        receiver::Receiver rx(acc, {"c8r50"});
        const auto ev = feed_all(rx, y, chunk);
        const std::string c = " (chunk " + std::to_string(chunk) + ")";
        check::equal(static_cast<int>(ev.size()), 2, "events" + c);
        if (ev.size() != 2) continue;
        const auto* h = std::get_if<receiver::HeaderEvent>(&ev[0]);
        const auto* b = std::get_if<receiver::BurstEvent>(&ev[1]);
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
    receiver::Receiver quiet(acc);
    check::is_true(feed_all(quiet, std::vector<double>(5 * config::FS, 0.0), 800).empty() && !quiet.busy(), "silence");

    check::current_step = "receive_any";
    const auto r = receiver::receive_any(std::span(y).subspan(at - config::FS / 2), config::FS);
    check::is_true(r && std::holds_alternative<modem::Received>(*r) && std::get<modem::Received>(*r).spec == &s,
                   "receive_any");
    return check::report("test_receiver");
}
