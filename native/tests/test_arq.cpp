// The gear shifter with no Python: known answers from the reference (burst
// timing, encoding, slots), predictor sanity, and a recommendation that
// respects the cap. Parity: tests/test_native_arq.py.

#include <cmath>
#include <string>

#include "arq/policy.hpp"
#include "check.hpp"
#include "modem/timing.hpp"

using namespace data2g;

namespace {

arq::Measured clean(double snr_db, int nc) {
    arq::Measured m;
    m.snr_est = snr_db;
    m.spread_est = 0.1;
    const double c_snr = snr_db + 10 * std::log10(2500.0 / 50 / nc);
    for (std::size_t i = 0; i < arq::CONSTS.size(); ++i) m.mi[i] = arq::capacity(c_snr, arq::CONSTS[i]);
    return m;
}

}  // namespace

int main() {
    check::report_crashes_instead_of_prompting();
    check::Watchdog watchdog(120, "test_arq");

    check::current_step = "timing";
    check::equal(modem::burst_seconds(arq::mode_at("qpsk-r1/2").ofdm[0], 3), 4.088, "qpsk-r1/2 x3 seconds");
    check::equal(modem::header_samples("n4"), 28 * config::NSYM, "n4 header: 24 symbols, 4 pilots");
    check::equal(modem::header_samples("n10"), 12 * config::NSYM, "n10 header: 10 symbols, a pilot, the closing pilot");

    check::current_step = "modes";
    check::equal(static_cast<int>(arq::modes().size()), 48, "modes");
    check::equal(arq::encode("qpsk-r1/2"), 7, "encode qpsk-r1/2");
    check::equal(arq::encode("fsk8r50-r1/2"), 51, "encode fsk8r50-r1/2");
    for (const auto& m : arq::modes()) check::is_true(arq::decode(arq::encode(m.name)) == &m, std::string(m.name) + " roundtrip");
    check::is_true(arq::decode(0x3F) == nullptr, "no such cpm mode");
    check::equal(arq::slots_for(arq::mode_at("w48-16qam-r1/2"), 6.0), 9, "w48-16qam-r1/2 slots in 6 s");
    check::equal(arq::slots_for(arq::mode_at("fsk8r50-r1/2"), 3.0), 2, "fsk8r50-r1/2 slots in 3 s");
    check::equal(arq::ctl_slots(arq::mode_at("ack-4f")), 3, "ack-4f control slots");
    for (const auto* m : arq::allowed(0)) check::is_true(arq::width_hz(*m) <= 500, std::string(m->name) + " in 500 Hz");

    check::current_step = "predictor";
    check::close(std::vector{arq::capacity(0.0, "gray-qam4")}, std::vector{0.4916849579383967}, 1e-15, "capacity");
    check::equal(arq::capacity(100.0, "c64-w48-r12"), 1.0, "capacity clamps");
    const auto lo = arq::predict_outcome(clean(-2, 24), "w", 2.5, 6.0);
    const auto hi = arq::predict_outcome(clean(12, 24), "w", 2.5, 6.0);
    for (const char* name : {"qpsk-r1/2", "16qam-r1/2", "w48-16qam-r2/3"}) {
        const int i = arq::outcome_index(name);
        check::is_true(hi[i].burst * hi[i].cw > lo[i].burst * lo[i].cw, std::string(name) + " monotone in SNR");
    }

    check::current_step = "shifter";
    arq::GearShifter g;
    arq::StationView st;
    st.cap = 0;
    check::equal(std::string(g.choose(st, 0).first), std::string("n10-ack-4f"), "nothing heard: fallback");
    g.observe(clean(15, 10), "n10-qpsk-r1/2", 0.0);
    const auto r = g.recommend(st);
    const auto* m = arq::decode(r.data);
    check::is_true(m && arq::width_hz(*m) <= 500, "recommendation within the cap");
    st.peer_recommend = r.data;
    st.peer_size_hint = r.hint;
    check::equal(std::string(g.choose(st, 0).first), std::string(m->name), "follows the recommendation");
    check::equal(std::string(g.choose(st, 1).first), std::string("n10-ack-4f"), "escalation: robust");
    g.predicted = {{"qpsk-r1/3", {0.5, 0.9}}};
    g.outcome("qpsk-r1/3", 0, 7, true);
    check::is_true(g.bias_burst["qpsk-r1/3"] > 0 && g.bias["qpsk-r1/3"] < -0.8, "control is not a data codeword");

    return check::report("test_arq");
}
