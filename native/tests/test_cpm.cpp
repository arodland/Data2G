// CPM round trips with no Python: every mode's burst (without the TX
// bandpass, which is dsp's) locks at its front, reads its header, and its
// soft bits give back the coded bits; the wide toolbox grids' too, and their
// goldens (data2g/cpm.py's values). Parity: tests/test_native_cpm.py.

#include <algorithm>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "check.hpp"
#include "config/config.hpp"
#include "cpm/cpm.hpp"

using namespace data2g;

int main() {
    check::report_crashes_instead_of_prompting();
    check::Watchdog watchdog(300, "test_cpm");
    std::mt19937 rng(1);
    std::normal_distribution<double> noise(0.0, 0.05);

    std::vector<const cpm::Spec*> all;
    for (auto table : {tables::CPM_SPECS, tables::CPM_WIDE_SPECS})
        for (const auto& s : table) all.push_back(&s);
    for (const cpm::Spec* sp : all) {
        const auto& s = *sp;
        const std::string name(s.name);
        check::current_step = "round trip";
        const auto& g = cpm::grid_of(s);
        check::equal(g.m, 1 << g.bits, name + " m = 2^bits");
        const auto L = cpm::layout(g, cpm::stream_symbols(g, 2, true));
        check::equal(static_cast<int>(L.sync_rows.size() + L.data_rows.size() + 2 * g.hdr_len), L.n, name + " rows");

        for (bool dup : {false, true}) {
            const int n_data = 2;
            std::vector<std::vector<std::uint8_t>> coded;
            for (int k = 0; k < 1 + dup + n_data; ++k) {
                const int n = k < 1 + dup ? cpm::ctl(g).coded_bits : s.coded_bits;
                std::vector<std::uint8_t> c(n);
                for (auto& b : c) b = rng() & 1;
                if (dup && k == 1) c = coded[0];
                coded.push_back(c);
            }
            const auto x = cpm::modulate(s, coded, dup);
            const double secs = cpm::burst_seconds(s, static_cast<int>(coded.size()), dup);
            check::equal(static_cast<double>(x.size()) / config::FS + 2 * tables::CPM.ramp_s, secs, name + " burst_seconds");

            const long lead = 2000;
            std::vector<double> y(x.size() + 2 * lead);
            for (std::size_t i = 0; i < y.size(); ++i)
                y[i] = noise(rng) + (i >= lead && i < lead + x.size() ? x[i - lead] : 0.0);
            const auto lock = cpm::find(g, y);
            check::is_true(lock.has_value(), name + " locks");
            if (!lock) continue;
            check::is_true(std::abs(lock->start - lead) < g.T / 4, name + " lock at the front");
            check::is_true(lock->spec == &s && lock->n_data == n_data && lock->dup == dup, name + " header");
            check::equal(lock->end, lead + static_cast<long>(x.size()), name + " end");

            const auto r = cpm::receive(y, *lock);  // refine, soft, the coherent control (the wide grids' sub-windows)
            check::equal(r.Z.rows, static_cast<std::size_t>(cpm::layout(g, cpm::stream_symbols(g, n_data, dup)).n * g.L),
                         name + " sub-window bins");
            check::equal(r.soft.size(), coded.size(), name + " slots");
            int errors = 0;
            for (std::size_t k = 0; k < coded.size(); ++k) {
                if (static_cast<int>(k) < 1 + dup) {  // control: bit LLRs
                    for (std::size_t i = 0; i < coded[k].size(); ++i) errors += (r.soft[k][i] < 0) != (coded[k][i] == 1);
                    continue;
                }
                const int p = g.bits, q = 1 << p;  // data: GF(M) symbol likelihoods, the sent value's the largest
                for (std::size_t j = 0; j * p < coded[k].size(); ++j) {
                    int v = 0;
                    for (int b = 0; b < p; ++b) v = v << 1 | coded[k][j * p + b];
                    const double* l = &r.soft[k][j * q];
                    errors += static_cast<int>(std::max_element(l, l + q) - l) != v;
                }
            }
            check::equal(errors, 0, name + " hard decisions");
        }
    }

    check::current_step = "noise";
    std::vector<double> y(40 * 320);
    for (auto& v : y) v = noise(rng);
    for (auto table : {tables::CPM_GRIDS, tables::CPM_WIDE_GRIDS})
        for (const auto& g : table) check::is_true(!cpm::find(g, y), std::string(g.name) + " no lock on noise");

    // data2g/cpm.py's values for the wide grids (python -c "from data2g import cpm; ...")
    check::current_step = "wide goldens";
    const auto* w8 = cpm::grid("c8w50");
    const auto* w16 = cpm::grid("c16w125");
    check::is_true(w8 && w16 && !cpm::grid("c8w51"), "wide grids found by name");
    if (w8 && w16) {
        check::equal(w8->T, 320, "c8w50 T");
        check::equal(w8->hdr_len, 12, "c8w50 header symbols");
        check::equal(cpm::nsub(*w8), 160, "c8w50 sub-window");
        check::equal(cpm::baud(*w8), 25.0, "c8w50 baud");
        check::equal(w16->hdr_len, 16, "c16w125 header symbols");
        check::equal(cpm::layout(*w8, cpm::stream_symbols(*w8, 0, false)).n, 166, "c8w50 control-only rows");
        check::equal(cpm::hdr_span(*w8, 1), std::int64_t{8960}, "c8w50 hdr_span 1");
        check::equal(cpm::hdr_span(*w8, 2), std::int64_t{34560}, "c8w50 hdr_span 2");
        check::equal(cpm::hdr_span(*w16, 2), std::int64_t{29696}, "c16w125 hdr_span 2");
        check::equal(w8->noise_median, 1.6783469900166612, "gamma(2) median");
        const std::vector<int> h8 = {5, 2, 1, 3, 3, 0, 3, 4, 2, 3, 0, 0};  // value 64: mode index 1, no data
        const auto t8 = cpm::header_symbols(*w8, 64);
        check::is_true(std::equal(h8.begin(), h8.end(), t8.begin(), t8.end()), "c8w50 header tones of 64");
        check::equal(cpm::burst_seconds(*cpm::spec("fsk8w50-r1/2"), 3), 32.260000000000005, "fsk8w50-r1/2 3 slots");
        check::equal(cpm::burst_seconds(cpm::ctl(*w16), 1), 2.676, "fsk16w125-ctl alone");
        check::is_true(cpm::coherent(*w8) && cpm::coh_w(*w8, true) == 4 && cpm::coh_w(*w8) == 8, "c8w50 coherent windows");
        check::is_true(cpm::ctl(*w8).name == "fsk8w50-ctl", "c8w50's control");
        // off air unless DATA2G_CPM_ONAIR names it
        const auto& on = cpm::grids();
        const bool named = std::getenv("DATA2G_CPM_ONAIR") && std::string(std::getenv("DATA2G_CPM_ONAIR")).find("c8w50") != std::string::npos;
        check::equal(std::find(on.begin(), on.end(), w8) != on.end(), named, "c8w50 on air as DATA2G_CPM_ONAIR says");
    }
    return check::report("test_cpm");
}
