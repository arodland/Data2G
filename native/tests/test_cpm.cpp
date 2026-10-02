// CPM round trips with no Python: every mode's burst (without the TX
// bandpass, which is dsp's) locks at its front, reads its header, and its
// soft bits give back the coded bits. Parity: tests/test_native_cpm.py.

#include <random>
#include <string>
#include <vector>

#include "check.hpp"
#include "cpm/cpm.hpp"

using namespace data2g;

int main() {
    check::report_crashes_instead_of_prompting();
    check::Watchdog watchdog(300, "test_cpm");
    std::mt19937 rng(1);
    std::normal_distribution<double> noise(0.0, 0.05);

    for (const auto& s : tables::CPM_SPECS) {
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

            const auto r = cpm::soft(g, y, lock->start, lock->cfo, n_data, dup);
            check::equal(r.slots.size(), coded.size(), name + " slots");
            int errors = 0;
            for (std::size_t k = 0; k < coded.size(); ++k)
                for (std::size_t i = 0; i < coded[k].size(); ++i) errors += (r.slots[k][i] < 0) != (coded[k][i] == 1);
            check::equal(errors, 0, name + " hard decisions");
        }
    }

    check::current_step = "noise";
    std::vector<double> y(40 * 320);
    for (auto& v : y) v = noise(rng);
    for (const auto& g : tables::CPM_GRIDS) check::is_true(!cpm::find(g, y), std::string(g.name) + " no lock on noise");
    return check::report("test_cpm");
}
