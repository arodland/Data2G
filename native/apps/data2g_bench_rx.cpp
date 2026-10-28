// data2g-bench-rx: a received burst's decode latency by stage, as a session decodes it, for timing on a slow
// machine with no Python. Each file is one burst (made by a script of numpy's channel simulator):
//
//   data2g-bench-rx [--threads N] [--reps R] [--dd-budget S] BURST.d2gb ...
//   data2g-bench-rx --listen SECONDS [--cpm-only]   (the streaming Receiver on noise, per 20 ms feed)
//   data2g-bench-rx --shifter N                     (GearShifter::recommend on random observations)
//   data2g-bench-rx --listen S --expect             (... an expected slot 1 s ahead every 8 s, all CPM grids: #135)
//   data2g-bench-rx --primed BURST.d2gb ...         (+ cpm::prime_lock on the true start's window and PRIME_TRIES
//                                                    decode_known on slot 0, the last one right: #135)
//   data2g-bench-rx --cwid                          (cwid::audio of a long ID: #134)
//
// Per burst, the best of R runs: receive (head-limited, as receiver::Receiver), soft bits, the link's prepare
// (every slot's first pass on the pool) and the slot-by-slot decodes with their DD, then the payloads right.
//
// File (little-endian): "D2GB", i64 n_audio, i64 head, i32 n_cw, i32 payload_bytes, char[8] band (NUL
// padded), f64[n_audio] audio, n_cw payloads. Slot i was sent under MaskId{7, 1, i} at burst position i.
// A CPM burst (band: its grid) has control slot 0 (ctl_mask(1, 0, 7)) and is followed by char[16] mode,
// i64 start, f64 cfo: the true lock, used (and counted as "genie") when cpm::find misses it; then optionally
// i32 n_ctl (2: the control sent twice, ARQ_DUP, slots 0 and 1 at RV 0 and 1).

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

#include <cmath>
#include <random>

#include "arq/phy.hpp"
#include "arq/policy.hpp"
#include "cpm/cpm.hpp"
#include "receiver/receiver.hpp"
#include "modem/modem.hpp"
#include "util/pool.hpp"
#include "arq/modes.hpp"

// measurement-only feature tests: #135 (primed reception), #134 (CW ID)
#if __has_include("../tests/test_primed.cpp")
#define BENCH_PRIMED 1
#else
#define BENCH_PRIMED 0
#endif
#if __has_include("cwid/cwid.hpp")
#define BENCH_CWID 1
#include "cwid/cwid.hpp"
#else
#define BENCH_CWID 0
#endif

using namespace data2g;

namespace {

struct Burst {
    std::string name, band;
    std::vector<double> audio;
    std::int64_t head = 0;
    std::vector<arq::Bytes> payloads;
    std::string mode;  // CPM only
    std::int64_t start = 0;
    double cfo = 0.0;
    int n_ctl = 1;
};

Burst load(const char* path) {
    std::ifstream f(path, std::ios::binary);
    char magic[4];
    std::int64_t n = 0;
    std::int32_t n_cw = 0, nb = 0;
    char band[9] = {};
    Burst b;
    f.read(magic, 4);
    f.read(reinterpret_cast<char*>(&n), 8);
    f.read(reinterpret_cast<char*>(&b.head), 8);
    f.read(reinterpret_cast<char*>(&n_cw), 4);
    f.read(reinterpret_cast<char*>(&nb), 4);
    f.read(band, 8);
    if (!f || std::memcmp(magic, "D2GB", 4) != 0) {
        std::fprintf(stderr, "%s: not a burst file\n", path);
        std::exit(1);
    }
    b.audio.resize(static_cast<std::size_t>(n));
    f.read(reinterpret_cast<char*>(b.audio.data()), n * 8);
    b.payloads.assign(static_cast<std::size_t>(n_cw), arq::Bytes(static_cast<std::size_t>(nb)));
    for (auto& p : b.payloads) f.read(reinterpret_cast<char*>(p.data()), nb);
    if (cpm::grid(band)) {
        char mode[17] = {};
        f.read(mode, 16);
        f.read(reinterpret_cast<char*>(&b.start), 8);
        f.read(reinterpret_cast<char*>(&b.cfo), 8);
        b.mode = mode;
        std::int32_t nc = 0;
        if (f.read(reinterpret_cast<char*>(&nc), 4)) b.n_ctl = nc;
        else f.clear();
    }
    if (!f) {
        std::fprintf(stderr, "%s: truncated\n", path);
        std::exit(1);
    }
    b.name = path;
    if (const auto s = b.name.rfind('/'); s != std::string::npos) b.name = b.name.substr(s + 1);
    b.band = band;
    return b;
}

double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

struct Run {
    double receive = 0, soft = 0, prepare = 0, decode = 0;
    double acq = 0, refine = 0;  // CPM: the lock search on the head (live, during the burst), refine (in receive)
    double plock = 0, primed = 0;  // --primed: prime_lock on the window, PRIME_TRIES decode_known
    bool genie = false, plocked = false, pok = false;
    int ok = 0, refines = 0;
    double total() const { return receive + soft + prepare + decode; }
};

bool primed_on = false;

Run run(const Burst& b, std::optional<double> dd_budget) {
    Run out;
    arq::Heard h;
    int n_cw = 0;
    if (const cpm::Grid* g = cpm::grid(b.band)) {
        const std::span<const double> x(b.audio);
        double t = now();
        auto lock = cpm::find(*g, x.first(static_cast<std::size_t>(std::min<std::int64_t>(b.head, x.size()))),
                              std::nullopt, 150.0, true, 0, std::nullopt, true);
        if (lock && lock->wait) lock = cpm::late_lock(*g, x, *lock);
        out.acq = now() - t;
        const int n_data = static_cast<int>(b.payloads.size()) - b.n_ctl;
        const auto bad = [&](const std::optional<cpm::Lock>& l) {
            return !l || !l->spec || l->n_data != n_data || std::abs(l->start - b.start) > g->T;
        };
#if BENCH_PRIMED
        if (primed_on) {  // the expected slot: TOL_S either side of the true start (receiver::Receiver::primed)
            const auto tol = static_cast<std::int64_t>(receiver::ReplySlot::TOL_S * config::FS);
            t = now();
            auto pl = cpm::prime_lock(*g, x, std::max<std::int64_t>(0, b.start - tol), b.start + tol, b.cfo);
            out.plock = now() - t;
            if (bad(lock) && !bad(pl)) lock = pl, out.plocked = true;
        }
#endif
        if (bad(lock)) {
            const cpm::Spec* sp = cpm::spec(b.mode);
            lock = cpm::Lock{sp, n_data, b.n_ctl == 2, b.start, b.cfo, 0, 0, 0, b.start, b.start, false};
            out.genie = true;
        }
#if __has_include("gfldpc/gfldpc.hpp")  // cpm-acq-gap's refine, timed alone (receive runs it too)
        t = now();
        (void)cpm::refine(*g, x, lock->start, lock->cfo, *lock->spec, lock->n_data, lock->dup);
        out.refine = now() - t;
#endif
        t = now();
        auto r = cpm::receive(x, *lock);
        out.receive = now() - t;
        n_cw = r.n_cw;
        h.cpm = std::make_shared<const cpm::Received>(std::move(r));
    } else {
        const std::string_view bands[1] = {b.band};
        double t = now();
        auto r = modem::receive(b.audio, bands, nullptr, b.head);
        out.receive = now() - t;
        n_cw = r.n_cw;
        h.ofdm = std::make_shared<const modem::Received>(std::move(r));
    }
    if (n_cw != static_cast<int>(b.payloads.size())) {
        std::fprintf(stderr, "%s: heard %d codewords, sent %zu\n", b.name.c_str(), n_cw, b.payloads.size());
        std::exit(1);
    }
    double t;
    t = now();
    auto soft = arq::soft_bits(h);
    out.soft = now() - t;
    arq::SoftStore store;
    arq::ModemRx rx(h, &store, dd_budget, soft);
    std::vector<arq::DecodeRequest> reqs;
    const bool dup = h.cpm && b.n_ctl == 2;
    for (int i = 0; i < n_cw; ++i)
        if (!(dup && i == 1))  // a dup control's RV 1: read as the link does, a pair when slot 0 fails alone
            reqs.push_back({i, h.cpm && i == 0 ? arq::ctl_mask(1, 0, 7) : arq::MaskId{7, 1, i}, 0,
                            arq::SoftKey{false, 1, i, 0}});
    t = now();
    rx.prepare(reqs);
    out.prepare = now() - t;
    t = now();
    const auto right = [&](const std::optional<arq::Bytes>& got, int slot) {
        const auto& want = b.payloads[static_cast<std::size_t>(slot)];  // a CPM control payload: padded
        return got && got->size() <= want.size() && std::equal(got->begin(), got->end(), want.begin());
    };
    for (const auto& q : reqs) {
        auto got = rx.decode(q.slot, q.mask, q.rv, &q.key);
        if (!got && dup && q.slot == 0) {  // Station::ctl_pair
            const arq::SoftKey k{true, 0, 0, 0};
            rx.decode(0, q.mask, 0, &k);
            got = rx.decode(1, q.mask, 1, &k);
            rx.forget(k);
        }
        out.ok += right(got, q.slot) * (dup && q.slot == 0 ? 2 : 1);
    }
    out.decode = now() - t;
#if BENCH_PRIMED
    if (primed_on) {  // a failed control's retries (Station::primed): PRIME_TRIES hypotheses, the last one right
        const auto& want = b.payloads[0];
        const arq::MaskId m = h.cpm ? arq::ctl_mask(1, 0, 7) : arq::MaskId{7, 1, 0};
        const std::size_t cpb =
            h.cpm ? static_cast<std::size_t>(arq::ctl_payload_bytes(arq::mode_at(b.mode))) : want.size();
        const arq::Bytes tmpl(want.begin(), want.begin() + static_cast<std::ptrdiff_t>(std::min(cpb, want.size())));
        arq::Bytes tmask(tmpl.size(), 0xFF);
        tmask[2] = 0, tmask[3] = 0x0F;  // 12 core bits unknown (cum, lost, K's top)
        t = now();
        for (int k = 0; k < 6; ++k) {
            arq::Bytes tt = tmpl;
            if (k < 5) tt[0] ^= static_cast<std::uint8_t>(1 << k);  // a wrong core field
            const auto got = rx.decode_known(0, m, tt, tmask, dup);
            if (k == 5) out.pok = got && std::equal(got->begin(), got->end(), want.begin());
        }
        out.primed = now() - t;
    }
#endif
    out.refines = rx.dd_refines();
    return out;
}

[[noreturn]] void usage(int code) {
    std::fprintf(code ? stderr : stdout,
                 "usage: data2g-bench-rx [--threads N] [--reps R] [--dd-budget S] BURST.d2gb ...\n"
                 "  --threads     the decode pool's size (default 4)\n"
                 "  --reps        runs per burst, the fastest reported (default 3)\n"
                 "  --dd-budget   seconds of DD per burst (default 1; 0: none; inf: no limit)\n");
    std::exit(code);
}

// The streaming Receiver on Gaussian noise, 20 ms feeds: CPU per feed (the listening load per audio feed).
bool cwsense_on = BENCH_CWID;  // --no-cwsense: #134's tree without its detector (#135's listening load alone)

void listen(double seconds, bool cpm_only, int expect) {
    std::vector<std::string_view> grids;
    for (const auto& g : tables::CPM_GRIDS) grids.push_back(g.name);
    receiver::Receiver rx(cpm_only ? modem::Accept{} : modem::Accept::of(), grids);
#if BENCH_CWID
    cwid::CwSense cw;
#endif
    std::int64_t fed = 0;
    int windows = 0;
    std::mt19937_64 rng(1);
    std::normal_distribution<double> nd(0.0, 0.1);
    const int n = config::FS / 50;
    std::vector<double> x(n), ts;
    const int feeds = static_cast<int>(seconds * 50);
    for (int k = 0; k < feeds; ++k) {
        for (auto& v : x) v = nd(rng);
#if BENCH_PRIMED
        if (expect && k % 400 == 100) {  // every 8 s (prime_lock waits ~3.9 s for both header copies), a reply
            // expected 1 s on (Engine::expect_reply), in the first `expect` grids
            const std::int64_t at = fed + config::FS, tol = static_cast<std::int64_t>(0.15 * config::FS);
            rx.expect(std::vector<std::string_view>(grids.begin(), grids.begin() + std::min<std::size_t>(expect, grids.size())),
                      at - tol, at + tol, 0.0);
            ++windows;
        }
#endif
        const double t = now();
#if BENCH_CWID
        if (cwsense_on) (void)cw.feed(x);
#endif
        (void)rx.feed(x);
        ts.push_back(now() - t);
        fed += n;
    }
    ts.erase(ts.begin(), ts.begin() + std::min<std::size_t>(ts.size(), 250));  // the first 5 s
    double sum = 0;
    for (double v : ts) sum += v;
    std::sort(ts.begin(), ts.end());
    const auto q = [&](double p) { return 1e6 * ts[static_cast<std::size_t>(p * (ts.size() - 1))]; };
    std::printf("listen %s%s%s%d: %.0f us per 20 ms feed (%.2f%% of a core), p50 %.0f p99 %.0f p99.9 %.0f max %.0f us over "
                "%zu feeds, %d windows\n",
                cpm_only ? "cpm only" : "all bands", cwsense_on ? " +cwsense" : "", expect ? " +expect " : "", expect,
                1e6 * sum / ts.size(), 100 * sum / (ts.size() * 0.02), q(0.5), q(0.99), q(0.999), 1e6 * ts.back(),
                ts.size(), windows);
}

// GearShifter::recommend on random observations (runs/cor_latency.py's, simplified): half under the CPM floor.
void shifter(int steps) {
    std::mt19937_64 rng(1);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    const auto modes = arq::allowed(2);
    arq::GearShifter g;
    double now_s = 0;
    std::vector<double> ts;
    for (int step = 0; step < steps; ++step) {
        const arq::Mode* m = modes[static_cast<std::size_t>(u(rng) * modes.size())];
        arq::Measured me;
        me.snr_est = step % 2 ? -12 + 8 * u(rng) : -8 + 33 * u(rng);
        me.spread_est = u(rng) < 0.5 ? 0.0 : 3 * u(rng);
        me.delay_est_ms = 4 * u(rng);
        const double bits[4] = {2, 4, 6, 8};
        for (int c = 0; c < 4; ++c) me.mi[c] = std::min(bits[c], std::log2(1 + std::pow(10, me.snr_est / 10)));
        now_s += 5;
        g.observe(me, m->name, now_s);
        g.observe_energy(-12 + 42 * u(rng), now_s);
        arq::StationView st;
        st.cap = 2;
        st.pending = u(rng) < 0.7;
        st.chat = u(rng) < 0.2;
        const double t = now();
        (void)g.recommend(st);
        ts.push_back(now() - t);
    }
    ts.erase(ts.begin(), ts.begin() + std::min<std::size_t>(ts.size(), 20));
    std::sort(ts.begin(), ts.end());
    const auto q = [&](double p) { return 1e3 * ts[static_cast<std::size_t>(p * (ts.size() - 1))]; };
    std::printf("recommend: median %.2f ms, p90 %.2f ms, max %.2f ms over %zu\n", q(0.5), q(0.9), 1e3 * ts.back(),
                ts.size());
}

}  // namespace

#if BENCH_CWID
void cwid_audio() {  // a long call at the slowest and default speeds, best of 5
    for (double wpm : {cwid::MIN_WPM, cwid::WPM}) {
        double best = 1e9;
        std::size_t n = 0;
        for (int k = 0; k < 5; ++k) {
            const double t = now();
            n = cwid::audio(cwid::id_text("VE3ABC/QRP"), wpm).size();
            best = std::min(best, now() - t);
        }
        std::printf("cwid audio %.0f wpm: %.2f ms for %.2f s of audio\n", wpm, 1e3 * best,
                    static_cast<double>(n) / config::FS);
    }
}
#endif

int main(int argc, char** argv) {
    int threads = 4, reps = 3, shifter_n = 0;
    double listen_s = 0;
    bool cpm_only = false, cw = false;
    int expect = 0;
    std::optional<double> dd_budget = arq::DD_BUDGET_S;
    std::vector<const char*> files;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const bool more = i + 1 < argc;
        if (a == "-h" || a == "--help") usage(0);
        else if (a == "--threads" && more) threads = std::atoi(argv[++i]);
        else if (a == "--reps" && more) reps = std::max(1, std::atoi(argv[++i]));
        else if (a == "--listen" && more) listen_s = std::atof(argv[++i]);
        else if (a == "--cpm-only") cpm_only = true;
        else if (a == "--expect" && more) expect = std::atoi(argv[++i]);
        else if (a == "--primed") primed_on = true;
        else if (a == "--cwid") cw = true;
        else if (a == "--no-cwsense") cwsense_on = false;
        else if (a == "--shifter" && more) shifter_n = std::atoi(argv[++i]);
        else if (a == "--dd-budget" && more) {
            const std::string v = argv[++i];
            dd_budget = v == "inf" ? std::nullopt : std::optional<double>(std::atof(v.c_str()));
        } else if (a[0] == '-') usage(2);
        else files.push_back(argv[i]);
    }
    pool::set_threads(threads);
#if BENCH_CWID
    if (cw) cwid_audio();
#endif
    if (cw && listen_s <= 0 && shifter_n <= 0) return 0;
    if (listen_s > 0) listen(listen_s, cpm_only, expect);
    if (shifter_n > 0) shifter(shifter_n);
    if (listen_s > 0 || shifter_n > 0) return 0;
    if (files.empty()) usage(2);
    std::printf("%-44s %5s %7s %7s %8s %6s %8s %8s %8s %4s %7s %7s\n", "burst", "ok", "acq", "refine", "receive",
                "soft", "prepare", "decode", "total", "dd", "plock", "primed");
    for (const char* path : files) {
        const Burst b = load(path);
        Run best;
        for (int k = 0; k < reps; ++k) {
            const Run r = run(b, dd_budget);
            if (k == 0 || r.total() < best.total()) best = r;
        }
        std::printf("%-44s %2d/%-2zu %7.1f %7.1f %8.1f %6.1f %8.1f %8.1f %8.1f %4d %7.1f %7.1f%s%s%s\n",
                    b.name.c_str(), best.ok, b.payloads.size(), 1e3 * best.acq, 1e3 * best.refine,
                    1e3 * best.receive, 1e3 * best.soft, 1e3 * best.prepare, 1e3 * best.decode, 1e3 * best.total(),
                    best.refines, 1e3 * best.plock, 1e3 * best.primed, best.genie ? " genie" : "",
                    best.plocked ? " plock" : "", best.pok ? " pok" : "");
    }
}
