// data2g-bench-rx: a received burst's decode latency by stage, as a session decodes it, for timing on a slow
// machine with no Python. Each file is one burst (made by a script of numpy's channel simulator):
//
//   data2g-bench-rx [--threads N] [--reps R] [--dd-budget S] BURST.d2gb ...
//
// Per burst, the best of R runs: receive (head-limited, as receiver::Receiver), soft bits, the link's prepare
// (every slot's first pass on the pool) and the slot-by-slot decodes with their DD, then the payloads right.
//
// File (little-endian): "D2GB", i64 n_audio, i64 head, i32 n_cw, i32 payload_bytes, char[8] band (NUL
// padded), f64[n_audio] audio, n_cw payloads. Slot i was sent under MaskId{7, 1, i} at burst position i.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <optional>
#include <cmath>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

#include "arq/phy.hpp"
#include "modem/modem.hpp"
#include "receiver/receiver.hpp"
#include "util/pool.hpp"

using namespace data2g;

namespace {

struct Burst {
    std::string name, band;
    std::vector<double> audio;
    std::int64_t head = 0;
    std::vector<arq::Bytes> payloads;
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
    int ok = 0, refines = 0;
    double total() const { return receive + soft + prepare + decode; }
};

Run run(const Burst& b, std::optional<double> dd_budget) {
    Run out;
    const std::string_view bands[1] = {b.band};
    double t = now();
    auto r = modem::receive(b.audio, bands, nullptr, b.head);
    out.receive = now() - t;
    const int n_cw = r.n_cw;
    if (n_cw != static_cast<int>(b.payloads.size())) {
        std::fprintf(stderr, "%s: heard %d codewords, sent %zu\n", b.name.c_str(), n_cw, b.payloads.size());
        std::exit(1);
    }
    arq::Heard h{std::make_shared<const modem::Received>(std::move(r)), nullptr};
    t = now();
    auto soft = arq::soft_bits(h);
    out.soft = now() - t;
    arq::SoftStore store;
    arq::ModemRx rx(h, &store, dd_budget, soft);
    std::vector<arq::DecodeRequest> reqs;
    for (int i = 0; i < n_cw; ++i) reqs.push_back({i, arq::MaskId{7, 1, i}, 0, arq::SoftKey{false, 1, i, 0}});
    t = now();
    rx.prepare(reqs);
    out.prepare = now() - t;
    t = now();
    for (const auto& q : reqs) {
        const auto got = rx.decode(q.slot, q.mask, q.rv, &q.key);
        out.ok += got && *got == b.payloads[static_cast<std::size_t>(q.slot)];
    }
    out.decode = now() - t;
    out.refines = rx.dd_refines();
    return out;
}

// --cpm: a synthetic CPM burst (`mode`, control + n_data data codewords, noise sigma per unit-RMS sample) fed to
// a receiver in 0.1 s blocks, streamed (cpm::Stream) or not: the end-of-burst latency, from the feed that holds
// its end to every data slot decoded (receive, ModemRx, the slots' passes), and slots decoded.
struct CpmRun {
    double end = 0;
    int ok = 0, early = 0;
};

CpmRun run_cpm(const std::string& mode, int n_data, double sigma, bool stream, std::optional<double> dd_budget) {
    const cpm::Spec& cs = *cpm::spec(mode);
    const cpm::Grid& g = cpm::grid_of(cs);
    arq::TxBurst b;
    b.submode = mode;
    auto bytes = [](int n, unsigned seed) {
        arq::Bytes p(static_cast<std::size_t>(n));
        for (auto& v : p) v = static_cast<std::uint8_t>((seed = seed * 1103515245u + 12345u) >> 16);
        return p;
    };
    b.slots.push_back({arq::ctl_mask(0, 0, 3), 0, bytes(codes::spec(cpm::ctl(g)).payload_bytes, 9)});
    for (int i = 1; i <= n_data; ++i)
        b.slots.push_back({arq::data_mask(0, i, 3), 0, bytes(codes::spec(cs).payload_bytes, static_cast<unsigned>(i))});
    std::vector<double> y(config::FS, 0.0);
    const auto x = arq::tx_audio(b);
    y.insert(y.end(), x.begin(), x.end());
    y.insert(y.end(), config::FS, 0.0);
    std::uint64_t s = 7;  // xorshift + Box-Muller: the same noise every run
    auto u = [&] {
        s ^= s << 13, s ^= s >> 7, s ^= s << 17;
        return (static_cast<double>(s >> 11) + 0.5) / 9007199254740992.0;
    };
    for (double& v : y) v += sigma * std::sqrt(-2 * std::log(u())) * std::cos(2 * 3.141592653589793 * u());
    receiver::Receiver rx(modem::Accept::of({}, 120.0), {g.name}, false);
    if (stream) rx.stream_cpm(arq::cpm_pass_fn());
    CpmRun out;
    const std::size_t blk = config::FS / 10;
    for (std::size_t i = 0; i < y.size(); i += blk) {
        const double t = now();
        auto ev = rx.feed(std::span<const double>(y).subspan(i, std::min(blk, y.size() - i)));
        for (auto& e : ev) {
            auto* be = std::get_if<receiver::BurstEvent>(&e);
            if (!be || !be->rx) continue;
            const auto& r = std::get<cpm::Received>(*be->rx);
            out.early = static_cast<int>(r.passes.size());
            arq::ModemRx m(arq::Heard{nullptr, std::make_shared<const cpm::Received>(r)}, nullptr, dd_budget);
            for (int j = 1; j <= n_data; ++j)
                out.ok += m.decode_plain(j, arq::data_mask(0, j, 3)) == b.slots[static_cast<std::size_t>(j)].payload;
            out.end = now() - t;
            return out;
        }
    }
    return out;
}

[[noreturn]] void usage(int code) {
    std::fprintf(code ? stderr : stdout,
                 "usage: data2g-bench-rx [--threads N] [--reps R] [--dd-budget S] BURST.d2gb ...\n"
                 "       data2g-bench-rx [--threads N] [--reps R] [--dd-budget S] --cpm MODE N_DATA SIGMA ...\n"
                 "  --cpm         a synthetic CPM burst, batch vs streamed end-of-burst latency (repeatable)\n"
                 "  --threads     the decode pool's size (default 4)\n"
                 "  --reps        runs per burst, the fastest reported (default 3)\n"
                 "  --dd-budget   seconds of DD per burst (default 1; 0: none; inf: no limit)\n");
    std::exit(code);
}

}  // namespace

int main(int argc, char** argv) {
    int threads = 4, reps = 3;
    std::optional<double> dd_budget = arq::DD_BUDGET_S;
    std::vector<const char*> files;
    std::vector<std::tuple<std::string, int, double>> cpms;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const bool more = i + 1 < argc;
        if (a == "-h" || a == "--help") usage(0);
        else if (a == "--cpm" && i + 3 < argc) {
            cpms.emplace_back(argv[i + 1], std::atoi(argv[i + 2]), std::atof(argv[i + 3]));
            i += 3;
        }
        else if (a == "--threads" && more) threads = std::atoi(argv[++i]);
        else if (a == "--reps" && more) reps = std::max(1, std::atoi(argv[++i]));
        else if (a == "--dd-budget" && more) {
            const std::string v = argv[++i];
            dd_budget = v == "inf" ? std::nullopt : std::optional<double>(std::atof(v.c_str()));
        } else if (a[0] == '-') usage(2);
        else files.push_back(argv[i]);
    }
    if (files.empty() && cpms.empty()) usage(2);
    pool::set_threads(threads);
    if (!cpms.empty())
        std::printf("%-16s %6s %5s %10s %10s %6s\n", "cpm burst", "sigma", "ok", "batch ms", "stream ms", "early");
    for (const auto& [mode, n_data, sigma] : cpms) {
        CpmRun bat, str;
        for (int k = 0; k < reps; ++k) {
            const CpmRun a = run_cpm(mode, n_data, sigma, false, dd_budget), c = run_cpm(mode, n_data, sigma, true, dd_budget);
            if (k == 0 || a.end < bat.end) bat = a;
            if (k == 0 || c.end < str.end) str = c;
        }
        std::printf("%-16s %6.2f %2d/%-2d %10.1f %10.1f %6d%s\n", mode.c_str(), sigma, str.ok, n_data, 1e3 * bat.end,
                    1e3 * str.end, str.early, bat.ok == str.ok ? "" : "  (batch decoded differently!)");
    }
    if (files.empty()) return 0;
    std::printf("%-36s %5s %8s %6s %8s %8s %8s %4s\n", "burst", "ok", "receive", "soft", "prepare", "decode", "total", "dd");
    for (const char* path : files) {
        const Burst b = load(path);
        Run best;
        for (int k = 0; k < reps; ++k) {
            const Run r = run(b, dd_budget);
            if (k == 0 || r.total() < best.total()) best = r;
        }
        std::printf("%-36s %2d/%-2zu %8.1f %6.1f %8.1f %8.1f %8.1f %4d\n", b.name.c_str(), best.ok, b.payloads.size(),
                    1e3 * best.receive, 1e3 * best.soft, 1e3 * best.prepare, 1e3 * best.decode, 1e3 * best.total(),
                    best.refines);
    }
}
