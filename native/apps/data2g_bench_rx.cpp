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
#include <string>
#include <vector>

#include "arq/phy.hpp"
#include "modem/modem.hpp"
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

[[noreturn]] void usage(int code) {
    std::fprintf(code ? stderr : stdout,
                 "usage: data2g-bench-rx [--threads N] [--reps R] [--dd-budget S] BURST.d2gb ...\n"
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
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const bool more = i + 1 < argc;
        if (a == "-h" || a == "--help") usage(0);
        else if (a == "--threads" && more) threads = std::atoi(argv[++i]);
        else if (a == "--reps" && more) reps = std::max(1, std::atoi(argv[++i]));
        else if (a == "--dd-budget" && more) {
            const std::string v = argv[++i];
            dd_budget = v == "inf" ? std::nullopt : std::optional<double>(std::atof(v.c_str()));
        } else if (a[0] == '-') usage(2);
        else files.push_back(argv[i]);
    }
    if (files.empty()) usage(2);
    pool::set_threads(threads);
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
