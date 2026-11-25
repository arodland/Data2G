// data2g/modem.py's scattered pilots (config.SC_BANDS, the DATA2G_PILOTS study flag): the layout. modem.cpp
// has the rest (transmit, demod_frames, receive, find_copy, pilot_coherence), beside the pilot-row code.
#include "modem/modem.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>

#include "config/config.hpp"

namespace data2g::modem {

using namespace config;
using std::int64_t;
using std::size_t;

namespace {

// modem.SC_STRIDES (the reasons are there)
constexpr std::array<std::pair<int, int>, 4> SC_STRIDES = {{{24, 49}, {48, 97}, {10, 37}, {4, 17}}};

bool env_on(const char* name) {  // Python's os.environ.get(name, "1") == "1"
    const char* v = std::getenv(name);
    return !v || std::string_view(v) == "1";
}

ScCells make_cells(int nc) {
    const int N = SYMS_PER_FRAME * nc;
    const auto layout = sc_layout();
    int g = 2 * nc + 1;  // k3d2's
    if (layout == "mix") {
        g = 0;
        for (const auto& [n, s] : SC_STRIDES)
            if (n == nc) g = s;
        if (!g) throw std::logic_error("scattered pilots: no stride for " + std::to_string(nc) + " carriers");
    } else if (layout == "gold") {
        const int g0 = static_cast<int>(std::nearbyint(N / 1.6180339887));
        g = 0;
        for (int d = 0; d < N && !g; ++d)
            for (int v : {g0 + d, g0 - d})
                if (std::gcd(v, N) == 1) {
                    g = v;
                    break;
                }
    }
    std::vector<char> pil(static_cast<size_t>(N), 0);
    for (int i = 0; i < nc; ++i) pil[static_cast<size_t>((static_cast<int64_t>(i) * g % N + N) % N)] = 1;
    ScCells c;
    c.own.assign(static_cast<size_t>(nc), -1);
    for (int i = 0; i < N; ++i) {
        if (pil[static_cast<size_t>(i)]) {
            c.P.push_back(i);
            c.PK.push_back(i % nc);
            if (c.own[static_cast<size_t>(i % nc)] >= 0) throw std::logic_error("a pilot layout must give each carrier one pilot a frame");
            c.own[static_cast<size_t>(i % nc)] = i;
        } else {
            c.Dc.push_back(i);
        }
    }
    if (std::count(c.own.begin(), c.own.end(), -1)) throw std::logic_error("a pilot layout must give each carrier one pilot a frame");
    return c;
}

}  // namespace

std::vector<std::uint8_t> sc_payload(int n, std::uint64_t& state) {
    std::vector<std::uint8_t> out;
    while (static_cast<int>(out.size()) < n) {
        state += 0x9E3779B97F4A7C15ull;
        std::uint64_t z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        z ^= z >> 31;
        for (int b = 0; b < 8; ++b) out.push_back(static_cast<std::uint8_t>(z >> (8 * b)));
    }
    out.resize(static_cast<size_t>(n));
    return out;
}

const ScFlags& sc_flags() {
    static const ScFlags f = [] {
        ScFlags out;
        out.spread_loo = env_on("DATA2G_SC_SPREAD_LOO");
        out.copy_coherent = env_on("DATA2G_SC_COPY_COHERENT");
        out.copy_rows = env_on("DATA2G_SC_COPY_ROWS");
        if (const char* v = std::getenv("DATA2G_SC_RESTORE_ACE")) out.restore_ace = std::stod(v);
        return out;
    }();
    return f;
}

bool sc_spec(const Spec& spec, int n_f) { return sc_band(spec.band) && n_f > 0; }

const ScCells& sc_cells(int nc) {
    static const auto all = [] {
        std::map<int, ScCells> m;
        for (const auto& b : BANDS) m.emplace(b.nc, make_cells(b.nc));
        return m;
    }();
    return all.at(nc);
}

bool sc_combs(int nc) {
    const auto& c = sc_cells(nc);
    if (nc % 3) return false;
    for (size_t i = 0; i < c.P.size(); ++i)
        if (c.P[i] / nc != 2 * (c.PK[i] % 3)) return false;
    return true;
}

std::vector<cd> copy_open(const waveform::Band& b) {
    std::vector<cd> out(b.pilot.size());
    for (size_t k = 0; k < out.size(); ++k) out[k] = std::conj(b.pilot[k]);
    return out;
}

Mat<int> sc_pil(std::string_view band, int n_air, std::optional<int> kc) {
    if (!sc_band(band)) return {};
    const int nc = config::band(band).nc;
    const auto& cells = sc_cells(nc);
    Mat<int> pil(static_cast<size_t>(n_air + 1), static_cast<size_t>(nc));
    for (int f = 0; f < n_air; ++f)
        if (!kc || f != *kc)
            for (int k = 0; k < nc; ++k) pil[static_cast<size_t>(f)][k] = cells.own[static_cast<size_t>(k)] / nc;
    return pil;
}

}  // namespace data2g::modem
