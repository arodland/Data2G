#include "convert.hpp"
#include "cpm/cpm.hpp"

namespace data2g::bind {

namespace {

// Grids and specs cross the boundary by name.
const cpm::Grid& grid(const std::string& name) {
    const auto* g = cpm::grid(name);
    if (!g) throw py::key_error("no cpm grid " + name);
    return *g;
}

const cpm::Spec& cspec(const std::string& name) {
    const auto* s = cpm::spec(name);
    if (!s) throw py::key_error("no cpm spec " + name);
    return *s;
}

std::span<const double> view(const In<double>& x) { return {x.data(), static_cast<std::size_t>(x.size())}; }

py::array_t<std::int64_t> ints(std::span<const int> v) { return np<std::int64_t>(v); }

}  // namespace

void bind_cpm(py::module_& m) {
    auto c = m.def_submodule("cpm");
    c.def("grids", [] {
        py::list out;
        for (const auto& g : tables::CPM_GRIDS) {
            py::dict d;
            d["name"] = std::string(g.name);
            d["m"] = g.m;
            d["rate"] = g.rate;
            d["center"] = g.center;
            d["bp"] = g.bp;
            d["clip_db"] = g.clip_db;
            d["T"] = g.T;
            d["bits"] = g.bits;
            d["f0"] = g.f0;
            d["sync_threshold"] = g.sync_threshold;
            d["header_threshold"] = g.header_threshold;
            d["hdr_len"] = g.hdr_len;
            d["costas_len"] = g.costas_len;
            d["preamble"] = np<std::int64_t>(g.preamble);
            d["mid_block"] = np<std::int64_t>(g.mid_block);
            out.append(d);
        }
        return out;
    });
    c.def("specs", [] {
        py::list out;
        for (auto table : {tables::CPM_SPECS, tables::CPM_CTL})
            for (const auto& s : table) {
                py::dict d;
                d["name"] = std::string(s.name);
                d["grid"] = std::string(s.grid);
                d["code"] = std::string(s.code);
                d["index"] = s.index;
                d["k"] = s.k;
                d["coded_bits"] = s.coded_bits;
                d["n_sym"] = s.n_sym;
                out.append(d);
            }
        return out;
    });
    c.def("header_symbols", [](const std::string& g, int value) {
        return np<std::int64_t>(cpm::header_symbols(grid(g), value));
    });
    c.def("header_value", &cpm::header_value);
    c.def("stream_symbols", [](const std::string& g, int n_data, bool dup) {
        return cpm::stream_symbols(grid(g), n_data, dup);
    });
    c.def("layout", [](const std::string& g, int n_sym) {
        const auto L = cpm::layout(grid(g), n_sym);
        py::list hdr;
        for (const auto& r : L.hdr_rows) hdr.append(ints(r));
        py::dict d;
        d["n"] = L.n;
        d["sync_rows"] = ints(L.sync_rows);
        d["sync_tones"] = ints(L.sync_tones);
        d["hdr_rows"] = py::tuple(hdr);
        d["data_rows"] = ints(L.data_rows);
        d["front"] = L.front;
        return d;
    });
    c.def("burst_seconds", [](const std::string& s, int n_cw, bool dup) {
        return cpm::burst_seconds(cspec(s), n_cw, dup);
    }, py::arg("spec"), py::arg("n_cw"), py::arg("dup") = false);
    c.def("to_tones", [](const std::string& g, const In<std::uint8_t>& bits) {
        return ints(cpm::to_tones(grid(g), {bits.data(), static_cast<std::size_t>(bits.size())}));
    });
    c.def("tones", [](const std::string& g, const In<int>& sym) {
        return np(cpm::tones(grid(g), {sym.data(), static_cast<std::size_t>(sym.size())}));
    });
    c.def("modulate", [](const std::string& s, const std::vector<In<std::uint8_t>>& coded, bool dup) {
        std::vector<std::vector<std::uint8_t>> v;
        for (const auto& a : coded) v.push_back(vec(a));
        return np(cpm::modulate(cspec(s), v, dup));
    }, "audio before the TX bandpass");
    c.def("energies", [](const std::string& g, const In<double>& x, long start, int n_sym, double cfo, int extra) {
        return np(cpm::energies(grid(g), view(x), start, n_sym, cfo, extra));
    }, py::arg("grid"), py::arg("x"), py::arg("start"), py::arg("n_sym"), py::arg("cfo"), py::arg("extra") = 0);
    c.def("shares", [](const In<double>& E) { return np(cpm::shares(mat(E))); });
    c.def("detect", [](const std::string& g, const In<double>& x, double reach_hz, bool fine, bool front_only, int n_sym,
                       double floor) {
        const auto d = cpm::detect(grid(g), view(x), reach_hz, fine, front_only, n_sym, floor);
        return py::make_tuple(d.score, d.start, d.cfo);
    }, py::arg("grid"), py::arg("x"), py::arg("reach_hz") = 150.0, py::arg("fine") = true,
       py::arg("front_only") = false, py::arg("n_sym") = 0, py::arg("floor") = -1.0);
    c.def("llrs", [](const std::string& g, const In<double>& E) { return np(cpm::llrs(grid(g), mat(E))); });
    c.def("read_header", [](const std::string& g, const In<double>& x, long s0, double cfo, int copies) {
        const auto h = cpm::read_header(grid(g), view(x), s0, cfo, copies);
        return py::make_tuple(std::string(h.spec->name), h.n_data, h.dup, h.score, h.runner_up);
    }, py::arg("grid"), py::arg("x"), py::arg("s0"), py::arg("cfo"), py::arg("copies") = 2);
    c.def("soft", [](const std::string& g, const In<double>& x, long s0, double cfo, int n_data, bool dup) {
        const auto r = cpm::soft(grid(g), view(x), s0, cfo, n_data, dup);
        py::list slots;
        for (const auto& s : r.slots) slots.append(np(s));
        return py::make_tuple(slots, np(r.E));
    });
    c.def("peak_ratio", [](const std::string& g, const In<double>& x, long s0, double cfo) {
        return cpm::peak_ratio(grid(g), view(x), s0, cfo);
    });
    c.def("find", [](const std::string& g, const In<double>& x, std::optional<double> threshold, double reach_hz,
                     bool front_only, long lo, std::optional<long> hi) -> py::object {
        const auto l = cpm::find(grid(g), view(x), threshold, reach_hz, front_only, lo, hi);
        if (!l) return py::none();
        py::dict d;
        d["spec"] = std::string(l->spec->name);
        d["n_data"] = l->n_data;
        d["dup"] = l->dup;
        d["start"] = l->start;
        d["cfo"] = l->cfo;
        d["score"] = l->score;
        d["header_score"] = l->header_score;
        d["header_margin"] = l->header_margin;
        d["end"] = l->end;
        d["header_end"] = l->header_end;
        return d;
    }, py::arg("grid"), py::arg("x"), py::arg("threshold") = py::none(), py::arg("reach_hz") = 150.0,
       py::arg("front_only") = true, py::arg("lo") = 0, py::arg("hi") = py::none());
    c.def("measure", [](const std::string& g, const In<double>& E, int n_sym) {
        const auto r = cpm::measure(grid(g), mat(E), n_sym);
        py::dict d;
        d["snr_est"] = r.snr_est;
        d["spread_est"] = r.spread_est;
        d["frames"] = r.frames;
        d["snr"] = np(r.snr);
        return d;
    });
}

}  // namespace data2g::bind
