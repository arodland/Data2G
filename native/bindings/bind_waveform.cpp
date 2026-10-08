#include <pybind11/functional.h>

#include "convert.hpp"
#include "dsp/dsp.hpp"
#include "waveform/dsp.hpp"
#include "waveform/ofdm.hpp"
#include "waveform/sync.hpp"

namespace data2g::bind {

namespace {

using waveform::cdouble;

const waveform::Band& band_of(const std::string& name) {
    try {
        return waveform::band(name);
    } catch (const std::out_of_range&) {
        throw py::key_error("no band " + name);
    }
}

// A Mat handed to numpy without a copy (StreamDetector.stat, every hop).
template <typename T>
py::array_t<T> np_move(Mat<T>&& m) {
    auto* v = new std::vector<T>(std::move(m.data));
    py::capsule owner(v, [](void* p) { delete static_cast<std::vector<T>*>(p); });
    return py::array_t<T>({static_cast<py::ssize_t>(m.rows), static_cast<py::ssize_t>(m.cols)}, v->data(), owner);
}

// StreamDetector's per-bin rows -> a 2-D array
template <typename T, typename Row>
py::array_t<T> rows(std::size_t r, Row row) {
    const std::size_t c = r ? row(0).size() : 0;
    py::array_t<T> a({static_cast<py::ssize_t>(r), static_cast<py::ssize_t>(c)});
    for (std::size_t i = 0; i < r; ++i) std::ranges::copy(row(i), a.mutable_data() + i * c);
    return a;
}

py::tuple acquisition(const waveform::Acquisition& a) {
    py::list alts;
    for (const auto& [s, f] : a.alternatives) alts.append(py::make_tuple(s, f));
    return py::make_tuple(a.preamble_start, a.freq_offset, a.metric, alts);
}

}  // namespace

void bind_waveform(py::module_& m) {
    auto d = m.def_submodule("dsp");
    d.def("pairwise_sum", [](const In<double>& a) { return dsp::pairwise_sum(view(a)); });
    d.def("pairwise_sum_complex", [](const In<cdouble>& a) { return dsp::pairwise_sum(view(a)); });
    d.def("quantile", [](const In<double>& a, double q) { return dsp::quantile(vec(a), q); });
    d.def("firwin_bandpass", [](int n, double lo, double hi, double fs) { return np(dsp::firwin_bandpass(n, lo, hi, fs)); });
    d.def("hilbert", [](const In<double>& x) { return np(dsp::hilbert(view(x))); });
    d.def("convolve_same", [](const In<double>& a, const In<double>& v) { return np(dsp::convolve_same(view(a), view(v))); });
    d.def("next_fast_len", &dsp::next_fast_len);
    d.def("fftconvolve_valid", [](const In<cdouble>& a, const In<cdouble>& v) {
        return np(dsp::fftconvolve_valid(view(a), view(v)));
    });

    auto w = m.def_submodule("waveform");
    py::register_exception<waveform::SyncError>(w, "SyncError");

    py::class_<waveform::Band>(w, "Band")
        .def_property_readonly("name", [](const waveform::Band& b) { return std::string(b.spec->name); })
        .def_property_readonly("nc", &waveform::Band::nc)
        .def_property_readonly("freqs", [](const waveform::Band& b) { return np(b.freqs); })
        .def_property_readonly("bb", [](const waveform::Band& b) { return np(b.bb); })
        .def_property_readonly("mod", [](const waveform::Band& b) { return np(b.mod); })
        .def_property_readonly("demod", [](const waveform::Band& b) { return np(b.demod); })
        .def_property_readonly("pilot", [](const waveform::Band& b) { return np(b.pilot); })
        .def_property_readonly("preamble_template", [](const waveform::Band& b) { return np(b.preamble_template); })
        .def_property_readonly("preamble_samples", &waveform::Band::preamble_samples)
        .def_property_readonly("preamble_threshold", &waveform::Band::preamble_threshold)
        .def_property_readonly("tx_bandpass", &waveform::Band::tx_bandpass)
        .def("modulate_symbols", [](const waveform::Band& b, const In<cdouble>& s) {
            return np(b.modulate_symbols(mat(s)));
        })
        .def("demod_window", [](const waveform::Band& b, const In<cdouble>& z, std::int64_t start, std::int64_t backoff) {
            return np(b.demod_window(view(z), start, backoff));
        }, py::arg("z"), py::arg("start"), py::arg("backoff") = 0)
        .def("preamble_waveform", [](const waveform::Band& b) { return np(b.preamble_waveform()); });
    w.def("band", &band_of, py::return_value_policy::reference, py::arg("name") = "w");

    w.def("to_baseband", [](const In<double>& x, std::int64_t n0) { return np(waveform::to_baseband(view(x), n0)); },
          py::arg("x"), py::arg("n0") = 0);
    w.def("freq_correct", [](const In<cdouble>& z, double f) { return np(waveform::freq_correct(view(z), f)); });
    w.def("tx_condition", [](const In<double>& x, double headroom, const std::vector<double>& overshoot,
                             std::size_t lo, std::size_t hi, std::pair<double, double> bandpass, py::object project,
                             const std::vector<double>& closing) {
        waveform::Projector proj;
        if (!project.is_none())
            proj = [project](std::span<const double> v) {
                return vec(project(np<double>(v)).cast<In<double>>());
            };
        return np(waveform::tx_condition(view(x), headroom, overshoot, lo, hi, bandpass, proj, closing));
    }, py::arg("x"), py::arg("clip_headroom_db"), py::arg("overshoot"), py::arg("active_lo"), py::arg("active_hi"),
       py::arg("bandpass"), py::arg("project") = py::none(), py::arg("closing") = std::vector<double>{});
    w.def("papr_db", [](const In<double>& x) { return waveform::papr_db(view(x)); });

    w.def("cfo_grid", [](double reach) { return np(waveform::cfo_grid(reach)); }, py::arg("reach") = config::ACQUIRE_REACH_HZ);
    w.def("unit_template", [](const std::string& b) { return np(waveform::unit_template(band_of(b))); });
    w.def("repeat_corr", [](const In<cdouble>& z, const In<cdouble>& t, double f) {
        return np(waveform::repeat_corr(view(z), view(t), f));
    });
    w.def("repeat_corrs", [](const In<cdouble>& z, const In<cdouble>& t, const std::vector<double>& freqs) {
        return np(waveform::repeat_corrs(view(z), view(t), freqs));
    });
    w.def("raw_stat", [](const In<cdouble>& z, const std::string& b, double reach, int repeats,
                         std::optional<std::size_t> levels_from, bool outs) {
        auto r = waveform::raw_stat(view(z), band_of(b), reach, repeats, levels_from, outs);
        return py::make_tuple(np(r.S), np(r.q), np(r.freqs), outs ? py::object(np(r.outs)) : py::none());
    }, py::arg("z"), py::arg("band"), py::arg("reach") = config::ACQUIRE_REACH_HZ, py::arg("repeats") = 0,
       py::arg("levels_from") = py::none(), py::arg("outs") = false);
    w.def("detection_stat", [](const In<cdouble>& z, const std::string& b, double reach, int repeats) {
        auto [S, f] = waveform::detection_stat(view(z), band_of(b), reach, repeats);
        return py::make_tuple(np(S), np(f));
    }, py::arg("z"), py::arg("band"), py::arg("reach") = config::ACQUIRE_REACH_HZ, py::arg("repeats") = 0);
    w.def("first_path", [](const In<double>& p, std::size_t peak, int search, double frac, bool cyclic) {
        return waveform::first_path(view(p), peak, search, frac, cyclic);
    }, py::arg("power"), py::arg("peak"), py::arg("search") = config::FIRST_PATH_SEARCH,
       py::arg("frac") = config::FIRST_PATH_FRAC, py::arg("cyclic") = false);
    w.def("refine", [](const In<cdouble>& z, const std::string& b, std::int64_t n, double f) {
        return waveform::refine(view(z), band_of(b), n, f);
    });
    w.def("crossings", [](const In<double>& D, double thr, std::size_t span, std::size_t limit) {
        return waveform::crossings(view(D), thr, span, limit);
    });
    w.def("acquire", [](const In<cdouble>& z, const std::string& b, std::optional<double> threshold, double reach,
                        std::optional<std::pair<std::int64_t, std::int64_t>> search, std::optional<In<double>> S) {
        std::optional<Mat<double>> s;
        if (S) s = mat(*S);
        return acquisition(waveform::acquire(view(z), band_of(b), threshold, reach, search, s ? &*s : nullptr));
    }, py::arg("z"), py::arg("band"), py::arg("threshold") = py::none(), py::arg("reach") = config::ACQUIRE_REACH_HZ,
       py::arg("search") = py::none(), py::arg("S") = py::none());

    using SD = waveform::StreamDetector;
    py::class_<SD>(w, "StreamDetector", py::dynamic_attr())
        .def(py::init([](const std::string& b, double reach) { return new SD(band_of(b), reach); }),
             py::arg("band"), py::arg("reach") = config::ACQUIRE_REACH_HZ)
        .def_readonly_static("CHUNKS", &SD::CHUNKS)
        .def("reset", &SD::reset)
        .def("feed", [](SD& d, const In<cdouble>& z) { d.feed(view(z)); })
        .def("trim", &SD::trim)
        .def("level", &SD::level)
        .def("stat", [](const SD& d, std::int64_t lo, std::int64_t hi) { return np_move(d.stat(lo, hi)); })
        .def_property_readonly("band_name", [](const SD& d) { return std::string(d.band.spec->name); })
        .def_readonly("reach", &SD::reach)
        .def_readonly("span", &SD::span)
        .def_readwrite("fed", &SD::fed)
        .def_readonly("s0", &SD::s0)
        .def_readonly("c0", &SD::c0)
        .def_property_readonly("S", [](const SD& d) { return rows<double>(d.bins(), [&](std::size_t i) { return d.S(i); }); })
        .def_property_readonly("C", [](const SD& d) { return rows<cdouble>(d.bins(), [&](std::size_t i) { return d.C(i); }); })
        .def_property_readonly("tail", [](const SD& d) { return np(d.tail); })
        .def_property_readonly("levels", [](const SD& d) {
            py::list out;
            for (const auto& q : d.levels) out.append(np(q));
            return out;
        });
}

}  // namespace data2g::bind
