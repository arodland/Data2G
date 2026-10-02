// data2g.equalizer, same names, arguments and defaults (bb defaults to the
// wide band's, as BB_FREQS).
#include <optional>

#include "convert.hpp"
#include "equalizer/equalizer.hpp"

namespace data2g::bind {

namespace {

using equalizer::cd;

const std::vector<double>& wide_bb() {
    static const std::vector<double> b = equalizer::bb(config::BANDS[0]);
    return b;
}

std::vector<double> bb_of(const std::optional<In<double>>& bb) { return bb ? vec(*bb) : wide_bb(); }

py::array_t<cd> np3(const Mat<cd>& m, std::size_t a, std::size_t b) {
    return np(m).reshape({static_cast<py::ssize_t>(a), static_cast<py::ssize_t>(b), static_cast<py::ssize_t>(m.cols)});
}

py::array_t<double> np3(const Mat<double>& m, std::size_t a, std::size_t b) {
    return np(m).reshape({static_cast<py::ssize_t>(a), static_cast<py::ssize_t>(b), static_cast<py::ssize_t>(m.cols)});
}

}  // namespace

void bind_equalizer(py::module_& m) {
    namespace eq = equalizer;
    auto e = m.def_submodule("equalizer");
    const auto no_bb = py::arg("bb") = std::optional<In<double>>();
    e.attr("FRAME_S") = eq::FRAME_S;
    e.attr("BB_FREQS") = np(wide_bb());
    e.def("bb", [](const std::string& name) {
        for (const auto& b : config::BANDS)
            if (b.name == name) return np(eq::bb(b));
        throw py::key_error("no band " + name);
    });
    e.def("residual_cfo", [](const In<cd>& h) { return eq::residual_cfo(mat(h)); });
    e.def("delay_profile", [](const In<cd>& h, const std::optional<In<double>>& bb) {
        return np(eq::delay_profile(mat(h), bb_of(bb)));
    }, py::arg("h_pilot"), no_bb);
    e.def("delay_support", [](const In<cd>& h, double floor_db, const std::optional<In<double>>& bb) {
        return eq::delay_support(mat(h), bb_of(bb), floor_db);
    }, py::arg("h_pilot"), py::arg("floor_db") = -15.0, no_bb);
    e.def("window_shift", &eq::window_shift);
    e.def("support_basis", [](const In<double>& bb, int d0, int d1) {
        const auto b = eq::support_basis(vec(bb), d0, d1);
        return py::make_tuple(np(b->u), b->r, b->r_full, np(b->keep));
    });
    e.def("_freq_smooth", [](const In<cd>& h, eq::Support s, const std::optional<In<double>>& bb) {
        auto r = eq::freq_smooth(mat(h), s, bb_of(bb));
        return py::make_tuple(np(r.hs), r.n0, r.r, np(r.keep));
    }, py::arg("h_pilot"), py::arg("support"), no_bb);
    e.def("preamble_noise", [](const In<cd>& h) { return eq::preamble_noise(mat(h)); });
    e.def("preamble_noise_k", [](const In<cd>& h) { return np(eq::preamble_noise_k(mat(h))); });
    e.def("per_carrier_noise", [](const In<double>& p, int samples) {
        return np(eq::per_carrier_noise(vec(p), samples));
    }, py::arg("power_k"), py::arg("samples"));
    e.def("_doppler_corr", [](const In<double>& dt, double spread_hz) {
        py::array_t<double> out(std::vector<py::ssize_t>(dt.shape(), dt.shape() + dt.ndim()));
        for (py::ssize_t i = 0; i < dt.size(); ++i) out.mutable_data()[i] = eq::doppler_corr(dt.data()[i], spread_hz);
        return out;
    }, py::arg("dt"), py::arg("spread_hz"));
    e.def("measure_spread", [](const In<cd>& hs, double n0_s) { return eq::measure_spread(mat(hs), n0_s); });
    e.def("estimate", [](const In<cd>& h, eq::Support s, const std::optional<In<double>>& bb, double n0_pre,
                         const std::optional<In<double>>& n0_pre_k) {
        const std::vector<double> pk = n0_pre_k ? vec(*n0_pre_k) : std::vector<double>();
        auto r = eq::estimate(mat(h), s, bb_of(bb), n0_pre, pk);
        constexpr std::size_t S = config::SYMS_PER_FRAME - 1;
        py::dict d;
        d["h"] = np3(r.h, static_cast<std::size_t>(r.n_f), S);
        d["mse"] = np3(r.mse, static_cast<std::size_t>(r.n_f), S);
        d["n0"] = r.n0;
        d["n0_k"] = np(r.n0_k);
        d["p_sig"] = r.p_sig;
        d["spread_hz"] = r.spread_hz;
        return d;
    }, py::arg("h_pilot"), py::arg("support"), no_bb, py::arg("n0_pre") = eq::INF,
       py::arg("n0_pre_k") = std::optional<In<double>>());
    e.def("time_shift_phase", [](const In<double>& shift, const std::optional<In<double>>& bb) {
        return np(eq::time_shift_phase(vec(shift), bb_of(bb)));
    }, py::arg("shift"), no_bb);
    e.def("refine", [](const In<cd>& h_pilot, const In<double>& t_pilot, const In<cd>& z, const In<double>& w,
                       const In<double>& t_rows, eq::Support s, const py::dict& est,
                       const std::optional<In<double>>& bb) {
        const auto tr = mat(t_rows);
        const auto bbv = bb_of(bb);
        std::pair<Mat<cd>, Mat<double>> r;
        {
            const auto hp = mat(h_pilot);
            const auto tp = vec(t_pilot);
            const auto zm = mat(z);
            const auto wm = mat(w);
            const double p_sig = est["p_sig"].cast<double>(), spread = est["spread_hz"].cast<double>(),
                         n0 = est["n0"].cast<double>();
            py::gil_scoped_release nogil;
            r = eq::refine(hp, tp, zm, wm, tr, s, p_sig, spread, n0, bbv);
        }
        return py::make_tuple(np3(r.first, tr.rows, tr.cols), np3(r.second, tr.rows, tr.cols));
    }, py::arg("h_pilot"), py::arg("t_pilot"), py::arg("z"), py::arg("w"), py::arg("t_rows"), py::arg("support"),
       py::arg("est"), no_bb);
}

}  // namespace data2g::bind
