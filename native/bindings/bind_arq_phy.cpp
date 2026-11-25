// data2g.arq.phy over the C++ core (native/core/arq/phy). A received burst
// crosses as modem.receive's / cpm.receive's dict; its per-slot soft bits
// are cached in r["_soft"] in Python's format, so Python and C++ decoders of
// one burst share them. ModemRx's store stays the caller's Python dict,
// {key: (buffer (1, L), highest RV, submode, (slot, rv, mask_id))}.
#include <pybind11/functional.h>
#include <cmath>

#include "arq_convert.hpp"
#include "arq/ping.hpp"
#include "arq/probe.hpp"

namespace data2g::bind {

namespace {

using namespace data2g::arq;
using cd = std::complex<double>;

std::string name_of(const py::handle& spec) {
    return py::isinstance<py::str>(spec) ? spec.cast<std::string>() : spec.attr("name").cast<std::string>();
}

template <typename T>
Mat<T> rows(const py::handle& o) {  // (..., nc) -> (rows, nc)
    const auto a = o.cast<In<T>>();
    const std::size_t nc = a.ndim() ? a.shape(a.ndim() - 1) : 1;
    Mat<T> m(nc ? a.size() / nc : 0, nc);
    std::copy(a.data(), a.data() + a.size(), m.data.begin());
    return m;
}

bool is_cpm(const py::dict& r) { return r.contains("family") && r["family"].cast<std::string>() == "cpm"; }

}  // namespace

Heard heard_of(const py::dict& r) {
    Heard h;
    if (is_cpm(r)) {
        auto c = std::make_shared<cpm::Received>();
        const auto& m = mode_at(name_of(r["spec"]));
        if (!m.is_cpm()) throw py::value_error("not a CPM mode");
        c->spec = m.cpm;
        c->dup = r["dup"].cast<bool>();
        for (auto s : r["soft"]) c->soft.push_back(vec(s.cast<In<double>>()));
        c->E = mat(r["E"].cast<In<double>>());
        c->n_cw = static_cast<int>(c->soft.size());
        c->n_ctl_slots = 1 + c->dup;
        if (r.contains("Z")) c->Z = mat(r["Z"].cast<In<cd>>());  // coherent DD (cpm.receive's bins)
        h.cpm = std::move(c);
        return h;
    }
    auto o = std::make_shared<modem::Received>();
    o->spec = &spec(name_of(r["spec"]));
    o->n_cw = r["n_cw"].cast<int>();
    o->raw = rows<cd>(r["raw"]);
    const auto e = r["est"].cast<py::dict>();
    o->est.h = rows<cd>(e["h"]);
    o->est.mse = rows<double>(e["mse"]);
    o->est.n_f = static_cast<int>(o->est.h.rows / config::DATA_SYMS_PER_FRAME);
    o->est.nc = static_cast<int>(o->est.h.cols);
    o->est.n0 = e["n0"].cast<double>();
    if (e.contains("n0_k")) o->est.n0_k = vec(e["n0_k"].cast<In<double>>());
    o->est.n0_snr = e.contains("n0_snr") ? e["n0_snr"].cast<double>() : o->est.n0;
    o->est.p_sig = e["p_sig"].cast<double>();
    o->est.spread_hz = e.contains("spread_hz") ? e["spread_hz"].cast<double>() : 0.0;
    o->est.filter_spread_hz = e.contains("filter_spread_hz") ? e["filter_spread_hz"].cast<double>() : o->est.spread_hz;
    o->est.slack = e.contains("slack") ? e["slack"].cast<int>() : equalizer::SUPPORT_SLACK;
    o->est.clip_ratio = e["clip_ratio"].cast<double>();
    o->est.gain = e.contains("gain") ? e["gain"].cast<double>() : 1.0;
    o->band = config::band(r["band"].cast<std::string>()).name;
    if (r.contains("hp")) o->hp = rows<cd>(r["hp"]);
    if (r.contains("kc") && !r["kc"].is_none()) o->kc = r["kc"].cast<int>();
    const auto sup = r["support"].cast<py::sequence>();
    o->support = {sup[0].cast<int>(), sup[1].cast<int>()};
    const auto seen = (r.contains("seen_support") ? r["seen_support"] : r["support"]).cast<py::sequence>();
    o->seen_support = {seen[0].cast<int>(), seen[1].cast<int>()};
    o->frames_in = r.contains("frames_in") ? r["frames_in"].cast<int>() : o->n_cw * o->spec->frames_per_cw;
    h.ofdm = std::move(o);
    return h;
}

py::object soft_py(const Heard& h, const SlotSoft& s) {
    if (h.cpm) {
        py::list out;
        for (const auto& v : s) out.append(np(v));
        return out;
    }
    Mat<double> m(s.size(), s.empty() ? 0 : s[0].size());
    for (std::size_t i = 0; i < s.size(); ++i) std::copy(s[i].begin(), s[i].end(), m[i]);
    return np(m);
}

// r's soft bits, made once and kept in r["_soft"] (phy.ModemRx's cache).
std::shared_ptr<const SlotSoft> cached_soft(py::dict r, const Heard& h) {
    if (r.contains("_soft")) {
        auto out = std::make_shared<SlotSoft>();
        if (h.cpm)
            for (auto s : r["_soft"]) out->push_back(vec(s.cast<In<double>>()));
        else {
            const auto m = mat(r["_soft"].cast<In<double>>());
            for (std::size_t i = 0; i < m.rows; ++i) out->emplace_back(m[i], m[i] + m.cols);
        }
        return out;
    }
    std::shared_ptr<const SlotSoft> s;
    {
        py::gil_scoped_release nogil;
        s = soft_bits(h);
    }
    r["_soft"] = soft_py(h, *s);
    return s;
}

std::function<double()> clock_of(const py::object& c) {
    if (c.is_none()) return monotonic;
    return [c] {
        py::gil_scoped_acquire gil;
        return c().cast<double>();
    };
}

namespace {

// phy.ModemRx: the C++ decoder, its store the caller's dict.
class PyModemRx {
public:
    PyModemRx(py::dict r, py::object store, std::optional<double> dd_budget, bool dd, const py::object& clock)
        : store_(std::move(store)) {
        auto h = heard_of(r);
        auto soft = h.ofdm && h.ofdm->partial() ? nullptr : cached_soft(r, h);  // a partial burst's: per slot
        rx_ = std::make_unique<ModemRx>(std::move(h), nullptr, dd_budget, std::move(soft), dd, clock_of(clock));
    }

    std::vector<int> advance(const py::dict& r, std::optional<double> budget_s) {
        auto h = heard_of(r);
        py::gil_scoped_release nogil;
        return rx_->advance(std::move(h), budget_s);
    }

    py::object decode(int slot, const py::handle& mask, int rv, const py::object& key) {
        const MaskId m = mask_of(mask);
        std::optional<Bytes> out;
        if (key.is_none()) {
            {
                py::gil_scoped_release nogil;
                out = rx_->decode_plain(slot, m);
            }
            return out ? py::object(pyb(*out)) : py::none();
        }
        std::optional<SoftEntry> e;
        py::tuple t;
        py::object got = store_.attr("get")(key);
        if (!got.is_none()) {
            t = got.cast<py::tuple>();
            // slot -1: not written back unless the decode stores
            e = SoftEntry{vec(t[0].cast<In<double>>()), t[1].cast<int>(), t[2].cast<std::string>(), -1, 0, {}};
        }
        {
            py::gil_scoped_release nogil;
            out = rx_->decode_stored(slot, m, rv, e);
        }
        if (out) return pyb(*out);
        if (e && e->slot >= 0) {
            py::array_t<double> buf({py::ssize_t{1}, static_cast<py::ssize_t>(e->buf.size())});
            std::copy(e->buf.begin(), e->buf.end(), buf.mutable_data());
            store_[key] = py::make_tuple(buf, e->top, e->submode, py::make_tuple(slot, rv, py::reinterpret_borrow<py::object>(mask)));
        }
        return py::none();
    }

    py::list raw(int slot) {
        std::vector<Bytes> out;
        {
            py::gil_scoped_release nogil;
            out = rx_->raw(slot);
        }
        py::list l;
        for (const auto& b : out) l.append(pyb(b));
        return l;
    }
    py::list raw_pair(int slot) {
        std::vector<Bytes> out;
        {
            py::gil_scoped_release nogil;
            out = rx_->raw_pair(slot);
        }
        py::list l;
        for (const auto& b : out) l.append(pyb(b));
        return l;
    }
    py::object decode_pair(int slot, const py::handle& mask) {
        const MaskId m = mask_of(mask);
        std::optional<Bytes> out;
        {
            py::gil_scoped_release nogil;
            out = rx_->decode_pair(slot, m);
        }
        return out ? py::object(pyb(*out)) : py::none();
    }
    bool converged(int slot, bool pair) {
        py::gil_scoped_release nogil;
        return rx_->converged(slot, pair);
    }
    // phy.ModemRx.decode_repeat over the Python store
    py::object decode_repeat(int slot, const py::handle& mask, const py::object& key) {
        py::object held = store_.attr("get")(key);
        if (!held.is_none() && held.cast<py::tuple>()[2].cast<std::string>() != rx_->submode()) store_.attr("pop")(key);
        const bool had = rx_->has_coded(slot);
        py::object got = decode(slot, mask, 0, key);
        if (got.is_none() || !rx_->own_score(slot, mask_of(mask), bytes_of(got))) {
            if (!had) rx_->erase_coded(slot);
            return py::none();
        }
        return got;
    }
    py::object measure_data(const py::dict& md) {
        const auto m = measured(md);
        std::optional<Measured> out;
        {
            py::gil_scoped_release nogil;
            out = rx_->measure_data(m);
        }
        return out ? py::object(to_dict(*out)) : py::none();
    }
    py::object decode_known(int slot, const py::handle& mask, const py::object& tmpl, const py::object& tmask, bool pair) {
        const MaskId m = mask_of(mask);
        const Bytes t = bytes_of(tmpl), k = bytes_of(tmask);
        std::optional<Bytes> out;
        {
            py::gil_scoped_release nogil;
            out = rx_->decode_known(slot, m, t, k, pair);
        }
        return out ? py::object(pyb(*out)) : py::none();
    }
    int n_ctl_slots() const { return rx_->n_ctl_slots(); }
    void forget(const py::object& key) { store_.attr("pop")(key, py::none()); }
    const std::string& submode() const { return rx_->submode(); }
    int n_cw() const { return rx_->n_cw(); }

private:
    std::unique_ptr<ModemRx> rx_;
    py::object store_;
};

}  // namespace

namespace {

arq::probe::Report report_of(const py::dict& d) {
    arq::probe::Report m;
    m.lo_hz = d["lo_hz"].cast<double>();
    m.hi_hz = d["hi_hz"].cast<double>();
    m.snr_nb = d["snr_nb"].cast<double>();
    m.snr_top = d["snr_top"].cast<double>();
    m.snr_tilt = d["snr_tilt"].cast<double>();
    m.noise_tilt = d["noise_tilt"].cast<double>();
    const auto snr = d["snr"].cast<std::vector<double>>(), sinad = d["sinad"].cast<std::vector<double>>();
    for (std::size_t i = 0; i < m.snr.size(); ++i) m.snr[i] = snr.at(i), m.sinad[i] = sinad.at(i);
    m.rms_dbfs = d["rms_dbfs"].cast<double>();
    m.peak_dbfs = d["peak_dbfs"].cast<double>();
    m.clip_permille = d["clip_permille"].cast<int>();
    return m;
}

py::dict report_py(const arq::probe::Report& m) {
    py::dict d;
    d["lo_hz"] = m.lo_hz;
    d["hi_hz"] = m.hi_hz;
    d["snr_nb"] = m.snr_nb;
    d["snr_top"] = m.snr_top;
    d["snr_tilt"] = m.snr_tilt;
    d["noise_tilt"] = m.noise_tilt;
    d["snr"] = std::vector<double>(m.snr.begin(), m.snr.end());
    d["sinad"] = std::vector<double>(m.sinad.begin(), m.sinad.end());
    d["rms_dbfs"] = m.rms_dbfs;
    d["peak_dbfs"] = m.peak_dbfs;
    d["clip_permille"] = m.clip_permille;
    return d;
}

arq::ping::Level level_of(const py::dict& d) {
    return {d["rms_dbfs"].cast<double>(), d["peak_dbfs"].cast<double>(), d["clip_permille"].cast<int>()};
}

}  // namespace

void bind_arq_ping(py::module_& m) {
    namespace pg = arq::ping;
    namespace pr = arq::probe;
    auto g = m.def_submodule("ping", "data2g.arq.ping");
    g.def("tx_level", [](const std::string& path) { return pg::tx_level(path); });
    g.def("ping_body", [](const std::string& frm, const std::string& to, const py::object& report) {
        return pyb(pg::ping_body(frm, to, report.is_none() ? std::nullopt : std::optional<Bytes>(bytes_of(report))));
    });
    g.def("unpack_ping", [](const py::bytes& body) {
        const Bytes b = bytes_of(body);
        try {
            auto [frm, to, rep] = pg::unpack_ping(b);
            return py::make_tuple(frm, to, rep ? py::object(pyb(*rep)) : py::object(py::none()));
        } catch (const std::invalid_argument& e) {
            throw py::value_error(e.what());
        }
    });
    g.def("level", [](const In<double>& x) {
        const auto l = pg::level(vec(x));
        py::dict d;
        d["rms_dbfs"] = l.rms_dbfs;
        d["peak_dbfs"] = l.peak_dbfs;
        d["clip_permille"] = l.clip_permille;
        return d;
    });
    auto q = m.def_submodule("probe", "data2g.arq.probe");
    q.def("n_samples", [] { return pr::n_samples(); });
    q.def("tones", [] { return pr::tones(); });
    q.def("holes", [] { return pr::holes(); });
    q.def("waveform", [](const std::string& path, double level) { return pr::waveform(path, level); });
    q.def("measure", [](const In<double>& y, const py::dict& lvl) {
        const auto v = vec(y);
        const auto l = level_of(lvl);
        pr::Report m;
        {
            py::gil_scoped_release nogil;
            m = pr::measure(v, l);
        }
        return report_py(m);
    });
    q.def("pack_report", [](const py::dict& d) { return pyb(pr::pack_report(report_of(d))); });
    q.def("unpack_report", [](const py::bytes& b) {
        try {
            return report_py(pr::unpack_report(bytes_of(b)));
        } catch (const std::invalid_argument& e) {
            throw py::value_error(e.what());
        }
    });
    q.def("fmt", [](const py::dict& d) { return pr::fmt(report_of(d)); });
    q.def("best_step", [](const py::dict& d) { return pr::best_step(report_of(d)); });
    q.def("verdict", [](const py::dict& d, const std::string& path) { return pr::verdict(report_of(d), path); });
}

void bind_arq_phy(py::module_& m) {
    bind_arq_ping(m);
    auto p = m.def_submodule("phy", "data2g.arq.phy");
    p.attr("DD_ITERS") = DD_ITERS;
    p.attr("DD_BUDGET_S") = DD_BUDGET_S;
    p.def("dd_default", &dd_default);
    p.def("mask_value", [](const py::handle& t) { return mask_value(mask_of(t)); });
    p.def("tx_audio", [](const py::handle& burst, const py::object& tx_path, const py::object& level) {
        const auto b = burst_cpp(burst);
        const std::string path = tx_path.is_none() ? std::string() : tx_path.cast<std::string>();
        const std::optional<double> g = level.is_none() ? std::nullopt : std::optional<double>(level.cast<double>());
        std::vector<double> x;
        {
            py::gil_scoped_release nogil;
            x = tx_audio(*b, path, g);
        }
        return np(x);
    }, py::arg("burst"), py::arg("tx_path") = py::none(), py::arg("level") = py::none());
    p.def("soft_bits", [](const py::dict& r) {
        const auto h = heard_of(r);
        return soft_py(h, *soft_bits(h));
    });
    p.def("measure", [](const py::dict& r) { return to_dict(measure(heard_of(r))); });
    p.def("data_mer", [](const py::dict& r, const py::dict& coded_bits, const py::dict& md) {
        const auto h = heard_of(r);
        if (!h.ofdm) throw py::value_error("data_mer: OFDM burst required");
        std::map<int, std::vector<std::uint8_t>> cb;
        for (const auto& [k, v] : coded_bits) {
            const auto a = v.cast<In<std::uint8_t>>();
            cb[k.cast<int>()] = std::vector<std::uint8_t>(a.data(), a.data() + a.size());
        }
        const auto m = measured(md);
        Measured out;
        {
            py::gil_scoped_release nogil;
            out = data_mer(*h.ofdm, cb, m);
        }
        return to_dict(out);
    });
    py::class_<PyModemRx>(p, "ModemRx")
        .def(py::init<py::dict, py::object, std::optional<double>, bool, py::object>(), py::arg("r"), py::arg("store"),
             py::arg("dd_budget") = py::none(), py::arg("dd") = dd_default(), py::arg("clock") = py::none())
        .def("decode", &PyModemRx::decode)
        .def("advance", &PyModemRx::advance, py::arg("r"), py::arg("budget_s") = py::none())
        .def("raw", &PyModemRx::raw)
        .def("raw_pair", &PyModemRx::raw_pair)
        .def("decode_pair", &PyModemRx::decode_pair)
        .def("converged", &PyModemRx::converged, py::arg("slot"), py::arg("pair") = false)
        .def("decode_repeat", &PyModemRx::decode_repeat, py::arg("slot"), py::arg("mask_id"), py::arg("key"))
        .def("decode_known", &PyModemRx::decode_known, py::arg("slot"), py::arg("mask_id"), py::arg("template"),
             py::arg("tmask"), py::arg("pair") = false)
        .def("measure_data", &PyModemRx::measure_data)
        .def_property_readonly("n_ctl_slots", &PyModemRx::n_ctl_slots)
        .def("forget", &PyModemRx::forget)
        .def_property_readonly("submode", &PyModemRx::submode)
        .def_property_readonly("n_cw", &PyModemRx::n_cw);
}

}  // namespace data2g::bind
