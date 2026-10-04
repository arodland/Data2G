// data2g.arq.phy and data2g.kisslink over the C++ core. A received burst
// crosses as modem.receive's / cpm.receive's dict; its per-slot soft bits
// are cached in r["_soft"] in Python's format, so Python and C++ decoders of
// one burst share them. ModemRx's store stays the caller's Python dict,
// {key: (buffer (1, L), highest RV, submode, (slot, rv, mask_id))}.
#include <pybind11/functional.h>
#include <cmath>

#include "arq_convert.hpp"
#include "kisslink/kisslink.hpp"

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
    o->est.p_sig = e["p_sig"].cast<double>();
    o->est.spread_hz = e.contains("spread_hz") ? e["spread_hz"].cast<double>() : 0.0;
    o->est.clip_ratio = e["clip_ratio"].cast<double>();
    o->est.gain = e.contains("gain") ? e["gain"].cast<double>() : 1.0;
    o->band = modem::band(r["band"].cast<std::string>()).name;
    if (r.contains("hp")) o->hp = rows<cd>(r["hp"]);
    if (r.contains("kc") && !r["kc"].is_none()) o->kc = r["kc"].cast<int>();
    const auto sup = r["support"].cast<py::sequence>();
    o->support = {sup[0].cast<int>(), sup[1].cast<int>()};
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

// phy.ModemRx: the C++ decoder, its store the caller's dict.
class PyModemRx {
public:
    PyModemRx(py::dict r, py::object store, std::optional<double> dd_budget, bool dd, const py::object& clock)
        : store_(std::move(store)) {
        auto h = heard_of(r);
        auto soft = cached_soft(r, h);
        rx_ = std::make_unique<ModemRx>(std::move(h), nullptr, dd_budget, std::move(soft), dd, clock_of(clock));
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
        try {
            py::gil_scoped_release nogil;
            out = rx_->decode_stored(slot, m, rv, e);
        } catch (const StoreMismatch&) {
            const std::string msg = py::str("soft bits of {} stored in {} ({}), resent in {} slot {} rv {}")
                                        .format(key, t[2], t[3], rx_->submode(), slot, rv).cast<std::string>();
            PyErr_SetString(PyExc_AssertionError, msg.c_str());
            throw py::error_already_set();
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
    void forget(const py::object& key) { store_.attr("pop")(key, py::none()); }
    const std::string& submode() const { return rx_->submode(); }
    int n_cw() const { return rx_->n_cw(); }

private:
    std::unique_ptr<ModemRx> rx_;
    py::object store_;
};

using kisslink::KissLink;

}  // namespace

void bind_arq_phy(py::module_& m) {
    auto p = m.def_submodule("phy", "data2g.arq.phy");
    p.attr("DD_ITERS") = DD_ITERS;
    p.attr("DD_BUDGET_S") = DD_BUDGET_S;
    p.def("dd_default", &dd_default);
    p.def("mask_value", [](const py::handle& t) { return mask_value(mask_of(t)); });
    p.def("tx_audio", [](const py::handle& burst) {
        const auto b = burst_cpp(burst);
        std::vector<double> x;
        {
            py::gil_scoped_release nogil;
            x = tx_audio(*b);
        }
        return np(x);
    });
    p.def("soft_bits", [](const py::dict& r) {
        const auto h = heard_of(r);
        return soft_py(h, *soft_bits(h));
    });
    p.def("measure", [](const py::dict& r) { return to_dict(measure(heard_of(r))); });
    py::class_<PyModemRx>(p, "ModemRx")
        .def(py::init<py::dict, py::object, std::optional<double>, bool, py::object>(), py::arg("r"), py::arg("store"),
             py::arg("dd_budget") = py::none(), py::arg("dd") = dd_default(), py::arg("clock") = py::none())
        .def("decode", &PyModemRx::decode)
        .def("raw", &PyModemRx::raw)
        .def("forget", &PyModemRx::forget)
        .def_property_readonly("submode", &PyModemRx::submode)
        .def_property_readonly("n_cw", &PyModemRx::n_cw);

    auto k = m.def_submodule("kisslink", "data2g.kisslink");
    k.def("parse_ax25", [](const py::bytes& f) -> py::object {
        const auto a = kisslink::parse_ax25(bytes_view(f));
        if (!a) return py::none();
        return py::make_tuple(a->dst, a->src, a->next_hop, a->sender, a->connected);
    });
    k.def("station_hash", [](const std::string& c) { return kisslink::station_hash(c); });
    k.def("group_key", [](const std::string& g) { return kisslink::group_key(g); });
    k.def("group_name", [](const std::string& g) { return kisslink::group_name(g); });
    k.def("pack_pair", [](const std::string& g, const std::string& c) { return pyb(kisslink::pack_pair(g, c)); });
    k.def("unpack_pair", [](const py::bytes& b) {
        const auto [g, c] = kisslink::unpack_pair(bytes_view(b));
        return py::make_tuple(g, c);
    });
    py::class_<kisslink::Port>(k, "Port")
        .def_readonly("group", &kisslink::Port::group)
        .def_readonly("mode", &kisslink::Port::mode)
        .def_property_readonly("auto", [](const kisslink::Port& p) { return p.shift; })
        .def_property_readonly("from_call", [](const kisslink::Port& p) -> py::object {
            return p.from_call ? py::object(py::str(*p.from_call)) : py::none();
        })
        .def_property_readonly("key", &kisslink::Port::key);
    // Acks are Python objects (a KISS client and its tag): the link carries an
    // id, and the object waits in a dict on the KissLink's Python side.
    auto ack_objs = [](const py::object& self) -> py::dict {
        if (!py::hasattr(self, "_ack_objs")) self.attr("_ack_objs") = py::dict();
        return self.attr("_ack_objs");
    };
    auto acks_py = [ack_objs](const py::object& self, const std::vector<std::pair<int, std::int64_t>>& acks, bool take) {
        py::dict d = ack_objs(self);
        py::list out;
        for (const auto& [port, id] : acks) {
            const py::int_ key(id);
            out.append(py::make_tuple(port, d.contains(key) ? py::object(d[key]) : py::none()));
            if (take && d.contains(key)) PyDict_DelItem(d.ptr(), key.ptr());
        }
        return out;
    };
    py::class_<KissLink>(k, "KissLink", py::dynamic_attr())
        .def(py::init([](int cap, const py::object& clock, int n_sent, const py::object& broadcast, int persist,
                         double slot_s, double busy_limit_s) {
                 auto* l = new KissLink(cap, broadcast.is_none() ? std::string() : broadcast.cast<std::string>(),
                                        clock_of(clock));
                 l->n_sent = n_sent;
                 l->persist = persist;
                 l->slot_s = slot_s;
                 l->busy_limit_s = busy_limit_s;
                 return l;
             }),
             py::arg("cap") = 2, py::kw_only(), py::arg("clock") = py::none(), py::arg("n_sent") = 0,
             py::arg("broadcast") = py::none(), py::arg("persist") = 63, py::arg("slot_s") = kisslink::SLOT_S,
             py::arg("busy_limit_s") = 60.0)
        .def_readonly("cap", &KissLink::cap)
        .def_property_readonly("queue", [ack_objs](const py::object& self) {
            const auto& l = self.cast<const KissLink&>();
            py::dict d = ack_objs(self);
            py::list out;
            for (const auto& q : l.queue) {
                py::object ack = py::none();
                if (q.ack && d.contains(py::int_(*q.ack))) ack = d[py::int_(*q.ack)];
                out.append(py::make_tuple(q.port, pyb(q.frame), ack));
            }
            return out;
        })
        .def_property_readonly("peers", [](const KissLink& l) {
            py::object ns = py::module_::import("types").attr("SimpleNamespace");
            py::dict out;
            for (const auto& [h, p] : l.peers) {
                py::object report = py::none();
                if (p.report) report = py::make_tuple(p.report->first, p.report->second);
                out[py::int_(h)] = ns(py::arg("heard") = p.heard, py::arg("report") = report);
            }
            return out;
        })
        .def_property_readonly("ports", [](const KissLink& l) {
            py::dict out;
            for (const auto& [n, p] : l.ports) out[py::int_(n)] = py::cast(p);
            return out;
        })
        .def_property_readonly("events", [](const KissLink& l) { return l.events; })
        .def_property_readonly("acks", [acks_py](const py::object& self) {
            return acks_py(self, self.cast<const KissLink&>().acks, false);
        })
        .def("take_events", &KissLink::take_events)
        .def("take_acks", [acks_py](const py::object& self) { return acks_py(self, self.cast<KissLink&>().take_acks(), true); })
        .def_property_readonly("me", [](const KissLink& l) { return l.me; })
        .def_readwrite("n_sent", &KissLink::n_sent)
        .def_readonly("broadcast", &KissLink::broadcast)
        .def_readwrite("persist", &KissLink::persist)
        .def_readwrite("slot_s", &KissLink::slot_s)
        .def_readwrite("busy_limit_s", &KissLink::busy_limit_s)
        .def("command", [](KissLink& l, int cmd, const py::bytes& payload) { l.command(cmd, bytes_view(payload)); })
        .def("open", &KissLink::open, py::arg("group"), py::arg("from_call") = std::nullopt)
        .def("close", &KissLink::close)
        .def("set_mode", &KissLink::set_mode, py::arg("n"), py::arg("mode"), py::arg("auto") = false)
        .def("enqueue",
             [ack_objs](const py::object& self, const py::handle& f, int port, const py::object& ack) {
                 std::optional<std::int64_t> id;
                 if (!ack.is_none()) {
                     py::dict d = ack_objs(self);
                     const std::int64_t next = py::hasattr(self, "_ack_next") ? self.attr("_ack_next").cast<std::int64_t>() + 1 : 1;
                     self.attr("_ack_next") = py::int_(next);
                     d[py::int_(next)] = ack;
                     id = next;
                 }
                 self.cast<KissLink&>().enqueue(bytes_of(f), port, id);
             },
             py::arg("frame"), py::arg("port") = 0, py::arg("ack") = py::none())
        .def("on_sent", [](KissLink& l, const py::handle& burst) { l.on_sent(burst_cpp(burst)); })
        .def("missed", &KissLink::missed)
        .def("next_burst", [](KissLink& l) { return burst_py(l.next_burst()); })
        .def("on_burst", [](KissLink& l, py::dict r, const py::object&) -> py::object {
            auto h = heard_of(r);
            auto soft = cached_soft(r, h);
            // Python's PHY.DD_BUDGET_S as it stands, so a test patching it applies here too
            const auto budget = py::module_::import("data2g.arq.phy").attr("DD_BUDGET_S").cast<double>();
            std::optional<std::vector<std::pair<int, Bytes>>> frames;
            {
                py::gil_scoped_release nogil;  // the clock takes the GIL back if it is Python's
                frames = l.on_burst(h, std::move(soft), std::isfinite(budget) ? std::optional(budget) : std::nullopt);
            }
            if (!frames) return py::none();
            py::list out;
            for (const auto& [port, f] : *frames) out.append(py::make_tuple(port, pyb(f)));
            return out;
        }, py::arg("r"), py::arg("rx") = py::none());
}

}  // namespace data2g::bind
