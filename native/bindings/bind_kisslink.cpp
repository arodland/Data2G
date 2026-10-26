// data2g.kisslink over the C++ core (native/core/kisslink): KissLink, AX.25
// parsing, group keys and data framing. A heard burst crosses as
// bind_arq_phy.cpp's does (heard_of, cached_soft); acks are Python objects
// kept on the KissLink's Python side, the C++ link carrying an id.
#include <pybind11/functional.h>
#include <cmath>

#include "arq_convert.hpp"
#include "kisslink/kisslink.hpp"

namespace data2g::bind {

void bind_kisslink(py::module_& m) {
    using kisslink::KissLink;

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
    k.def("cobs", [](const py::bytes& f) { return pyb(kisslink::cobs(bytes_view(f))); });
    k.def("uncobs", [](const py::bytes& b) -> py::object {
        const auto f = kisslink::uncobs(bytes_view(b));
        return f ? py::object(pyb(*f)) : py::none();
    });
    k.def("framed_len", &kisslink::framed_len);
    k.def("pack_frames", [](const py::list& frames, std::size_t pb) {
        std::vector<Bytes> f;
        for (const auto& x : frames) f.push_back(bytes_of(x));
        py::list out;
        for (const auto& p : kisslink::pack_frames(f, pb)) out.append(pyb(p));
        return out;
    });
    k.def("unpack_frames", [](const py::list& payloads, const std::vector<bool>& ok) {
        std::vector<Bytes> pl;
        for (const auto& x : payloads) pl.push_back(bytes_of(x));
        auto [frames, lost] = kisslink::unpack_frames(pl, ok);
        py::list out;
        for (const auto& f : frames) out.append(pyb(f));
        return py::make_tuple(out, lost);
    });
    py::class_<kisslink::Port>(k, "Port")
        .def_readonly("group", &kisslink::Port::group)
        .def_readonly("mode", &kisslink::Port::mode)
        .def_property_readonly("auto", [](const kisslink::Port& p) { return p.shift; })
        .def_readonly("dup", &kisslink::Port::dup)
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
        .def("set_mode", &KissLink::set_mode, py::arg("n"), py::arg("mode"), py::arg("auto") = false,
             py::arg("dup") = false)
        .def("reset", &KissLink::reset)
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
        .def("missed", &KissLink::missed, py::arg("submode"), py::arg("n_cw"), py::arg("snr") = py::none())
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
