// data2g.arq's modes, predictor and policy (the gear shifter). Modes cross
// by name; measurements as predictor dicts; a station as any object with
// link.Station's attributes (missing ones take their defaults).
#include "arq/modes.hpp"
#include "arq/policy.hpp"
#include "arq/predictor.hpp"
#include "convert.hpp"

namespace data2g::bind {

namespace {

const arq::Mode& amode(const std::string& name) {
    const auto* m = arq::mode(name);
    if (!m) throw py::key_error("no mode " + name);
    return *m;
}

arq::Measured measured(const py::dict& d) {
    arq::Measured m;
    m.snr_est = d["snr_est"].cast<double>();
    m.spread_est = d["spread_est"].cast<double>();
    m.delay_est_ms = d["delay_est_ms"].cast<double>();
    for (std::size_t i = 0; i < arq::CONSTS.size(); ++i)
        m.mi[i] = d[py::str("mi_" + std::string(arq::CONSTS[i]))].cast<double>();
    if (d.contains("headroom")) m.headroom = d["headroom"].cast<double>();
    if (d.contains("frames")) m.frames = d["frames"].cast<double>();
    return m;
}

py::dict to_dict(const arq::Measured& m) {
    py::dict d;
    d["snr_est"] = m.snr_est;
    d["spread_est"] = m.spread_est;
    d["delay_est_ms"] = m.delay_est_ms;
    for (std::size_t i = 0; i < arq::CONSTS.size(); ++i) d[py::str("mi_" + std::string(arq::CONSTS[i]))] = m.mi[i];
    d["headroom"] = m.headroom;
    d["frames"] = m.frames;
    return d;
}

// predictor's prev: (measured, band, age) or None.
std::optional<arq::Prev> prev(const py::object& p) {
    if (p.is_none()) return std::nullopt;
    auto t = p.cast<py::tuple>();
    return arq::Prev{measured(t[0].cast<py::dict>()), t[1].cast<std::string>(), t[2].cast<double>()};
}

template <typename T>
T attr(const py::object& o, const char* name, T dflt) {
    if (!py::hasattr(o, name)) return dflt;
    const py::object v = o.attr(name);
    return v.is_none() ? dflt : v.cast<T>();
}

std::optional<int> opt_int(const py::object& o, const char* name) {
    if (!py::hasattr(o, name) || o.attr(name).is_none()) return std::nullopt;
    return o.attr(name).cast<int>();
}

arq::StationView station(const py::object& s) {
    arq::StationView v;
    v.cap = s.attr("cap").cast<int>();
    if (py::hasattr(s, "tx")) v.pending = py::bool_(s.attr("tx").attr("pending")());
    v.peer_recommend = opt_int(s, "peer_recommend");
    v.peer_reply_recommend = opt_int(s, "peer_reply_recommend");
    v.peer_size_hint = attr(s, "peer_size_hint", 1);
    v.peer_wants_dup = py::bool_(attr<py::object>(s, "peer_wants_dup", py::bool_(false)));
    v.chat = py::bool_(attr<py::object>(s, "chat", py::bool_(false))) ||
             py::bool_(attr<py::object>(s, "peer_chat", py::bool_(false)));
    v.peer_queued = attr(s, "peer_queued", 0L);
    if (py::hasattr(s, "rx")) v.held = static_cast<long>(py::len(s.attr("rx").attr("buf")));
    return v;
}

using Shifter = arq::GearShifter;

py::dict map_to_dict(const Shifter::Map& m) {
    py::dict d;
    for (const auto& [k, v] : m) d[py::str(k)] = v;
    return d;
}

Shifter::Map dict_to_map(const py::dict& d) {
    Shifter::Map m;
    for (const auto& [k, v] : d) m[k.cast<std::string>()] = v.cast<double>();
    return m;
}

}  // namespace

void bind_arq(py::module_& m) {
    auto a = m.def_submodule("arq", "data2g.arq: modes, predictor, policy");

    // -- modes
    a.def("modes", [] {
        std::vector<std::string> out;
        for (const auto& md : arq::modes()) out.emplace_back(md.name);
        return out;
    });
    a.def("is_cpm", [](const std::string& n) { return amode(n).is_cpm(); });
    a.def("burst_seconds", [](const std::string& n, int n_cw, bool dup) { return arq::burst_seconds(amode(n), n_cw, dup); },
          py::arg("name"), py::arg("n_cw"), py::arg("dup") = false);
    a.def("payload_bytes", [](const std::string& n) { return arq::payload_bytes(amode(n)); });
    a.def("ctl_payload_bytes", [](const std::string& n) { return arq::ctl_payload_bytes(amode(n)); });
    a.def("max_ctl", [](const std::string& n) { return arq::max_ctl(amode(n)); });
    a.def("min_cw", [](const std::string& n, bool data) { return arq::min_cw(amode(n), data); });
    a.def("rv_cycle", [](const std::string& n) { return arq::rv_cycle(amode(n)); });

    // -- predictor
    a.attr("CONSTS") = py::make_tuple(std::string(arq::CONSTS[0]), std::string(arq::CONSTS[1]),
                                      std::string(arq::CONSTS[2]), std::string(arq::CONSTS[3]));
    a.def("const_family", [](const std::string& n) { return std::string(arq::const_family(n)); });
    a.def("capacity", [](const In<double>& snr_db, const std::string& c) {
        py::array_t<double> out(snr_db.size());
        for (py::ssize_t i = 0; i < snr_db.size(); ++i) out.mutable_data()[i] = arq::capacity(snr_db.data()[i], c);
        return out;
    });
    a.def("effective_mi", [](const In<std::complex<double>>& h, const In<double>& var, const std::string& c) {
        return arq::effective_mi({h.data(), static_cast<std::size_t>(h.size())},
                                 {var.data(), static_cast<std::size_t>(var.size())}, c);
    });
    a.def("outcome_inputs", [](const py::dict& md, const std::string& band, double gap, double seconds,
                               const py::object& p, std::optional<std::vector<std::string>> bands) {
        const auto pv = prev(p);
        std::vector<std::string_view> bv;
        if (bands) bv.assign(bands->begin(), bands->end());
        return np(arq::outcome_inputs(measured(md), band, gap, seconds, pv ? &*pv : nullptr,
                                      bands ? std::span<const std::string_view>(bv) : tables::OUTCOME_BANDS));
    }, py::arg("measured"), py::arg("band"), py::arg("gap"), py::arg("seconds"), py::arg("prev") = py::none(),
       py::arg("bands") = py::none());
    a.def("outcome_logits", [](const In<double>& x) {
        return np(arq::outcome_logits({x.data(), static_cast<std::size_t>(x.size())}));
    });
    a.def("outcome_modes", [] { return std::vector<std::string>(tables::OUTCOME_MODES.begin(), tables::OUTCOME_MODES.end()); });
    a.def("outcome_bands", [] { return std::vector<std::string>(tables::OUTCOME_BANDS.begin(), tables::OUTCOME_BANDS.end()); });
    a.def("outcome_knows", [](const std::string& n) { return arq::outcome_knows(n); });
    a.def("predict_outcome", [](const py::dict& md, const std::string& band, double gap, double seconds,
                                const py::object& p) {
        const auto pv = prev(p);
        const auto out = arq::predict_outcome(measured(md), band, gap, seconds, pv ? &*pv : nullptr);
        py::dict d;
        for (std::size_t i = 0; i < out.size(); ++i)
            d[py::str(std::string(tables::OUTCOME_MODES[i]))] = py::make_tuple(out[i].burst, out[i].cw);
        return d;
    }, py::arg("measured"), py::arg("band"), py::arg("gap"), py::arg("seconds"), py::arg("prev") = py::none());

    // -- policy
    a.def("cap_hz", &arq::cap_hz);
    a.def("fallback", [](int cap) { return std::string(arq::fallback(cap)); });
    a.def("connect_mode", [](int cap, int tries) { return std::string(arq::connect_mode(cap, tries)); },
          py::arg("cap"), py::arg("tries") = 0);
    a.def("width_hz", [](const std::string& n) { return arq::width_hz(amode(n)); });
    a.def("allowed", [](int cap) {
        std::vector<std::string> out;
        for (const auto* md : arq::allowed(cap)) out.emplace_back(md->name);
        return out;
    });
    a.def("encode", [](const std::string& n) { return arq::encode(n); });
    a.def("decode", [](int rec) -> std::optional<std::string> {
        const auto* md = arq::decode(rec);
        return md ? std::optional<std::string>(md->name) : std::nullopt;
    });
    a.def("ctl_slots", [](const std::string& n) { return arq::ctl_slots(amode(n)); });
    a.def("slots_for", [](const std::string& n, double seconds, bool data, bool dup) {
        return arq::slots_for(amode(n), seconds, data, dup);
    }, py::arg("name"), py::arg("seconds"), py::arg("data") = true, py::arg("dup") = false);

    py::class_<Shifter>(a, "GearShifter")
        .def(py::init<>())
        .def_readwrite("gap_s", &Shifter::gap_s)
        .def_readwrite("use_cpm", &Shifter::use_cpm)
        .def_readwrite("min_success", &Shifter::min_success)
        .def_readwrite("measured_band", &Shifter::measured_band)
        .def_readwrite("measured_at", &Shifter::measured_at)
        .def_readwrite("want_dup", &Shifter::want_dup)
        .def_readwrite("peer_had_data", &Shifter::peer_had_data)
        .def_property("measured",
            [](const Shifter& s) -> py::object { return s.measured ? py::object(to_dict(*s.measured)) : py::none(); },
            [](Shifter& s, const py::object& d) {
                s.measured = d.is_none() ? std::nullopt : std::optional(measured(d.cast<py::dict>()));
            })
        .def_property("prev",
            [](const Shifter& s) -> py::object {
                if (!s.prev) return py::none();
                return py::make_tuple(to_dict(s.prev->m), s.prev->band, s.prev->at);
            },
            [](Shifter& s, const py::object& p) {
                if (p.is_none()) return s.prev.reset();
                auto t = p.cast<py::tuple>();
                s.prev = Shifter::Heard{measured(t[0].cast<py::dict>()), t[1].cast<std::string>(), t[2].cast<double>()};
            })
        .def_property("bias", [](const Shifter& s) { return map_to_dict(s.bias); },
                      [](Shifter& s, const py::dict& d) { s.bias = dict_to_map(d); })
        .def_property("bias_burst", [](const Shifter& s) { return map_to_dict(s.bias_burst); },
                      [](Shifter& s, const py::dict& d) { s.bias_burst = dict_to_map(d); })
        .def_property("predicted",
            [](const Shifter& s) {
                py::dict d;
                for (const auto& [k, v] : s.predicted) d[py::str(k)] = py::make_tuple(v.first, v.second);
                return d;
            },
            [](Shifter& s, const py::dict& d) {
                s.predicted.clear();
                for (const auto& [k, v] : d) s.predicted[k.cast<std::string>()] = v.cast<std::pair<double, double>>();
            })
        .def_property("log",
            [](const Shifter& s) {
                py::list out;
                for (const auto& e : s.log) out.append(py::make_tuple(e.data, e.hint, e.reply));
                return out;
            },
            [](Shifter& s, const py::list& l) {
                s.log.clear();
                for (const auto& e : l) {
                    auto t = e.cast<py::tuple>();
                    s.log.push_back({t[0].cast<std::string>(), t[1].cast<int>(), t[2].cast<std::string>()});
                }
            })
        .def("choose", [](const Shifter& s, const py::object& st, int escalation) {
            const auto [name, n] = s.choose(station(st), escalation);
            return py::make_tuple(std::string(name), n);
        })
        .def("next_capacity", [](const Shifter& s, const py::object& st) { return s.next_capacity(station(st)); })
        .def("observe", [](Shifter& s, const py::dict& md, const std::string& submode, double now) {
            s.observe(measured(md), submode, now);
        })
        .def("outcome", [](Shifter& s, const std::string& submode, int decoded, int sent, std::optional<bool> usable) {
            s.outcome(submode, decoded, sent, usable);
        }, py::arg("submode"), py::arg("decoded"), py::arg("sent"), py::arg("usable") = py::none())
        .def("recommend", [](Shifter& s, const py::object& st) {
            const auto r = s.recommend(station(st));
            return py::make_tuple(r.data, r.hint, r.reply);
        });
}

}  // namespace data2g::bind
