// data2g.arq values <-> Python, shared by the arq bindings: bytes, mask
// ids, link.TxBurst, predictor measurement dicts.
#pragma once

#include "arq/engine.hpp"
#include "arq/link.hpp"
#include "arq/predictor.hpp"
#include "arq/session.hpp"
#include "convert.hpp"

namespace data2g::bind {

// bind_arq_link's adapters: a Python policy object, a random.Random, and a
// Session whose stations see a test's build override.
std::shared_ptr<arq::Policy> py_policy(py::object o);
// bind_engine's Engine object as the C++ engine (bind_host).
arq::Engine& engine_of(const py::handle& h);
std::shared_ptr<arq::Rng> py_rng(py::object o);
std::shared_ptr<arq::Session> py_session(const std::string& call, std::shared_ptr<arq::Policy> policy,
                                         std::shared_ptr<arq::Rng> rng, const std::vector<std::string>& aliases,
                                         double stats_interval_s);

using arq::Bytes;
using arq::MaskId;
using arq::TxBurst;
using arq::TxBurstPtr;

inline Bytes bytes_of(const py::handle& h) {
    const std::string_view s = py::isinstance<py::bytes>(h) ? std::string_view(h.cast<py::bytes>())
                                                            : std::string_view(py::bytes(py::reinterpret_borrow<py::object>(h)));
    return {s.begin(), s.end()};
}

inline py::bytes pyb(const Bytes& b) { return py::bytes(reinterpret_cast<const char*>(b.data()), b.size()); }

inline py::tuple mask_tuple(const MaskId& m) { return py::make_tuple(m.key, m.direction, m.seq); }

inline MaskId mask_of(const py::handle& t) {
    auto s = t.cast<py::sequence>();
    return {s[0].cast<int>(), s[1].cast<int>(), s[2].cast<int>()};
}

inline py::object link_attr(const char* name) { return py::module_::import("data2g.arq.link").attr(name); }

inline py::object burst_py(const TxBurstPtr& b) {
    if (!b) return py::none();
    auto Slot = link_attr("Slot");
    py::list slots;
    for (const auto& s : b->slots) slots.append(Slot(mask_tuple(s.mask_id), s.rv, pyb(s.payload)));
    return link_attr("TxBurst")(b->submode, slots, b->burst_seq);
}

inline TxBurstPtr burst_cpp(const py::handle& o) {
    if (o.is_none()) return nullptr;
    auto b = std::make_shared<TxBurst>();
    b->submode = o.attr("submode").cast<std::string>();
    for (auto s : o.attr("slots")) b->slots.push_back({mask_of(s.attr("mask_id")), s.attr("rv").cast<int>(), bytes_of(s.attr("payload"))});
    b->burst_seq = o.attr("burst_seq").cast<std::int64_t>();
    return b;
}

inline arq::Measured measured(const py::dict& d) {
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

inline py::dict to_dict(const arq::Measured& m) {
    py::dict d;
    d["snr_est"] = m.snr_est;
    d["spread_est"] = m.spread_est;
    d["delay_est_ms"] = m.delay_est_ms;
    for (std::size_t i = 0; i < arq::CONSTS.size(); ++i) d[py::str("mi_" + std::string(arq::CONSTS[i]))] = m.mi[i];
    d["headroom"] = m.headroom;
    d["frames"] = m.frames;
    return d;
}

}  // namespace data2g::bind
