// data2g.receiver: search_span, receive_any, NoiseProfile and the streaming
// Receiver. Results are Python's dicts with submodes / CPM specs by name
// (tests/native_subs/receiver.py restores the spec objects); events are (kind, dict).
#include "convert.hpp"
#include "modem_dicts.hpp"
#include "receiver/noise_profile.hpp"
#include "receiver/receiver.hpp"

namespace data2g::bind {

namespace {

// Names as views into the static tables (the core keeps string_views).
std::vector<std::string_view> grid_views(const std::vector<std::string>& names) {
    std::vector<std::string_view> out;
    for (const auto& n : names) {
        const auto* g = cpm::grid(n);
        if (!g) throw py::key_error("no CPM grid " + n);
        out.push_back(g->name);
    }
    return out;
}

std::vector<std::string_view> band_views(const std::vector<std::string>& names) {
    std::vector<std::string_view> out;
    for (const auto& n : names) out.push_back(config::band(n).name);
    return out;
}

modem::Accept accept_or_all(const py::object& a) {
    auto acc = modem_accept_of(a);
    if (!acc) throw py::type_error("Receiver needs an Accept");
    return *acc;
}

py::dict cpm_lock_dict(const cpm::Lock& l) {
    py::dict d;
    d["spec"] = std::string(l.spec->name);
    d["n_data"] = l.n_data;
    d["dup"] = l.dup;
    d["start"] = l.start;
    d["cfo"] = l.cfo;
    d["score"] = l.score;
    d["header_score"] = l.header_score;
    d["header_margin"] = l.header_margin;
    d["end"] = l.end;
    d["header_end"] = l.header_end;
    d["band"] = std::string(l.spec->grid);
    d["family"] = "cpm";
    d["n_cw"] = 1 + l.dup + l.n_data;
    return d;
}

py::dict pending_dict(const receiver::Pending& p) { return p.is_cpm() ? cpm_lock_dict(p.cpm()) : modem_lock_dict(p.ofdm()); }

py::dict rx_dict(const receiver::Rx& rx) {
    if (const auto* r = std::get_if<modem::Received>(&rx)) return modem_received_dict(*r);
    const auto& r = std::get<cpm::Received>(rx);
    py::dict d;
    d["family"] = "cpm";
    d["spec"] = std::string(r.spec->name);
    d["band"] = std::string(r.spec->grid);
    d["n_cw"] = r.n_cw;
    d["n_ctl_slots"] = r.n_ctl_slots;
    d["dup"] = r.dup;
    py::list soft;
    for (const auto& s : r.soft) soft.append(np(s));
    d["soft"] = soft;
    d["E"] = np(r.E);
    d["cfo"] = r.cfo;
    d["preamble_start"] = r.preamble_start;
    d["header_end"] = r.header_end;
    return d;
}

py::list events(const std::vector<receiver::Event>& evs) {
    py::list out;
    for (const auto& e : evs) {
        if (const auto* h = std::get_if<receiver::HeaderEvent>(&e)) {
            py::dict d = pending_dict(h->header);
            d["stream_end"] = h->stream_end;
            out.append(py::make_tuple("header", d));
            continue;
        }
        const auto& b = std::get<receiver::BurstEvent>(e);
        py::dict d;
        d["header"] = pending_dict(b.header);
        d["rx"] = b.rx ? py::object(rx_dict(*b.rx)) : py::object(py::none());
        d["audio"] = np(b.audio);
        out.append(py::make_tuple("burst", d));
    }
    return out;
}

}  // namespace

void bind_receiver(py::module_& m) {
    auto rcv = m.def_submodule("receiver", "data2g.receiver");
    rcv.def("search_span", [](const std::vector<std::string>& bands, const std::vector<std::string>& grids) {
        return receiver::search_span(band_views(bands), grid_views(grids));
    }, py::arg("bands"), py::arg("cpm_grids") = std::vector<std::string>{});
    rcv.def("receive_any", [](const In<double>& y, std::int64_t lead,
                            const std::optional<std::vector<std::string>>& grids) -> py::object {
        std::optional<std::vector<std::string_view>> g;
        if (grids) g = grid_views(*grids);
        std::optional<receiver::Rx> r;
        {
            py::gil_scoped_release nogil;
            r = receiver::receive_any({y.data(), static_cast<std::size_t>(y.size())}, lead, g);
        }
        return r ? py::object(rx_dict(*r)) : py::object(py::none());
    }, py::arg("y"), py::arg("lead") = 0, py::arg("cpm_grids") = py::none());

    py::class_<receiver::NoiseProfile>(rcv, "NoiseProfile")
        .def(py::init<>())
        .def("feed", [](receiver::NoiseProfile& p, const In<double>& x, double t_start) {
            p.feed({x.data(), static_cast<std::size_t>(x.size())}, t_start);
        })
        .def("mark", &receiver::NoiseProfile::mark)
        .def("span_z", [](const receiver::NoiseProfile& p, double start, double end, double lo, double hi) {
            return p.span_z(p.span_spectra(start, end), lo, hi);  // engine.py's span_z(span_spectra(start, end), lo, hi)
        })
        .def("snapshot", [](const receiver::NoiseProfile& p) -> py::object {
            const auto s = p.snapshot();
            if (!s) return py::none();
            py::dict d;
            d["noise_db"] = std::vector<double>(s->db.begin(), s->db.end());
            d["noise_tail_db"] = std::vector<double>(s->tail_db.begin(), s->tail_db.end());
            d["impulses_per_min"] = s->impulses_per_min;
            d["noise_blocks"] = s->blocks;
            return d;
        });

    py::class_<receiver::Receiver>(rcv, "Receiver")
        .def(py::init([](const py::object& accept, const std::vector<std::string>& grids, bool blank) {
            return receiver::Receiver(accept_or_all(accept), grid_views(grids), blank);
        }), py::arg("accept"), py::arg("cpm_grids") = std::vector<std::string>{}, py::arg("blank") = true)
        .def("feed", [](receiver::Receiver& r, const In<double>& x) {
            std::vector<receiver::Event> evs;
            {
                py::gil_scoped_release nogil;
                evs = r.feed({x.data(), static_cast<std::size_t>(x.size())});
            }
            return events(evs);
        })
        .def("reset", &receiver::Receiver::reset)
        .def("expect", [](receiver::Receiver& r, const std::vector<std::string>& grids, std::int64_t lo, std::int64_t hi,
                          double cfo) { r.expect(grid_views(grids), lo, hi, cfo); })
        .def_property_readonly("n_expected", &receiver::Receiver::n_expected)
        .def("drop", &receiver::Receiver::drop)
        .def_property_readonly("busy", &receiver::Receiver::busy)
        .def_property_readonly("channel_busy", &receiver::Receiver::channel_busy)
        .def_property_readonly("on_air", &receiver::Receiver::on_air)
        .def_property_readonly("n_blanked", &receiver::Receiver::n_blanked)
        .def_property_readonly("pending", [](const receiver::Receiver& r) -> py::object {
            return r.pending() ? py::object(pending_dict(*r.pending())) : py::object(py::none());
        });
}

}  // namespace data2g::bind
