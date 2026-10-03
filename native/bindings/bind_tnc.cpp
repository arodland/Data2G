// data2g.tnc: KISS framing, burst packing, search_span, receive_any and the
// streaming Receiver. Results are Python's dicts with submodes / CPM specs
// by name (conftest restores the spec objects); events are (kind, dict).
#include "convert.hpp"
#include "modem_dicts.hpp"
#include "tnc/tnc.hpp"

namespace data2g::bind {

namespace {

std::span<const std::uint8_t> view(const py::bytes& b) { return bytes_view(b); }

std::vector<tnc::Bytes> bytes_list(const std::vector<py::bytes>& v) {
    std::vector<tnc::Bytes> out;
    for (const auto& b : v) {
        const auto s = view(b);
        out.emplace_back(s.begin(), s.end());
    }
    return out;
}

py::list py_bytes_list(const std::vector<tnc::Bytes>& v) {
    py::list out;
    for (const auto& b : v) out.append(to_bytes(b));
    return out;
}

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
    for (const auto& n : names) out.push_back(modem::band(n).name);
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

py::dict pending_dict(const tnc::Pending& p) { return p.is_cpm() ? cpm_lock_dict(p.cpm()) : modem_lock_dict(p.ofdm()); }

py::dict rx_dict(const tnc::Rx& rx) {
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

py::list events(const std::vector<tnc::Event>& evs) {
    py::list out;
    for (const auto& e : evs) {
        if (const auto* h = std::get_if<tnc::HeaderEvent>(&e)) {
            out.append(py::make_tuple("header", pending_dict(h->header)));
            continue;
        }
        const auto& b = std::get<tnc::BurstEvent>(e);
        py::dict d;
        d["header"] = pending_dict(b.header);
        d["rx"] = b.rx ? py::object(rx_dict(*b.rx)) : py::object(py::none());
        d["audio"] = np(b.audio);
        out.append(py::make_tuple("burst", d));
    }
    return out;
}

}  // namespace

void bind_tnc(py::module_& m) {
    auto t = m.def_submodule("tnc", "data2g.tnc");
    t.def("kiss_encode", [](const py::bytes& data, int port) { return to_bytes(tnc::kiss_encode(view(data), port)); },
          py::arg("data"), py::arg("port") = 0);
    py::class_<tnc::KissDecoder>(t, "KissDecoder")
        .def(py::init<>())
        .def("feed", [](tnc::KissDecoder& d, const py::bytes& data) {
            py::list out;
            for (const auto& [cmd, payload] : d.feed(view(data))) out.append(py::make_tuple(cmd, to_bytes(payload)));
            return out;
        });
    t.def("capacity", [](const std::string& s, int max_cw) { return tnc::capacity(spec(s), max_cw); },
          py::arg("spec"), py::arg("max_cw") = config::MAX_CODEWORDS);
    t.def("pack", [](const std::vector<py::bytes>& packets, const std::string& s) {
        return py_bytes_list(tnc::pack(bytes_list(packets), spec(s)));
    });
    t.def("unpack", [](const std::vector<py::bytes>& payloads, const std::vector<bool>& ok) {
        auto [packets, lost] = tnc::unpack(bytes_list(payloads), ok);
        return py::make_tuple(py_bytes_list(packets), lost);
    });
    t.def("search_span", [](const std::vector<std::string>& bands, const std::vector<std::string>& grids) {
        return tnc::search_span(band_views(bands), grid_views(grids));
    }, py::arg("bands"), py::arg("cpm_grids") = std::vector<std::string>{});
    t.def("receive_any", [](const In<double>& y, std::int64_t lead,
                            const std::optional<std::vector<std::string>>& grids) -> py::object {
        std::optional<std::vector<std::string_view>> g;
        if (grids) g = grid_views(*grids);
        std::optional<tnc::Rx> r;
        {
            py::gil_scoped_release nogil;
            r = tnc::receive_any({y.data(), static_cast<std::size_t>(y.size())}, lead, g);
        }
        return r ? py::object(rx_dict(*r)) : py::object(py::none());
    }, py::arg("y"), py::arg("lead") = 0, py::arg("cpm_grids") = py::none());

    py::class_<tnc::NoiseProfile>(t, "NoiseProfile")
        .def(py::init<>())
        .def("feed", [](tnc::NoiseProfile& p, const In<double>& x, double t_start) {
            p.feed({x.data(), static_cast<std::size_t>(x.size())}, t_start);
        })
        .def("mark", &tnc::NoiseProfile::mark)
        .def("snapshot", [](const tnc::NoiseProfile& p) -> py::object {
            const auto s = p.snapshot();
            if (!s) return py::none();
            py::dict d;
            d["noise_db"] = std::vector<double>(s->db.begin(), s->db.end());
            d["noise_tail_db"] = std::vector<double>(s->tail_db.begin(), s->tail_db.end());
            d["noise_blocks"] = s->blocks;
            return d;
        });

    py::class_<tnc::Receiver>(t, "Receiver")
        .def(py::init([](const py::object& accept, const std::vector<std::string>& grids, bool blank) {
            return tnc::Receiver(accept_or_all(accept), grid_views(grids), blank);
        }), py::arg("accept"), py::arg("cpm_grids") = std::vector<std::string>{}, py::arg("blank") = true)
        .def("feed", [](tnc::Receiver& r, const In<double>& x) {
            std::vector<tnc::Event> evs;
            {
                py::gil_scoped_release nogil;
                evs = r.feed({x.data(), static_cast<std::size_t>(x.size())});
            }
            return events(evs);
        })
        .def("reset", &tnc::Receiver::reset)
        .def_property_readonly("busy", &tnc::Receiver::busy)
        .def_property_readonly("channel_busy", &tnc::Receiver::channel_busy)
        .def_property_readonly("on_air", &tnc::Receiver::on_air)
        .def_property_readonly("n_blanked", &tnc::Receiver::n_blanked)
        .def_property_readonly("pending", [](const tnc::Receiver& r) -> py::object {
            return r.pending() ? py::object(pending_dict(*r.pending())) : py::object(py::none());
        });
}

}  // namespace data2g::bind
