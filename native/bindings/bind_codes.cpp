#include "convert.hpp"

namespace data2g::bind {

void bind_codes(py::module_& m) {
    auto c = m.def_submodule("codes");
    c.attr("PLAIN") = codes::PLAIN;
    c.def("crc16", [](const py::bytes& b) { return codes::crc16(bytes_view(b)); });
    c.def("crc24", [](const py::bytes& b, std::uint32_t crc) { return codes::crc24(bytes_view(b), crc); },
          py::arg("data"), py::arg("crc") = 0xFFFFFF);
    c.def("crc32", [](const py::bytes& b) { return codes::crc32(bytes_view(b)); });
    c.def("with_crc", [](const py::bytes& payload, int n_crc, std::int64_t mask) {
        return to_bytes(codes::with_crc(bytes_view(payload), n_crc, static_cast<std::uint32_t>(mask)));
    }, py::arg("payload"), py::arg("n_crc"), py::arg("mask") = 0);
    c.def("scramble_seed", &codes::scramble_seed, py::arg("index") = 0);
    c.def("scrambler", [](int k, int seed) { return np(codes::scrambler(k, seed)); },
          py::arg("k"), py::arg("seed") = 0x1FF);
    c.def("interleaver", [](const std::string& name) { return np<std::int64_t>(codes::interleaver(spec(name))); });
    c.def("info_pos", [](const std::string& name) { return np<std::int64_t>(codes::info_pos(spec(name))); });

    auto cfg = m.def_submodule("config");
    cfg.def("submodes", [] {
        py::list out;
        for (const auto& s : config::SUBMODES) {
            py::dict d;
            d["index"] = s.index;
            d["name"] = std::string(s.name);
            d["code"] = std::string(s.code);
            d["constellation"] = std::string(s.constellation);
            d["band"] = std::string(s.band);
            d["frames_per_cw"] = s.frames_per_cw;
            d["k"] = s.k;
            d["coded_bits"] = s.coded_bits;
            d["headroom"] = s.headroom;
            out.append(d);
        }
        return out;
    });
}

}  // namespace data2g::bind
