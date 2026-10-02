// data2g_native: the C++ core as Python functions, for tests/conftest.py's
// `pytest --native` substitutions and tests/test_native_parity.py.
// Submodes are named by SubmodeSpec.name.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <stdexcept>
#include <string>

#include "codes/codes.hpp"

namespace py = pybind11;
using namespace data2g;

namespace {

std::span<const std::uint8_t> bytes_view(const py::bytes& b) {
    const std::string_view s(b);
    return {reinterpret_cast<const std::uint8_t*>(s.data()), s.size()};
}

const config::Submode& spec(const std::string& name) {
    const auto* s = codes::submode(name);
    if (!s) throw py::key_error("no submode " + name);
    return *s;
}

template <typename T, typename Out = T>
py::array_t<Out> to_array(std::span<const T> v) {
    py::array_t<Out> a(static_cast<py::ssize_t>(v.size()));
    std::copy(v.begin(), v.end(), a.mutable_data());
    return a;
}

}  // namespace

PYBIND11_MODULE(data2g_native, m) {
    // Bumped when the module's Python-facing signatures change, so conftest
    // refuses a stale build instead of failing confusingly.
    m.attr("__abi__") = 1;

    auto c = m.def_submodule("codes");
    c.attr("PLAIN") = codes::PLAIN;
    c.def("crc16", [](const py::bytes& b) { return codes::crc16(bytes_view(b)); });
    c.def("crc24", [](const py::bytes& b, std::uint32_t crc) { return codes::crc24(bytes_view(b), crc); },
          py::arg("data"), py::arg("crc") = 0xFFFFFF);
    c.def("crc32", [](const py::bytes& b) { return codes::crc32(bytes_view(b)); });
    c.def("with_crc", [](const py::bytes& payload, int n_crc, std::int64_t mask) {
        const auto out = codes::with_crc(bytes_view(payload), n_crc, static_cast<std::uint32_t>(mask));
        return py::bytes(reinterpret_cast<const char*>(out.data()), out.size());
    }, py::arg("payload"), py::arg("n_crc"), py::arg("mask") = 0);
    c.def("scramble_seed", &codes::scramble_seed, py::arg("index") = 0);
    c.def("scrambler", [](int k, int seed) { return to_array<std::uint8_t>(std::span<const std::uint8_t>(codes::scrambler(k, seed))); },
          py::arg("k"), py::arg("seed") = 0x1FF);
    c.def("interleaver", [](const std::string& name) { return to_array<std::uint16_t, std::int64_t>(codes::interleaver(spec(name))); });
    c.def("info_pos", [](const std::string& name) { return to_array<std::uint16_t, std::int64_t>(codes::info_pos(spec(name))); });

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
