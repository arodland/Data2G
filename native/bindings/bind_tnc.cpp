// data2g.tnc: the KISS byte codec.
#include "convert.hpp"
#include "tnc/tnc.hpp"

namespace data2g::bind {

namespace {

std::span<const std::uint8_t> view(const py::bytes& b) { return bytes_view(b); }

}  // namespace

void bind_tnc(py::module_& m) {
    auto t = m.def_submodule("tnc", "data2g.tnc");
    t.def("kiss_encode", [](const py::bytes& data, int port, int cmd) { return to_bytes(tnc::kiss_encode(view(data), port, cmd)); },
          py::arg("data"), py::arg("port") = 0, py::arg("cmd") = tnc::KISS_DATA);
    py::class_<tnc::KissDecoder>(t, "KissDecoder")
        .def(py::init<>())
        .def("feed", [](tnc::KissDecoder& d, const py::bytes& data) {
            py::list out;
            for (const auto& [cmd, payload] : d.feed(view(data))) out.append(py::make_tuple(cmd, to_bytes(payload)));
            return out;
        });
}

}  // namespace data2g::bind
