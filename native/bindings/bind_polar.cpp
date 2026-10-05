#include "convert.hpp"
#include "polar/polar.hpp"

#include <optional>

namespace data2g::bind {

void bind_polar(py::module_& m) {
    using polar::PolarCode;
    using polar::SCLDecoder;
    auto p = m.def_submodule("polar");
    p.def("transform", [](const In<std::uint8_t>& u) {
        auto x = mat(u);
        for (std::size_t r = 0; r < x.rows; ++r) polar::transform(x.row(r));
        return np(x);
    });
    p.def("ga_info_pos", [](int k, int e) { return np<std::int64_t>(polar::ga_info_pos(k, e)); });
    p.def("ir_copies", [](int k, int e) { return np<std::int64_t>(polar::ir_copies(k, e)); });  // flattened

    // Shaped like polar.PolarCode where codes.py and decoders_torch read it.
    // info_pos None: the frozen GA design for (k, e).
    py::class_<PolarCode>(p, "PolarCode")
        .def(py::init([](int k, int e, std::optional<std::vector<std::uint16_t>> info) {
                 return info ? PolarCode(k, e, *info) : PolarCode(k, e, polar::ga_info_pos(k, e));
             }),
             py::arg("k"), py::arg("e"), py::arg("info_pos") = py::none())
        .def_readonly("k", &PolarCode::k)
        .def_readonly("e", &PolarCode::e)
        .def_readonly("n", &PolarCode::n)
        .def_property_readonly("info_pos", [](const PolarCode& c) { return np<std::int64_t>(std::span<const std::uint16_t>(c.info_pos)); })
        .def_property_readonly("sent", [](const PolarCode& c) { return np<std::int64_t>(std::span<const std::uint16_t>(c.sent)); })
        .def_property_readonly("punctured", [](const PolarCode& c) { return np<std::int64_t>(std::span<const std::uint16_t>(c.punctured)); })
        .def_property_readonly("copies", [](const PolarCode& c) {
            py::array_t<std::int64_t> out({static_cast<py::ssize_t>(c.copies.size()), py::ssize_t{2}});
            auto* o = out.mutable_data();
            for (const auto& [src, dst] : c.copies) *o++ = src, *o++ = dst;
            return out;
        })
        // polar.IRPolarCode(base, copies=copies), copies (src, dst) flattened
        .def_static("ir", [](const PolarCode& base, std::vector<std::uint16_t> copies) { return PolarCode::ir(base, copies); })
        .def("encode", [](const PolarCode& c, const In<std::uint8_t>& bits) { return np(c.encode(mat(bits))); });
    p.def("polar_code", [](const std::string& name) { return polar::polar_code(spec(name)); });

    py::class_<SCLDecoder>(p, "SCLDecoder")
        .def(py::init<PolarCode, int>(), py::arg("code"), py::arg("list_size") = 8)
        .def_property_readonly("code", &SCLDecoder::code)
        .def_property_readonly("L", &SCLDecoder::list_size)
        .def("decode", [](const SCLDecoder& d, const In<float>& llr) {
            if (llr.ndim() != 2) throw std::invalid_argument("expected (B, e) LLRs");
            auto in = mat(llr);
            polar::SclResult r;
            {
                py::gil_scoped_release unlocked;
                r = d.decode(in);
            }
            const py::ssize_t B = static_cast<py::ssize_t>(r.paths.rows), L = d.list_size(), k = d.code().k;
            py::array_t<std::uint8_t> paths({B, L, k});
            std::copy(r.paths.data.begin(), r.paths.data.end(), paths.mutable_data());
            return py::make_tuple(paths, np(r.metric));
        });
}

}  // namespace data2g::bind
