#include <optional>

#include "convert.hpp"
#include "ldpc/ldpc.hpp"

namespace data2g::bind {

namespace {

// A base matrix as numpy int64, like ldpc.tables().
py::array_t<std::int64_t> base(const ldpc::Code& c, int rows, int cols) {
    py::array_t<std::int64_t> a({rows, cols});
    auto v = a.mutable_unchecked<2>();
    for (int r = 0; r < rows; ++r)
        for (int col = 0; col < cols; ++col) v(r, col) = c.shift(r, col);
    return a;
}

py::array_t<std::int64_t> ints(const std::vector<int>& v) { return np<std::int64_t>(std::span<const int>(v)); }

py::array_t<bool> flags(const std::vector<std::uint8_t>& v) { return np<bool>(std::span<const std::uint8_t>(v)); }

}  // namespace

void bind_ldpc(py::module_& m) {
    auto l = m.def_submodule("ldpc");
    l.attr("CH_CLAMP") = ldpc::CH_CLAMP;
    l.attr("BIG") = ldpc::BIG;
    l.def("phi", [](const In<float>& x) {
        auto v = vec(x);
        ldpc::phi(v);
        return np(v);
    });
    l.def("layout", [](int k, int n, std::optional<int> bg) { return ldpc::layout(k, n, bg.value_or(0)); },
          py::arg("k"), py::arg("n"), py::arg("bg") = py::none());

    // The attributes and methods of ldpc.QCLDPC, so it can stand in for one.
    py::class_<ldpc::Code>(l, "QCLDPC")
        .def_readonly("z", &ldpc::Code::z)
        .def_readonly("kb", &ldpc::Code::kb)
        .def_readonly("k", &ldpc::Code::k)
        .def_readonly("n", &ldpc::Code::n)
        .def_readonly("mb", &ldpc::Code::mb)
        .def_property_readonly("n_cols", &ldpc::Code::n_cols)
        .def_property_readonly("base", [](const ldpc::Code& c) { return base(c, c.mb, c.kb + c.mb); })
        .def_property_readonly("full_base", [](const ldpc::Code& c) { return base(c, c.full_rows(), c.full_cols()); })
        .def_property_readonly("sent", [](const ldpc::Code& c) { return ints(c.sent()); })
        .def_property_readonly("edges", [](const ldpc::Code& c) {
            const auto [r, col] = c.edges();
            return py::make_tuple(ints(r), ints(col));
        })
        .def("mother", [](const ldpc::Code& c, std::optional<int> n) { return c.mother(n.value_or(0)); },
             py::arg("n") = py::none())
        .def("encode", [](const ldpc::Code& c, const In<std::uint8_t>& bits) { return np(c.encode(mat(bits))); })
        .def("encode_full", [](const ldpc::Code& c, const In<std::uint8_t>& bits) { return np(c.encode_full(mat(bits))); })
        .def("syndrome_ok", [](const ldpc::Code& c, const In<std::uint8_t>& full) { return flags(c.syndrome_ok(mat(full))); });

    l.def("qc_code", [](int k, int n, std::optional<int> bg) {
        try {
            return ldpc::qc_code(k, n, bg.value_or(0));
        } catch (const std::out_of_range& e) {
            throw py::key_error(e.what());  // as ldpc.qc_code
        }
    }, py::arg("k"), py::arg("n"), py::arg("bg") = py::none());

    // ldpc.MinSumDecoder: decode(llr_sent, iters, alpha, posterior) -> (bits
    // uint8 (B, k), ok bool (B,)), plus posterior float32 (B, n) if asked.
    py::class_<ldpc::Decoder>(l, "MinSumDecoder")
        .def(py::init<const ldpc::Code&>())
        .def("decode", [](const ldpc::Decoder& d, const In<float>& llr, int iters, py::object alpha, bool posterior) {
            std::vector<float> a;
            if (!alpha.is_none()) {
                // np.ndim(alpha): one value, or one per iteration
                const auto arr = py::array_t<float, py::array::forcecast>::ensure(alpha);
                if (!arr) throw py::type_error("alpha: expected a number or a sequence");
                a.assign(arr.data(), arr.data() + arr.size());
                if (arr.ndim() > 0 && static_cast<int>(a.size()) < iters) throw py::index_error("alpha: one per iteration");
            }
            const auto in = mat(llr);
            ldpc::Decoded r;
            {
                py::gil_scoped_release release;
                r = d.decode(in, iters, a, posterior);
            }
            if (posterior) return py::tuple(py::make_tuple(np(r.bits), flags(r.ok), np(r.posterior)));
            return py::tuple(py::make_tuple(np(r.bits), flags(r.ok)));
        }, py::arg("llr_sent"), py::arg("iters") = 30, py::arg("alpha") = py::none(), py::arg("posterior") = false);
}

}  // namespace data2g::bind
