#include "convert.hpp"
#include <optional>

namespace data2g::bind {

namespace {

const codes::Spec& cspec(const std::string& name) {
    const auto* s = codes::spec(name);
    if (!s) throw py::key_error("no code spec " + name);
    return *s;
}

// A scalar or per-row argument (crc_mask, index, rvs); None: the default.
template <typename T>
std::vector<T> per_row(const py::object& o) {
    if (o.is_none()) return {};
    const auto a = In<std::int64_t>::ensure(o);
    if (!a) throw py::type_error("expected an integer or an integer array");
    std::vector<T> v(static_cast<std::size_t>(a.size()));
    for (std::size_t i = 0; i < v.size(); ++i) v[i] = static_cast<T>(a.data()[i]);
    return v;
}

py::list payload_list(const std::vector<codes::Payload>& ps) {
    py::list out;
    for (const auto& p : ps) out.append(py::make_tuple(to_bytes(p.data), p.ok));
    return out;
}

py::array_t<bool> flags(const std::vector<std::uint8_t>& v) { return np<bool>(std::span<const std::uint8_t>(v)); }

// spread / despread on (rows, n_cw * N) uint8 or float64, dtype kept.
template <bool Spread>
py::array shuffle(const py::array& x, int n_cw, int m) {
    auto go = [&]<typename T>(T) -> py::array {
        const auto a = mat(In<T>(x));
        Mat<T> out(a.rows, a.cols);
        for (std::size_t r = 0; r < a.rows; ++r) {
            const auto v = Spread ? codes::spread(a.row(r), n_cw, m) : codes::despread(a.row(r), n_cw, m);
            std::copy(v.begin(), v.end(), out[r]);
        }
        return np(out);
    };
    if (x.dtype().is(py::dtype::of<std::uint8_t>())) return go(std::uint8_t{});
    if (x.dtype().is(py::dtype::of<double>())) return go(double{});
    throw py::type_error("spread: uint8 or float64 only");
}

void bind_codec(py::module_& c) {
    using codes::Spec;
    auto by_name = [](auto f) { return [f](const std::string& name) { return f(cspec(name)); }; };
    c.def("crc_bits", by_name([](const Spec& s) { return s.crc_bits; }));
    c.def("payload_bytes", by_name([](const Spec& s) { return s.payload_bytes; }));
    c.def("rv_cycle", by_name([](const Spec& s) { return codes::rv_cycle(s); }));
    c.def("buffer_len", by_name([](const Spec& s) { return codes::buffer_len(s); }));
    c.def("rv_positions", [](const std::string& name, int rv) {
        return np<std::int64_t>(std::span<const int>(codes::rv_positions(cspec(name), rv)));
    });
    c.def("info_bits", [](const std::string& name, const py::bytes& payload, std::int64_t mask, int index) {
        return np(codes::info_bits(cspec(name), bytes_view(payload), static_cast<std::uint32_t>(mask), index));
    }, py::arg("spec"), py::arg("payload"), py::arg("crc_mask") = 0, py::arg("index") = 0);
    c.def("encode", [](const std::string& name, const py::bytes& payload, int rv, std::int64_t mask, int index) {
        return np(codes::encode(cspec(name), bytes_view(payload), rv, static_cast<std::uint32_t>(mask), index));
    }, py::arg("spec"), py::arg("payload"), py::arg("rv") = 0, py::arg("crc_mask") = 0, py::arg("index") = 0);
    c.def("encode_info", [](const std::string& name, const In<std::uint8_t>& bits, int rv) {
        return np(codes::encode_info(cspec(name), mat(bits), rv));
    }, py::arg("spec"), py::arg("bits"), py::arg("rv") = 0);
    c.def("flip", [](const std::string& name, int index, int rv) { return np(codes::flip(cspec(name), index, rv)); },
          py::arg("spec"), py::arg("index"), py::arg("rv") = 0);
    c.def("spread", &shuffle<true>);
    c.def("despread", &shuffle<false>);

    // codes.combine; a float64 C-contiguous writable buf is added to in place, as np.add.at does.
    c.def("combine", [](const std::string& name, py::object buf, const In<double>& soft, py::object rvs) {
        const auto& s = cspec(name);
        Mat<double> b;
        std::optional<py::array_t<double>> target;  // a default array_t is a real (empty) array
        if (!buf.is_none()) {
            auto arr = py::cast<py::array>(buf);
            if (arr.dtype().is(py::dtype::of<double>()) && (arr.flags() & py::array::c_style) && arr.writeable() &&
                arr.ndim() == 2)
                target = py::array_t<double>::ensure(arr);
            b = mat(In<double>(arr));
        }
        codes::combine(s, b, mat(soft), per_row<int>(rvs));
        if (!target) return np(b);
        std::copy(b.data.begin(), b.data.end(), target->mutable_data());
        return *target;
    }, py::arg("spec"), py::arg("buf"), py::arg("soft"), py::arg("rvs"));

    c.def("payloads", [](const std::string& name, const In<std::uint8_t>& bits, const In<std::uint8_t>& conv,
                         py::object masks, py::object index) {
        return payload_list(codes::payloads(cspec(name), mat(bits), vec(conv), per_row<std::uint32_t>(masks),
                                            per_row<int>(index)));
    }, py::arg("spec"), py::arg("bits"), py::arg("converged"), py::arg("crc_mask") = py::none(),
       py::arg("index") = py::none());
    c.def("decode_llrs", [](const std::string& name, const In<float>& llr, int iters, py::object masks, py::object index) {
        const auto& s = cspec(name);
        const auto in = mat(llr);
        const auto m = per_row<std::uint32_t>(masks);
        const auto i = per_row<int>(index);
        codes::Info r;
        {
            py::gil_scoped_release unlocked;
            r = codes::decode_llrs(s, in, iters, m, i);
        }
        return py::make_tuple(np(r.bits), flags(r.ok));
    }, py::arg("spec"), py::arg("llr"), py::arg("iters") = codes::ITERS, py::arg("crc_mask") = py::none(),
       py::arg("index") = py::none());
    c.def("decode_many", [](const std::string& name, const In<float>& soft, py::object masks, py::object index) {
        const auto& s = cspec(name);
        const auto in = mat(soft);
        const auto m = per_row<std::uint32_t>(masks);
        const auto i = per_row<int>(index);
        std::vector<codes::Payload> r;
        {
            py::gil_scoped_release unlocked;
            r = codes::decode_many(s, in, m, i);
        }
        return payload_list(r);
    }, py::arg("spec"), py::arg("soft"), py::arg("crc_mask") = py::none(), py::arg("index") = py::none());
    c.def("decode_buffer", [](const std::string& name, const In<double>& buf, int max_rv, py::object masks,
                              py::object index) {
        const auto& s = cspec(name);
        const auto in = mat(buf);
        const auto m = per_row<std::uint32_t>(masks);
        const auto i = per_row<int>(index);
        std::vector<codes::Payload> r;
        {
            py::gil_scoped_release unlocked;
            r = codes::decode_buffer(s, in, max_rv, m, i);
        }
        return payload_list(r);
    }, py::arg("spec"), py::arg("buf"), py::arg("max_rv") = 0, py::arg("crc_mask") = py::none(),
       py::arg("index") = py::none());
    c.def("decode_raw", [](const std::string& name, const In<float>& soft, py::object index) {
        const auto& s = cspec(name);
        const auto in = mat(soft);
        const auto i = per_row<int>(index);
        codes::Raw r;
        {
            py::gil_scoped_release unlocked;
            r = codes::decode_raw(s, in, i);
        }
        const py::ssize_t B = static_cast<py::ssize_t>(r.cands.rows), L = r.list, k = s.k;
        py::array_t<std::uint8_t> cands({B, L, k});
        std::copy(r.cands.data.begin(), r.cands.data.end(), cands.mutable_data());
        py::array_t<bool> usable({B, L});
        std::copy(r.usable.data.begin(), r.usable.data.end(), usable.mutable_data());
        return py::make_tuple(cands, usable);
    }, py::arg("spec"), py::arg("soft"), py::arg("index") = py::none());
    c.def("check", [](const std::string& name, const In<std::uint8_t>& cands, const In<std::uint8_t>& usable,
                      std::int64_t mask) -> py::object {
        const auto r = codes::check(cspec(name), vec(cands), vec(usable), static_cast<std::uint32_t>(mask));
        return r ? py::object(to_bytes(*r)) : py::object(py::none());
    });
    // (..., n) bits -> descrambled, same shape.
    c.def("descramble", [](const std::string& name, const In<std::uint8_t>& bits, int index) {
        const auto& s = cspec(name);
        py::array_t<std::uint8_t> out(std::vector<py::ssize_t>(bits.shape(), bits.shape() + bits.ndim()));
        const std::size_t n = bits.ndim() ? static_cast<std::size_t>(bits.shape(bits.ndim() - 1)) : 1;
        for (std::size_t r = 0; n && r < static_cast<std::size_t>(bits.size()) / n; ++r) {
            const auto v = codes::descramble(s, std::span(bits.data() + r * n, n), index);
            std::copy(v.begin(), v.end(), out.mutable_data() + r * n);
        }
        return out;
    });
    c.def("crc_ok", [](const std::string& name, const In<std::uint8_t>& bits, py::object masks, py::object index) {
        return flags(codes::crc_ok(cspec(name), mat(bits), per_row<std::uint32_t>(masks), per_row<int>(index)));
    }, py::arg("spec"), py::arg("bits"), py::arg("crc_mask") = py::none(), py::arg("index") = py::none());
}

}  // namespace

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
    // Submodes, CPM modes and CPM control codewords, by name.
    c.def("interleaver", [](const std::string& name) { return np<std::int64_t>(cspec(name).perm); });
    c.def("info_pos", [](const std::string& name) { return np<std::int64_t>(cspec(name).info_pos); });
    bind_codec(c);

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
