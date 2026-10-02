#include "convert.hpp"
#include "modem/timing.hpp"

namespace data2g::bind {

void bind_timing(py::module_& m) {
    auto t = m.def_submodule("timing", "data2g.modem's burst timing");
    t.def("hosts", [](const std::string& band) { return modem::hosts(band); });
    t.def("header_layout", [](const std::string& band) {
        const auto v = modem::header_layout(band);
        py::array_t<bool> a(static_cast<py::ssize_t>(v.size()));
        std::copy(v.begin(), v.end(), a.mutable_data());
        return a;
    });
    t.def("header_samples", [](const std::string& band) { return modem::header_samples(band); });
    t.def("copy_frame", [](const std::string& band, int n_f) { return modem::copy_frame(band, n_f); });
    t.def("frames_on_air", [](const std::string& name, int n_cw) { return modem::frames_on_air(spec(name), n_cw); });
    t.def("burst_end", [](long p0, const std::string& name, int n_cw) { return modem::burst_end(p0, spec(name), n_cw); });
    t.def("head_samples", [](const std::string& band) { return modem::head_samples(band); });
    t.def("burst_seconds", [](const std::string& name, int n_cw) { return modem::burst_seconds(spec(name), n_cw); });
}

}  // namespace data2g::bind
