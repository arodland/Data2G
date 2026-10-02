#include "audio/filters.hpp"
#include "convert.hpp"

namespace data2g::bind {

void bind_audio(py::module_& m) {
    auto a = m.def_submodule("audio");
    py::class_<audio::Decimator>(a, "Decimator")
        .def(py::init<int>(), py::arg("rate"))
        .def("__call__", [](audio::Decimator& d, const In<double>& x) { return np(d(vec(x))); })
        .def_property_readonly("d", &audio::Decimator::factor)
        .def_property_readonly("taps", [](const audio::Decimator& d) { return np(d.taps()); });
    py::class_<audio::Interpolator>(a, "Interpolator")
        .def(py::init<int>(), py::arg("rate"))
        .def("__call__", [](audio::Interpolator& f, const In<double>& x) { return np(f(vec(x))); })
        .def_property_readonly("u", &audio::Interpolator::factor)
        .def_property_readonly("taps", [](const audio::Interpolator& f) { return np(f.taps()); });
    py::class_<audio::Blanker>(a, "Blanker")
        .def(py::init<>())
        .def("__call__", [](audio::Blanker& b, const In<double>& x) { return np(b(vec(x))); })
        .def_readwrite("env", &audio::Blanker::env)
        .def_readwrite("n_blanked", &audio::Blanker::n_blanked);
}

}  // namespace data2g::bind
