// data2g.cwid over native/core/cwid, for tests/test_native_cwid.py.
#include "convert.hpp"
#include "cwid/cwid.hpp"

namespace data2g::bind {

void bind_cwid(py::module_& m) {
    auto c = m.def_submodule("cwid", "data2g.cwid");
    c.def("id_text", &cwid::id_text);
    c.def("keying", &cwid::keying);
    c.def("audio", [](const std::string& text, double wpm, double freq) { return np(cwid::audio(text, wpm, freq)); },
          py::arg("text"), py::arg("wpm") = cwid::WPM, py::arg("freq") = cwid::FREQ_HZ);
    c.def("seconds", &cwid::seconds, py::arg("text"), py::arg("wpm") = cwid::WPM);
    py::class_<cwid::CwSense>(c, "CwSense")
        .def(py::init<double>(), py::arg("center") = cwid::FREQ_HZ)
        .def("reset", &cwid::CwSense::reset)
        .def("classify", [](cwid::CwSense& s, const In<double>& x) { return s.classify(view(x)); })
        .def("feed", [](cwid::CwSense& s, const In<double>& x) { return s.feed(view(x)); })
        .def_property_readonly("active", &cwid::CwSense::active)
        .def_property_readonly("hits_from", &cwid::CwSense::hits_from);
}

}  // namespace data2g::bind
