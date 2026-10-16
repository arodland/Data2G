// data2g_native: the C++ core as Python functions, for `pytest --native`'s
// substitutions (tests/native_subs/) and the tests/test_native_*.py parity
// tests. Never installed: tests/native_subs/__init__.py puts build/python on
// sys.path.
// One bind_<module>.cpp per Python module it stands in for (native/README.md
// maps them); register it here.

#include "convert.hpp"
#include "util/pool.hpp"

namespace data2g::bind {
void bind_codes(py::module_&);
void bind_config(py::module_&);
void bind_constellation(py::module_&);
void bind_cpm(py::module_&);
void bind_polar(py::module_&);
void bind_equalizer(py::module_&);
void bind_dsp(py::module_&);
void bind_waveform(py::module_&);
void bind_ldpc(py::module_&);
void bind_timing(py::module_&);
void bind_arq(py::module_&);
void bind_arq_link(py::module_&);
void bind_audio(py::module_&);
void bind_modem(py::module_&);
void bind_tnc(py::module_&);
void bind_receiver(py::module_&);
void bind_arq_phy(py::module_&);
void bind_kisslink(py::module_&);
void bind_engine(py::module_&);
void bind_host(py::module_&);
}

PYBIND11_MODULE(data2g_native, m) {
    // Bumped when the module's Python-facing signatures change, so
    // tests/native_subs (its ABI) refuses a stale build instead of failing
    // confusingly.
    m.attr("__abi__") = 3;
    data2g::bind::bind_codes(m);
    data2g::bind::bind_config(m);
    data2g::bind::bind_constellation(m);
    data2g::bind::bind_cpm(m);
    data2g::bind::bind_polar(m);
    data2g::bind::bind_equalizer(m);
    data2g::bind::bind_dsp(m);
    data2g::bind::bind_waveform(m);
    data2g::bind::bind_ldpc(m);
    data2g::bind::bind_timing(m);
    data2g::bind::bind_arq(m);
    data2g::bind::bind_arq_link(m);
    data2g::bind::bind_audio(m);
    data2g::bind::bind_modem(m);
    data2g::bind::bind_tnc(m);
    data2g::bind::bind_receiver(m);
    data2g::bind::bind_arq_phy(m);
    data2g::bind::bind_kisslink(m);
    data2g::bind::bind_engine(m);
    data2g::bind::bind_host(m);
    m.def("set_threads", &data2g::pool::set_threads, pybind11::arg("n"), "Shared pool size, caller included; < 1: default.");
    m.def("threads", &data2g::pool::threads);
}
