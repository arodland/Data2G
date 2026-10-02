// data2g_native: the C++ core as Python functions, for tests/conftest.py's
// `pytest --native` substitutions and tests/test_native_parity.py.
// One bind_<module>.cpp per core module; register it here.

#include "convert.hpp"

namespace data2g::bind {
void bind_codes(py::module_&);
void bind_constellation(py::module_&);
void bind_cpm(py::module_&);
void bind_polar(py::module_&);
void bind_equalizer(py::module_&);
void bind_waveform(py::module_&);
void bind_ldpc(py::module_&);
void bind_timing(py::module_&);
void bind_arq(py::module_&);
void bind_arq_link(py::module_&);
void bind_audio(py::module_&);
void bind_modem(py::module_&);
void bind_tnc(py::module_&);
void bind_arq_phy(py::module_&);
void bind_engine(py::module_&);
}

PYBIND11_MODULE(data2g_native, m) {
    // Bumped when the module's Python-facing signatures change, so conftest
    // refuses a stale build instead of failing confusingly.
    m.attr("__abi__") = 1;
    data2g::bind::bind_codes(m);
    data2g::bind::bind_constellation(m);
    data2g::bind::bind_cpm(m);
    data2g::bind::bind_polar(m);
    data2g::bind::bind_equalizer(m);
    data2g::bind::bind_waveform(m);
    data2g::bind::bind_ldpc(m);
    data2g::bind::bind_timing(m);
    data2g::bind::bind_arq(m);
    data2g::bind::bind_arq_link(m);
    data2g::bind::bind_audio(m);
    data2g::bind::bind_modem(m);
    data2g::bind::bind_tnc(m);
    data2g::bind::bind_arq_phy(m);
    data2g::bind::bind_engine(m);
}
