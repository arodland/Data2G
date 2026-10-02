// data2g_native: the C++ core as Python functions, for tests/conftest.py's
// `pytest --native` substitutions and tests/test_native_parity.py.
// One bind_<module>.cpp per core module; register it here.

#include "convert.hpp"

namespace data2g::bind {
void bind_codes(py::module_&);
void bind_constellation(py::module_&);
void bind_cpm(py::module_&);
}

PYBIND11_MODULE(data2g_native, m) {
    // Bumped when the module's Python-facing signatures change, so conftest
    // refuses a stale build instead of failing confusingly.
    m.attr("__abi__") = 1;
    data2g::bind::bind_codes(m);
    data2g::bind::bind_constellation(m);
    data2g::bind::bind_cpm(m);
}
