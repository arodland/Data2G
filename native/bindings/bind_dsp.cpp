// native/core/dsp (numpy / scipy primitives, no Python module of its own)
// as data2g_native.dsp, for tests/test_native_waveform.py to check against
// numpy and scipy themselves.
#include "convert.hpp"
#include "dsp/dsp.hpp"

namespace data2g::bind {

using cdouble = std::complex<double>;

void bind_dsp(py::module_& m) {
    auto d = m.def_submodule("dsp");
    d.def("pairwise_sum", [](const In<double>& a) { return dsp::pairwise_sum(view(a)); });
    d.def("pairwise_sum_complex", [](const In<cdouble>& a) { return dsp::pairwise_sum(view(a)); });
    d.def("quantile", [](const In<double>& a, double q) { return dsp::quantile(vec(a), q); });
    d.def("firwin_bandpass", [](int n, double lo, double hi, double fs) { return np(dsp::firwin_bandpass(n, lo, hi, fs)); });
    d.def("hilbert", [](const In<double>& x) { return np(dsp::hilbert(view(x))); });
    d.def("convolve_same", [](const In<double>& a, const In<double>& v) { return np(dsp::convolve_same(view(a), view(v))); });
    d.def("next_fast_len", &dsp::next_fast_len);
    d.def("fftconvolve_valid", [](const In<cdouble>& a, const In<cdouble>& v) {
        return np(dsp::fftconvolve_valid(view(a), view(v)));
    });
}

}  // namespace data2g::bind
