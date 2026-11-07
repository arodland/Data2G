// data2g.hand over native/core/hand, for tests/test_native_hand.py's parity checks.
#include "convert.hpp"
#include "hand/hand.hpp"

namespace data2g::bind {

void bind_hand(py::module_& m) {
    auto h = m.def_submodule("hand");
    h.def("channel_freq", &hand::channel_freq);
    h.def("modulate", [](int tag, std::uint64_t payload, double freq) {
        return np(hand::modulate(tag, hand::payload_bits(payload), freq));
    });
    h.def("tx_audio", [](int tag, std::uint64_t payload, int channel, double cfo) {
        return np(hand::tx_audio(tag, payload, channel, cfo));
    });
    h.def("decode_window", [](const In<double>& y, int tag, const std::vector<int>& channels, std::int64_t nominal) {
        py::list out;
        for (const auto& r : hand::decode_window(vec(y), tag, channels, nominal))
            out.append(py::dict(py::arg("c") = r.c, py::arg("payload") = r.payload, py::arg("snr") = r.snr, py::arg("dt_ms") = r.dt_ms));
        return out;
    });
}

}  // namespace data2g::bind
