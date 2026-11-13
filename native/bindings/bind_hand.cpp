// data2g.hand over native/core/hand, for tests/test_native_hand.py's parity checks.
#include "convert.hpp"
#include "hand/hand.hpp"

namespace data2g::bind {

void bind_hand(py::module_& m) {
    auto h = m.def_submodule("hand");
    h.def("channel_freq", &hand::channel_freq);
    h.def("channels_within", &hand::channels_within);
    h.def("crc_init", &hand::crc_init);
    h.def("modulate", [](int tag, std::uint64_t payload, int c) {
        return np(hand::modulate(tag, hand::payload_bits(payload), c));
    });
    h.def("tx_audio", [](int tag, std::uint64_t payload, int channel, double cfo) {
        return np(hand::tx_audio(tag, payload, channel, cfo));
    });
    h.def("decode_window", [](const In<double>& y, int tag, const std::vector<int>& channels, std::int64_t nominal,
                              const std::string& round) {
        const hand::Round grids = round == "cheap" ? hand::ROUND_CHEAP : hand::ROUND_FULL;
        py::list out;
        for (const auto& r : hand::decode_window(vec(y), tag, channels, nominal, grids))
            out.append(py::dict(py::arg("c") = r.c, py::arg("payload") = r.payload, py::arg("snr") = r.snr, py::arg("dt_ms") = r.dt_ms));
        return out;
    }, py::arg("y"), py::arg("tag"), py::arg("channels"), py::arg("nominal"), py::arg("round") = "full");
}

}  // namespace data2g::bind
