// data2g.config's submode table as data2g_native.config.submodes(): the
// generated one (native/core/generated/config.hpp), for tests/test_native_parity.py to
// compare with config.SUBMODES.
#include "convert.hpp"

namespace data2g::bind {

void bind_config(py::module_& m) {
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
