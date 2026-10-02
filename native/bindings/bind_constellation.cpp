#include "constellation/constellation.hpp"
#include "convert.hpp"

namespace data2g::bind {

namespace {

using constellation::cd;

// Constellations cross the boundary by name.
const constellation::Constellation& constel(const std::string& name) {
    const auto* c = constellation::find(name);
    if (!c) throw py::key_error("no constellation " + name);
    return *c;
}

}  // namespace

void bind_constellation(py::module_& m) {
    auto c = m.def_submodule("constellation");
    c.def("names", [] {
        std::vector<std::string> out;
        for (const auto& t : tables::CONSTELLATIONS) out.emplace_back(t.name);
        return out;
    });
    c.def("bits_per_symbol", [](const std::string& name) { return constel(name).m; });
    c.def("points", [](const std::string& name) { return np<cd>(constel(name).points); });
    c.def("ace_dirs", [](const std::string& name) {
        const auto& t = constel(name);
        py::array_t<cd> a({static_cast<py::ssize_t>(t.points.size()), py::ssize_t{2}});
        std::copy(t.ace.begin(), t.ace.end(), a.mutable_data());
        return a;
    });
    c.def("modulate", [](const In<std::uint8_t>& bits, const std::string& name) {
        return np(constellation::modulate(vec(bits), constel(name)));
    });
    c.def("llr", [](const In<cd>& y, const In<cd>& h, const In<double>& var, const std::string& name) {
        return np(constellation::llr(vec(y), vec(h), vec(var), constel(name)));
    });
    c.def("ace_project", [](const In<cd>& got, const In<cd>& want, const In<cd>& dirs) {
        return np(constellation::ace_project(vec(got), vec(want), vec(dirs)));
    });
}

}  // namespace data2g::bind
