// data2g.host.Host over the C++ Host, on bind_engine's Engine (sync mode:
// the host touches the session stage). What BUFFER counts is read through
// the session's Python view when a test has replaced its station, and from
// a Python policy's next_capacity when the engine's policy is Python (the
// --native substitution's), as host.py reads them.
#include <pybind11/stl.h>

#include "arq_convert.hpp"
#include "host/host.hpp"

namespace data2g::bind {

namespace {

class PyHost : public host::Host {
public:
    PyHost(py::object engine, std::optional<int> credit) : Host(engine_of(engine), credit), engine_obj(std::move(engine)) {}
    py::object engine_obj;

protected:
    Queued queued() override {
        py::object s = engine_obj.attr("session");
        if (!py::dict(py::getattr(s, "__dict__")).contains("_station_override")) return Host::queued();
        Queued q;
        q.unsent = q.unacked = static_cast<std::int64_t>(py::len(s.attr("_pending_write")));
        py::object st = s.attr("station");
        if (!st.is_none()) {
            q.station = true;
            py::object tx = st.attr("tx");
            const auto n = static_cast<std::int64_t>(py::len(tx.attr("buf")));
            q.unsent += tx.attr("buf_off").cast<std::int64_t>() + n - tx.attr("stream_end").cast<std::int64_t>();
            q.unacked += n;
        }
        return q;
    }
    std::optional<std::int64_t> next_capacity() override {
        py::object s = engine_obj.attr("session");
        py::object policy = s.attr("policy");
        if (policy.is_none()) return Host::next_capacity();  // a C++ policy
        py::object st = s.attr("station");
        if (st.is_none() || !py::hasattr(policy, "next_capacity")) return std::nullopt;
        return policy.attr("next_capacity")(st).cast<std::int64_t>();
    }
};

}  // namespace

void bind_host(py::module_& m) {
    auto h = m.def_submodule("host", "data2g.host");
    h.attr("VERSION") = std::string(host::VERSION);
    h.attr("ALIVE_S") = host::ALIVE_S;
    h.attr("BUFFER_REPEAT_S") = host::BUFFER_REPEAT_S;
    h.def("ignored", [](const std::string& c) { return host::ignored(c); });
    h.def("bw_cap", [](const std::string& c) { return host::bw_cap(c); });
    py::class_<PyHost>(h, "Host")
        .def(py::init([](py::object engine, std::optional<int> credit) {
                 if (engine_of(engine).config().worker) throw py::value_error("Host: a worker-mode engine's session is the worker's");
                 return new PyHost(std::move(engine), credit);
             }),
             py::arg("engine"), py::arg("buffer_credit") = py::none())
        .def("command", [](PyHost& s, const std::string& line) { s.command(line); })
        .def("client_gone", &PyHost::client_gone)
        .def("data_in", [](PyHost& s, const py::object& b) { s.data_in(bytes_of(b)); })
        .def("after_step", &PyHost::after_step)
        .def("take_cmd", [](PyHost& s) {
            auto out = std::move(s.out_cmd);
            s.out_cmd.clear();
            return out;
        })
        .def("take_data", [](PyHost& s) {
            auto b = pyb(s.out_data);
            s.out_data.clear();
            return b;
        })
        .def_readonly("engine", &PyHost::engine_obj)
        .def_readwrite("cap", &PyHost::cap)
        .def_readwrite("listening", &PyHost::listening)
        .def_readwrite("buffer_credit", &PyHost::buffer_credit);
}

}  // namespace data2g::bind
