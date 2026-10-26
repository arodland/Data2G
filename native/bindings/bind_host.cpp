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
    Queued queued(bool vara) override {
        py::object s = engine_obj.attr("session");
        if (!py::dict(py::getattr(s, "__dict__")).contains("_station_override")) return Host::queued(vara);
        Queued q;
        q.unsent = q.unacked = static_cast<std::int64_t>(py::len(s.attr("_pending_write")));
        py::object st = s.attr("station");
        if (!st.is_none()) {  // session.py's queued_bytes over the test's station
            q.station = true;
            py::object tx = st.attr("tx");
            const auto [u, a] = tx.attr("queued")(vara).cast<std::pair<std::int64_t, std::int64_t>>();
            q.unsent += u;
            q.unacked += a;
        }
        return q;
    }
    std::optional<std::int64_t> next_capacity() override {
        py::object s = engine_obj.attr("session");
        py::object policy = s.attr("policy");
        if (policy.is_none()) return Host::next_capacity();  // a C++ policy
        py::object st = s.attr("station");
        if (st.is_none()) return std::nullopt;
        return policy.attr("next_capacity")(st).cast<std::optional<std::int64_t>>();
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
        .def("command", [](PyHost& s, const std::string& line, const std::string& peer) { s.command(line, peer); },
             py::arg("line"), py::arg("peer") = "127.0.0.1")
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
        .def_readwrite("buffer_credit", &PyHost::buffer_credit)
        .def_readwrite("data2g", &PyHost::data2g)
        .def_readwrite("output_volume", &PyHost::output_volume)
        .def_property_readonly("accept_addr", [](const PyHost& s) -> py::object {
            return s.accept_addr ? py::object(py::make_tuple(s.accept_addr->host, s.accept_addr->port)) : py::none();
        })
        // host.py's dial: (host, port) -> bool
        .def_property("dial", [](const PyHost&) { return py::none(); }, [](PyHost& s, py::object f) {
            if (f.is_none()) s.dial = nullptr;
            else s.dial = [f](const host::Addr& a) { return f(py::make_tuple(a.host, a.port)).cast<bool>(); };
        });
}

}  // namespace data2g::bind
