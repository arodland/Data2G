// data2g.arq.engine over the C++ Engine. Two ways to make one:
// - pure C++ (policy, rng, tx_audio None): GearPolicy, DefaultRng, C++ TX
//   audio; step() releases the GIL, and `worker` is allowed.
// - Python-compatible (tests/conftest.py's substitution): `rng` a
//   random.Random, drawn from as engine.py draws; sessions are bind_arq_link's
//   Session with its Python view; `policy` a factory of Python policies;
//   `tx_audio` read at each burst (a test's patch of phy.tx_audio). Sync
//   only: Python is called from the session stage.
#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include "arq/engine.hpp"
#include "arq_convert.hpp"

namespace data2g::bind {

namespace {

using namespace data2g::arq;

class PyEngine : public Engine {
public:
    PyEngine(std::string call, EngineConfig cfg, EngineHooks hooks, py::object tx_audio, py::object kiss, bool compat)
        : Engine(std::move(call), std::move(cfg), std::move(hooks)),
          tx_audio_fn(std::move(tx_audio)), kiss_obj(std::move(kiss)), compat(compat) {}
    ~PyEngine() override {
        py::gil_scoped_release nogil;  // the worker may be waiting for it (logging)
        stop();
    }

    py::object tx_audio_fn, kiss_obj, receiver = py::none();
    py::object session_obj;  // one Python view per session, so attributes set on it last
    const Session* session_of = nullptr;
    bool compat;

protected:
    std::vector<tnc::Receiver::Item> receiver_feed(std::span<const double> x) override {
        if (receiver.is_none()) return Engine::receiver_feed(x);
        if (py::len(receiver.attr("feed")(np<double>(x))))
            throw std::logic_error("a Python receiver's events can't cross into the C++ engine");
        return {};
    }
    bool receiver_busy() override { return receiver.is_none() ? Engine::receiver_busy() : receiver.attr("busy").cast<bool>(); }
    bool receiver_channel_busy() override {
        return receiver.is_none() ? Engine::receiver_channel_busy() : receiver.attr("channel_busy").cast<bool>();
    }
    void receiver_reset() override {
        if (receiver.is_none()) return Engine::receiver_reset();
        receiver.attr("reset")();
    }
    std::vector<double> tx_audio(const TxBurst& b) override {
        if (tx_audio_fn.is_none()) return Engine::tx_audio(b);
        return vec(tx_audio_fn(burst_py(std::make_shared<TxBurst>(b))).cast<In<double>>());
    }
};

// step() without the GIL unless Python is called from inside it.
template <typename F>
auto unlocked(PyEngine& e, F f) {
    if (e.compat || !e.receiver.is_none()) return f();
    py::gil_scoped_release nogil;
    return f();
}

}  // namespace

arq::Engine& engine_of(const py::handle& h) { return h.cast<PyEngine&>(); }

void bind_engine(py::module_& m) {
    auto e = m.def_submodule("engine", "data2g.arq.engine");
    e.attr("MAX_BURST_S") = MAX_BURST_S;
    e.def("to_f16", [](const In<double>& x) {
        py::array_t<std::uint16_t> out(x.size());
        for (py::ssize_t i = 0; i < x.size(); ++i) out.mutable_data()[i] = to_half(x.data()[i]);
        return out;
    });
    e.def("json_num", &json_num);

    py::class_<PyEngine>(e, "Engine")
        .def(py::init([](std::string call, py::object policy, double ptt_delay_s, py::object record_dir,
                         std::optional<std::uint64_t> seed, double min_header_score, py::object kiss,
                         double stats_interval_s, py::object rng, py::object tx_audio, std::optional<bool> dd, bool worker) {
                 EngineConfig cfg;
                 cfg.ptt_delay_s = ptt_delay_s;
                 if (!record_dir.is_none()) cfg.record_dir = py::str(record_dir).cast<std::string>();
                 cfg.seed = seed;
                 cfg.min_header_score = min_header_score;
                 if (!kiss.is_none()) cfg.kiss = kiss.cast<kisslink::KissLink*>();
                 cfg.stats_interval_s = stats_interval_s;
                 if (dd) cfg.dd = *dd;
                 cfg.worker = worker;
                 const bool compat = !policy.is_none() || !rng.is_none() || !tx_audio.is_none();
                 if (compat && worker) throw py::value_error("worker: Python policies, rng and tx_audio need sync mode");
                 EngineHooks hooks;
                 if (!policy.is_none()) hooks.policy = [policy] { return py_policy(policy()); };
                 if (!rng.is_none()) {
                     hooks.rng = py_rng(rng);
                     hooks.session_rng = [](double s) { return py_rng(py::module_::import("random").attr("Random")(s)); };
                     hooks.session = py_session;
                 }
                 return new PyEngine(std::move(call), std::move(cfg), std::move(hooks), tx_audio, kiss, compat);
             }),
             py::arg("call"), py::arg("policy") = py::none(), py::arg("ptt_delay_s") = 0.1, py::arg("record_dir") = py::none(),
             py::arg("seed") = py::none(), py::arg("min_header_score") = 0.0, py::arg("kiss") = py::none(),
             py::arg("stats_interval_s") = 60.0, py::kw_only(), py::arg("rng") = py::none(), py::arg("tx_audio") = py::none(),
             py::arg("dd") = py::none(), py::arg("worker") = false)
        .def("step", [](PyEngine& self, const In<double>& x) {
            const std::vector<double> v = vec(x);
            auto o = unlocked(self, [&] { return self.step(v); });
            return py::make_tuple(np(o.audio), o.ptt);
        })
        .def("next_event", &PyEngine::next_event)
        .def("listen", &PyEngine::listen, py::arg("on") = true)
        .def("set_call", [](PyEngine& self, const std::string& call, py::args aliases) {
            self.set_call(call, aliases.cast<std::vector<std::string>>());
        })
        .def("abort", &PyEngine::abort)
        .def("connect", &PyEngine::connect)
        .def("set_chat", &PyEngine::set_chat)
        .def("events", &PyEngine::events)
        .def("send_cq", &PyEngine::send_cq)
        .def_property_readonly("now", &PyEngine::now)
        .def_property("n", &PyEngine::n, &PyEngine::set_n)
        .def_property_readonly("call", &PyEngine::call)
        .def_property_readonly("aliases", [](const PyEngine& s) { return py::tuple(py::cast(s.aliases())); })
        .def_property_readonly("session", [](PyEngine& self) {
            if (self.session_of != &self.session()) {
                self.session_obj = py::cast(self.session_ptr());
                self.session_of = &self.session();
            }
            return self.session_obj;
        })
        .def_property_readonly("tx", [](const PyEngine& self) -> py::object {
            const auto& t = self.tx();
            if (!t) return py::none();
            return py::make_tuple(burst_py(t->burst), t->pos);  // engine.py's [burst, audio, position], less the audio
        })
        .def_property_readonly("_extra", [](const PyEngine& self) {
            py::list out;
            for (const auto& b : self.extra()) out.append(burst_py(b));
            return out;
        })
        .def("take_kiss_rx", [](PyEngine& self) {
            py::list out;
            for (const auto& f : self.kiss_rx()) out.append(pyb(f));
            self.kiss_rx().clear();
            return out;
        })
        .def_property_readonly("kiss", [](const PyEngine& self) { return self.kiss_obj; })
        .def_property_readonly("busy", &PyEngine::busy)
        .def_property_readonly("channel_busy", &PyEngine::channel_busy)
        .def_property("receiver", [](const PyEngine& self) { return self.receiver; },
                      [](PyEngine& self, py::object r) {
                          if (self.config().worker) throw py::value_error("worker: no Python receiver");
                          self.receiver = std::move(r);
                      });
}

}  // namespace data2g::bind
