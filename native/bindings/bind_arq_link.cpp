// data2g.arq frames, link and session over the C++ core. Station and
// Session are wrapped to look like the Python dataclasses to everything
// that reads them (policies, the engine, the host, the tests):
// - the policy, the received burst and the session's random.Random stay
//   Python objects, called through adapters in the same order and with the
//   same arguments as the Python, so seeded runs draw the same numbers;
// - a TxBurst crosses as data2g.arq.link.TxBurst / Slot;
// - logging goes to Python's "data2g.link" / "data2g.session" loggers.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "arq/session.hpp"
#include "arq_convert.hpp"

namespace data2g::bind {

using namespace data2g::arq;

namespace {

// --- adapters to the Python objects ---------------------------------------------------

class PyPolicy : public Policy {
public:
    explicit PyPolicy(py::object o) : obj(std::move(o)) {}
    py::object obj;

    std::pair<std::string, int> choose(Station& st, int escalation) override {
        auto r = obj.attr("choose")(py::cast(&st, py::return_value_policy::reference), escalation).cast<py::sequence>();
        return {r[0].cast<std::string>(), r[1].cast<int>()};
    }
    int payload_bytes(const std::string& m) override { return obj.attr("payload_bytes")(m).cast<int>(); }
    int ctl_payload_bytes(const std::string& m) override {
        return py::hasattr(obj, "ctl_payload_bytes") ? obj.attr("ctl_payload_bytes")(m).cast<int>() : payload_bytes(m);
    }
    int max_ctl(const std::string& m) override { return py::hasattr(obj, "max_ctl") ? obj.attr("max_ctl")(m).cast<int>() : 4; }
    std::optional<Recommendation> recommend(Station& st) override {
        if (!py::hasattr(obj, "recommend")) return std::nullopt;
        auto r = obj.attr("recommend")(py::cast(&st, py::return_value_policy::reference)).cast<py::sequence>();
        return Recommendation{r[0].cast<int>(), r[1].cast<int>(), r[2].is_none() ? std::nullopt : std::optional<int>(r[2].cast<int>())};
    }
    bool want_dup() override { return py::bool_(py::getattr(obj, "want_dup", py::bool_(false))); }
    int rv_cycle(const std::string& m) override { return obj.attr("rv_cycle")(m).cast<int>(); }
    bool has_outcome() override { return py::hasattr(obj, "outcome"); }
    void outcome(const std::string& m, int decoded, int sent, bool usable) override {
        obj.attr("outcome")(m, decoded, sent, py::arg("usable") = usable);
    }
    std::string mode_name(int rec) override {
        return py::hasattr(obj, "mode_name") ? py::str(obj.attr("mode_name")(rec)).cast<std::string>() : std::to_string(rec);
    }
    bool has_airtime() override { return py::hasattr(obj, "airtime"); }
    std::optional<double> snr_est() override {
        auto m = py::getattr(obj, "measured", py::none());
        if (!py::bool_(m) || !m.contains("snr_est")) return std::nullopt;
        return m["snr_est"].cast<double>();
    }
    // dup only when set: session.py's on_header passes two arguments
    double airtime(const std::string& m, int n_cw, bool dup) override {
        return (dup ? obj.attr("airtime")(m, n_cw, true) : obj.attr("airtime")(m, n_cw)).cast<double>();
    }
    std::string connect_mode(int cap, int tries) override { return obj.attr("connect_mode")(cap, tries).cast<std::string>(); }
    std::optional<double> reply_hold(Station& st, const TxBurst& burst) override {
        if (!py::hasattr(obj, "reply_hold")) return std::nullopt;
        return obj.attr("reply_hold")(py::cast(&st, py::return_value_policy::reference),
                                      burst_py(std::make_shared<TxBurst>(burst))).cast<double>();
    }
    void observe(const Measured& m, const std::string& submode, double now) override {
        obj.attr("observe")(to_dict(m), submode, now);
    }
};

std::shared_ptr<Policy> policy_of(py::object o) { return std::make_shared<PyPolicy>(std::move(o)); }
py::object policy_py(const std::shared_ptr<Policy>& p) {
    auto* a = dynamic_cast<PyPolicy*>(p.get());
    return a ? a->obj : py::none();
}

class PyRx : public RxBurst {
public:
    explicit PyRx(py::object o) : obj(std::move(o)), sub(obj.attr("submode").cast<std::string>()), n(obj.attr("n_cw").cast<int>()) {}
    py::object obj;
    std::string sub;
    int n;

    const std::string& submode() const override { return sub; }
    int n_cw() const override { return n; }
    py::object key(const SoftKey* k) const {
        if (!k) return py::none();
        if (k->ctl) return py::make_tuple("ctl", py::module_::import("builtins").attr("id")(obj), k->index);
        return py::make_tuple(k->peer, k->seq);
    }
    std::optional<Bytes> decode(int slot, const MaskId& mask, int rv, const SoftKey* k) override {
        auto r = obj.attr("decode")(slot, mask_tuple(mask), rv, key(k));
        if (r.is_none()) return std::nullopt;
        return bytes_of(r);
    }
    void forget(const SoftKey& k) override { obj.attr("forget")(key(&k)); }
};

class PyRng : public Rng {
public:
    explicit PyRng(py::object o) : obj(std::move(o)) {}
    py::object obj;
    std::int64_t randrange(std::int64_t n) override { return obj.attr("randrange")(n).cast<std::int64_t>(); }
    double uniform(double a, double b) override { return obj.attr("uniform")(a, b).cast<double>(); }
};

// link.COMPRESS, which tests monkeypatch, read at each build as Python does.
void sync_compress(Station& s) { s.compress = py::bool_(link_attr("COMPRESS")); }

// A Station whose internal builds (from on_timeout, or the session) see an
// instance override of the Python method, as Python's self.build would.
class PyStation : public Station {
public:
    using Station::Station;
    TxBurstPtr build(bool fresh) override {
        py::handle self = py::detail::get_object_handle(static_cast<Station*>(this), py::detail::get_type_info(typeid(Station)));
        if (self) {
            py::dict d = py::getattr(self, "__dict__");
            if (d.contains("build")) {
                py::object f = d["build"];
                return burst_cpp(fresh ? f() : f(py::arg("fresh") = false));
            }
        }
        sync_compress(*this);
        return Station::build(fresh);
    }
};

// session.py's tunables as Python has them now (a study patches the module).
SessionTuning py_tuning() {
    auto S = py::module_::import("data2g.arq.session");
    auto d = [&](const char* n) { return S.attr(n).cast<double>(); };
    auto i = [&](const char* n) { return S.attr(n).cast<int>(); };
    auto pair = [&](const char* n) { return S.attr(n).cast<std::pair<double, double>>(); };
    SessionTuning t;
    t.connect_tries = i("CONNECT_TRIES");
    t.disc_tries = i("DISC_TRIES");
    t.reply_start_s = d("REPLY_START_S");
    t.idle_close_s = d("IDLE_CLOSE_S");
    std::tie(t.keepalive_lo_s, t.keepalive_hi_s) = pair("KEEPALIVE_S");
    std::tie(t.chat_keepalive_lo_s, t.chat_keepalive_hi_s) = pair("CHAT_KEEPALIVE_S");
    t.keepalive_doubling = py::bool_(S.attr("KEEPALIVE_DOUBLING"));
    t.link_lost_s = d("LINK_LOST_S");
    t.wake_guard_s = d("WAKE_GUARD_S");
    t.wake_jitter_s = d("WAKE_JITTER_S");
    t.wake_tries = i("WAKE_TRIES");
    t.chat_wake_tries = i("CHAT_WAKE_TRIES");
    t.repeat_max_s = d("REPEAT_MAX_S");
    return t;
}

// A Session that takes session.py's tunables when made, and whose timeouts
// go through an instance override of _on_timeout, as Python's self._on_timeout
// would (scripts/linksim.py wraps it to count timeouts).
class PySession : public Session {
public:
    template <typename... A>
    explicit PySession(A&&... a) : Session(std::forward<A>(a)...) {
        tune = py_tuning();
    }

    void on_timeout(double now) override {
        {
            py::gil_scoped_acquire gil;
            py::handle self = py::detail::get_object_handle(static_cast<Session*>(this), py::detail::get_type_info(typeid(Session)));
            if (self) {
                py::dict d = py::getattr(self, "__dict__");
                if (d.contains("_on_timeout")) {
                    d["_on_timeout"](now);
                    return;
                }
            }
        }
        Session::on_timeout(now);
    }

    // One Python object per C++ burst, so a repeat (station.last_sent sent
    // again) is the same object, as in Python: scripts/linksim.py and
    // test_arq_session's channel key on burst identity. Holding the pointer
    // keeps its address from being reused while cached.
    py::object burst_obj(const TxBurstPtr& b) {
        if (!b) return py::none();
        for (const auto& [p, o] : bursts)
            if (p == b) return o;
        py::object o = burst_py(b);
        bursts.emplace_back(b, o);
        if (bursts.size() > 8) bursts.erase(bursts.begin());  // repeats are of the latest few
        return o;
    }
    std::vector<std::pair<TxBurstPtr, py::object>> bursts;

protected:
    std::shared_ptr<Station> make_station(int direction, bool master_, int key) override {
        return std::make_shared<PyStation>(direction, policy, master_, key, std::nullopt, cap, chat);
    }
};

template <typename K, typename V, typename F>
py::dict dict_of(const std::map<K, V>& m, F f) {
    py::dict d;
    for (const auto& [k, v] : m) d[py::cast(k)] = f(v);
    return d;
}

py::set set_of(const std::set<std::int64_t>& s) {
    py::set out;
    for (auto x : s) out.add(py::int_(x));
    return out;
}

py::object counter(const std::map<std::string, std::int64_t>& m) {
    py::dict d;
    for (const auto& [k, v] : m) d[py::str(k)] = v;
    return py::module_::import("collections").attr("Counter")(d);
}

}  // namespace

std::shared_ptr<Policy> py_policy(py::object o) { return policy_of(std::move(o)); }
std::shared_ptr<Rng> py_rng(py::object o) { return std::make_shared<PyRng>(std::move(o)); }
std::shared_ptr<Session> py_session(const std::string& call, std::shared_ptr<Policy> policy, std::shared_ptr<Rng> rng,
                                    const std::vector<std::string>& aliases, double stats_interval_s) {
    return std::make_shared<PySession>(call, std::move(policy), 1.0, std::move(rng), aliases, stats_interval_s);
}

void bind_arq_link(py::module_& m) {
    // the GIL taken: an engine's worker logs too
    set_log_sink({[](const char* name, int level) {
                      py::gil_scoped_acquire gil;
                      return py::module_::import("logging").attr("getLogger")(name).attr("isEnabledFor")(level).cast<bool>();
                  },
                  [](const char* name, int level, const std::string& msg) {
                      py::gil_scoped_acquire gil;
                      py::module_::import("logging").attr("getLogger")(name).attr("log")(level, "%s", msg);
                  }});

    // bind_arq (the gear shifter) made the submodule and registers first
    auto a = py::reinterpret_borrow<py::module_>(m.attr("arq"));

    // --- frames ---
    py::class_<Core, std::shared_ptr<Core>>(a, "Core")
        .def(py::init([](int ftype, int n_ctl, int burst_seq, int acted_on, int cum, bool reply_lost, int k, int recommend,
                         int size_hint) {
                 return std::make_shared<Core>(Core{ftype, n_ctl, burst_seq, acted_on, cum, reply_lost, k, recommend, size_hint});
             }),
             py::arg("ftype") = ARQ, py::arg("n_ctl") = 1, py::arg("burst_seq") = 0, py::arg("acted_on") = 0, py::arg("cum") = 0,
             py::arg("reply_lost") = false, py::arg("k") = 0, py::arg("recommend") = 0, py::arg("size_hint") = 1)
        .def_readwrite("ftype", &Core::ftype)
        .def_readwrite("n_ctl", &Core::n_ctl)
        .def_readwrite("burst_seq", &Core::burst_seq)
        .def_readwrite("acted_on", &Core::acted_on)
        .def_readwrite("cum", &Core::cum)
        .def_readwrite("reply_lost", &Core::reply_lost)
        .def_readwrite("k", &Core::k)
        .def_readwrite("recommend", &Core::recommend)
        .def_readwrite("size_hint", &Core::size_hint)
        .def("pack", [](const Core& c) { return pyb(c.pack()); })
        .def_static("unpack", [](const py::bytes& b) { return std::make_shared<Core>(Core::unpack(bytes_of(b))); })
        .def("__eq__", [](const Core& x, const py::object& y) { return py::isinstance<Core>(y) && x == y.cast<const Core&>(); })
        .def("__repr__", [](const Core& c) {
            return format("Core(ftype=%d, n_ctl=%d, burst_seq=%d, acted_on=%d, cum=%d, reply_lost=%s, k=%d, recommend=%d, size_hint=%d)",
                          c.ftype, c.n_ctl, c.burst_seq, c.acted_on, c.cum, c.reply_lost ? "True" : "False", c.k, c.recommend,
                          c.size_hint);
        });

    // Control holds its Core by reference and its ext as a dict, as the dataclass does.
    struct PyControl {
        std::shared_ptr<Core> core;
        py::dict ext;
    };
    auto to_ext = [](const py::dict& d) {
        Ext e;
        for (auto [k, v] : d) e[k.cast<int>()] = bytes_of(v);
        return e;
    };
    py::class_<PyControl>(a, "Control")
        .def(py::init([](std::shared_ptr<Core> core, std::optional<py::dict> ext) {
                 return PyControl{std::move(core), ext ? *ext : py::dict()};
             }),
             py::arg("core"), py::arg("ext") = py::none())
        .def_readwrite("core", &PyControl::core)
        .def_readwrite("ext", &PyControl::ext)
        .def("pack", [to_ext](PyControl& c, int pb) {
            Control ctl{*c.core, to_ext(c.ext)};
            auto out = ctl.pack(pb);
            c.core->n_ctl = ctl.core.n_ctl;
            py::list l;
            for (const auto& p : out) l.append(pyb(p));
            return l;
        })
        .def_static("unpack", [](const py::iterable& payloads) {
            std::vector<Bytes> ps;
            for (auto p : payloads) ps.push_back(bytes_of(p));
            const Control c = Control::unpack(ps);
            return PyControl{std::make_shared<Core>(c.core), dict_of(c.ext, pyb)};
        });

    a.def("pack_bitmap", [](const py::iterable& received, int cum) {
        std::set<int> r;
        for (auto x : received) r.insert(x.cast<int>());
        return pyb(pack_bitmap(r, cum));
    });
    a.def("unpack_bitmap", [](const py::bytes& b, int cum) {
        py::set out;
        for (int x : unpack_bitmap(bytes_of(b), cum)) out.add(py::int_(x));
        return out;
    });
    a.def("pack_rv", [](const std::vector<int>& rvs) { return pyb(pack_rv(rvs)); });
    a.def("unpack_rv", [](const py::bytes& b, int k) { return unpack_rv(bytes_of(b), k); });
    a.def("pack_flags", [](const py::iterable& flags) {
        std::vector<bool> f;
        for (auto x : flags) f.push_back(py::bool_(py::reinterpret_borrow<py::object>(x)));
        return pyb(pack_flags(f));
    });
    a.def("unpack_flags", [](const py::bytes& b, int n) {
        py::list out;
        for (bool x : unpack_flags(bytes_of(b), n)) out.append(py::bool_(x));
        return out;
    });
    a.def("deflate", [](const py::bytes& hist, const py::bytes& data) { return pyb(deflate(bytes_of(hist), bytes_of(data))); });
    a.def("deflate_fit", [](const py::bytes& hist, const py::bytes& data, int pb) -> py::object {
        auto r = deflate_fit(bytes_of(hist), bytes_of(data), pb);
        if (!r) return py::none();
        return py::make_tuple(r->first, pyb(r->second));
    });
    a.def("inflate", [](const py::bytes& hist, const py::bytes& payload) { return pyb(inflate(bytes_of(hist), bytes_of(payload))); });
    a.def("pack_call", [](const std::string& c) { return pyb(pack_call(c)); });
    a.def("unpack_call", [](const py::bytes& b) { return unpack_call(bytes_of(b)); });
    a.def("pack_connect", [](const py::bytes& b) { return pyb(pack_connect(bytes_of(b))); });
    a.def("unpack_connect", [](const py::bytes& b) { return pyb(unpack_connect(bytes_of(b))); });
    a.attr("COMPACT_BYTES") = COMPACT_BYTES;
    a.def("to_records", [](const py::bytes& b) { return pyb(to_records(bytes_of(b))); });
    py::class_<RecordReader>(a, "RecordReader")
        .def(py::init<>())
        .def_property_readonly("buf", [](const RecordReader& r) { return pyb(r.buf); })
        .def_readwrite("delivered", &RecordReader::delivered)
        .def("feed", [](RecordReader& r, const py::bytes& b) { return pyb(r.feed(bytes_of(b))); });

    // --- link ---
    a.def("unwrap", &unwrap);
    a.def("ctl_mask", [](int d, int i, int key) { return mask_tuple(ctl_mask(d, i, key)); }, py::arg("direction"), py::arg("i"),
          py::arg("key") = 0);
    a.def("data_mask",
          [](int d, std::int64_t seq, int key, bool comp, int epoch) { return mask_tuple(data_mask(d, seq, key, comp, epoch)); },
          py::arg("direction"), py::arg("seq"), py::arg("key") = 0, py::arg("comp") = false, py::arg("epoch") = 0);

    py::class_<Codeword>(a, "Codeword")
        .def_readonly("seq", &Codeword::seq)
        .def_readonly("start", &Codeword::start)
        .def_readonly("length", &Codeword::length)
        .def_readonly("submode", &Codeword::submode)
        .def_property_readonly("payload", [](const Codeword& c) { return pyb(c.payload); })
        .def_readonly("heard", &Codeword::heard)
        .def_readonly("comp", &Codeword::comp)
        .def_readonly("cstart", &Codeword::cstart)
        .def_readonly("first_bn", &Codeword::first_bn)
        .def_readonly("comp_known", &Codeword::comp_known);

    py::class_<TxSide>(a, "TxSide")
        .def_property_readonly("buf", [](const TxSide& t) { return pyb(t.buf); })
        .def_readonly("buf_off", &TxSide::buf_off)
        .def_readonly("stream_end", &TxSide::stream_end)
        .def_property_readonly("cws", [](const TxSide& t) { return dict_of(t.cws, [](const Codeword& c) { return py::cast(c); }); })
        .def_readonly("base", &TxSide::base)
        .def_readonly("next", &TxSide::next)
        .def_property_readonly("ack", [](const TxSide& t) -> py::object {
            if (!t.ack) return py::none();
            return py::make_tuple(t.ack->first, py::module_::import("builtins").attr("frozenset")(set_of(t.ack->second)));
        })
        .def_readonly("acked", &TxSide::acked)
        .def_readonly("acked_wire", &TxSide::acked_wire)
        .def_readonly("acked_plain", &TxSide::acked_plain)
        .def_property_readonly("hist", [](const TxSide& t) { return pyb(t.hist); })
        .def_readonly("hist_off", &TxSide::hist_off)
        .def("pending", &TxSide::pending)
        .def("missing", &TxSide::missing);

    py::class_<RxSide>(a, "RxSide")
        .def_readonly("cum", &RxSide::cum)
        .def_property_readonly("buf", [](const RxSide& r) {
            return dict_of(r.buf, [](const std::pair<Bytes, bool>& v) { return py::make_tuple(pyb(v.first), v.second); });
        })
        .def_property_readonly("reader", [](RxSide& r) { return &r.reader; }, py::return_value_policy::reference_internal)
        .def_property_readonly("out", [](const RxSide& r) { return pyb(r.out); })
        .def_property_readonly("hist", [](const RxSide& r) { return pyb(r.hist); })
        .def_readonly("wire", &RxSide::wire)
        .def_readonly("plain", &RxSide::plain)
        .def_property_readonly("comp_seqs", [](const RxSide& r) { return set_of(r.comp_seqs); });

    py::class_<Station, std::shared_ptr<Station>>(a, "Station", py::dynamic_attr())
        .def(py::init([](int direction, py::object policy, bool master, int key, std::optional<int> max_misses, int cap, bool chat) {
                 return std::shared_ptr<Station>(
                     std::make_shared<PyStation>(direction, policy_of(std::move(policy)), master, key, max_misses, cap, chat));
             }),
             py::arg("direction"), py::arg("policy"), py::arg("master") = false, py::arg("key") = 0,
             py::arg("max_misses") = LINK_LOST_MISSES, py::arg("cap") = 2, py::arg("chat") = false)
        .def_readonly("direction", &Station::direction)
        .def_property_readonly("policy", [](const Station& s) { return policy_py(s.policy); })
        .def_readonly("master", &Station::master)
        .def_readonly("key", &Station::key)
        .def_readwrite("max_misses", &Station::max_misses)
        .def_readwrite("cap", &Station::cap)
        .def_readonly("last_rx_data", &Station::last_rx_data)
        .def_readwrite("peer_recommend", &Station::peer_recommend)
        .def_readwrite("peer_size_hint", &Station::peer_size_hint)
        .def_readwrite("chat", &Station::chat)
        .def_readonly("peer_chat", &Station::peer_chat)
        .def_readonly("peer_queued", &Station::peer_queued)
        .def_readonly("peer_wants_dup", &Station::peer_wants_dup)
        .def_readwrite("peer_reply_recommend", &Station::peer_reply_recommend)
        .def_property_readonly("tx", [](Station& s) { return &s.tx; }, py::return_value_policy::reference_internal)
        .def_property_readonly("rx", [](Station& s) { return &s.rx; }, py::return_value_policy::reference_internal)
        .def_property_readonly("stats", [](const Station& s) { return counter(s.stats); })
        .def_property_readonly("state", [](const Station& s) { return s.state == LinkState::FAILED ? "failed" : "active"; })
        .def_readonly("fail_reason", &Station::fail_reason)
        .def_readonly("bursts_sent", &Station::bursts_sent)
        .def_property_readonly("last_sent", [](const Station& s) { return burst_py(s.last_sent); })
        .def_readonly("peer_burst", &Station::peer_burst)
        .def_readonly("reply_lost", &Station::reply_lost)
        .def_readonly("misses", &Station::misses)
        .def_readonly("reply_escalation", &Station::reply_escalation)
        .def_readonly("esc_floor", &Station::esc_floor)
        .def_readonly("_clean", &Station::clean_)
        .def_readonly("_sent_esc", &Station::sent_esc_)
        .def_readonly("no_progress", &Station::no_progress)
        .def_readonly("resyncs", &Station::resyncs)
        .def_readonly("resync_due", &Station::resync_due)
        .def_readonly("_latest", &Station::latest)
        .def_readonly("_confirmed", &Station::confirmed)
        .def_property_readonly("_sent_seqs", [](const Station& s) { return dict_of(s.sent_seqs, [](const auto& v) { return py::cast(v); }); })
        .def_property_readonly("peer", &Station::peer)
        .def("write", [](Station& s, const py::object& b) { s.write(bytes_of(b)); })
        .def("read", [](Station& s) { return pyb(s.read()); })
        .def("build", [](Station& s, bool fresh) {
            sync_compress(s);
            return burst_py(s.Station::build(fresh));
        }, py::arg("fresh") = true)
        .def("on_timeout", [](Station& s, bool allow_repeat) { return burst_py(s.on_timeout(allow_repeat)); },
             py::arg("allow_repeat") = true)
        .def("handle", [](Station& s, py::object rx) {
            PyRx r(std::move(rx));
            return s.handle(r);
        })
        .def("answered", &Station::answered)
        .def("_ctl_pair", [](Station& s, py::object rx, int slot, int i) -> py::object {
            PyRx r(std::move(rx));
            auto p = s.ctl_pair(r, slot, i);
            return p ? py::object(pyb(*p)) : py::none();
        })
        .def("_new_available", &Station::new_available);

    // --- session ---
    a.def("session_key", &session_key);
    a.def("_frame_desc", [](const py::bytes& b) { return frame_desc(bytes_of(b)); });

    // The session's Python view keeps one wrapper per station in its
    // __dict__, so attributes set on it (a test's build override) last.
    auto station_py = [](Session& s, py::object self) -> py::object {
        py::dict d = py::getattr(self, "__dict__");
        if (d.contains("_station_override")) return d["_station_override"];
        if (!s.station) return py::none();
        if (d.contains("_station_obj")) {
            py::object o = d["_station_obj"];
            if (o.cast<Station*>() == s.station.get()) return o;
        }
        py::object o = py::cast(s.station);
        d["_station_obj"] = o;
        return o;
    };
    py::class_<Session, std::shared_ptr<Session>>(a, "Session", py::dynamic_attr())
        .def(py::init([](std::string call, py::object policy, double t_turn, py::object rng, std::vector<std::string> aliases,
                         double stats_interval_s) {
                 std::shared_ptr<Rng> r;
                 if (!rng.is_none()) r = std::make_shared<PyRng>(rng);
                 auto* s = new PySession(std::move(call), policy_of(std::move(policy)), t_turn, r, std::move(aliases), stats_interval_s);
                 return static_cast<Session*>(s);
             }),
             py::arg("call"), py::arg("policy"), py::arg("t_turn") = 1.0, py::arg("rng") = py::none(),
             py::arg("aliases") = std::vector<std::string>{}, py::arg("stats_interval_s") = 60.0)
        .def_readwrite("call", &Session::call)
        .def_property_readonly("policy", [](const Session& s) { return policy_py(s.policy); })
        .def_readwrite("t_turn", &Session::t_turn)
        .def_property_readonly("rng", [](const Session& s) -> py::object {
            auto* r = dynamic_cast<PyRng*>(s.rng.get());
            return r ? r->obj : py::none();
        })
        .def_property_readonly("state", [](const Session& s) { return state_name(s.state); })
        .def_readwrite("peer", &Session::peer)
        .def_readwrite("cap", &Session::cap)
        .def_property("station", [station_py](py::object self) { return station_py(self.cast<Session&>(), self); },
                      [](py::object self, py::object v) { py::getattr(self, "__dict__")["_station_override"] = v; })
        .def_property_readonly("events", [](py::object self) {
            // a list the engine clears in place (events[:] = []): new events
            // are moved into it at each read
            auto& s = self.cast<Session&>();
            py::dict d = py::getattr(self, "__dict__");
            if (!d.contains("_events_list")) d["_events_list"] = py::list();
            py::list l = d["_events_list"];
            for (const auto& e : s.events) l.append(e);
            s.events.clear();
            return l;
        })
        .def_readonly("close_reason", &Session::close_reason)
        .def_readwrite("chat", &Session::chat)  // Python's attribute: set_chat() is the one that reaches the station
        .def_property_readonly("aliases", [](const Session& s) { return py::tuple(py::cast(s.aliases)); })
        .def_readwrite("stats_interval_s", &Session::stats_interval_s)
        .def_property_readonly("_out", [](const Session& s) { return burst_py(s.out); })
        .def_readonly("_due", &Session::due)
        .def_readonly("_deadline", &Session::deadline)
        .def_readonly("_build_at", &Session::build_at)
        .def_property_readonly("_master", &Session::master)
        .def_property_readonly("_pending_write", [](const Session& s) { return pyb(s.pending_write); })
        .def_readonly("_nonce", &Session::nonce)
        .def_readonly("_tries", &Session::tries)
        .def_readonly("_wakes", &Session::wakes)
        .def("_on_timeout", [](Session& s, double now) { s.Session::on_timeout(now); })
        .def("set_chat", &Session::set_chat)
        .def("listen", &Session::listen)
        .def("connect", &Session::connect)
        .def("disconnect", &Session::disconnect)
        .def("write", [](Session& s, const py::object& b) { s.write(bytes_of(b)); })
        .def("read", [](Session& s) { return pyb(s.read()); })
        .def("poll", [](Session& s, double now) {
            auto* p = dynamic_cast<PySession*>(&s);
            return p ? p->burst_obj(s.poll(now)) : burst_py(s.poll(now));
        })
        .def("next_event", &Session::next_event)
        .def("on_tx_end", [](Session& s, const py::object& burst, double now) { s.on_tx_end(burst_cpp(burst), now); })
        .def("on_header", &Session::on_header)
        .def("on_rx", [](Session& s, py::object rx, double now) {
            PyRx r(std::move(rx));
            s.on_rx(r, now);
        });
}

}  // namespace data2g::bind
