// data2g.modem, dict-shaped as in Python. Submodes go out by name (conftest
// maps them back to SubmodeSpec); going in, "spec" may be a name or any
// object with .name. `accept` is None or anything with .max_cw ((name, cw)
// pairs) and .min_score (modem.Accept).
#include <cmath>
#include <limits>
#include <map>
#include <optional>

#include "convert.hpp"
#include "modem/modem.hpp"

namespace data2g::bind {

namespace {

using modem::cd;

template <typename T>
std::span<const T> view(const In<T>& a) {
    return {a.data(), static_cast<std::size_t>(a.size())};
}

std::string name_of(const py::handle& spec) {
    return py::isinstance<py::str>(spec) ? spec.cast<std::string>() : spec.attr("name").cast<std::string>();
}

const config::Submode& spec_of(const py::handle& s) { return spec(name_of(s)); }

std::optional<modem::Accept> accept_of(const py::object& a) {
    if (a.is_none()) return std::nullopt;
    modem::Accept out;
    for (const auto& item : a.attr("max_cw")) {
        const auto t = item.cast<py::tuple>();
        out.max_cw.emplace_back(&spec(t[0].cast<std::string>()), t[1].cast<int>());
    }
    out.min_score = a.attr("min_score").cast<double>();
    return out;
}

// Band names as views into config::BANDS (the core keeps string_views).
std::string_view band_view(const std::string& name) { return modem::band(name).name; }

std::vector<std::string_view> bands_of(const std::optional<std::vector<std::string>>& bands) {
    std::vector<std::string_view> out;
    if (bands)
        for (const auto& b : *bands) out.push_back(band_view(b));
    return out;
}

py::array_t<cd> np3(const Mat<cd>& m, std::size_t per) {
    return np(m).reshape({static_cast<py::ssize_t>(m.rows / per), static_cast<py::ssize_t>(per),
                          static_cast<py::ssize_t>(m.cols)});
}

py::array_t<double> np3(const Mat<double>& m, std::size_t per) {
    return np(m).reshape({static_cast<py::ssize_t>(m.rows / per), static_cast<py::ssize_t>(per),
                          static_cast<py::ssize_t>(m.cols)});
}

// (rows, nc) from an (..., nc) array
Mat<cd> mat2(const In<cd>& a) {
    const std::size_t nc = a.ndim() ? a.shape(a.ndim() - 1) : 1;
    Mat<cd> m(nc ? a.size() / nc : 0, nc);
    std::copy(a.data(), a.data() + a.size(), m.data.begin());
    return m;
}

Mat<double> mat2(const In<double>& a) {
    const std::size_t nc = a.ndim() ? a.shape(a.ndim() - 1) : 1;
    Mat<double> m(nc ? a.size() / nc : 0, nc);
    std::copy(a.data(), a.data() + a.size(), m.data.begin());
    return m;
}

py::object header_tuple(const modem::Header& h) {
    return py::make_tuple(h.word, py::make_tuple(std::string(h.spec->name), h.n_cw), h.score);
}

py::tuple acq_tuple(const waveform::Acquisition& a) {
    py::list alts;
    for (const auto& [s, f] : a.alternatives) alts.append(py::make_tuple(s, f));
    return py::make_tuple(a.preamble_start, a.freq_offset, a.metric, alts);
}

py::dict header_dict(const modem::HeaderRead& r) {
    py::dict d;
    d["word"] = r.hdr.word;
    d["hdr"] = r.valid ? py::object(py::make_tuple(std::string(r.hdr.spec->name), r.hdr.n_cw)) : py::object(py::none());
    d["score"] = r.hdr.score;
    d["pending_copy"] = r.pending_copy;
    d["y"] = np(r.y);
    d["y_all"] = np(r.y_all);
    d["h_pre"] = np(r.h_pre);
    d["h_first"] = np(r.h_first);
    d["p0"] = r.p0;
    d["start"] = r.start;
    d["band"] = std::string(r.band);
    d["n0_pre"] = r.n0_pre;
    d["n0_pre_k"] = np(r.n0_pre_k);
    return d;
}

py::dict lock_dict(const modem::Lock& l) {
    py::dict d;
    d["spec"] = std::string(l.spec->name);
    d["n_cw"] = l.n_cw;
    d["start"] = l.start;
    d["score"] = l.score;
    d["band"] = std::string(l.band);
    d["end"] = l.end;
    d["p0"] = l.p0;
    d["cfo"] = l.cfo;
    if (l.copy) {
        py::dict c;
        c["word"] = l.copy->word;
        c["pc"] = l.copy->pc;
        d["copy"] = c;
    }
    return d;
}

template <typename T>
T get(const py::dict& d, const char* k, T dflt) {
    return d.contains(k) ? d[k].cast<T>() : dflt;
}

modem::Lock lock_of(const py::dict& d) {
    modem::Lock l;
    l.spec = &spec_of(d["spec"]);
    l.n_cw = d["n_cw"].cast<int>();
    l.start = get<std::int64_t>(d, "start", 0);
    l.end = get<std::int64_t>(d, "end", 0);
    l.p0 = d["p0"].cast<std::int64_t>();
    l.score = get<double>(d, "score", 0.0);
    l.band = d.contains("band") ? band_view(d["band"].cast<std::string>()) : l.spec->sync_band;
    l.cfo = d["cfo"].cast<double>();
    if (d.contains("copy")) {
        const auto c = d["copy"].cast<py::dict>();
        l.copy = modem::CopyRef{c["word"].cast<int>(), c["pc"].cast<std::int64_t>()};
    }
    return l;
}

py::dict est_dict(const modem::DataEstimate& e) {
    constexpr std::size_t S = config::DATA_SYMS_PER_FRAME;
    py::dict d;
    d["h"] = np3(e.h, S);
    d["mse"] = np3(e.mse, S);
    d["n0"] = e.n0;
    d["n0_k"] = np(e.n0_k);
    d["p_sig"] = e.p_sig;
    d["spread_hz"] = e.spread_hz;
    d["clip_ratio"] = e.clip_ratio;
    d["gain"] = e.gain;
    d["band"] = std::string(e.band);
    return d;
}

// The DataEstimate fields noise_var / decode_received read, from a dict.
modem::DataEstimate est_of(const py::dict& d) {
    modem::DataEstimate e;
    e.h = mat2(d["h"].cast<In<cd>>());
    e.mse = mat2(d["mse"].cast<In<double>>());
    e.n0 = d["n0"].cast<double>();
    if (d.contains("n0_k")) e.n0_k = vec(d["n0_k"].cast<In<double>>());
    e.p_sig = get<double>(d, "p_sig", 0.0);
    e.clip_ratio = d["clip_ratio"].cast<double>();
    return e;
}

py::dict received_dict(const modem::Received& r) {
    py::dict d;
    d["spec"] = std::string(r.spec->name);
    d["n_cw"] = r.n_cw;
    d["raw"] = np3(r.raw, config::SYMS_PER_FRAME);
    d["est"] = est_dict(r.est);
    d["acq"] = acq_tuple(r.acq);
    d["band"] = std::string(r.band);
    d["hp"] = np(r.hp);
    d["kc"] = r.kc ? py::object(py::int_(*r.kc)) : py::object(py::none());
    d["cfo"] = r.cfo;
    d["p0"] = r.p0;
    d["shift"] = r.shift;
    d["steps"] = np(r.steps);
    d["phi_ref"] = r.phi_ref;
    d["support"] = py::make_tuple(r.support.first, r.support.second);
    d["preamble_start"] = r.preamble_start;
    d["score"] = r.score;
    return d;
}

py::tuple burst_tuple(const modem::Burst& b) {
    py::list payloads, ok;
    for (std::size_t i = 0; i < b.payloads.size(); ++i) {
        payloads.append(to_bytes(b.payloads[i]));
        ok.append(static_cast<bool>(b.crc_ok[i]));
    }
    return py::make_tuple(std::string(b.spec->name), payloads, ok, b.freq_offset, b.preamble_start, b.snr_db,
                          np(b.soft));
}

// stats: {band: S or None}
struct Stats {
    std::vector<Mat<double>> mats;
    std::vector<modem::BandStat> v;
};

Stats stats_of(const std::optional<std::map<std::string, std::optional<In<double>>>>& stats) {
    Stats s;
    if (!stats) return s;
    s.mats.reserve(stats->size());
    for (const auto& [b, S] : *stats)
        if (S) {
            s.mats.push_back(mat(*S));
            s.v.push_back({band_view(b), &s.mats.back()});
        }
    return s;
}

using OptBands = std::optional<std::vector<std::string>>;
using OptStats = std::optional<std::map<std::string, std::optional<In<double>>>>;

}  // namespace

void bind_modem(py::module_& m) {
    namespace mo = modem;
    auto d = m.def_submodule("modem", "data2g.modem");
    d.def("crc6", &mo::crc6);
    d.def("codeword", [](int word, const std::string& band) { return np(mo::codeword(word, band)); });
    d.def("header_bits", [](int submode, int n_cw, const std::string& band) {
        return np<std::int64_t>(std::span<const std::uint8_t>(mo::header_bits(submode, n_cw, band)));
    }, py::arg("submode"), py::arg("n_cw"), py::arg("band") = "w");
    d.def("valid_words", [](const std::string& band, const py::object& accept) {
        const auto a = accept_of(accept);
        return np<std::int64_t>(std::span<const int>(mo::valid_words(band, a ? &*a : nullptr)));
    }, py::arg("band"), py::arg("accept") = py::none());
    d.def("signs", [](const In<std::int64_t>& words, const std::string& band) {
        const auto n = mo::header_cols(band).size();
        py::array_t<float> out({static_cast<py::ssize_t>(words.size()), static_cast<py::ssize_t>(n)});
        for (py::ssize_t i = 0; i < words.size(); ++i) {
            const auto c = mo::codeword(static_cast<int>(words.data()[i]), band);
            for (std::size_t j = 0; j < n; ++j) out.mutable_data()[static_cast<std::size_t>(i) * n + j] = 1.0f - 2.0f * c[j];
        }
        return out;
    });
    d.def("header_corr", [](const In<double>& soft, const std::string& band, const py::object& accept) {
        const auto a = accept_of(accept);
        return np(mo::header_corr(view(soft), band, a ? &*a : nullptr));
    }, py::arg("soft"), py::arg("band") = "w", py::arg("accept") = py::none());
    d.def("decode_header", [](const In<double>& soft, const std::string& band, const py::object& accept) {
        const auto a = accept_of(accept);
        return header_tuple(mo::decode_header(view(soft), band, a ? &*a : nullptr));
    }, py::arg("soft"), py::arg("band") = "w", py::arg("accept") = py::none());

    d.def("modulate", [](const std::vector<py::bytes>& payloads, const py::object& s, const std::vector<int>& rvs) {
        std::vector<std::vector<std::uint8_t>> p;
        for (const auto& b : payloads) {
            const auto v = bytes_view(b);
            p.emplace_back(v.begin(), v.end());
        }
        return np(mo::modulate(p, spec_of(s), rvs));
    }, py::arg("payloads"), py::arg("submode"), py::arg("rvs") = std::vector<int>{});
    d.def("modulate_bits", [](const In<std::uint8_t>& bits, const py::object& s) {
        return np(mo::modulate_bits(view(bits), spec_of(s)));
    });
    d.def("burst_waveform", [](const In<cd>& data, const py::object& s) {
        return np(mo::burst_waveform(mat2(data), spec_of(s)));
    });
    d.def("ace_cells", [](const py::object& s, int n_f) {
        const auto starts = mo::ace_cells(spec_of(s), n_f);
        const auto n = static_cast<py::ssize_t>(starts.size());
        py::array_t<std::int64_t> full({n, static_cast<py::ssize_t>(config::NSYM)});
        for (py::ssize_t i = 0; i < n; ++i)
            for (int t = 0; t < config::NSYM; ++t) full.mutable_data()[i * config::NSYM + t] = starts[static_cast<std::size_t>(i)] + t;
        return full;
    });

    d.def("bin_phase_step", [](const In<cd>& h) { return mo::bin_phase_step(view(h)); });
    d.def("demod_frames", [](const In<cd>& z, std::int64_t p, int n_f, int shift, double phi_ref,
                             const std::optional<In<double>>& steps_in, const std::string& band) {
        const std::vector<double> si = steps_in ? vec(*steps_in) : std::vector<double>();
        const auto f = mo::demod_frames(view(z), p, n_f, shift, phi_ref, si, band);
        return py::make_tuple(np3(f.raw, config::SYMS_PER_FRAME), np(f.hp), np(f.steps));
    }, py::arg("z"), py::arg("p"), py::arg("n_f"), py::arg("shift"), py::arg("phi_ref"),
       py::arg("steps_in") = py::none(), py::arg("band") = "w");
    d.def("read_header", [](const In<cd>& z, std::int64_t start, const std::string& band, const py::object& accept) {
        const auto a = accept_of(accept);
        return header_dict(mo::read_header(view(z), start, band, a ? &*a : nullptr));
    }, py::arg("z"), py::arg("start"), py::arg("band") = "w", py::arg("accept") = py::none());
    d.def("copy_llr", [](const In<cd>& z, std::int64_t p, const std::string& band, int n_hdr) -> py::object {
        const auto r = mo::copy_llr(view(z), p, band, n_hdr);
        return r ? py::object(np(*r)) : py::object(py::none());
    });
    d.def("best_header", [](const In<cd>& z0, const OptBands& bands, bool complete, const py::object& accept,
                            const OptStats& stats) {
        const auto a = accept_of(accept);
        const auto st = stats_of(stats);
        const auto bs = bands_of(bands);
        mo::BestHeader r;
        {
            py::gil_scoped_release nogil;
            r = mo::best_header(view(z0), bs, complete, a ? &*a : nullptr, st.v);
        }
        return py::make_tuple(header_dict(r.hd), acq_tuple(r.acq), np(r.z));
    }, py::arg("z0"), py::arg("bands") = py::none(), py::arg("complete") = true, py::arg("accept") = py::none(),
       py::arg("stats") = py::none());
    d.def("find_burst", [](const In<double>& x, const OptBands& bands, const py::object& accept, const OptStats& stats) {
        const auto a = accept_of(accept);
        const auto st = stats_of(stats);
        const auto bs = bands_of(bands);
        mo::Lock l;
        {
            py::gil_scoped_release nogil;
            l = mo::find_burst(view(x), bs, a ? &*a : nullptr, st.v);
        }
        return lock_dict(l);
    }, py::arg("x"), py::arg("bands") = py::none(), py::arg("accept") = py::none(), py::arg("stats") = py::none());
    d.def("pilot_coherence", [](const In<double>& x, const py::dict& lock, int n_max, bool latest) {
        return mo::pilot_coherence(view(x), lock_of(lock), n_max, latest);
    }, py::arg("x"), py::arg("lock"), py::arg("n_max") = 8, py::arg("latest") = false);
    d.def("find_copy", [](const In<double>& x, const std::string& band, const py::object& accept,
                          const std::optional<In<cd>>& C, std::optional<double> level) {
        const auto a = accept_of(accept);
        std::optional<Mat<cd>> c;
        if (C) c = mat(*C);
        double peak = std::numeric_limits<double>::quiet_NaN();
        std::optional<mo::Lock> l;
        {
            py::gil_scoped_release nogil;
            l = mo::find_copy(view(x), band_view(band), a ? &*a : nullptr, c ? &*c : nullptr, level, &peak);
        }
        return py::make_tuple(l ? py::object(lock_dict(*l)) : py::object(py::none()),
                              std::isnan(peak) ? py::object(py::none()) : py::object(py::float_(peak)));
    }, py::arg("x"), py::arg("band"), py::arg("accept") = py::none(), py::arg("C") = py::none(),
       py::arg("level") = py::none());
    d.def("cfo_aliases", &mo::cfo_aliases);
    d.def("copy_header", [](const In<cd>& z, const py::dict& lock) {
        return header_dict(mo::copy_header(view(z), lock_of(lock)));
    });
    d.def("receive", [](const In<double>& x, const OptBands& bands, const py::object& accept,
                        std::optional<std::int64_t> head, const std::optional<py::dict>& copy) {
        const auto a = accept_of(accept);
        const auto bs = bands_of(bands);
        std::optional<mo::Lock> c;
        if (copy) c = lock_of(*copy);
        mo::Received r;
        {
            py::gil_scoped_release nogil;
            r = mo::receive(view(x), bs, a ? &*a : nullptr, head, c ? &*c : nullptr);
        }
        return received_dict(r);
    }, py::arg("x"), py::arg("bands") = py::none(), py::arg("accept") = py::none(), py::arg("head") = py::none(),
       py::arg("copy") = py::none());
    d.def("resolve_alias", &mo::resolve_alias);
    d.def("data_channel", [](const In<cd>& h_pilot, equalizer::Support support, const std::string& band, double n0_pre,
                             const std::optional<py::tuple>& clip, const std::optional<In<double>>& n0_pre_k,
                             std::optional<int> n_frames) {
        std::optional<mo::ClipConsts> c;
        if (clip) {
            c.emplace();
            for (const auto& [k, v] : (*clip)[0].cast<py::dict>()) c->gains.emplace_back(k.cast<int>(), v.cast<double>());
            c->gain = (*clip)[1].cast<double>();
            c->ratio = (*clip)[2].cast<double>();
        }
        const std::vector<double> pk = n0_pre_k ? vec(*n0_pre_k) : std::vector<double>();
        return est_dict(mo::data_channel(mat(h_pilot), support, band_view(band), n0_pre, c ? &*c : nullptr, pk,
                                         n_frames.value_or(-1)));
    }, py::arg("h_pilot"), py::arg("support"), py::arg("band") = "w", py::arg("n0_pre") = equalizer::INF,
       py::arg("clip") = py::none(), py::arg("n0_pre_k") = py::none(), py::arg("n_frames") = py::none());
    d.def("noise_var", [](const In<cd>& h, const In<double>& n0_k, double clip_ratio) {
        mo::DataEstimate e;
        e.n0_k = vec(n0_k);
        e.clip_ratio = clip_ratio;
        const auto v = mo::noise_var(mat2(h), e);
        return np(v).reshape(std::vector<py::ssize_t>(h.shape(), h.shape() + h.ndim()));
    });
    d.def("soft_bits", [](const In<cd>& raw, const In<cd>& h, const In<double>& var, const py::object& s) {
        return np(mo::soft_bits(mat2(raw), mat2(h), mat2(var), spec_of(s)));
    });
    d.def("decode_received", [](const py::dict& r) {
        mo::Received rx;
        rx.spec = &spec_of(r["spec"]);
        rx.n_cw = r["n_cw"].cast<int>();
        rx.raw = mat2(r["raw"].cast<In<cd>>());
        rx.est = est_of(r["est"].cast<py::dict>());
        rx.cfo = r["cfo"].cast<double>();
        rx.preamble_start = r["preamble_start"].cast<std::int64_t>();
        mo::Burst b;
        {
            py::gil_scoped_release nogil;
            b = mo::decode_received(rx);
        }
        return burst_tuple(b);
    });
    d.def("demodulate", [](const In<double>& x, const OptBands& bands, const py::object& accept) {
        const auto a = accept_of(accept);
        const auto bs = bands_of(bands);
        mo::Burst b;
        {
            py::gil_scoped_release nogil;
            b = mo::demodulate(view(x), bs, a ? &*a : nullptr);
        }
        return burst_tuple(b);
    }, py::arg("x"), py::arg("bands") = py::none(), py::arg("accept") = py::none());
}

}  // namespace data2g::bind
