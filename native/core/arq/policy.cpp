#include "arq/policy.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>

#include "cpm/cpm.hpp"

namespace data2g::arq {

namespace {

// frames.BANDS_CODE: sync bands, 2 bits.
constexpr std::pair<std::string_view, int> BANDS_CODE[] = {{"w", 0}, {"n10", 1}, {"w48", 2}};

int band_code(std::string_view sync_band) {
    for (const auto& [b, c] : BANDS_CODE)
        if (b == sync_band) return c;
    throw std::out_of_range("no band code for " + std::string(sync_band));
}

// Python's round(x, 2): the correctly rounded decimal, ties to even.
double round2(double x) {
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.2f", x);
    return std::strtod(buf, nullptr);
}

double logit_shift(double q, double b) {
    q = std::clamp(q, 1e-6, 1 - 1e-6);
    return 1 / (1 + std::exp(-(std::log(q / (1 - q)) + b)));
}

double get(const GearShifter::Map& m, std::string_view k) {
    const auto it = m.find(k);
    return it == m.end() ? 0.0 : it->second;
}

int ceil_div(long a, long b) { return static_cast<int>((a + b - 1) / b); }  // a >= 0, b > 0

}  // namespace

int cap_hz(int cap) {
    switch (cap) {
        case 0: return 500;
        case 1: return 1200;
        case 2: return 2400;
    }
    throw std::out_of_range("no cap " + std::to_string(cap));
}

std::string_view fallback(int cap) {
    cap_hz(cap);
    return cap == 0 ? "n10-ack-4f" : "ack-4f";
}

std::string_view connect_mode(int cap, int tries) {
    cap_hz(cap);
    if (tries) return ROBUST_CONNECT;
    return cap == 0 ? "n10-qpsk-r1/3" : "qpsk-r1/5";
}

double width_hz(const Mode& m) {
    if (m.is_cpm()) {
        const auto& g = cpm::grid_of(*m.cpm);
        return g.m * g.rate + 2 * g.bp;
    }
    if (m.band == "n4") return 200;
    if (m.band == "n10") return 500;
    if (m.band == "w") return 1200;
    if (m.band == "w48") return 2400;
    throw std::out_of_range("no width for band " + std::string(m.band));
}

std::vector<const Mode*> allowed(int cap) {
    const int hz = cap_hz(cap);
    std::vector<const Mode*> out;
    for (const auto& m : modes())
        if (width_hz(m) <= hz) out.push_back(&m);
    return out;
}

int encode(std::string_view submode) {
    const Mode& m = mode_at(submode);
    if (m.is_cpm()) return CPM_CODE << 4 | static_cast<int>(m.cpm - tables::CPM_SPECS.data());
    return band_code(m.ofdm->sync_band) << 4 | m.ofdm->index;
}

const Mode* decode(int rec) {
    if (rec >> 4 == CPM_CODE) {
        const auto i = static_cast<std::size_t>(rec & 15);
        return i < tables::CPM_SPECS.size() ? mode(tables::CPM_SPECS[i].name) : nullptr;
    }
    for (const auto& [band, code] : BANDS_CODE)
        if (code == rec >> 4) {
            for (const auto& s : config::SUBMODES)
                if (s.sync_band == band && s.index == (rec & 15)) return mode(s.name);
            return nullptr;
        }
    return nullptr;
}

int ctl_slots(const Mode& m) { return std::min(max_ctl(m), ceil_div(CTL_BYTES, ctl_payload_bytes(m))); }

int slots_for(const Mode& m, double seconds, bool data, bool dup) {
    if (m.is_cpm()) seconds *= CPM_SIZE_SCALE;
    int n = 1;
    while (n < 64 && burst_seconds(m, n + 1) <= seconds) ++n;
    n = std::max({n, min_cw(m, data), ctl_slots(m) + data});
    if (m.is_cpm()) n = std::min(n + (dup && data), 1 + dup + tables::CPM.max_data);
    return n;
}

std::pair<std::string_view, int> GearShifter::choose(const StationView& st, int escalation) const {
    const auto rec = st.pending ? st.peer_recommend : st.peer_reply_recommend;
    if (escalation || !rec) return {fallback(st.cap), 2};
    const Mode* m = decode(*rec);
    const auto ok = allowed(st.cap);
    if (!m || std::find(ok.begin(), ok.end(), m) == ok.end()) return {fallback(st.cap), 2};
    return {m->name, slots_for(*m, SIZE_S.at(st.peer_size_hint), st.pending, st.peer_wants_dup)};
}

int GearShifter::next_capacity(const StationView& st) const {
    const Mode* m = st.peer_recommend ? decode(*st.peer_recommend) : nullptr;
    const auto ok = allowed(st.cap);
    if (!m || std::find(ok.begin(), ok.end(), m) == ok.end()) m = &mode_at(fallback(st.cap));
    const int n = slots_for(*m, SIZE_S.at(st.peer_size_hint));
    return (n - ctl_slots(*m)) * payload_bytes(*m);
}

void GearShifter::observe(const Measured& m, std::string_view submode, double now) {
    if (measured) prev = Heard{*measured, measured_band, measured_at};
    measured = m;
    measured_band = std::string(mode_at(submode).band);
    measured_at = now;
}

void GearShifter::outcome(std::string_view submode, int decoded, int sent, std::optional<bool> usable) {
    const auto it = predicted.find(submode);
    if (it == predicted.end() || (sent == 0 && !usable)) return;
    const bool ok = usable ? *usable : decoded > 0;
    const auto [pb, p] = it->second;
    const std::string key(submode);
    bias_burst[key] = std::clamp(get(bias_burst, key) + BIAS_STEP * (ok - pb), -BIAS_MAX, BIAS_MAX);
    if (!ok || sent == 0) return;
    bias[key] = std::clamp(get(bias, key) + BIAS_STEP * (static_cast<double>(decoded) / sent - p), -BIAS_MAX, BIAS_MAX);
}

GearRecommendation GearShifter::recommend(const StationView& st) {
    if (!measured) {
        const int fb = encode(fallback(st.cap));
        return {fb, 1, fb};
    }
    std::vector<const Mode*> cands;
    for (const Mode* s : allowed(st.cap))
        if (!s->is_cpm() || (use_cpm && outcome_knows(s->name))) cands.push_back(s);
    std::optional<Prev> pv;
    if (prev && measured_at - prev->at <= PREV_MAX_S) pv = Prev{prev->m, prev->band, measured_at - prev->at};

    std::map<double, std::vector<Outcome>> memo;  // by the burst's rounded length
    auto predicted_at = [&](const Mode& s, int n_cw) -> const Outcome& {
        const double sec = round2(burst_seconds(s, n_cw));
        auto it = memo.find(sec);
        if (it == memo.end())
            it = memo.emplace(sec, predict_outcome(*measured, measured_band, gap_s, sec, pv ? &*pv : nullptr)).first;
        const int i = outcome_index(s.name);
        if (i < 0) throw std::out_of_range("outcome model lacks " + std::string(s.name));
        return it->second[static_cast<std::size_t>(i)];
    };
    auto q_burst = [&](const Mode& s, int n_cw) { return logit_shift(predicted_at(s, n_cw).burst, get(bias_burst, s.name)); };
    auto q_cw = [&](const Mode& s, int n_cw) { return logit_shift(predicted_at(s, n_cw).cw, get(bias, s.name)); };

    // my reply: the cheapest in expectation, a lost one costing a timeout
    const Mode* reply = nullptr;
    double reply_c = std::numeric_limits<double>::infinity();
    for (const Mode* s : cands) {
        const double t = burst_seconds(*s, 1);
        const double ok = q_burst(*s, 1);
        const double c = t + (1 - ok) * (TIMEOUT_S + t) / std::max(ok, 1e-3);
        if (c < reply_c) reply = s, reply_c = c;
    }

    const Mode* cur = log.empty() ? nullptr : &mode_at(log.back().data);
    const long held = cur ? st.held * payload_bytes(*cur) : 0;
    const bool chat = st.chat;
    const long queued = std::max<long>(CHAT_BYTES, st.peer_queued);
    const Mode* best = nullptr;
    int best_hint = 0;
    double best_v = chat ? -std::numeric_limits<double>::infinity() : -1.0;
    for (const Mode* s : cands) {
        const int pb = payload_bytes(*s), c = ctl_slots(*s);
        for (int hint = 0; hint < static_cast<int>(SIZE_S.size()); ++hint) {
            int n = slots_for(*s, SIZE_S[hint]);
            int k = 0;
            if (chat) {
                k = ceil_div(queued, pb);
                if (n < k + c && hint < static_cast<int>(SIZE_S.size()) - 1) continue;
                n = std::min(n, k + c);
            }
            const double pn = q_cw(*s, n);
            const double ok_ctl = q_burst(*s, n);
            if (ok_ctl * pn < min_success) continue;
            const bool dup = s->is_cpm() && ok_ctl < DUP_BELOW;
            const double tb = burst_seconds(*s, n + dup, dup);
            const double t = tb + 2 * TURN_S + reply_c + (1 - ok_ctl) * TIMEOUT_S;
            double v;
            if (chat) {
                const double ok_all = std::max(ok_ctl * std::pow(pn, static_cast<double>(n - c)), 1e-3);
                v = -t * ceil_div(k, std::max(n - c, 1)) / ok_all + 1e-6 * (n - c) * pb;
            } else {
                v = (ok_ctl * pn * (n - c) * pb - static_cast<double>(s != cur ? held : 0)) / t;
            }
            if (v > best_v) best = s, best_hint = hint, best_v = v;
            if (chat) break;
        }
    }
    std::string_view best_name;
    if (!best) {
        best_name = fallback(st.cap), best_hint = 1;
    } else {
        best_name = best->name;
        const int nb = slots_for(*best, SIZE_S[best_hint]);
        predicted = {{std::string(best->name), {q_burst(*best, nb), q_cw(*best, nb)}}};
        want_dup = predicted.begin()->second.first < DUP_BELOW;
        if (reply) predicted[std::string(reply->name)] = {q_burst(*reply, 1), q_cw(*reply, 1)};
    }
    const std::string_view reply_name = reply ? reply->name : fallback(st.cap);
    log.push_back({std::string(best_name), best_hint, std::string(reply_name)});
    return {encode(best_name), best_hint, encode(reply_name)};
}

}  // namespace data2g::arq
