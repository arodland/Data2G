#include "arq/phy.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>

#include "constellation/constellation.hpp"
#include "cpm/cpm.hpp"
#include "equalizer/equalizer.hpp"
#include "waveform/dsp.hpp"

namespace data2g::arq {

namespace {

using cd = std::complex<double>;
constexpr std::size_t S = config::DATA_SYMS_PER_FRAME;

const codes::Spec& cspec(std::string_view name) {
    const auto* s = codes::spec(name);
    if (!s) throw std::out_of_range("no code spec " + std::string(name));
    return *s;
}

const codes::Spec& ctl_code(const Mode& m) {
    return m.is_cpm() ? cspec(cpm::ctl(cpm::grid_of(*m.cpm)).name) : codes::spec(*m.ofdm);
}

const codes::Spec& data_code(const Mode& m) { return m.is_cpm() ? cspec(m.cpm->name) : codes::spec(*m.ofdm); }

Mat<float> row_f(std::span<const double> v) {
    Mat<float> out(1, v.size());
    std::copy(v.begin(), v.end(), out.data.begin());
    return out;
}

// 30 (1 - 2 bit): a decoded codeword's bits as large LLRs.
std::vector<double> known(std::span<const std::uint8_t> bits) {
    std::vector<double> out(bits.size());
    for (std::size_t i = 0; i < bits.size(); ++i) out[i] = 30.0 * (1 - 2.0 * bits[i]);
    return out;
}

struct Post {
    std::vector<std::uint8_t> bits;  // info bits as decoded (still scrambled unless from a buffer)
    bool ok;
    std::vector<double> post;  // a-posteriori LLRs of the slot's coded bits, mapping order
};

// phy._decode_post: one LLR decode as decode_raw (buf nullptr: soft alone)
// or decode_buffer (RVs up to top, the slot sent at rv) would, with the
// posterior DD needs from the same pass.
Post decode_post(const codes::Spec& s, const std::vector<double>* buf, int top, int rv, std::span<const double> soft) {
    Mat<float> llr;
    const ldpc::Decoder* dec;
    if (!buf) {
        llr = Mat<float>(1, soft.size());
        for (std::size_t i = 0; i < soft.size(); ++i) llr[0][s.perm[i]] = static_cast<float>(soft[i]);
        dec = &codes::ldpc_decoder(s);
    } else {
        const int extent = std::min(codes::buffer_len(s), (std::min(top, codes::rv_cycle(s) - 1) + 1) * s.coded_bits);
        llr = row_f(std::span(*buf).first(static_cast<std::size_t>(extent)));
        dec = &codes::ldpc_decoder(s, extent);
    }
    const auto d = dec->decode(llr, codes::ITERS, {}, true);
    std::vector<double> code(d.posterior.data.begin(), d.posterior.data.end());
    if (buf) {
        const auto pos = codes::rv_positions(s, rv);
        std::vector<double> c(pos.size());
        for (std::size_t j = 0; j < pos.size(); ++j) c[j] = code[static_cast<std::size_t>(pos[j])];
        code = std::move(c);
    }
    std::vector<double> post(static_cast<std::size_t>(s.coded_bits));
    for (std::size_t i = 0; i < post.size(); ++i) post[i] = code[s.perm[i]];
    return {std::vector<std::uint8_t>(d.bits.data.begin(), d.bits.data.end()), d.ok[0] != 0, std::move(post)};
}

}  // namespace

bool dd_default() {
    static const bool on = [] {
        const char* v = std::getenv("DATA2G_DD");
        return !v || std::strcmp(v, "1") == 0;
    }();
    return on;
}

double monotonic() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

std::uint32_t mask_value(const MaskId& m) {
    if (m.key == 0) return m.seq == SEQ_MOD + 4 ? 0xC0C0C0u : 0;  // a compact CONNECT's (link COMPACT_CONNECT)
    const std::uint8_t b[4] = {static_cast<std::uint8_t>(m.key >> 8), static_cast<std::uint8_t>(m.key & 255),
                               static_cast<std::uint8_t>(m.direction), static_cast<std::uint8_t>(m.seq)};
    const std::uint32_t v = codes::crc32(b);
    return v ? v : 1;
}

std::vector<double> tx_audio(const TxBurst& burst) {
    const Mode& md = mode_at(burst.submode);
    const std::size_t n = burst.slots.size();
    if (md.is_cpm()) {
        // control slots (masks >= SEQ_MOD) in the grid's short codeword; two
        // of them are one codeword at RV 0 and 1 (ARQ_DUP)
        const auto n_ctl = std::count_if(burst.slots.begin(), burst.slots.end(),
                                         [](const Slot& s) { return s.mask_id.seq >= SEQ_MOD; });
        std::vector<std::vector<std::uint8_t>> coded;
        for (std::size_t i = 0; i < n; ++i) {
            const auto& s = burst.slots[i];
            coded.push_back(codes::encode(static_cast<std::ptrdiff_t>(i) < n_ctl ? ctl_code(md) : data_code(md),
                                          s.payload, s.rv, mask_value(s.mask_id), static_cast<int>(i)));
        }
        // the bandwidth cap's TX filter (cpm.bandpass): overshoot as CLIP_OVERSHOOT, its last factor repeated
        const auto& g = cpm::grid_of(*md.cpm);
        const auto& f = cpm::tx_filter(g, burst.cap);
        std::vector<double> overshoot(config::CLIP_OVERSHOOT.begin(), config::CLIP_OVERSHOOT.end());
        overshoot.resize(static_cast<std::size_t>(f.passes), config::CLIP_OVERSHOOT.back());
        return waveform::tx_condition(cpm::modulate(*md.cpm, coded, n_ctl == 2, burst.cap), g.clip_db, overshoot, 0,
                                      std::numeric_limits<std::size_t>::max(),
                                      {g.f0 - f.bp, g.f0 + (g.m - 1) * g.rate + f.bp});
    }
    const auto& s = codes::spec(*md.ofdm);
    std::vector<std::uint8_t> bits;
    for (std::size_t i = 0; i < n; ++i) {
        const auto c = codes::encode(s, burst.slots[i].payload, burst.slots[i].rv, mask_value(burst.slots[i].mask_id),
                                     static_cast<int>(i));
        bits.insert(bits.end(), c.begin(), c.end());
    }
    return modem::modulate_bits(codes::spread<std::uint8_t>(bits, static_cast<int>(n), md.ofdm->bits_per_cu), *md.ofdm);
}

const Mode& Heard::mode() const { return mode_at(ofdm ? ofdm->spec->name : cpm->spec->name); }
int Heard::n_cw() const { return ofdm ? ofdm->n_cw : static_cast<int>(cpm->soft.size()); }

std::shared_ptr<const SlotSoft> soft_bits(const Heard& r) {
    if (r.cpm) return std::make_shared<const SlotSoft>(r.cpm->soft);
    const auto& R = *r.ofdm;
    auto var = modem::noise_var(R.est.h, R.est);
    for (std::size_t i = 0; i < var.data.size(); ++i) var.data[i] += R.est.mse.data[i];
    const auto burst = modem::soft_bits(R.raw, R.est.h, var, *R.spec);
    const auto flat = codes::despread<double>(burst, R.n_cw, R.spec->bits_per_cu);
    const std::size_t n = flat.size() / static_cast<std::size_t>(R.n_cw);
    auto out = std::make_shared<SlotSoft>();
    for (int c = 0; c < R.n_cw; ++c) out->emplace_back(flat.begin() + c * n, flat.begin() + (c + 1) * n);
    return out;
}

Measured measure(const Heard& r) {
    Measured out;
    if (r.cpm) {
        const auto& g = cpm::grid_of(*r.cpm->spec);
        const auto m = cpm::measure(g, r.cpm->E, static_cast<int>(r.cpm->E.rows));
        out.snr_est = m.snr_est;
        out.spread_est = m.spread_est;
        out.frames = m.frames;
        std::vector<cd> h(m.snr.size());
        for (std::size_t i = 0; i < h.size(); ++i) h[i] = std::sqrt(m.snr[i]);
        const std::vector<double> ones(h.size(), 1.0);
        for (std::size_t c = 0; c < CONSTS.size(); ++c) out.mi[c] = effective_mi(h, ones, CONSTS[c]);
        return out;
    }
    // effective MI against thermal noise and estimation error, not the
    // transmitter's clip noise (it belongs to the submode sent)
    const auto& R = *r.ofdm;
    const auto& est = R.est;
    std::vector<double> var(est.mse.data.size());
    for (std::size_t i = 0; i < var.size(); ++i) var[i] = est.n0 + est.mse.data[i];
    const int nc = modem::band(R.band).nc;
    out.snr_est = 10 * std::log10(est.p_sig / est.n0 * nc * config::RS / config::SNR_REF_BW_HZ);
    out.spread_est = est.spread_hz;
    out.delay_est_ms = static_cast<double>(R.support.second - R.support.first) / config::FS * 1000;
    out.headroom = R.spec->headroom;
    out.frames = R.n_cw * R.spec->frames_per_cw;
    for (std::size_t c = 0; c < CONSTS.size(); ++c) out.mi[c] = effective_mi(est.h.data, var, CONSTS[c]);
    return out;
}

// --- ModemRx -------------------------------------------------------------------

ModemRx::ModemRx(Heard r, SoftStore* store, std::optional<double> dd_budget, std::shared_ptr<const SlotSoft> soft,
                 bool dd, Clock clock)
    : r_(std::move(r)), store_(store), soft_(soft ? std::move(soft) : soft_bits(r_)), dd_(dd), clock_(std::move(clock)) {
    if (dd_budget) dd_until_ = clock_() + *dd_budget;
    const Mode& md = r_.mode();
    submode_ = std::string(md.name);
    n_cw_ = r_.n_cw();
    n_ctl_slots_ = r_.cpm ? 1 + r_.cpm->dup : 0;
    data_spec_ = &data_code(md);
    ctl_spec_ = &ctl_code(md);
}

const codes::Spec& ModemRx::spec(int slot) const { return slot < n_ctl_slots_ ? *ctl_spec_ : *data_spec_; }

bool ModemRx::dd(const codes::Spec& s) const {
    return dd_ && r_.ofdm && r_.ofdm->hp.rows && !s.polar && !n_ctl_slots_;
}

bool ModemRx::late() const { return dd_until_ && clock_() >= *dd_until_; }

// The slot's soft bits on the estimate in use. A refined estimate's are
// made per slot as asked: codes::spread deals codeword symbols round-robin,
// so the slot's are every n_cw-th of the burst's.
const std::vector<double>& ModemRx::slot_soft(int slot) {
    if (!cur_.est) return (*soft_)[static_cast<std::size_t>(slot)];
    auto& cache = *cur_.soft;
    auto it = cache.find(slot);
    if (it != cache.end()) return it->second;
    const auto& R = *r_.ofdm;
    const auto& est = *cur_.est;
    const std::size_t nc = R.raw.cols, total = est.h.data.size(), n = static_cast<std::size_t>(n_cw_);
    std::vector<cd> y, h;
    std::vector<double> var;
    for (std::size_t j = static_cast<std::size_t>(slot); j < total; j += n) {
        const std::size_t row = j / nc, c = j % nc;
        const cd hj = est.h.data[j];
        y.push_back(R.raw[row / S * config::SYMS_PER_FRAME + 1 + row % S][c]);
        h.push_back(hj);
        var.push_back((est.n0_k.empty() ? est.n0 : est.n0_k[c]) + est.clip_ratio * std::norm(hj) + est.mse.data[j]);
    }
    return cache[slot] = constellation::llr(y, h, var, *constellation::find(R.spec->constellation));
}

std::optional<Bytes> ModemRx::decode(int slot, const MaskId& mask, int rv, const SoftKey* key) {
    if (!key) return decode_plain(slot, mask);
    std::optional<SoftEntry> stored;
    if (store_) {
        auto it = store_->find(*key);
        if (it != store_->end()) stored = it->second;
    }
    auto out = decode_stored(slot, mask, rv, stored);
    if (!out && store_ && stored) (*store_)[*key] = std::move(*stored);
    return out;
}

void ModemRx::forget(const SoftKey& key) {
    if (store_) store_->erase(key);
}

std::optional<Bytes> ModemRx::decode_plain(int slot, const MaskId& mask) {
    if (slot >= n_cw_) return std::nullopt;
    // CPM: its header says which slots are control (their own short
    // codeword); a blind ARQ_DUP pair probe on a data slot is a miss
    if (n_ctl_slots_ && (mask.seq >= SEQ_MOD) != (slot < n_ctl_slots_)) return std::nullopt;
    const std::uint32_t m = mask_value(mask);
    auto it = memo_.find({slot, m});
    if (it != memo_.end()) return it->second;
    const auto& s = spec(slot);
    const Raw& d = decoded(slot, s);
    return memo_[{slot, m}] = codes::check(s, d.cands, d.usable, m);
}

std::vector<Bytes> ModemRx::raw(int slot) {
    if (slot >= n_cw_) return {};
    const auto& s = spec(slot);
    const Raw& d = decoded(slot, s);
    const std::size_t k = static_cast<std::size_t>(s.k), nb = static_cast<std::size_t>(s.payload_bytes);
    std::vector<Bytes> out;
    for (std::size_t l = 0; l < d.usable.size(); ++l) {
        if (!d.usable[l]) continue;
        Bytes p(nb, 0);
        for (std::size_t i = 0; i < 8 * nb; ++i) p[i / 8] |= static_cast<std::uint8_t>((d.cands[l * k + i] & 1) << (7 - i % 8));
        out.push_back(std::move(p));
    }
    return out;
}

// The slot decoded alone, mask left open, with DD while it fails to
// converge. A converged codeword is what was on air, whoever it was for: DD
// learns from it.
const ModemRx::Raw& ModemRx::decoded(int slot, const codes::Spec& s) {
    auto it = raw_.find(slot);
    if (it != raw_.end()) return it->second;
    const DdState saved = cur_;
    Raw out;
    for (int i = 0; i <= DD_ITERS; ++i) {
        if (!dd(s)) {
            const int idx[1] = {slot};
            auto d = codes::decode_raw(s, row_f(slot_soft(slot)), idx);
            out = {std::move(d.cands.data), std::move(d.usable.data)};
            break;
        }
        auto p = decode_post(s, nullptr, 0, 0, slot_soft(slot));
        out = {codes::descramble(s, p.bits, slot), {static_cast<std::uint8_t>(p.ok)}};
        if (p.ok) {
            Mat<std::uint8_t> b(1, p.bits.size());
            b.data = p.bits;
            post_[slot] = known(codes::encode_info(s, b, 0).data);
            break;
        }
        if (i == DD_ITERS || late()) break;
        refine(slot, s, std::move(p.post));
    }
    if (std::none_of(out.usable.begin(), out.usable.end(), [](auto u) { return u != 0; })) undo(slot, saved);
    return raw_[slot] = std::move(out);
}

std::optional<Bytes> ModemRx::decode_stored(int slot, const MaskId& mask, int rv, std::optional<SoftEntry>& stored) {
    if (slot >= n_cw_) return std::nullopt;
    if (n_ctl_slots_ && (mask.seq >= SEQ_MOD) != (slot < n_ctl_slots_)) return std::nullopt;
    const std::uint32_t m = mask_value(mask);
    const auto& s = spec(slot);
    if (stored && stored->submode != submode_) {  // a resend in another mode: what was kept can't combine
        log_write("data2g.arq.phy", 30,
                  format("soft bits stored in %s (slot %d rv %d), resent in %s slot %d rv %d: dropped",
                         stored->submode.c_str(), stored->slot, stored->rv, submode_.c_str(), slot, rv));
        stored.reset();
    }
    const int top = std::max(stored ? stored->top : 0, rv);
    const DdState saved = cur_;
    // the buffer holds unscrambled soft bits: each slot's flipped by its own
    // scrambling, so a resend in any slot combines
    const auto fl = codes::flip(s, slot, rv);
    const int rvs[1] = {rv};
    auto combined = [&] {
        Mat<double> buf;
        if (stored) {
            buf = Mat<double>(1, stored->buf.size());
            buf.data = stored->buf;
        }
        const auto& soft = slot_soft(slot);
        Mat<double> x(1, soft.size());
        for (std::size_t i = 0; i < soft.size(); ++i) x.data[i] = fl[i] * soft[i];
        codes::combine(s, buf, x, rvs);
        return std::move(buf.data);
    };
    std::vector<double> buf;
    const std::uint32_t masks[1] = {m};
    const int plain[1] = {codes::PLAIN};
    for (int i = 0; i <= DD_ITERS; ++i) {
        buf = combined();
        if (!dd(s)) {
            Mat<double> b(1, buf.size());
            b.data = buf;
            auto p = codes::decode_buffer(s, b, top, masks, plain)[0];
            if (p.ok) return std::move(p.data);
            break;
        }
        auto d = decode_post(s, &buf, top, rv, {});
        Mat<std::uint8_t> bits(1, d.bits.size());
        bits.data = d.bits;
        const std::uint8_t conv[1] = {static_cast<std::uint8_t>(d.ok)};
        auto p = codes::payloads(s, bits, conv, masks, plain)[0];
        if (p.ok) {
            learn(slot, s, p.data, rv, m);
            return std::move(p.data);
        }
        if (i == DD_ITERS || late()) break;
        for (std::size_t j = 0; j < d.post.size(); ++j) d.post[j] *= fl[j];  // back to the bits as sent
        refine(slot, s, std::move(d.post));
    }
    if (cur_.est != saved.est) {
        undo(slot, saved);
        buf = combined();
    }
    stored = SoftEntry{std::move(buf), top, submode_, slot, rv, mask};
    return std::nullopt;
}

void ModemRx::learn(int slot, const codes::Spec& s, const Bytes& payload, int rv, std::uint32_t m) {
    if (dd(s)) post_[slot] = known(codes::encode(s, payload, rv, m, slot));
}

// DD after a failed decode of `slot`: its posterior joins the other slots'
// as soft pilots, the channel is re-estimated and the soft bits remade.
void ModemRx::refine(int slot, const codes::Spec& s, std::vector<double> post) {
    post_[slot] = std::move(post);
    if (!blind_) {
        // RV 0 slots the link has not asked about yet: parity checks alone
        // say they decoded (a resend at another RV simply fails).
        // split: one batch here, as Python (an LDPC batch stops together);
        // per-codeword decodes would parallelize at a small cost in parity.
        blind_ = true;
        std::vector<int> todo;
        for (int i = 0; i < n_cw_; ++i)
            if (!post_.count(i)) todo.push_back(i);
        if (!todo.empty()) {
            Mat<float> llr(todo.size(), static_cast<std::size_t>(s.coded_bits));
            for (std::size_t t = 0; t < todo.size(); ++t) {
                const auto& v = (*soft_)[static_cast<std::size_t>(todo[t])];
                std::copy(v.begin(), v.end(), llr[t]);
            }
            const auto info = codes::decode_llrs(s, llr);
            for (std::size_t t = 0; t < todo.size(); ++t)
                if (info.ok[t]) {
                    Mat<std::uint8_t> b(1, info.bits.cols);
                    std::copy_n(info.bits[t], info.bits.cols, b.data.begin());
                    post_[todo[t]] = known(codes::encode_info(s, b, 0).data);
                }
        }
    }
    cur_ = {dd_estimate(), std::make_shared<std::map<int, std::vector<double>>>()};
}

// A failed decode's posterior can be confidently wrong, so neither it nor
// the estimate made from it outlives the attempt.
void ModemRx::undo(int slot, const DdState& saved) {
    post_.erase(slot);
    cur_ = saved;
}

// phy._dd_estimate: the burst's estimate re-made by equalizer::refine with
// post_'s soft symbols as pilots.
ModemRx::Est ModemRx::dd_estimate() const {
    const auto& R = *r_.ofdm;
    const auto& spec = *R.spec;
    const auto& est = R.est;
    const auto& pts = *constellation::find(spec.constellation);
    const int m = pts.m;
    const std::size_t M = pts.points.size(), cb = static_cast<std::size_t>(spec.coded_bits);
    std::vector<double> llr(static_cast<std::size_t>(n_cw_) * cb, 0.0);
    std::vector<std::uint8_t> have(llr.size(), 0);
    for (const auto& [s, v] : post_) {
        std::copy(v.begin(), v.end(), llr.begin() + static_cast<std::ptrdiff_t>(s * cb));
        std::fill_n(have.begin() + static_cast<std::ptrdiff_t>(s * cb), cb, 1);
    }
    const auto L = codes::spread<double>(llr, n_cw_, m);
    const auto K = codes::spread<std::uint8_t>(have, n_cw_, m);
    const std::size_t nc = R.raw.cols, rows = est.h.rows, n_sym = rows * nc;
    const double g = est.gain, p = est.p_sig;
    Mat<cd> z(rows, nc);
    Mat<double> w(rows, nc);
    std::vector<double> la1(static_cast<std::size_t>(m)), la0(static_cast<std::size_t>(m)), lp(M), prob(M);
    SymCache& cache = sym_cache_;
    if (cache.valid.size() != n_sym) {
        cache.L.assign(n_sym * static_cast<std::size_t>(m), 0.0);
        cache.z.assign(n_sym, 0.0);
        cache.w.assign(n_sym, 0.0);
        cache.valid.assign(n_sym, 0);
    }
    // split: per symbol, independent
    for (std::size_t j = 0; j < n_sym; ++j) {
        // no posterior for this symbol's codeword: no soft pilot (z, w stay 0), whatever the math says
        if (!K[j * m]) continue;
        const std::size_t row = j / nc, c = j % nc;
        double* cL = &cache.L[j * static_cast<std::size_t>(m)];
        if (cache.valid[j] && std::equal(cL, cL + m, &L[j * m])) {
            z[row][c] = cache.z[j];
            w[row][c] = cache.w[j];
            continue;
        }
        std::copy(&L[j * m], &L[j * m] + m, cL);
        cache.valid[j] = 1;
        cache.z[j] = 0.0;
        cache.w[j] = 0.0;
        // log P(label) per point, labels MSB first (modulate's order)
        for (int b = 0; b < m; ++b) {
            const double l = std::clamp(L[j * m + b], -30.0, 30.0);
            la1[b] = -std::log1p(std::exp(-std::abs(l))) - std::max(l, 0.0);   // -logaddexp(0, l)
            la0[b] = -std::log1p(std::exp(-std::abs(l))) - std::max(-l, 0.0);  // -logaddexp(0, -l)
        }
        double top = -INFINITY;
        for (std::size_t a = 0; a < M; ++a) {
            double s1 = 0.0, s0 = 0.0;
            for (int b = 0; b < m; ++b) {
                if ((a >> (m - 1 - b)) & 1)
                    s1 += la1[b];
                else
                    s0 += la0[b];
            }
            lp[a] = s1 + s0;
            top = std::max(top, lp[a]);
        }
        double sum = 0.0;
        for (std::size_t a = 0; a < M; ++a) sum += prob[a] = std::exp(lp[a] - top);
        cd x = 0.0;
        double e2 = 0.0;
        for (std::size_t a = 0; a < M; ++a) {
            prob[a] /= sum;
            x += prob[a] * pts.points[a];
            e2 += prob[a] * std::norm(pts.points[a]);
        }
        const double v = e2 - std::norm(x);
        if (!(std::abs(x) > 1e-3)) continue;
        cache.z[j] = R.raw[row / S * config::SYMS_PER_FRAME + 1 + row % S][c] / (g * x);
        const double n0 = est.n0_k.empty() ? est.n0 : est.n0_k[c];
        cache.w[j] = g * g * std::norm(x) / (n0 + est.clip_ratio * (g * g) * p + g * g * p * v);
        z[row][c] = cache.z[j];
        w[row][c] = cache.w[j];
    }
    const std::size_t n_f = rows / S;
    Mat<double> t_rows(n_f, S);
    for (std::size_t f = 0; f < n_f; ++f) {
        const std::size_t air = f + (R.kc && static_cast<int>(f) >= *R.kc ? 1 : 0);
        for (std::size_t s = 0; s < S; ++s)
            t_rows[f][s] = static_cast<double>((air * config::SYMS_PER_FRAME + s + 1) * config::NSYM) / config::FS;
    }
    std::vector<double> t_pilot(R.hp.rows);
    for (std::size_t i = 0; i < t_pilot.size(); ++i)
        t_pilot[i] = static_cast<double>(i * config::SYMS_PER_FRAME * config::NSYM) / config::FS;
    auto [h, mse] = equalizer::refine(R.hp, t_pilot, z, w, t_rows, R.support, est.p_sig, est.spread_hz, est.n0,
                                      equalizer::bb(modem::band(spec.band)));
    auto out = std::make_shared<modem::DataEstimate>(est);
    for (std::size_t i = 0; i < h.data.size(); ++i) {
        out->h.data[i] = g * h.data[i];
        out->mse.data[i] = g * g * mse.data[i];
    }
    return out;
}

}  // namespace data2g::arq
