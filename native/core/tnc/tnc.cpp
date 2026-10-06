#include "tnc/tnc.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <string>

#include "dsp/dsp.hpp"
#include "dsp/fft.hpp"
#include "waveform/dsp.hpp"

namespace data2g::tnc {

using std::int64_t;

// --- KISS -------------------------------------------------------------------

Bytes kiss_encode(std::span<const std::uint8_t> data, int port, int cmd) {
    Bytes out{FEND};
    for (int i = -1; i < static_cast<int>(data.size()); ++i) {
        const std::uint8_t b = i < 0 ? static_cast<std::uint8_t>(port << 4 | cmd) : data[static_cast<std::size_t>(i)];
        if (b == FESC) out.insert(out.end(), {FESC, TFESC});
        else if (b == FEND) out.insert(out.end(), {FESC, TFEND});
        else out.push_back(b);
    }
    out.push_back(FEND);
    return out;
}

std::vector<std::pair<int, Bytes>> KissDecoder::feed(std::span<const std::uint8_t> data) {
    std::vector<std::pair<int, Bytes>> out;
    for (const std::uint8_t b : data) {
        if (b == FEND) {
            if (!buf_.empty()) out.emplace_back(buf_[0], Bytes(buf_.begin() + 1, buf_.end()));
            buf_.clear();
            esc_ = false;
        } else if (esc_) {
            buf_.push_back(b == TFEND ? FEND : b == TFESC ? FESC : b);
            esc_ = false;
        } else if (b == FESC) {
            esc_ = true;
        } else {
            buf_.push_back(b);
        }
    }
    return out;
}

// --- framing ----------------------------------------------------------------

int capacity(const modem::Spec& spec, int max_cw) { return max_cw * spec.payload_bytes; }

std::vector<Bytes> pack(const std::vector<Bytes>& packets, const modem::Spec& spec) {
    const std::size_t p = static_cast<std::size_t>(spec.payload_bytes);
    Bytes stream;
    for (const auto& x : packets) {
        if (x.size() > 0xFFFF) throw std::invalid_argument("packet longer than 65535 bytes");  // struct.error in Python
        stream.push_back(static_cast<std::uint8_t>(x.size() >> 8));
        stream.push_back(static_cast<std::uint8_t>(x.size() & 0xFF));
        stream.insert(stream.end(), x.begin(), x.end());
    }
    const auto cap = static_cast<std::size_t>(capacity(spec));
    if (stream.size() > cap)
        throw std::invalid_argument(std::to_string(stream.size()) + " bytes exceed a burst's " + std::to_string(cap));
    stream.resize((stream.size() + p - 1) / p * p, 0);
    std::vector<Bytes> out;
    for (std::size_t i = 0; i < stream.size(); i += p) out.emplace_back(stream.begin() + i, stream.begin() + i + p);
    return out;
}

std::pair<std::vector<Bytes>, int> unpack(const std::vector<Bytes>& payloads, const std::vector<bool>& ok) {
    Bytes stream;
    std::vector<bool> good;
    for (std::size_t i = 0; i < payloads.size(); ++i) {
        stream.insert(stream.end(), payloads[i].begin(), payloads[i].end());
        good.insert(good.end(), payloads[i].size(), ok.at(i));
    }
    const auto all_good = [&](std::size_t a, std::size_t b) {
        return std::all_of(good.begin() + a, good.begin() + std::min(b, good.size()), [](bool g) { return g; });
    };
    std::vector<Bytes> out;
    int lost = 0;
    std::size_t pos = 0;
    while (pos + 2 <= stream.size()) {
        if (!all_good(pos, pos + 2)) {
            ++lost;  // at least this one; the rest can not be found
            break;
        }
        const std::size_t n = static_cast<std::size_t>(stream[pos]) << 8 | stream[pos + 1];
        if (n == 0 || pos + 2 + n > stream.size()) break;
        if (all_good(pos + 2, pos + 2 + n)) out.emplace_back(stream.begin() + pos + 2, stream.begin() + pos + 2 + n);
        else ++lost;
        pos += 2 + n;
    }
    return {out, lost};
}

// --- receive ----------------------------------------------------------------

namespace {

const cpm::Grid& grid_named(std::string_view name) {
    const auto* g = cpm::grid(name);
    if (!g) throw std::out_of_range("no CPM grid " + std::string(name));
    return *g;
}

int64_t grid_span(const cpm::Grid& g) {
    const auto lay = cpm::layout(g, cpm::stream_symbols(g, 0, false));
    return static_cast<int64_t>(lay.hdr_rows[0].back() + 2) * g.T;
}

double mean(std::span<const double> v) { return dsp::pairwise_sum(v) / static_cast<double>(v.size()); }

}  // namespace

int64_t search_span(std::span<const std::string_view> bands, std::span<const std::string_view> cpm_grids) {
    int64_t n = 0;
    for (const auto b : bands) n = std::max<int64_t>(n, modem::head_samples(b));
    for (const auto g : cpm_grids) n = std::max(n, grid_span(grid_named(g)));
    return n;
}

std::optional<Rx> receive_any(std::span<const double> y, int64_t lead,
                              std::optional<std::vector<std::string_view>> cpm_grids) {
    std::vector<std::string_view> grids;
    if (cpm_grids) grids = *cpm_grids;
    else
        for (const auto& g : tables::CPM_GRIDS) grids.push_back(g.name);
    std::vector<std::string_view> all_bands;
    for (const auto& b : config::BANDS) all_bands.push_back(b.name);
    const int64_t head = lead + search_span(all_bands, grids);
    const auto len = static_cast<int64_t>(y.size());
    std::optional<modem::Received> r;
    try {
        r = modem::receive(y, {}, nullptr, std::min(head, len));
        if (r->score >= Receiver::SUSPECT_SCORE) return Rx(std::move(*r));
    } catch (const modem::SyncError&) {
        r.reset();
    }
    for (const auto g : grids) {
        const auto lock = cpm::find(grid_named(g), y.first(static_cast<std::size_t>(std::clamp<int64_t>(head, 0, len))));
        if (lock) return Rx(cpm::receive(y, *lock));  // a CPM lock beats a suspect OFDM header
    }
    if (!r) {
        // no preamble: the header copy of a burst whose head faded (as Receiver's find_copy)
        std::optional<modem::Lock> best;
        for (const auto b : config::HEADER_COPY_BANDS) {
            const auto c = modem::find_copy(y, b);
            if (c && c->start <= lead && c->end <= len && (!best || c->score > best->score)) best = c;
        }
        if (best) {
            try {
                return Rx(modem::receive(y, {}, nullptr, std::nullopt, &*best));
            } catch (const modem::SyncError&) {
            }
        }
    }
    if (r) return Rx(std::move(*r));
    return std::nullopt;
}

int64_t Pending::start() const { return is_cpm() ? cpm().start : ofdm().start; }
int64_t Pending::end() const { return is_cpm() ? cpm().end : ofdm().end; }
double Pending::score() const { return is_cpm() ? cpm().score : ofdm().score; }
int Pending::n_cw() const { return is_cpm() ? 1 + cpm().dup + cpm().n_data : ofdm().n_cw; }

void Pending::shift(int64_t d) {
    if (is_cpm()) {
        auto& l = std::get<1>(lock);
        l.start += d;
        l.end += d;
    } else {
        auto& l = std::get<0>(lock);
        l.start += d;
        l.end += d;
        l.p0 += d;
        if (l.copy) l.copy->pc += d;
    }
}

BurstEvent decode(DecodeRequest req, const modem::Accept& accept) {
    BurstEvent ev{req.header, std::nullopt, {}};
    Pending p = req.header;
    p.shift(-req.at);
    if (p.is_cpm()) {
        ev.rx = cpm::receive(req.seg, p.cpm());
    } else {
        try {
            if (p.copy()) {
                ev.rx = modem::receive(req.seg, {}, &accept, std::nullopt, &p.ofdm());
            } else {
                const std::string_view band[] = {p.ofdm().band};
                ev.rx = modem::receive(req.seg, band, &accept, std::min<int64_t>(req.head, static_cast<int64_t>(req.seg.size())));
            }
        } catch (const modem::SyncError&) {
            // the burst is lost: rx stays empty
        }
    }
    if (req.audio_lo == 0 && req.audio_hi == static_cast<int64_t>(req.seg.size())) ev.audio = std::move(req.seg);
    else ev.audio.assign(req.seg.begin() + req.audio_lo, req.seg.begin() + req.audio_hi);
    return ev;
}

Receiver::Receiver(modem::Accept accept, std::vector<std::string_view> cpm_grids, bool blank)
    : accept_(std::move(accept)), bands_(accept_.bands()) {
    for (const auto g : cpm_grids) grids_.push_back(&grid_named(g));
    // the least audio worth searching (a whole preamble and header), and
    // what a trim keeps (so one still arriving survives it)
    min_search_ = search_span(bands_, cpm_grids);
    keep_ = min_search_ + config::FS;
    detectors_.reserve(bands_.size());
    for (const auto b : bands_) detectors_.push_back({b, waveform::StreamDetector(waveform::band(b))});
    if (blank) blanker_.emplace();
    reset();
}

// --- NoiseProfile ----------------------------------------------------------------------

NoiseProfile::NoiseProfile() : win_(BLOCK) {
    for (int n = 0; n < BLOCK; ++n) win_[n] = 0.5 - 0.5 * std::cos(2 * std::numbers::pi * n / (BLOCK - 1));  // np.hanning
}

void NoiseProfile::feed(std::span<const double> x, double t_start) {
    const auto s = static_cast<std::int64_t>(std::nearbyint(t_start * config::FS));
    if (!buf_.empty() && s0_ + static_cast<std::int64_t>(buf_.size()) == s) {
        buf_.insert(buf_.end(), x.begin(), x.end());
    } else {
        buf_.assign(x.begin(), x.end());
        s0_ = s;
    }
    std::size_t off = 0;
    std::vector<dsp::cdouble> z(BLOCK);
    while (buf_.size() - off >= static_cast<std::size_t>(BLOCK)) {
        for (int n = 0; n < BLOCK; ++n) z[n] = buf_[off + n] * win_[n];
        const auto spec = dsp::fft(z, true);
        Block b{s0_, s0_ + BLOCK, {}, 0};
        {  // impulses: 10 ms pieces whose peak is over IMPULSE_X x the median piece RMS
            std::vector<double> rms, peak;
            for (int q = 0; q < BLOCK; q += PIECE) {
                double ss = 0.0, pk = 0.0;
                for (int n = 0; n < PIECE; ++n) {
                    const double v = buf_[off + q + n];
                    ss += v * v;
                    pk = std::max(pk, std::fabs(v));
                }
                rms.push_back(std::sqrt(ss / PIECE));
                peak.push_back(pk);
            }
            const double ref = dsp::quantile(rms, 0.5);
            if (ref > 0)
                for (const double pk : peak) b.impulses += pk > IMPULSE_X * ref;
        }
        for (std::size_t i = 0; i < BANDS_HZ.size(); ++i) {
            // rfft bins k (k * FS / BLOCK Hz) with lo <= f < hi
            const int bin_hz = config::FS / BLOCK;
            const int k0 = (BANDS_HZ[i].first + bin_hz - 1) / bin_hz, k1 = (BANDS_HZ[i].second + bin_hz - 1) / bin_hz;
            double sum = 0.0;
            for (int k = k0; k < k1; ++k) sum += std::norm(spec[static_cast<std::size_t>(k)]);
            b.p[i] = sum / (k1 - k0);
        }
        pending_.push_back(b);
        off += BLOCK;
        s0_ += BLOCK;
    }
    buf_.erase(buf_.begin(), buf_.begin() + static_cast<std::ptrdiff_t>(off));
    const std::int64_t now = s + static_cast<std::int64_t>(x.size());
    const auto commit = static_cast<std::int64_t>(std::nearbyint(COMMIT_S * config::FS));
    while (!pending_.empty() && pending_.front().end <= now - commit) {
        const Block b = pending_.front();
        pending_.pop_front();
        const bool marked = std::any_of(busy_.begin(), busy_.end(),
                                        [&](const auto& m) { return m.first < b.end && b.start < m.second; });
        if (!marked) {
            kept_.push_back(b.p);
            kept_impulses_.push_back(b.impulses);
            if (kept_.size() > WINDOW) kept_.pop_front(), kept_impulses_.pop_front();
        }
    }
    std::erase_if(busy_, [&](const auto& m) { return m.second <= now - 2 * commit; });
}

void NoiseProfile::mark(double start, double end) {
    busy_.emplace_back(static_cast<std::int64_t>(std::nearbyint(start * config::FS)),
                       static_cast<std::int64_t>(std::nearbyint(end * config::FS)));
}

std::optional<NoiseSnapshot> NoiseProfile::snapshot() const {
    if (kept_.size() < MIN_BLOCKS) return std::nullopt;
    NoiseSnapshot out;
    out.blocks = static_cast<int>(kept_.size());
    double imp = 0.0;
    for (const int n : kept_impulses_) imp += n;
    out.impulses_per_min = 60.0 * imp / (static_cast<double>(kept_.size()) * BLOCK / config::FS);
    for (std::size_t i = 0; i < BANDS_HZ.size(); ++i) {
        std::vector<double> v;
        v.reserve(kept_.size());
        for (const auto& p : kept_) v.push_back(p[i]);
        const double med = std::max(dsp::quantile(v, 0.5), 1e-30);  // digital silence
        out.db[i] = 10 * std::log10(med);
        out.tail_db[i] = 10 * std::log10(dsp::quantile(v, 0.9) / med);
    }
    return out;
}

void Receiver::reset() {
    buf_.clear();
    off_ = 0;
    pending_.reset();
    pilots_ok_ = confirmed_ = false;
    powers_.clear();
    ps_ = 0.0;
    pn_ = 0;
    fresh_ = HOP;  // search at once
    last_start_ = -1;
    decided_.clear();
    for (auto& d : detectors_) d.d.reset();
}

bool Receiver::on_air() const {
    if (powers_.size() < 50) return false;
    const double floor = dsp::quantile(std::vector<double>(powers_.begin(), powers_.end()), 0.05);
    const double last[] = {powers_[powers_.size() - 3], powers_[powers_.size() - 2], powers_.back()};
    return mean(last) > floor * std::pow(10.0, ON_AIR_DB / 10);
}

int64_t Receiver::decided(std::string_view key) const {
    const auto it = decided_.find(key);
    return it == decided_.end() ? 0 : it->second;
}

void Receiver::check_pilots() {
    const Pending& p = *pending_;
    if (p.is_cpm()) return;
    modem::Lock l = p.ofdm();
    l.p0 -= off_;  // only p0, as Python's dict(p, p0=...)
    const auto c = modem::pilot_coherence(buf_, l, modem::PILOT_PAIRS, true);
    if (static_cast<int>(c.size()) >= modem::PILOT_PAIRS) {
        const bool ok = mean(c) > modem::pilot_noise(l.spec->band);
        // a copy lock is one header copy: confirmed (no further search) at
        // the single-copy commit score
        confirmed_ = ok && p.score() >= (p.copy() ? modem::COPY_COMMIT_SCORE : SUSPECT_SCORE);
        pilots_ok_ = ok;
    }
}

// Python's buf = buf[-n:] if n < len(buf), off advanced by len(buf) - n.
void Receiver::trim(int64_t n) {
    const int64_t L = len();
    if (n < L) {
        off_ += L - n;
        const int64_t a = n > 0 ? L - n : n == 0 ? 0 : std::min(-n, L);  // buf[-n:]
        buf_.erase(buf_.begin(), buf_.begin() + a);
    }
    for (auto& d : detectors_) d.d.trim(off_);
}

Receiver::Stats Receiver::stats(int64_t w0) {
    Stats out;
    out.mats.reserve(detectors_.size());
    for (auto& [band, d] : detectors_) {
        if (d.fed < off_) {  // its audio was trimmed away: start over
            d.reset();
            d.fed = off_;
        }
        const auto from = static_cast<std::size_t>(d.fed - off_);
        d.feed(waveform::to_baseband(std::span<const double>(buf_).subspan(from), d.fed));
        // no search looks further back than keep_ (the statistic grew with
        // a long burst in the buffer)
        d.trim(off_ + len() - keep_ - d.span);
        const int64_t n = len() - w0 - d.span + 1;
        if (n <= 0) continue;
        const int64_t lo = off_ + w0;
        Mat<double> S = d.stat(lo, lo + n);
        const auto masked = static_cast<std::size_t>(std::clamp<int64_t>(decided(band) - lo, 0, n));
        for (std::size_t r = 0; r < S.rows; ++r) std::fill(S[r], S[r] + masked, -1.0);
        out.mats.push_back(std::move(S));
        out.v.push_back({band, &out.mats.back()});
    }
    return out;
}

void Receiver::searched() {
    const int64_t end = off_ + len();
    for (const auto& det : detectors_) {
        const auto& b = modem::band(det.band);
        decided_[det.band] = std::max(decided(det.band),
                                      end - modem::head_samples(det.band) - modem::preamble_samples(b) - REVISIT);
    }
    for (const auto* g : grids_)
        decided_[g->name] = std::max(decided(g->name), end - grid_span(*g) - g->T);
}

std::optional<modem::Lock> Receiver::find_copy() const {
    std::optional<modem::Lock> best;
    int64_t best_k = 0;
    for (const auto band : config::HEADER_COPY_BANDS) {
        const auto it = std::find_if(detectors_.begin(), detectors_.end(), [&](const Detector& d) { return d.band == band; });
        if (it == detectors_.end()) continue;
        const auto& d = it->d;
        const auto level = d.level();
        if (!level) continue;
        const int64_t a = std::max(d.c0, off_);  // the stream index both the buffer and C cover from
        const auto x = std::span<const double>(buf_).subspan(static_cast<std::size_t>(a - off_));
        const int64_t have = static_cast<int64_t>(d.C(0).size()) - (a - d.c0);
        const auto cols = static_cast<std::size_t>(
            std::max<int64_t>(0, std::min(have, static_cast<int64_t>(x.size()) - config::M + 1)));
        std::vector<const modem::cd*> rows;
        for (std::size_t i = 0; i < d.bins(); ++i) rows.push_back(d.C(i).data() + (a - d.c0));
        const auto lock = modem::find_copy(x, band, &accept_, rows, cols, level);
        if (lock && (!best || lock->score > best->score)) {
            best = lock;
            best_k = a - off_;
        }
    }
    if (best) {
        Pending p{*best};
        p.shift(best_k);
        return p.ofdm();
    }
    return best;
}

std::optional<cpm::Lock> Receiver::find_cpm() const {
    for (const auto* g : grids_) {
        const auto lock = cpm::find(*g, buf_, std::nullopt, 150.0, true, std::max<int64_t>(0, decided(g->name) - off_),
                                    len() - grid_span(*g) + 2 * g->T);
        if (lock) return lock;
    }
    return std::nullopt;
}

bool Receiver::supersede(std::vector<Item>& out, bool whole) {
    fresh_ = 0;
    const Pending p = *pending_;
    if (p.is_cpm()) return false;  // scores aren't comparable across families
    const auto band = p.ofdm().band;
    const int64_t hdr_end = p.start() + modem::preamble_samples(modem::band(band)) + modem::header_samples(band) - off_;
    const int64_t w0 = whole ? hdr_end : std::max(hdr_end, len() - keep_);
    if (len() - w0 < keep_ / 2) return false;
    std::optional<modem::Lock> q;
    try {
        const auto st = stats(w0);
        q = modem::find_burst(std::span<const double>(buf_).subspan(static_cast<std::size_t>(w0)), bands_, &accept_, st.v);
        q->start += w0 + off_;
        q->end += w0 + off_;
        q->p0 += w0 + off_;
    } catch (const modem::SyncError&) {
    }
    searched();
    if (p.copy() && (!q || q->score < p.score() + SUPERSEDE_MARGIN)) {
        // a copy lock can be a copy read off the wrong frame, taken before
        // the burst's own copy arrived: the true one, later, replaces it
        if (auto c = find_copy()) {
            Pending cp{*c};
            cp.shift(off_);
            q = cp.ofdm();
        }
    }
    if (!q || q->score < p.score() + SUPERSEDE_MARGIN || q->start == p.start()) return false;
    if (!whole) {  // completing: the caller has already handled p
        const auto a = static_cast<std::size_t>(std::max<int64_t>(0, p.start() - off_));
        const auto b = static_cast<std::size_t>(std::clamp<int64_t>(q->start - off_, 0, len()));
        out.push_back(BurstEvent{p, std::nullopt, std::vector<double>(buf_.begin() + a, buf_.begin() + std::max(a, b))});
    }
    pending_ = Pending{*q};
    pilots_ok_ = q->copy.has_value() || q->score >= SUSPECT_SCORE;
    confirmed_ = false;
    out.push_back(HeaderEvent{*pending_, off_ + len()});
    return true;
}

std::vector<Event> Receiver::feed(std::span<const double> x) {
    std::vector<Event> out;
    for (auto& it : feed_deferred(x)) {
        if (auto* r = std::get_if<DecodeRequest>(&it)) out.emplace_back(decode(std::move(*r), accept_));
        else if (auto* h = std::get_if<HeaderEvent>(&it)) out.emplace_back(std::move(*h));
        else out.emplace_back(std::move(std::get<BurstEvent>(it)));
    }
    return out;
}

std::vector<Receiver::Item> Receiver::feed_deferred(std::span<const double> x_in) {
    std::vector<double> blanked;
    std::span<const double> x = x_in;
    if (blanker_) {
        blanked = (*blanker_)(x_in);
        x = blanked;
    }
    // 0.1 s block powers for the noise floor (feeds may be any size)
    {
        std::vector<double> sq(x.size());
        for (std::size_t i = 0; i < x.size(); ++i) sq[i] = x[i] * x[i];
        std::span<const double> rest = sq;
        while (!rest.empty()) {
            const auto take = std::min<std::size_t>(rest.size(), static_cast<std::size_t>(config::FS / 10 - pn_));
            ps_ += dsp::pairwise_sum(rest.first(take));
            pn_ += static_cast<int>(take);
            rest = rest.subspan(take);
            if (pn_ == config::FS / 10) {
                powers_.push_back(ps_ / pn_);
                if (powers_.size() > FLOOR_BLOCKS) powers_.pop_front();
                ps_ = 0.0;
                pn_ = 0;
            }
        }
    }
    buf_.insert(buf_.end(), x.begin(), x.end());
    fresh_ += static_cast<int64_t>(x.size());
    std::vector<Item> out;
    while (true) {
        if (!pending_) {
            // a burst's end is checked on every call; new preambles only every HOP samples
            if (len() < min_search_ || fresh_ < HOP) break;
            fresh_ = 0;
            // digital silence has no noise to normalize by, and filter
            // ringing then reads as preambles
            {
                const auto tail = std::span<const double>(buf_).last(static_cast<std::size_t>(min_search_));
                std::vector<double> sq(tail.size());
                for (std::size_t i = 0; i < tail.size(); ++i) sq[i] = tail[i] * tail[i];
                if (std::sqrt(mean(sq)) < SILENCE_RMS) {
                    trim(keep_);
                    for (auto& d : detectors_) {  // no noise level in silence: skip it
                        d.d.reset();
                        d.d.fed = off_ + len();
                    }
                    break;
                }
            }
            std::optional<Pending> p;
            try {
                const auto st = stats();
                p = Pending{modem::find_burst(buf_, bands_, &accept_, st.v)};
                if (p->score() < SUSPECT_SCORE)
                    if (auto c = find_cpm()) p = Pending{*c};  // a strong CPM burst read as a weak OFDM header
            } catch (const modem::SyncError&) {
                if (auto c = find_cpm()) p = Pending{*c};
            }
            searched();
            if (!p)
                if (auto c = find_copy()) p = Pending{*c};  // the preamble faded: the frame pilots and header copy
            if (!p) {
                trim(keep_);
                break;
            }
            if (p->start() + off_ <= last_start_) break;  // the suspect burst just handled, still in the kept window
            p->shift(off_);
            pending_ = p;
            // BUSY at once on a clear header (or a copy lock: its pilots
            // passed already); a suspect one waits for its pilots
            pilots_ok_ = p->is_cpm() || p->copy() || p->score() >= SUSPECT_SCORE;
            confirmed_ = false;
            out.push_back(HeaderEvent{*p, off_ + len()});
        }
        const Pending p = *pending_;
        if (off_ + len() < p.end() + config::LEADIN_SAMPLES) {
            if (fresh_ >= HOP) {
                check_pilots();
                // the later-header search is for false locks: a confirmed burst skips it
                if (confirmed_) fresh_ = 0;
                else if (supersede(out)) continue;
            }
            break;
        }
        DecodeRequest req;
        req.header = p;
        if (p.is_cpm()) {
            req.seg = buf_;
            req.at = off_;
            req.audio_lo = std::max<int64_t>(0, p.start() - off_);
            req.audio_hi = std::max(req.audio_lo, std::min(p.end() - off_, len()));
            out.push_back(std::move(req));
            last_start_ = p.start();
            trim(len() - (p.end() - off_));
            pending_.reset();
            continue;
        }
        const int64_t s0 = std::max<int64_t>(0, p.start() - config::LEADIN_SAMPLES - off_);
        const int64_t s1 = std::clamp<int64_t>(p.end() + config::LEADIN_SAMPLES - off_, s0, len());
        req.seg.assign(buf_.begin() + s0, buf_.begin() + s1);
        req.at = off_ + s0;
        req.head = p.start() - off_ - s0 + modem::head_samples(p.ofdm().band) + config::NSYM + 3 * config::M;
        req.audio_hi = s1 - s0;
        out.push_back(std::move(req));
        last_start_ = p.start();
        // a better header inside this burst's span (it was a false lock, and
        // the real one began before it ended) is next, not trimmed away
        if (!confirmed_ && supersede(out, true)) {
            trim(len() - std::max<int64_t>(0, pending_->start() - config::LEADIN_SAMPLES - off_));
            continue;
        }
        // a suspect burst may have claimed an end past a real preamble that
        // began inside it: keep the last search window (a confident one
        // keeps nothing of itself)
        int64_t cut = p.end() - off_;
        if (p.score() < SUSPECT_SCORE) cut = std::min(cut, len() - keep_);
        trim(len() - std::max<int64_t>(0, cut));
        pending_.reset();
    }
    return out;
}

}  // namespace data2g::tnc
