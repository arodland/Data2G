// data2g/receiver.py: the live receive front end. Receiver: feed() audio
// at FS as it arrives, get events back: a burst's header as it is found
// (preamble search over the StreamDetectors' statistics, a CPM early lock,
// or the header copy of a burst whose head faded), BUSY while it arrives,
// then the whole burst's receive. feed_deferred() leaves each receive as a
// DecodeRequest, so arq::Engine can run decode() on its worker. Also the
// offline search (search_span, receive_any). NoiseProfile is
// noise_profile.hpp.
//
// Positions in events are stream sample indices (samples fed since reset).
#pragma once

#include <cstdint>
#include <deque>
#include <map>
#include <optional>
#include <span>
#include <string_view>
#include <variant>
#include <vector>

#include "audio/filters.hpp"
#include "cpm/cpm.hpp"
#include "modem/modem.hpp"
#include "waveform/sync.hpp"

namespace data2g::receiver {

inline constexpr double SILENCE_RMS = 1e-4;  // -80 dBFS: digital silence, no search

// Samples holding a whole preamble and header (every copy) of any of these.
std::int64_t search_span(std::span<const std::string_view> bands, std::span<const std::string_view> cpm_grids = {});

// What a burst's receive gives: OFDM up to soft bits, or CPM's soft bits.
using Rx = std::variant<modem::Received, cpm::Received>;

// receiver.receive_any: one burst starting within `lead` samples of y's start,
// OFDM first; nullopt if none. cpm_grids nullopt: every grid.
std::optional<Rx> receive_any(std::span<const double> y, std::int64_t lead = 0,
                              std::optional<std::vector<std::string_view>> cpm_grids = std::nullopt);

// A burst being received: find_burst's / find_copy's lock or a CPM early
// lock. A CPM lock's header_end stays the buffer index it was found at (as
// Python's dict(p, start=..., end=...) leaves it).
struct Pending {
    std::variant<modem::Lock, cpm::Lock> lock;

    bool is_cpm() const { return lock.index() == 1; }
    const modem::Lock& ofdm() const { return std::get<0>(lock); }
    const cpm::Lock& cpm() const { return std::get<1>(lock); }
    std::int64_t start() const;
    std::int64_t end() const;
    double score() const;
    int n_cw() const;  // CPM: slots, control included
    bool copy() const { return !is_cpm() && ofdm().copy.has_value(); }
    void shift(std::int64_t d);  // every position (but a CPM header_end) by d
};

struct HeaderEvent {
    Pending header;
    std::int64_t stream_end = 0;  // the stream index of the audio fed so far (the caller's now)
};
struct BurstEvent {
    Pending header;
    std::optional<Rx> rx;  // nullopt: lost
    std::vector<double> audio;  // the segment received
    double receive_ms = 0;      // wall time modem::receive took (sync check, header, channel estimate), for the log
};
using Event = std::variant<HeaderEvent, BurstEvent>;

// A complete burst's audio and how to receive it: what the receiver hands
// to a decode step (in line from feed(); feed_deferred() hands it back for
// arq::Engine to decode on its worker). Nothing the receiver does after a
// burst reads its rx (supersede and trim read only the audio), so decode()
// can run elsewhere and its BurstEvent be posted later, ordered by
// header.start().
struct DecodeRequest {
    Pending header;              // stream indices
    std::vector<double> seg;     // the audio
    std::int64_t at = 0;         // seg[0]'s stream index
    std::int64_t head = 0;       // OFDM, no copy: the preamble and header lie in seg[:head]
    std::int64_t audio_lo = 0, audio_hi = 0;  // the event's audio: seg[audio_lo:audio_hi]
};
BurstEvent decode(DecodeRequest req, const modem::Accept& accept);

// The streaming receiver: feed() audio at FS as it arrives, get events
// back, in order: a HeaderEvent as a burst starts, then its BurstEvent.
// busy(): a burst is arriving (our own replies wait for it);
// channel_busy(): the host's BUSY (the pending burst's pilots or in-band
// energy say it is really there).
class Receiver {
public:
    static constexpr int HOP = config::FS / 4;
    static constexpr int MARGIN = 320;  // audio kept either side of an OFDM burst's span (receiver.Receiver.MARGIN)
    static constexpr double ON_AIR_DB = 3.0;
    static constexpr std::size_t FLOOR_BLOCKS = 1200;
    static constexpr int REVISIT = 2 * HOP;
    static constexpr int CPM_TAIL = config::FS / 20;  // kept before a CPM burst's end (receiver.Receiver.CPM_TAIL)
    static constexpr double SUPERSEDE_MARGIN = 0.05;
    static constexpr double SUSPECT_SCORE = 0.36;

    explicit Receiver(modem::Accept accept, std::vector<std::string_view> cpm_grids = {}, bool blank = true);

    std::vector<Event> feed(std::span<const double> x);
    // feed() with each burst to decode left as its DecodeRequest, in order.
    using Item = std::variant<HeaderEvent, BurstEvent, DecodeRequest>;
    std::vector<Item> feed_deferred(std::span<const double> x);
    void reset();

    bool busy() const { return pending_.has_value(); }
    bool channel_busy() const { return pending_ && (pilots_ok_ || on_air()); }
    bool on_air() const;
    const std::optional<Pending>& pending() const { return pending_; }
    // The (blanked) audio fed so far: its end's stream index, and n samples from stream index
    // `from` (empty unless all of them are still held).
    std::int64_t stream_end() const { return off_ + len(); }
    std::span<const double> audio(std::int64_t from, std::int64_t n) const {
        if (from < off_ || n < 0 || from + n > stream_end()) return {};
        return std::span<const double>(buf_).subspan(static_cast<std::size_t>(from - off_), static_cast<std::size_t>(n));
    }
    const modem::Accept& accept() const { return accept_; }
    std::int64_t n_blanked() const { return blanker_ ? blanker_->n_blanked : 0; }

private:
    struct Detector {
        std::string_view band;
        waveform::StreamDetector d;
    };
    struct Stats {  // per band, the statistic for buf[w0:]'s starts
        std::vector<Mat<double>> mats;
        std::vector<modem::BandStat> v;
    };

    void check_pilots();
    void trim(std::int64_t n);
    Stats stats(std::int64_t w0 = 0);
    void searched();
    bool supersede(std::vector<Item>& out, bool whole = false);
    std::optional<modem::Lock> find_copy() const;
    std::optional<cpm::Lock> find_cpm();
    std::int64_t decided(std::string_view key) const;
    std::int64_t decided_mid(const cpm::Grid& g) const;
    std::int64_t len() const { return static_cast<std::int64_t>(buf_.size()); }

    modem::Accept accept_;
    std::vector<std::string_view> bands_;
    std::vector<const cpm::Grid*> grids_;
    std::int64_t min_search_, keep_, hold_;
    std::vector<Detector> detectors_;
    std::optional<audio::Blanker> blanker_;

    std::vector<double> buf_;
    std::int64_t off_ = 0;  // stream index of buf_[0]
    std::optional<Pending> pending_;
    bool pilots_ok_ = false, confirmed_ = false;
    std::deque<double> powers_;  // in-band power of 0.1 s blocks
    double ps_ = 0.0;
    int pn_ = 0;
    std::int64_t fresh_ = HOP, last_start_ = -1;
    std::map<std::string_view, std::int64_t> decided_;
    std::map<std::string_view, std::int64_t> decided_mid_;  // per CPM grid, its mid block's (Python's "<grid>/mid")
    // CPM fronts whose first header copy fell short (cpm::find's candidates,
    // stream indices): read again with both copies once the second is in
    struct CpmWait {
        const cpm::Grid* g;
        cpm::Lock cand;
        std::int64_t first;  // its earliest tile start: kept until read
    };
    std::vector<CpmWait> cpm_wait_;
};


}  // namespace data2g::receiver
