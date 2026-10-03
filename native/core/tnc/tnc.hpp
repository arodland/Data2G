// data2g/tnc.py's radio side: KISS framing, burst packing, and the
// streaming Receiver (preamble/header search over a rolling buffer, BUSY,
// the header-copy and CPM paths). The host-app parts (rigctld, the KISS
// TCP server, sound devices) belong to the app port; the Decimator and
// Blanker are audio/filters.hpp.
//
// Positions in events are stream sample indices (samples fed since reset).
#pragma once

#include <array>
#include <cstdint>
#include <deque>
#include <map>
#include <optional>
#include <span>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "audio/filters.hpp"
#include "cpm/cpm.hpp"
#include "modem/modem.hpp"
#include "waveform/sync.hpp"

namespace data2g::tnc {

using Bytes = std::vector<std::uint8_t>;

// --- KISS -------------------------------------------------------------------

inline constexpr std::uint8_t FEND = 0xC0, FESC = 0xDB, TFEND = 0xDC, TFESC = 0xDD;

Bytes kiss_encode(std::span<const std::uint8_t> data, int port = 0);  // a data frame (command 0)

// Bytes in, whole frames (command byte, data) out, across reads.
class KissDecoder {
public:
    std::vector<std::pair<int, Bytes>> feed(std::span<const std::uint8_t> data);

private:
    Bytes buf_;
    bool esc_ = false;
};

// --- framing: [length, 2 bytes big-endian][frame] back to back, zero-padded --

int capacity(const modem::Spec& spec, int max_cw = config::MAX_CODEWORDS);
// Packets -> codeword payloads for one burst. Throws std::invalid_argument past capacity().
std::vector<Bytes> pack(const std::vector<Bytes>& packets, const modem::Spec& spec);
// Payloads and their CRC results -> (whole packets, packets lost).
std::pair<std::vector<Bytes>, int> unpack(const std::vector<Bytes>& payloads, const std::vector<bool>& ok);

// --- receive ----------------------------------------------------------------

inline constexpr double SILENCE_RMS = 1e-4;  // -80 dBFS: digital silence, no search

// Samples holding a whole preamble and header (every copy) of any of these.
std::int64_t search_span(std::span<const std::string_view> bands, std::span<const std::string_view> cpm_grids = {});

// What a burst's receive gives: OFDM up to soft bits, or CPM's soft bits.
using Rx = std::variant<modem::Received, cpm::Received>;

// tnc.receive_any: one burst starting within `lead` samples of y's start,
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
};
struct BurstEvent {
    Pending header;
    std::optional<Rx> rx;  // nullopt: lost
    std::vector<double> audio;  // the segment received
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

// tnc.Receiver: feed() audio at FS as it arrives, get events back.
// data2g/tnc.py NoiseProfile: the passband's noise between bursts, per
// sub-band, from 0.1 s blocks of idle audio (median power, and the 90th
// percentile over it). Times are sample indices, round(t * FS).
struct NoiseSnapshot {
    std::array<double, 5> db{}, tail_db{};
    int blocks = 0;
};

class NoiseProfile {
public:
    static constexpr std::array<std::pair<int, int>, 5> BANDS_HZ = {
        {{350, 950}, {950, 1300}, {1300, 1750}, {1750, 2100}, {2100, 2700}}};
    static constexpr int BLOCK = config::FS / 10;
    static constexpr std::size_t WINDOW = 600;
    static constexpr double COMMIT_S = 3.0;
    static constexpr std::size_t MIN_BLOCKS = 20;
    static constexpr double RECOVER_S = 0.6;

    NoiseProfile();
    void feed(std::span<const double> x, double t_start);  // heard from t_start (s); a gap starts a new block
    void mark(double start, double end);                    // not noise from start to end (s)
    std::optional<NoiseSnapshot> snapshot() const;

private:
    struct Block {
        std::int64_t start, end;
        std::array<double, 5> p;
    };
    std::vector<double> win_, buf_;
    std::int64_t s0_ = 0;
    std::deque<Block> pending_;
    std::vector<std::pair<std::int64_t, std::int64_t>> busy_;
    std::deque<std::array<double, 5>> kept_;
};

class Receiver {
public:
    static constexpr int HOP = config::FS / 4;
    static constexpr double ON_AIR_DB = 3.0;
    static constexpr std::size_t FLOOR_BLOCKS = 1200;
    static constexpr int REVISIT = 2 * HOP;
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
    std::optional<cpm::Lock> find_cpm() const;
    std::int64_t decided(std::string_view key) const;
    std::int64_t len() const { return static_cast<std::int64_t>(buf_.size()); }

    modem::Accept accept_;
    std::vector<std::string_view> bands_;
    std::vector<const cpm::Grid*> grids_;
    std::int64_t min_search_, keep_;
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
};

}  // namespace data2g::tnc
