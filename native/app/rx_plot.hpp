// A received burst as constellation-plot points: the engine's
// arq::BurstHeard (once decoded) or arq::BurstPreview (live, as it arrives)
// turned into what gui/constellation.hpp draws. OFDM through
// modem::equalized and the mode's constellation::find; CPM through
// cpm::shares. Called on the session stage (heard) and the receive thread
// (live); pure functions, no Qt.
#pragma once

#include <complex>
#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace data2g::arq {
struct BurstHeard;
struct BurstPreview;
}  // namespace data2g::arq

namespace data2g::app {

// The newest burst's data symbols, for a constellation plot. OFDM: equalized
// IQ. CPM (tones): each symbol's 1 - the runner-up tone's share + i * the
// strongest's, so a clean one is 1 + 1i and a weaker one falls below the diagonal.
// live: the burst is still arriving (--live-constellation), OFDM frames
// equalized one at a time (modem::preview_frame); lost: it ended undecoded,
// the live view all there is.
struct RxConstellation {
    std::string mode;
    bool tones = false, live = false, lost = false;
    std::vector<std::complex<float>> symbols;  // at most MAX_SYMBOLS, spread over the burst
    std::vector<std::complex<float>> points;   // the ideal: the mode's constellation, or 1 + 0i
    static constexpr std::size_t MAX_SYMBOLS = 4000;
};
// A heard burst's plot; nullopt when it was lost.
std::optional<RxConstellation> rx_constellation(const arq::BurstHeard& b);
RxConstellation rx_constellation(const arq::BurstPreview& p);  // live
// Any symbol inside the plot's square? A broken channel estimate puts
// every one far outside (recordings/20261013-234857 burst 0: a 10.7 ms
// gap mid-burst, |eq| ~3e7), a NaN nowhere: then the live view stays.
bool plot_visible(const RxConstellation& c);

}  // namespace data2g::app
