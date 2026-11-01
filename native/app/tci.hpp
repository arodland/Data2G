// TCI (ExpertSDR3's Transceiver Control Interface, github.com/ExpertSDR3/TCI,
// protocol 2.1): the radio's audio, PTT and dial over one WebSocket, in place
// of a sound card and Hamlib (data2g-host --tci HOST:PORT).
//
// Its own QThread and event loop, as a sound card stream has (qtaudio.hpp):
// the server's TX_CHRONO frames ask for transmit audio and are answered at
// once, whatever the owner's thread is doing.
//
// - receive: after READY, AUDIO_START:0 for mono float32 at `rate`; RX audio
//   frames go through a CapturePipeline into the capture FIFO. While no RX
//   audio arrives (not connected yet, a server that stops it while
//   transmitting, a lost connection) silence is fed at real time, as PipeIo
//   does after its input ends, so the engine's clock runs on.
// - transmit: set_ptt(true) sends TRX:0,true,tci; each TX_CHRONO is answered
//   with as many samples pulled from the playback FIFO. A server that sends no
//   TX_CHRONO (AetherSDR, per Decodium 1.0.542) is fed on our clock instead,
//   from half a second after keying (at once if none was ever seen).
// - dial: the server's VFO:0,0,Hz notifications.
//
// Transceiver 0, channel 0 only. A lost connection is retried every 2 s.
#pragma once

#include <QThread>

#include <optional>
#include <string>

namespace data2g::audio {
class CaptureFifo;
class PlaybackFifo;
}  // namespace data2g::audio

namespace data2g::app {

class Tci {
public:
    // `rate`: 8000, 24000 or 48000 (a TCI rate and a multiple of FS), the
    // playback FIFO's. `modulation`: set after READY ("usb", "digu"); empty
    // leaves the operator's. Connects in the background.
    Tci(std::string host, int port, int rate, std::string modulation, audio::CaptureFifo& capture, audio::PlaybackFifo& playback);
    ~Tci();  // disconnects, the thread joined
    Tci(const Tci&) = delete;
    Tci& operator=(const Tci&) = delete;

    // A rig::Keyer::Ptt: blocks until the command is on the socket. Throws
    // std::runtime_error when not connected (the Keyer logs it and carries on).
    void set_ptt(bool on);
    // A rig::Keyer::MustRelease: a key was sent on this connection.
    bool keyed() const;
    // The dial the server last reported, or nothing.
    std::optional<double> frequency_hz() const;

    struct Worker;

private:
    QThread thread_;
    Worker* w_;  // lives on thread_
};

}  // namespace data2g::app
