// data2g/host.py's Host: VARA command semantics over an arq::Engine. No
// sockets, no audio (those are the app's, apps/data2g_host.cpp). command()
// and data_in() take the client's side; after_step() collects what goes
// back after each block. Everything here touches the engine's session
// stage, so in worker mode it runs only through Engine::post() and the
// after_block callback.
//
// Commands: MYCALL call..., LISTEN ON|OFF|CQ, CONNECT from to, DISCONNECT,
// ABORT, CQFRAME call bw, BW500|BW1200|BW2300|BW2750, CHAT ON|OFF, VERSION,
// and IGNORED's settings; broadcast (docs/broadcast.md): BCAST OPEN group
// [FROM call], BCAST CLOSE n, BCAST MODE n [AUTO] mode, MODES. Replies OK /
// WRONG. Notifications: CONNECTED src dst bw, DISCONNECTED, PTT ON|OFF, BUSY
// ON|OFF, BUFFER n, IAMALIVE, MODE; broadcast statuses (BCAST ...) once the
// client has sent a BCAST command.
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "arq/engine.hpp"

namespace data2g::host {

inline constexpr std::string_view VERSION = "Data2G 0.1";
inline constexpr double ALIVE_S = 60.0;          // IAMALIVE on the command port this often, as VARA does
inline constexpr double BUFFER_REPEAT_S = 30.0;  // a nonzero BUFFER is repeated this often

// VARA settings a client may send that have no Data2G meaning (yet): OK, logged.
bool ignored(std::string_view cmd);
// BW500 -> 0, BW1200 -> 1, BW2300/BW2750 -> 2; nullopt: not a BW command.
std::optional<int> bw_cap(std::string_view cmd);
// MODES (and --list-modes): one line per mode within the cap, narrowest
// first: MODE name bandwidth-Hz bytes-per-codeword max-codewords
// seconds-at-1 seconds-at-max.
std::vector<std::string> mode_lines(int cap);

class Host {
public:
    // `buffer_credit`: queued bytes BUFFER may leave out, at most (nullopt:
    // the next burst's whole capacity; 0: plain VARA, all of them).
    explicit Host(arq::Engine& engine, std::optional<int> buffer_credit = std::nullopt);
    virtual ~Host() = default;

    void command(std::string_view line);
    // The command client's TCP connection closed: it owned the session.
    void client_gone();
    void data_in(arq::ByteView data);
    void after_step(bool ptt);

    arq::Engine& engine;
    int cap = 2;
    bool listening = false;
    std::optional<int> buffer_credit;
    std::vector<std::string> out_cmd;  // lines for the command port, without CR
    arq::Bytes out_data;               // bytes for the data port

protected:
    // What BUFFER counts. Seams for a binding whose session or policy is a
    // Python object (tests patch them).
    struct Queued {
        std::int64_t unsent = 0, unacked = 0;
        bool station = false;
    };
    virtual Queued queued();
    // The policy's next_capacity(station); nullopt when it has none.
    virtual std::optional<std::int64_t> next_capacity();

private:
    // BCAST OPEN group [FROM call] | CLOSE n | MODE n [AUTO] mode -> the reply, nullopt for WRONG.
    std::optional<std::string> bcast_command(const std::vector<std::string>& args);

    bool bcast_ = false;  // the client has sent BCAST: broadcast statuses go to it
    bool ptt_ = false, busy_ = false;
    std::optional<std::int64_t> buffer_ = 0;  // nullopt: answer the next step whatever it is
    double buffer_t_ = 0.0;                   // engine time of the last BUFFER line
    std::optional<std::string> mode_;
    double alive_ = 0.0;                      // engine time of the last IAMALIVE
};

}  // namespace data2g::host
