#include "host/host.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <stdexcept>

namespace data2g::host {

namespace {

using arq::SessionState;

constexpr const char* LOG = "data2g.host";
constexpr int INFO = 20;

// ponytail: from memory of VARA clients, not a checked list (as host.py); check the log on first contact with Pat
constexpr std::array<std::string_view, 8> IGNORED = {"COMPRESSION", "PUBLIC",     "CWID",       "P2P",
                                                     "WINLINK",     "REGISTERED", "ENCRYPTION", "IGNOREKISSDCD"};

std::string bw_name(int cap) {
    static const std::array<const char*, 3> names = {"500", "1200", "2300"};
    return names.at(static_cast<std::size_t>(cap));  // host.py: BW_NAME[cap], a KeyError otherwise
}

std::string upper(std::string s) {
    for (auto& c : s) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    return s;
}

std::vector<std::string> split(std::string_view s) {  // str.split(): runs of whitespace
    std::vector<std::string> out;
    std::size_t i = 0;
    while (i < s.size()) {
        while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) ++i;
        std::size_t j = i;
        while (j < s.size() && !std::isspace(static_cast<unsigned char>(s[j]))) ++j;
        if (j > i) out.emplace_back(s.substr(i, j - i));
        i = j;
    }
    return out;
}

std::string_view strip(std::string_view s) {
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) s.remove_prefix(1);
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) s.remove_suffix(1);
    return s;
}

}  // namespace

bool ignored(std::string_view cmd) { return std::find(IGNORED.begin(), IGNORED.end(), cmd) != IGNORED.end(); }

std::optional<int> bw_cap(std::string_view cmd) {
    if (cmd == "BW500") return 0;
    if (cmd == "BW1200") return 1;
    if (cmd == "BW2300" || cmd == "BW2750") return 2;
    return std::nullopt;
}

Host::Host(arq::Engine& e, std::optional<int> credit) : engine(e), buffer_credit(credit) {}

void Host::command(std::string_view line) {
    const auto words = split(line);
    if (words.empty()) return;
    const std::string cmd = upper(words[0]);
    const std::vector<std::string> args(words.begin() + 1, words.end());
    const std::string a0 = args.empty() ? std::string() : upper(args[0]);
    auto& e = engine;
    bool ok = true;
    if (cmd == "MYCALL" && !args.empty()) {
        e.set_call(args[0], {args.begin() + 1, args.end()});  // VARA takes several: all answer connects
    } else if (cmd == "LISTEN" && a0 == "CQ") {
        // CQ frames are always reported
    } else if (cmd == "LISTEN" && (a0 == "ON" || a0 == "OFF")) {
        listening = a0 == "ON";
        e.listen(listening);
    } else if (cmd == "CONNECT" && args.size() == 2) {
        try {
            const auto aliases = e.aliases();
            e.set_call(args[0], aliases);
            e.connect(args[1], cap);
        } catch (const std::runtime_error&) {
            ok = false;
        }
    } else if (cmd == "DISCONNECT") {
        e.session().disconnect();
    } else if (cmd == "ABORT") {
        e.abort();
        out_cmd.emplace_back("DISCONNECTED");
        if (listening) e.listen();
    } else if (cmd == "CQFRAME" && args.size() == 2 && bw_cap("BW" + args[1])) {
        try {
            e.send_cq(args[0], *bw_cap("BW" + args[1]));
        } catch (const std::runtime_error&) {  // a session under way
            ok = false;
        } catch (const std::invalid_argument&) {  // a callsign that won't pack
            ok = false;
        }
    } else if (auto c = bw_cap(cmd)) {
        cap = *c;
    } else if (cmd == "CHAT" && (a0 == "ON" || a0 == "OFF")) {
        e.set_chat(a0 == "ON");
    } else if (ignored(cmd)) {
        arq::log_write(LOG, INFO, "accepted, not implemented: " + std::string(strip(line)));
    } else if (cmd == "VERSION") {
        out_cmd.push_back("VERSION " + std::string(VERSION));
        return;
    } else {
        ok = false;
    }
    out_cmd.emplace_back(ok ? "OK" : "WRONG");
}

void Host::client_gone() {
    auto& e = engine;
    arq::log_write(LOG, INFO, "command client gone");
    const auto s = e.session().state;
    if (s == SessionState::CONNECTED || s == SessionState::DISCONNECTING) e.session().disconnect();
    else if (s == SessionState::CONNECTING) e.abort();
    listening = false;
    const auto now = e.session().state;
    if (now == SessionState::LISTEN || now == SessionState::CLOSED) e.listen(false);
}

void Host::data_in(arq::ByteView data) {
    engine.session().write(data);
    // always answer data with a BUFFER line, changed or not: Pat counts what
    // it wrote until one arrives, and blocks once that count passes 7x its
    // next write (it waited forever on a 6-byte B2F line)
    buffer_.reset();
}

Host::Queued Host::queued() {
    const auto& s = engine.session();
    Queued q;
    q.unsent = q.unacked = static_cast<std::int64_t>(s.pending_write.size());
    if (const auto& st = s.station) {
        q.station = true;
        q.unsent += st->new_available();
        q.unacked += static_cast<std::int64_t>(st->tx.buf.size());  // from the first unacked codeword on
    }
    return q;
}

std::optional<std::int64_t> Host::next_capacity() {
    auto& s = engine.session();
    const auto* g = dynamic_cast<const arq::GearPolicy*>(s.policy.get());
    if (!g || !s.station) return std::nullopt;
    return g->next_capacity(*s.station);
}

void Host::after_step(bool ptt) {
    auto& e = engine;
    if (e.now() - alive_ >= ALIVE_S) {  // some VARA clients count on it
        alive_ = e.now();
        out_cmd.emplace_back("IAMALIVE");
    }
    for (const auto& ev : e.events()) {
        const auto w = split(ev);
        if (ev.starts_with("CONNECTED")) {
            const std::string& peer = w.at(1);
            const std::string& me = e.session().call;  // the call dialed, when MYCALL gave several
            const bool master = e.session().master();
            out_cmd.push_back("CONNECTED " + (master ? me : peer) + " " + (master ? peer : me) + " " + bw_name(e.session().cap));
        } else if (ev.starts_with("CQFRAME")) {
            out_cmd.push_back("CQFRAME " + w.at(1) + " " + bw_name(std::stoi(w.at(2))));
        } else if (ev.starts_with("DISCONNECTED")) {
            arq::log_write(LOG, INFO, ev);
            out_cmd.emplace_back("DISCONNECTED");
        }
    }
    if (e.session().state == SessionState::CLOSED && listening) e.listen();
    const auto got = e.session().read();
    out_data.insert(out_data.end(), got.begin(), got.end());
    if (ptt != ptt_) {
        ptt_ = ptt;
        out_cmd.emplace_back(ptt ? "PTT ON" : "PTT OFF");
        if (ptt && e.tx() && e.tx()->burst->submode != mode_) {
            mode_ = e.tx()->burst->submode;
            out_cmd.push_back("MODE " + *mode_);
        }
    }
    if (e.channel_busy() != busy_) {
        busy_ = e.channel_busy();
        out_cmd.emplace_back(busy_ ? "BUSY ON" : "BUSY OFF");
    }
    // BUFFER, as VARA defines it: bytes the peer hasn't acknowledged yet,
    // sent or not. Pat blocks writes while BUFFER >= 7x its write and its
    // Flush waits for BUFFER 0; counting everything kept Data2G bursts short.
    // So while more than the next burst waits, BUFFER is the unsent bytes
    // past it (the credit); once it has all gone, 1 until the peer acks it
    // all. --buffer-credit 0 reports the plain VARA figure. (host.py has the
    // full story.)
    const Queued q = queued();
    std::int64_t buffered = q.unacked;
    if (q.station && buffer_credit != 0) {
        if (auto c = next_capacity()) {
            std::int64_t credit = *c;
            if (buffer_credit) credit = std::min<std::int64_t>(credit, *buffer_credit);
            buffered = q.unsent > credit ? q.unsent - credit : std::min<std::int64_t>(q.unacked, 1);
        }
    }
    // Pat's write and Flush give up after a minute without a BUFFER line
    if (buffered != buffer_ || (buffered && e.now() - buffer_t_ >= BUFFER_REPEAT_S)) {
        buffer_ = buffered;
        buffer_t_ = e.now();
        out_cmd.push_back("BUFFER " + std::to_string(buffered));
    }
}

}  // namespace data2g::host
