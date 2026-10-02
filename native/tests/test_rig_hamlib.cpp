// The libhamlib backend with no radio: Hamlib's dummy rig (model 1)
// directly, then the host's real path, model 2 (NET rigctl) to a rigctld
// running the dummy, as tnc.Rigctld reaches localhost:4532 today.

#include <arpa/inet.h>
#include <netinet/in.h>
#include <signal.h>
#include <spawn.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <string>
#include <thread>

#include "check.hpp"
#include "rig/controller.hpp"
#include "rig/hamlib/hamlib.hpp"

extern char** environ;

using namespace data2g;

namespace {

void test_dummy() {
    check::current_step = "dummy";
    rig::HamlibConfig cfg;
    cfg.model = rig::MODEL_DUMMY;
    cfg.device = "";
    auto rig = rig::make_hamlib_backend(cfg);
    rig->open();
    rig->set_ptt(true);
    rig->set_ptt(false);
    check::is_true(rig->frequency_hz() > 0, "dummy: frequency");
    check::is_true(rig->description().find("Dummy") != std::string::npos, "dummy: description " + rig->description());
    rig->close();
    bool threw = false;
    try {
        rig->set_ptt(true);
    } catch (const rig::RigError&) {
        threw = true;
    }
    check::is_true(threw, "closed: PTT reports rather than crashes");
}

int free_port() {
    const int s = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    bind(s, reinterpret_cast<sockaddr*>(&a), sizeof a);
    socklen_t len = sizeof a;
    getsockname(s, reinterpret_cast<sockaddr*>(&a), &len);
    close(s);
    return ntohs(a.sin_port);
}

bool listening(int port) {
    const int s = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    a.sin_port = htons(static_cast<std::uint16_t>(port));
    const bool ok = connect(s, reinterpret_cast<sockaddr*>(&a), sizeof a) == 0;
    close(s);
    return ok;
}

void test_net_rigctl() {
    check::current_step = "rigctld";
    const std::string rigctld = DATA2G_RIGCTLD;
    if (rigctld.empty() || rigctld.find("NOTFOUND") != std::string::npos) {
        check::fail("rigctld", "not found at configure time");
        return;
    }
    const std::string port = std::to_string(free_port());
    const char* argv[] = {rigctld.c_str(), "-m", "1", "-T", "127.0.0.1", "-t", port.c_str(), nullptr};
    pid_t pid = 0;
    if (posix_spawn(&pid, rigctld.c_str(), nullptr, nullptr, const_cast<char**>(argv), environ) != 0) {
        check::fail("rigctld", "could not start " + rigctld);
        return;
    }
    for (int i = 0; i < 250 && !listening(std::stoi(port)); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(20));

    rig::HamlibConfig cfg;  // the defaults are the host's: model 2
    cfg.device = "127.0.0.1:" + port;
    std::string status;
    std::mutex m;
    rig::RigController controller({}, [&](const std::string& s, bool) {
        std::lock_guard<std::mutex> lock(m);
        status = s;
    });
    rig::RigConfig rc;
    rc.poll_interval_s = 0;
    controller.start(rig::make_hamlib_backend(cfg), rc);
    bool ok = true;
    try {
        controller.set_ptt(true);
        controller.set_ptt(false);
    } catch (const std::exception& e) {
        ok = false;
        check::fail("rigctld: PTT", e.what());
    }
    check::is_true(ok, "model 2 keys through rigctld");
    controller.stop();
    check::is_true(controller.wait_for_shutdown(5.0), "rigctld: worker closed the rig");
    {
        std::lock_guard<std::mutex> lock(m);
        check::is_true(status.find("127.0.0.1:" + port) != std::string::npos, "status names the rigctld: " + status);
    }
    kill(pid, SIGTERM);
    waitpid(pid, nullptr, 0);
}

}  // namespace

int main() {
    check::report_crashes_instead_of_prompting();
    check::Watchdog watchdog(100, "test_rig_hamlib");
    std::printf("hamlib %s\n", rig::hamlib_version().c_str());
    test_dummy();
    test_net_rigctl();
    return check::report("rig hamlib");
}
