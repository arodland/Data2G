// Data2G TNC for Android: the station of data2g-host (KISS and VARA ports on
// this phone) under a Qt Quick screen, kept running by TncService.

#include <QGuiApplication>
#include <QJniEnvironment>
#include <QJniObject>
#include <QQmlApplicationEngine>
#include <QtCore/qcoreapplication_platform.h>

#include <cstdlib>

#include "audio/android/androidaudio.hpp"
#include "modem.hpp"
#include "rig/android/androidrig.hpp"

using namespace data2g;

namespace {

// The VM and a Context for the audio and serial layers' Java halves. Here,
// on Qt's thread, because it has the app's class loader (a thread we make
// later finds only the system's); Qt owns JNI_OnLoad, hence calls.
void init_bridges() {
    auto* vm = QJniEnvironment::javaVM();
    audio::android::set_java_vm(reinterpret_cast<audio::android::JavaVM_*>(vm));
    rig::android::set_java_vm(reinterpret_cast<rig::android::JavaVM_*>(vm));
    QJniObject ctx = QNativeInterface::QAndroidApplication::context();
    for (const char* cls : {"org/cleverdomain/data2g/AudioBridge", "org/cleverdomain/data2g/SerialBridge"})
        QJniObject::callStaticMethod<void>(cls, "init", "(Landroid/content/Context;)V", ctx.object());
    QJniEnvironment().checkAndClearExceptions();
}

}  // namespace

int main(int argc, char* argv[]) {
    QGuiApplication app(argc, argv);
    // QSettings keys off these: set before anything reads one.
    QCoreApplication::setOrganizationName(QStringLiteral("Data2G"));
    QCoreApplication::setOrganizationDomain(QStringLiteral("cleverdomain.org"));
    QCoreApplication::setApplicationName(QStringLiteral("Data2G TNC"));
    QCoreApplication::setApplicationVersion(QStringLiteral(DATA2G_VERSION));
    init_bridges();
    androidapp::Modem::instance();  // its poll timer lives on this thread

    QQmlApplicationEngine engine;
    engine.loadFromModule("Data2G", "Main");
    if (engine.rootObjects().isEmpty()) return 1;
    const int rc = QGuiApplication::exec();
    // End without unwinding (SSTVAE's main.cpp): returning runs static
    // destructors under threads still using them, and Android's HWUI threads
    // outliving the graphics state left a tombstone on every ordinary exit.
    // The service released the radio and audio on its stop path already.
    std::_Exit(rc);
}
