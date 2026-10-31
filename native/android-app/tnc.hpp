// The modem, as QML sees it: its state, the station settings, and the
// activity log. A view owning nothing (modem.hpp has the modem): every
// property is read on demand, so a view made after a rotation, or after the
// activity came back from the background, shows what one there all along would.
//
// Start and Stop go through TncService, never straight to the Modem.
// Settings are saved at once and take effect at the next start.
#pragma once

#include <QObject>
#include <QString>
#include <QStringList>
#include <QTimer>
#include <QtQml/qqmlregistration.h>

#include <cstdint>

class Tnc : public QObject {
    Q_OBJECT
    QML_ELEMENT

    Q_PROPERTY(bool running READ running NOTIFY changed)    // or starting
    Q_PROPERTY(bool starting READ starting NOTIFY changed)
    Q_PROPERTY(QString status READ status NOTIFY changed)
    Q_PROPERTY(bool ptt READ ptt NOTIFY changed)
    Q_PROPERTY(bool busy READ busy NOTIFY changed)
    Q_PROPERTY(QString frequency READ frequency NOTIFY changed)  // the dial, when the rig is polled
    Q_PROPERTY(QString counters READ counters NOTIFY changed)    // audio trouble; empty when there is none

    Q_PROPERTY(QString callsign READ callsign WRITE setCallsign NOTIFY settingsChanged)
    Q_PROPERTY(QStringList inputDevices READ inputDevices NOTIFY devicesChanged)
    Q_PROPERTY(QStringList outputDevices READ outputDevices NOTIFY devicesChanged)
    // A name from the lists above; SYSTEM_DEFAULT for the platform's choice.
    Q_PROPERTY(QString inputDevice READ inputDevice WRITE setInputDevice NOTIFY settingsChanged)
    Q_PROPERTY(QString outputDevice READ outputDevice WRITE setOutputDevice NOTIFY settingsChanged)
    Q_PROPERTY(double txLevelDb READ txLevelDb WRITE setTxLevelDb NOTIFY settingsChanged)  // -30..0
    Q_PROPERTY(int kissPort READ kissPort CONSTANT)
    Q_PROPERTY(int varaPort READ varaPort CONSTANT)

    Q_PROPERTY(QStringList log READ log NOTIFY logChanged)

public:
    explicit Tnc(QObject* parent = nullptr);

    bool running() const;
    bool starting() const;
    QString status() const;
    bool ptt() const;
    bool busy() const;
    QString frequency() const;
    QString counters() const;

    QString callsign() const;
    void setCallsign(const QString& c);
    QStringList inputDevices() const { return inputs_; }
    QStringList outputDevices() const { return outputs_; }
    QString inputDevice() const;
    void setInputDevice(const QString& d);
    QString outputDevice() const;
    void setOutputDevice(const QString& d);
    double txLevelDb() const;
    void setTxLevelDb(double db);
    int kissPort() const;
    int varaPort() const;

    QStringList log() const { return log_; }

    // Asks for the microphone (and notifications), then TncService.
    Q_INVOKABLE void start();
    Q_INVOKABLE void stop();
    Q_INVOKABLE void refreshDevices();
    Q_INVOKABLE void clearLog();
    // Back while running: the task goes to the background (Background.java),
    // since ending the activity ends the process, and the modem with it.
    Q_INVOKABLE void moveToBackground();

Q_SIGNALS:
    void changed();
    void settingsChanged();
    void devicesChanged();
    void logChanged();
    void error(const QString& message);

private:
    void start_service();

    QStringList inputs_, outputs_;
    QStringList log_;
    std::uint64_t log_seen_ = 0;
    QTimer poll_;
};
