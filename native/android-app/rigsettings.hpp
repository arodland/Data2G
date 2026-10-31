// The rig settings, as QML sees them: SSTVAE's rig screen (its
// android-app/rigcontrol.hpp) over Data2G's saved settings (app::Args'
// rig fields), with no connection of its own: the modem opens the rig when it
// starts, as data2g-host does. Saved at once, used at the next start.
//
// Three ways to reach a radio, as in SSTVAE:
//   usb        a serial adapter or the radio's own USB port (usb-serial-for-android);
//              CAT, or DTR/RTS keying
//   bluetooth  RFCOMM to a paired radio; CAT keying only (no control lines)
//   network    host:port handed to Hamlib: rigctld (model 2) or a ser2net-style server
#pragma once

#include <QObject>
#include <QString>
#include <QStringList>
#include <QVariantList>
#include <QtQml/qqmlregistration.h>

class RigSettings : public QObject {
    Q_OBJECT
    QML_ELEMENT

    Q_PROPERTY(bool enabled READ enabled WRITE setEnabled NOTIFY changed)
    Q_PROPERTY(QString connection READ connection WRITE setConnection NOTIFY changed)
    Q_PROPERTY(int model READ model WRITE setModel NOTIFY changed)
    Q_PROPERTY(QString modelLabel READ modelLabel NOTIFY changed)
    Q_PROPERTY(QVariantList devices READ devices NOTIFY devicesChanged)  // {id, label, permitted}
    Q_PROPERTY(QString device READ device WRITE setDevice NOTIFY changed)
    Q_PROPERTY(bool devicePermitted READ devicePermitted NOTIFY devicesChanged)
    Q_PROPERTY(bool bluetoothReady READ bluetoothReady NOTIFY devicesChanged)
    Q_PROPERTY(QString host READ host WRITE setHost NOTIFY changed)  // host[:port]; Hamlib's default port 4532
    Q_PROPERTY(int baud READ baud WRITE setBaud NOTIFY changed)      // 0: the model's own
    Q_PROPERTY(QString pttMethod READ pttMethod WRITE setPttMethod NOTIFY changed)  // vox, cat, dtr, rts
    Q_PROPERTY(QStringList pttMethods READ pttMethods NOTIFY changed)
    Q_PROPERTY(QString pttAudio READ pttAudio WRITE setPttAudio NOTIFY changed)  // mic, data
    Q_PROPERTY(bool pttAudioSupported READ pttAudioSupported NOTIFY changed)
    Q_PROPERTY(QString error READ error NOTIFY changed)

public:
    explicit RigSettings(QObject* parent = nullptr);
    ~RigSettings() override;

    bool enabled() const;
    void setEnabled(bool on);
    QString connection() const;
    void setConnection(const QString& kind);
    int model() const;
    void setModel(int m);
    QString modelLabel() const;
    QVariantList devices() const { return devices_; }
    QString device() const;
    void setDevice(const QString& id);
    bool devicePermitted() const;
    bool bluetoothReady() const;
    QString host() const;
    void setHost(const QString& h);
    int baud() const;
    void setBaud(int b);
    QString pttMethod() const;
    void setPttMethod(const QString& m);
    QStringList pttMethods() const;
    QString pttAudio() const;
    void setPttAudio(const QString& a);
    bool pttAudioSupported() const;
    QString error() const { return error_; }

    // Rows {number, label} matching `query`, capped (Hamlib knows hundreds).
    Q_INVOKABLE QVariantList findModels(const QString& query) const;
    Q_INVOKABLE void refreshDevices();
    // The system's permission dialog for the selected device; the answer re-lists.
    Q_INVOKABLE void requestPermission();

Q_SIGNALS:
    void changed();
    void devicesChanged();

private:
    QVariantList devices_;
    QString error_;
};
