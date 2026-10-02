// The settings dialog, and the QSettings store behind it. The values are
// data2g-host's options (app::Args): what is saved is overridden by the
// same flags on data2g-gui's command line. The dialog edits a copy and
// writes back only through apply_to (on OK), so Cancel cancels.
//
// Devices are stored by name, which select_device() matches as a
// substring: indices renumber when a USB card comes or goes.
#pragma once

#include <QDialog>
#include <QStringList>

#include "app/station.hpp"

class QCheckBox;
class QComboBox;
class QDoubleSpinBox;
class QLineEdit;
class QSettings;
class QSpinBox;

namespace data2g::gui {

// The persisted subset of `base`'s fields: callsign, devices, sample rate,
// volume, PTT, ports, decode worker.
app::Args load_settings(QSettings& s, app::Args base = {});
void save_settings(QSettings& s, const app::Args& a);

class SettingsDialog : public QDialog {
    Q_OBJECT

public:
    // Device lists from the audio backend (empty without one).
    SettingsDialog(const app::Args& a, const QStringList& inputs, const QStringList& outputs, QWidget* parent = nullptr);
    void apply_to(app::Args& a) const;

private:
    QLineEdit* mycall_;
    QComboBox *input_, *output_, *rate_;
    QDoubleSpinBox* volume_;
    QLineEdit* rig_host_;
    QSpinBox *rig_port_, *ptt_on_, *ptt_off_;
    QCheckBox* vara_;
    QLineEdit* host_;
    QSpinBox* cmd_port_;
    QCheckBox* kiss_;
    QLineEdit* kiss_address_;
    QSpinBox* kiss_port_;
    QCheckBox* worker_;
};

}  // namespace data2g::gui
