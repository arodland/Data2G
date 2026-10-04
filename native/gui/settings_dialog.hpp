// The settings dialog, and the QSettings store behind it. The values are
// data2g-host's options (app::Args): what is saved is overridden by the
// same flags on data2g-gui's command line. The dialog edits a copy and
// writes back only through apply_to (on OK), so Cancel cancels.
//
// Devices are stored by name, which select_device() matches as a
// substring: indices renumber when a USB card comes or goes.
//
// The Rig tab is SSTVAE's (WSJT-X's Radio tab): a Hamlib model picker with
// search, the device and serial fields only where the model's port type
// uses them, PTT method, and Test CAT / Test PTT on a worker thread.
#pragma once

#include <QDialog>
#include <QStringList>

#include "app/station.hpp"

class QCheckBox;
class QComboBox;
class QDoubleSpinBox;
class QFormLayout;
class QLineEdit;
class QPushButton;
class QSettings;
class QSlider;
class QSpinBox;

namespace data2g::gui {

// The persisted subset of `base`'s fields: callsign, devices, sample rate,
// volume, PTT and rig, ports, decode worker.
app::Args load_settings(QSettings& s, app::Args base = {});
void save_settings(QSettings& s, const app::Args& a);

class SettingsDialog : public QDialog {
    Q_OBJECT

public:
    // Device lists from the audio backend (empty without one).
    SettingsDialog(const app::Args& a, const QStringList& inputs, const QStringList& outputs, QWidget* parent = nullptr);
    void apply_to(app::Args& a) const;

private:
    QWidget* rig_tab(const app::Args& a);
    int rig_model() const;
    void sync_rig();  // which rig fields show (model's port type) and are enabled (PTT method)
    void test_rig(bool key_ptt);
    QString rigctld_text() const;  // model 2's device when none is set: rigctld's host:port

    app::Args base_;  // what apply_to keeps for fields the dialog doesn't show
    QWidget *serial_row_, *serial_row2_, *lines_row_;
    QLineEdit* mycall_;
    QComboBox *input_, *output_, *rate_;
    QDoubleSpinBox* volume_;
    QSpinBox *ptt_on_, *ptt_off_;
    QFormLayout* rig_form_;
    QCheckBox *rig_on_, *rig_debug_;
    QComboBox *rig_model_, *baud_, *data_bits_, *stop_bits_, *parity_, *handshake_, *dtr_, *rts_, *ptt_method_, *ptt_audio_,
        *rig_mode_;
    QLineEdit *rig_device_, *ptt_device_;
    QSpinBox *timeout_, *retries_;
    QDoubleSpinBox* poll_;
    QPushButton *test_cat_ = nullptr, *test_ptt_ = nullptr;
    QCheckBox* vara_;
    QLineEdit* host_;
    QSpinBox* cmd_port_;
    QCheckBox* kiss_;
    QLineEdit* kiss_address_;
    QSpinBox* kiss_port_;
    QCheckBox* worker_;
    QSlider* noise_rule_;  // the noise rule's weight x 100 (NOISE_RULE_STEPS)
};

}  // namespace data2g::gui
