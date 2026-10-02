#include "settings_dialog.hpp"

#include <QCheckBox>
#include <QComboBox>
#include <QDialogButtonBox>
#include <QDoubleSpinBox>
#include <QFormLayout>
#include <QGroupBox>
#include <QLabel>
#include <QLineEdit>
#include <QSettings>
#include <QSpinBox>
#include <QVBoxLayout>

namespace data2g::gui {

namespace {

std::optional<std::string> opt(const QString& s) {
    return s.trimmed().isEmpty() ? std::nullopt : std::optional(s.trimmed().toStdString());
}
QString qs(const std::optional<std::string>& s) { return QString::fromStdString(s.value_or("")); }
QString qs(const std::string& s) { return QString::fromStdString(s); }

// "(system default)" first; a saved device that isn't there now is kept, marked.
void fill_devices(QComboBox* c, const QStringList& names, const std::optional<std::string>& current) {
    c->addItem(QObject::tr("(system default)"), QString());
    for (const auto& n : names) c->addItem(n, n);
    if (!current) return;
    const QString cur = qs(current);
    int i = c->findData(cur);
    if (i < 0) {
        c->addItem(QObject::tr("%1 (not found)").arg(cur), cur);
        i = c->count() - 1;
    }
    c->setCurrentIndex(i);
}

QSpinBox* spin(int lo, int hi, int v, const QString& suffix = {}) {
    auto* s = new QSpinBox;
    s->setRange(lo, hi);
    s->setValue(v);
    s->setSuffix(suffix);
    return s;
}

}  // namespace

app::Args load_settings(QSettings& s, app::Args a) {
    const auto str = [&](const char* k, const std::string& d) { return s.value(k, qs(d)).toString().toStdString(); };
    if (s.contains("mycall")) a.mycall = opt(s.value("mycall").toString());
    if (s.contains("input_device")) a.input_device = opt(s.value("input_device").toString());
    if (s.contains("output_device")) a.output_device = opt(s.value("output_device").toString());
    a.sample_rate = s.value("sample_rate", a.sample_rate).toInt();
    a.output_volume = s.value("output_volume", a.output_volume).toDouble();
    a.rigctld_host = str("rigctld_host", a.rigctld_host);
    a.rigctld_port = s.value("rigctld_port", a.rigctld_port).toInt();
    a.ptt_on_delay_ms = s.value("ptt_on_delay_ms", a.ptt_on_delay_ms).toInt();
    a.ptt_off_delay_ms = s.value("ptt_off_delay_ms", a.ptt_off_delay_ms).toInt();
    a.vara = s.value("vara", a.vara).toBool();
    a.host = str("host", a.host);
    a.command_port = s.value("command_port", a.command_port).toInt();
    a.kiss = s.value("kiss", a.kiss).toBool();
    a.kiss_address = str("kiss_address", a.kiss_address);
    a.kiss_port = s.value("kiss_port", a.kiss_port).toInt();
    a.decode_worker = s.value("decode_worker", a.decode_worker).toBool();
    return a;
}

void save_settings(QSettings& s, const app::Args& a) {
    s.setValue("mycall", qs(a.mycall));
    s.setValue("input_device", qs(a.input_device));
    s.setValue("output_device", qs(a.output_device));
    s.setValue("sample_rate", a.sample_rate);
    s.setValue("output_volume", a.output_volume);
    s.setValue("rigctld_host", qs(a.rigctld_host));
    s.setValue("rigctld_port", a.rigctld_port);
    s.setValue("ptt_on_delay_ms", a.ptt_on_delay_ms);
    s.setValue("ptt_off_delay_ms", a.ptt_off_delay_ms);
    s.setValue("vara", a.vara);
    s.setValue("host", qs(a.host));
    s.setValue("command_port", a.command_port);
    s.setValue("kiss", a.kiss);
    s.setValue("kiss_address", qs(a.kiss_address));
    s.setValue("kiss_port", a.kiss_port);
    s.setValue("decode_worker", a.decode_worker);
}

SettingsDialog::SettingsDialog(const app::Args& a, const QStringList& inputs, const QStringList& outputs, QWidget* parent)
    : QDialog(parent) {
    setWindowTitle(tr("Data2G settings"));
    auto* top = new QVBoxLayout(this);
    const auto group = [&](const QString& title) {
        auto* g = new QGroupBox(title);
        auto* f = new QFormLayout(g);
        top->addWidget(g);
        return f;
    };

    auto* station = group(tr("Station"));
    mycall_ = new QLineEdit(qs(a.mycall));
    mycall_->setPlaceholderText(QStringLiteral("NOCALL"));
    station->addRow(tr("Callsign"), mycall_);
    worker_ = new QCheckBox(tr("Decode on a worker thread"));
    worker_->setChecked(a.decode_worker);
    worker_->setToolTip(tr("Keeps preamble search and BUSY running during a burst's decode"));
    station->addRow(worker_);

    auto* audio = group(tr("Audio"));
    input_ = new QComboBox;
    fill_devices(input_, inputs, a.input_device);
    audio->addRow(tr("Input"), input_);
    output_ = new QComboBox;
    fill_devices(output_, outputs, a.output_device);
    audio->addRow(tr("Output"), output_);
    rate_ = new QComboBox;
    for (int r : {8000, 16000, 24000, 32000, 48000, 96000}) rate_->addItem(QString::number(r), r);
    if (rate_->findData(a.sample_rate) < 0) rate_->addItem(QString::number(a.sample_rate), a.sample_rate);
    rate_->setCurrentIndex(rate_->findData(a.sample_rate));
    audio->addRow(tr("Sample rate (Hz)"), rate_);
    volume_ = new QDoubleSpinBox;
    volume_->setRange(-60.0, 20.0);
    volume_->setDecimals(1);
    volume_->setSuffix(tr(" dB"));
    volume_->setValue(a.output_volume);
    volume_->setToolTip(tr("0 dB puts a burst's peak at full scale"));
    audio->addRow(tr("Output volume"), volume_);
    if (!a.audio_io.empty())
        audio->addRow(new QLabel(tr("Audio is --audio-io %1 for this run.").arg(qs(a.audio_io))));

    auto* ptt = group(tr("PTT (rigctld, Hamlib model 2)"));
    rig_host_ = new QLineEdit(qs(a.rigctld_host));
    ptt->addRow(tr("rigctld host"), rig_host_);
    rig_port_ = spin(0, 65535, a.rigctld_port);
    rig_port_->setSpecialValueText(tr("0 (no PTT)"));
    ptt->addRow(tr("rigctld port"), rig_port_);
    ptt_on_ = spin(0, 2000, a.ptt_on_delay_ms, tr(" ms"));
    ptt->addRow(tr("PTT on delay"), ptt_on_);
    ptt_off_ = spin(0, 2000, a.ptt_off_delay_ms, tr(" ms"));
    ptt->addRow(tr("PTT off delay"), ptt_off_);

    auto* vara = group(tr("VARA ports"));
    vara_ = new QCheckBox(tr("Serve VARA clients"));
    vara_->setChecked(a.vara);
    vara->addRow(vara_);
    host_ = new QLineEdit(qs(a.host));
    vara->addRow(tr("Address"), host_);
    cmd_port_ = spin(1, 65534, a.command_port);
    auto* data_port = new QLabel;
    const auto show_data = [data_port](int p) { data_port->setText(tr("data on %1").arg(p + 1)); };
    show_data(a.command_port);
    connect(cmd_port_, &QSpinBox::valueChanged, data_port, show_data);
    auto* row = new QHBoxLayout;
    row->addWidget(cmd_port_);
    row->addWidget(data_port);
    vara->addRow(tr("Command port"), row);

    auto* kiss = group(tr("KISS"));
    kiss_ = new QCheckBox(tr("Serve KISS clients"));
    kiss_->setChecked(a.kiss);
    kiss->addRow(kiss_);
    kiss_address_ = new QLineEdit(qs(a.kiss_address));
    kiss->addRow(tr("Address"), kiss_address_);
    kiss_port_ = spin(1, 65535, a.kiss_port);
    kiss->addRow(tr("Port"), kiss_port_);

    auto* buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
    connect(buttons, &QDialogButtonBox::accepted, this, &QDialog::accept);
    connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);
    top->addWidget(buttons);
}

void SettingsDialog::apply_to(app::Args& a) const {
    a.mycall = opt(mycall_->text());
    a.input_device = opt(input_->currentData().toString());
    a.output_device = opt(output_->currentData().toString());
    a.sample_rate = rate_->currentData().toInt();
    a.output_volume = volume_->value();
    a.rigctld_host = rig_host_->text().trimmed().toStdString();
    a.rigctld_port = rig_port_->value();
    a.ptt_on_delay_ms = ptt_on_->value();
    a.ptt_off_delay_ms = ptt_off_->value();
    a.vara = vara_->isChecked();
    a.host = host_->text().trimmed().toStdString();
    a.command_port = cmd_port_->value();
    a.kiss = kiss_->isChecked();
    a.kiss_address = kiss_address_->text().trimmed().toStdString();
    a.kiss_port = kiss_port_->value();
    a.decode_worker = worker_->isChecked();
}

}  // namespace data2g::gui
