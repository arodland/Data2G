#include "settings_dialog.hpp"

#include <QCheckBox>
#include <QComboBox>
#include <QCompleter>
#include <QDialogButtonBox>
#include <QDoubleSpinBox>
#include <QFormLayout>
#include <QGroupBox>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QPushButton>
#include <QSettings>
#include <QSlider>
#include <QSpinBox>
#include <QTabWidget>
#include <QThread>
#include <QVBoxLayout>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <memory>
#include <utility>

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

// One of app::rig_choices(field), shown as `labels` (same order; a value
// without one shows as itself). A saved value that isn't a choice is kept,
// so check() reports it rather than the dialog quietly changing it.
QComboBox* choice(const char* field, const std::string& current, const QStringList& labels) {
    auto* c = new QComboBox;
    const auto& values = app::rig_choices(field);
    for (std::size_t i = 0; i < values.size(); ++i) c->addItem(labels.value(static_cast<int>(i), qs(values[i])), qs(values[i]));
    if (c->findData(qs(current)) < 0) c->addItem(qs(current), qs(current));
    c->setCurrentIndex(c->findData(qs(current)));
    return c;
}
std::string chosen(const QComboBox* c) { return c->currentData().toString().toStdString(); }

QWidget* row(std::initializer_list<QWidget*> ws) {
    auto* w = new QWidget;
    auto* l = new QHBoxLayout(w);
    l->setContentsMargins(0, 0, 0, 0);
    for (auto* x : ws) l->addWidget(x);
    l->addStretch();
    return w;
}

void set_row_visible(QFormLayout* form, QWidget* field, bool visible) {
    field->setVisible(visible);
    if (QWidget* label = form->labelForField(field)) label->setVisible(visible);
}

// The string-valued rig options; the QSettings key is the Args field's name.
// (A named member-pointer type: MSVC can't parse one spelled inside pair<>.)
using StringField = std::string app::Args::*;
struct RigString {
    const char* key;
    StringField field;
};
const RigString RIG_STRINGS[] = {
    {"rig_device", &app::Args::rig_device},       {"rig_data_bits", &app::Args::rig_data_bits},
    {"rig_stop_bits", &app::Args::rig_stop_bits}, {"rig_parity", &app::Args::rig_parity},
    {"rig_handshake", &app::Args::rig_handshake}, {"rig_dtr", &app::Args::rig_dtr},
    {"rig_rts", &app::Args::rig_rts},             {"ptt_method", &app::Args::ptt_method},
    {"ptt_device", &app::Args::ptt_device},       {"ptt_audio", &app::Args::ptt_audio},
    {"rig_mode", &app::Args::rig_mode},
};

}  // namespace

// The noise rule's slider, in hundredths.
constexpr int NOISE_RULE_MAX = 150, NOISE_RULE_DEFAULT = 100, NOISE_RULE_DETENT = 5;

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
    a.rig = s.value("rig", a.rig).toBool();
    a.rig_model = s.value("rig_model", a.rig_model).toInt();
    a.rig_baud = s.value("rig_baud", a.rig_baud).toInt();
    for (const auto& [key, field] : RIG_STRINGS) a.*field = str(key, a.*field);
    a.rig_timeout_ms = s.value("rig_timeout_ms", a.rig_timeout_ms).toInt();
    a.rig_retries = s.value("rig_retries", a.rig_retries).toInt();
    a.rig_poll_interval = s.value("rig_poll_interval", a.rig_poll_interval).toDouble();
    a.rig_debug = s.value("rig_debug", a.rig_debug).toBool();
    a.vara = s.value("vara", a.vara).toBool();
    a.host = str("host", a.host);
    a.command_port = s.value("command_port", a.command_port).toInt();
    a.kiss = s.value("kiss", a.kiss).toBool();
    a.kiss_address = str("kiss_address", a.kiss_address);
    a.kiss_port = s.value("kiss_port", a.kiss_port).toInt();
    a.decode_worker = s.value("decode_worker", a.decode_worker).toBool();
    a.noise_rule = s.value("noise_rule", a.noise_rule).toDouble();
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
    s.setValue("rig", a.rig);
    s.setValue("rig_model", a.rig_model);
    s.setValue("rig_baud", a.rig_baud);
    for (const auto& [key, field] : RIG_STRINGS) s.setValue(key, qs(a.*field));
    s.setValue("rig_timeout_ms", a.rig_timeout_ms);
    s.setValue("rig_retries", a.rig_retries);
    s.setValue("rig_poll_interval", a.rig_poll_interval);
    s.setValue("rig_debug", a.rig_debug);
    s.setValue("vara", a.vara);
    s.setValue("host", qs(a.host));
    s.setValue("command_port", a.command_port);
    s.setValue("kiss", a.kiss);
    s.setValue("kiss_address", qs(a.kiss_address));
    s.setValue("kiss_port", a.kiss_port);
    s.setValue("decode_worker", a.decode_worker);
    s.setValue("noise_rule", a.noise_rule);
}

SettingsDialog::SettingsDialog(const app::Args& a, const QStringList& inputs, const QStringList& outputs, QWidget* parent)
    : QDialog(parent), base_(a) {
    setWindowTitle(tr("Data2G settings"));
    auto* outer = new QVBoxLayout(this);
    auto* tabs = new QTabWidget;
    outer->addWidget(tabs);
    auto* general = new QWidget;
    auto* top = new QVBoxLayout(general);
    tabs->addTab(general, tr("Station"));
    tabs->addTab(rig_tab(a), tr("Rig"));
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

    // The noise rule's weight, 0.00-1.50 in 0.01 steps: a detent at the
    // default (1.00) while dragging; 0 is off.
    auto* link = group(tr("Link"));
    noise_rule_ = new QSlider(Qt::Horizontal);
    noise_rule_->setObjectName(QStringLiteral("noise_rule"));
    noise_rule_->setRange(0, NOISE_RULE_MAX);
    noise_rule_->setSingleStep(1);
    noise_rule_->setPageStep(10);
    noise_rule_->setTickPosition(QSlider::TicksBelow);
    noise_rule_->setTickInterval(50);
    noise_rule_->setValue(static_cast<int>(std::lround(std::min(a.noise_rule, NOISE_RULE_MAX / 100.0) * 100)));
    noise_rule_->setToolTip(tr("A mode whose band is noisier in the receiver's noise profile than the band last "
                               "measured is predicted at a lower SNR. The weight is how much its often-loud "
                               "moments count. 0: off; 1.00: the default."));
    auto* noise_label = new QLabel;
    noise_label->setObjectName(QStringLiteral("noise_rule_value"));
    noise_label->setMinimumWidth(noise_label->fontMetrics().horizontalAdvance(tr("1.00 (default)")));
    const auto show_noise = [noise_label](int v) {
        noise_label->setText(v == 0 ? tr("off") : v == NOISE_RULE_DEFAULT ? tr("1.00 (default)")
                                                                          : QString::number(v / 100.0, 'f', 2));
    };
    show_noise(noise_rule_->value());
    connect(noise_rule_, &QSlider::valueChanged, noise_label, show_noise);
    connect(noise_rule_, &QSlider::sliderMoved, noise_rule_, [this](int v) {
        if (v != NOISE_RULE_DEFAULT && std::abs(v - NOISE_RULE_DEFAULT) <= NOISE_RULE_DETENT)
            noise_rule_->setValue(NOISE_RULE_DEFAULT);  // the detent: dragging only, so the arrow keys still step through
    });
    auto* noise_row = new QHBoxLayout;
    noise_row->addWidget(noise_rule_, 1);
    noise_row->addWidget(noise_label);
    link->addRow(tr("Noise rule"), noise_row);

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
    auto* port_row = new QHBoxLayout;
    port_row->addWidget(cmd_port_);
    port_row->addWidget(data_port);
    vara->addRow(tr("Command port"), port_row);

    auto* kiss = group(tr("KISS"));
    kiss_ = new QCheckBox(tr("Serve KISS clients"));
    kiss_->setChecked(a.kiss);
    kiss->addRow(kiss_);
    kiss_address_ = new QLineEdit(qs(a.kiss_address));
    kiss->addRow(tr("Address"), kiss_address_);
    kiss_port_ = spin(1, 65535, a.kiss_port);
    kiss->addRow(tr("Port"), kiss_port_);
    top->addStretch();

    auto* buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
    connect(buttons, &QDialogButtonBox::accepted, this, &QDialog::accept);
    connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);
    outer->addWidget(buttons);
}

// SSTVAE's Rig tab (gui/settings_dialog.cpp rig_tab), in a plain form.
QWidget* SettingsDialog::rig_tab(const app::Args& a) {
    auto* page = new QWidget;
    auto* form = rig_form_ = new QFormLayout(page);

    rig_on_ = new QCheckBox(tr("Use rig control (PTT through Hamlib)"));
    rig_on_->setChecked(app::rig_enabled(a));
    form->addRow(rig_on_);

    // Editable: the list may be missing (no Hamlib) or lack a saved model,
    // and the number must still be typeable and survive a round trip.
    rig_model_ = new QComboBox;
    rig_model_->setEditable(true);
    rig_model_->setInsertPolicy(QComboBox::NoInsert);
#ifdef DATA2G_HAVE_RIG
    for (const rig::RigModel& m : rig::list_models())
        rig_model_->addItem(tr("%1 (%2)").arg(qs(m.label())).arg(m.model), m.model);
#endif
    // 300-odd "<mfg> <model>" entries: match what is typed anywhere ("7300").
    rig_model_->completer()->setCaseSensitivity(Qt::CaseInsensitive);
    rig_model_->completer()->setFilterMode(Qt::MatchContains);
    if (const int i = rig_model_->findData(a.rig_model); i >= 0) rig_model_->setCurrentIndex(i);
    else rig_model_->setEditText(QString::number(a.rig_model));
    form->addRow(tr("Rig"), rig_model_);

    // Model 2 with no device is rigctld at --rigctld-host:--rigctld-port: shown as such.
    rig_device_ = new QLineEdit(a.rig_device.empty() && a.rig_model == rig::MODEL_NET_RIGCTL ? rigctld_text() : qs(a.rig_device));
    form->addRow(tr("Device"), rig_device_);

    baud_ = new QComboBox;
    baud_->addItem(tr("Default"), 0);
    for (int b : {1200, 2400, 4800, 9600, 19200, 38400, 57600, 115200}) baud_->addItem(QString::number(b), b);
    if (baud_->findData(a.rig_baud) < 0) baud_->addItem(QString::number(a.rig_baud), a.rig_baud);
    baud_->setCurrentIndex(baud_->findData(a.rig_baud));
    data_bits_ = choice("rig_data_bits", a.rig_data_bits, {tr("Default")});
    stop_bits_ = choice("rig_stop_bits", a.rig_stop_bits, {tr("Default")});
    parity_ = choice("rig_parity", a.rig_parity, {tr("Default"), tr("None"), tr("Odd"), tr("Even")});
    handshake_ = choice("rig_handshake", a.rig_handshake, {tr("Default"), tr("None"), tr("XON/XOFF"), tr("Hardware")});
    const QStringList levels = {tr("Default"), tr("High"), tr("Low")};
    dtr_ = choice("rig_dtr", a.rig_dtr, levels);
    rts_ = choice("rig_rts", a.rig_rts, levels);
    serial_row_ = row({new QLabel(tr("Baud")), baud_, new QLabel(tr("Data")), data_bits_, new QLabel(tr("Stop")), stop_bits_});
    form->addRow(tr("Serial"), serial_row_);
    serial_row2_ = row({new QLabel(tr("Parity")), parity_, new QLabel(tr("Handshake")), handshake_});
    form->addRow(QString(), serial_row2_);
    lines_row_ = row({new QLabel(tr("DTR")), dtr_, new QLabel(tr("RTS")), rts_});
    lines_row_->setToolTip(tr("Held for the whole session: how an interface powered from the control lines stays fed"));
    form->addRow(tr("Control lines"), lines_row_);

    ptt_method_ = choice("ptt_method", a.ptt_method, {tr("VOX"), tr("CAT"), tr("DTR"), tr("RTS")});
    ptt_device_ = new QLineEdit(qs(a.ptt_device));
    ptt_device_->setPlaceholderText(tr("(the device above)"));
    form->addRow(tr("PTT"), row({ptt_method_, new QLabel(tr("Port")), ptt_device_}));
    ptt_audio_ = choice("ptt_audio", a.ptt_audio, {tr("Mic / front"), tr("Data / rear")});
    form->addRow(tr("Transmit audio"), ptt_audio_);
    auto* note = new QLabel(tr("VOX: never key (the rig keys on the audio). DTR and RTS may use another port. "
                               "Transmit audio: the input CAT keying selects, on rigs with two."));
    note->setWordWrap(true);
    form->addRow(note);
    rig_mode_ = choice("rig_mode", a.rig_mode, {tr("None"), tr("USB"), tr("Data/Pkt")});
    form->addRow(tr("Mode on connect"), rig_mode_);

    ptt_on_ = spin(0, 2000, a.ptt_on_delay_ms, tr(" ms"));
    ptt_off_ = spin(0, 2000, a.ptt_off_delay_ms, tr(" ms"));
    form->addRow(tr("PTT delays"), row({new QLabel(tr("On")), ptt_on_, new QLabel(tr("Off")), ptt_off_}));
    timeout_ = spin(1, 60000, a.rig_timeout_ms, tr(" ms"));
    retries_ = spin(0, 10, a.rig_retries);
    form->addRow(tr("Hamlib"), row({new QLabel(tr("Timeout")), timeout_, new QLabel(tr("Retries")), retries_}));
    poll_ = new QDoubleSpinBox;
    poll_->setRange(0.0, 3600.0);
    poll_->setDecimals(1);
    poll_->setSuffix(tr(" s"));
    poll_->setSpecialValueText(tr("off (key only)"));
    poll_->setValue(a.rig_poll_interval);
    poll_->setToolTip(tr("Read the dial frequency this often, for the main window"));
    form->addRow(tr("Frequency poll"), poll_);
    rig_debug_ = new QCheckBox(tr("Hamlib's trace in the log"));
    rig_debug_->setChecked(a.rig_debug);
    form->addRow(rig_debug_);

#ifdef DATA2G_HAVE_RIG
    // They act at once, on a radio that may be on an antenna. The running
    // station holds the rig, so a serial one may report busy here.
    test_cat_ = new QPushButton(tr("Test CAT"));
    test_ptt_ = new QPushButton(tr("Test PTT (0.5 s)"));
    connect(test_cat_, &QPushButton::clicked, this, [this] { test_rig(false); });
    connect(test_ptt_, &QPushButton::clicked, this, [this] { test_rig(true); });
    auto* tests = new QHBoxLayout;
    tests->addStretch();
    tests->addWidget(test_cat_);
    tests->addWidget(test_ptt_);
    tests->addStretch();
    form->addRow(tests);
#endif

    connect(rig_model_, &QComboBox::currentTextChanged, this, [this] { sync_rig(); });
    connect(ptt_method_, &QComboBox::currentIndexChanged, this, [this] { sync_rig(); });
    sync_rig();
    return page;
}

QString SettingsDialog::rigctld_text() const {
    return tr("%1:%2").arg(qs(base_.rigctld_host)).arg(base_.rigctld_port ? base_.rigctld_port : 4532);
}

int SettingsDialog::rig_model() const {
    // An item carries its number; typed text is a label, a bare number, or a
    // label ending "(N)".
    const QString text = rig_model_->currentText().trimmed();
    if (const int i = rig_model_->findText(text); i >= 0) return rig_model_->itemData(i).toInt();
    bool ok = false;
    if (const int n = text.toInt(&ok); ok) return n;
    const int open = text.lastIndexOf(QLatin1Char('(')), close = text.lastIndexOf(QLatin1Char(')'));
    if (open >= 0 && close > open)
        if (const int n = text.mid(open + 1, close - open - 1).toInt(&ok); ok) return n;
    return base_.rig_model;
}

void SettingsDialog::sync_rig() {
    const int model = rig_model();
    // Without Hamlib to ask: model 2 is a network client, anything else may be serial.
    rig::PortType port = model == rig::MODEL_NET_RIGCTL ? rig::PortType::Network : rig::PortType::Serial;
    bool micdata = false;
#ifdef DATA2G_HAVE_RIG
    if (const auto info = rig::model_info(model)) port = info->port;
    micdata = rig::supports_ptt_audio_source(model);
#endif
    const bool serial = port == rig::PortType::Serial;
    set_row_visible(rig_form_, rig_device_, port != rig::PortType::None);
    rig_device_->setPlaceholderText(serial                               ? tr("/dev/ttyUSB0 or COM5")
                                    : port == rig::PortType::Network ? tr("host:port")
                                                                     : tr("(Hamlib's default)"));
    for (QWidget* w : {serial_row_, serial_row2_, lines_row_}) set_row_visible(rig_form_, w, serial);
    const std::string method = chosen(ptt_method_);
    ptt_device_->setEnabled(method == "dtr" || method == "rts");
    ptt_audio_->setEnabled(method == "cat" && micdata);
}

void SettingsDialog::test_rig(bool key_ptt) {
#ifdef DATA2G_HAVE_RIG
    app::Args a = base_;
    apply_to(a);
    a.rig = true;
    if (const auto bad = app::check(a)) {
        QMessageBox::warning(this, tr("Rig control"), qs(*bad));
        return;
    }
    const rig::HamlibConfig config = app::hamlib_config(a);
    test_cat_->setEnabled(false);
    test_ptt_->setEnabled(false);
    // A worker thread: a rig that is off costs the timeout, and the GUI never
    // waits on one. The answer reaches the dialog only if it still exists.
    auto result = std::make_shared<std::pair<bool, QString>>();
    QThread* t = QThread::create([config, key_ptt, result] {
        try {
            auto backend = rig::make_hamlib_backend(config);
            backend->open();
            if (key_ptt) {
                backend->set_ptt(true);
                QThread::msleep(500);
                backend->set_ptt(false);
                *result = {true, tr("PTT keyed and released.")};
            } else {
                *result = {true, tr("Connected to %1.\nDial frequency: %2 MHz")
                                     .arg(qs(backend->description()))
                                     .arg(backend->frequency_hz() / 1e6, 0, 'f', 4)};
            }
            backend->close();
        } catch (const std::exception& e) {
            *result = {false, QString::fromUtf8(e.what())};
        }
    });
    connect(t, &QThread::finished, this, [this, result] {
        test_cat_->setEnabled(true);
        test_ptt_->setEnabled(true);
        if (result->first) QMessageBox::information(this, tr("Rig control"), result->second);
        else QMessageBox::warning(this, tr("Rig control"), result->second);
    });
    connect(t, &QThread::finished, t, &QObject::deleteLater);
    t->start();
#else
    (void)key_ptt;
#endif
}

void SettingsDialog::apply_to(app::Args& a) const {
    a.mycall = opt(mycall_->text());
    a.input_device = opt(input_->currentData().toString());
    a.output_device = opt(output_->currentData().toString());
    a.sample_rate = rate_->currentData().toInt();
    a.output_volume = volume_->value();
    a.rig = rig_on_->isChecked();
    a.rig_model = rig_model();
    a.rigctld_host = base_.rigctld_host;
    a.rigctld_port = base_.rigctld_port;
    a.rig_device = rig_device_->text().trimmed().toStdString();
    // Model 2 still at rigctld's address: kept as --rigctld-host/--rigctld-port.
    if (a.rig_model == rig::MODEL_NET_RIGCTL && QString::fromStdString(a.rig_device) == rigctld_text()) {
        a.rig_device.clear();
        if (a.rig && !a.rigctld_port) a.rigctld_port = 4532;  // turned on after --rigctld-port 0
    }
    a.rig_baud = baud_->currentData().toInt();
    a.rig_data_bits = chosen(data_bits_);
    a.rig_stop_bits = chosen(stop_bits_);
    a.rig_parity = chosen(parity_);
    a.rig_handshake = chosen(handshake_);
    a.rig_dtr = chosen(dtr_);
    a.rig_rts = chosen(rts_);
    a.ptt_method = chosen(ptt_method_);
    a.ptt_device = ptt_device_->text().trimmed().toStdString();
    a.ptt_audio = chosen(ptt_audio_);
    a.rig_mode = chosen(rig_mode_);
    a.rig_timeout_ms = timeout_->value();
    a.rig_retries = retries_->value();
    a.rig_poll_interval = poll_->value();
    a.rig_debug = rig_debug_->isChecked();
    a.ptt_on_delay_ms = ptt_on_->value();
    a.ptt_off_delay_ms = ptt_off_->value();
    a.vara = vara_->isChecked();
    a.host = host_->text().trimmed().toStdString();
    a.command_port = cmd_port_->value();
    a.kiss = kiss_->isChecked();
    a.kiss_address = kiss_address_->text().trimmed().toStdString();
    a.kiss_port = kiss_port_->value();
    a.decode_worker = worker_->isChecked();
    // left where it opened: the value it came with (a command line may set more than the slider's top)
    const int shown = static_cast<int>(std::lround(std::min(base_.noise_rule, NOISE_RULE_MAX / 100.0) * 100));
    a.noise_rule = noise_rule_->value() == shown ? base_.noise_rule : noise_rule_->value() / 100.0;
}

}  // namespace data2g::gui
