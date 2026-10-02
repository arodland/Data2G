#include "waterfall.hpp"

#include <QMouseEvent>
#include <QPainter>
#include <QTimer>

#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>

#include "dsp/fft.hpp"
#include "generated/config.hpp"

namespace data2g::gui {

namespace {

constexpr int NFFT = 1024;
constexpr int BINS = NFFT / 2;  // 0 .. FS/2
constexpr double DISPLAY_HZ = config::FS / 2.0;
constexpr std::size_t METER_SAMPLES = config::FS / 10;  // one engine block
constexpr double DB_FLOOR = -95.0, DB_CEIL = -20.0;

using Rgb = std::array<std::uint8_t, 3>;

// 256-entry black -> blue -> green -> yellow -> white ramp.
const std::array<Rgb, 256>& colormap() {
    static const std::array<Rgb, 256> lut = [] {
        struct Stop {
            double at;
            Rgb color;
        };
        constexpr std::array<Stop, 5> stops{
            {{0.00, {0, 0, 0}}, {0.25, {0, 0, 140}}, {0.50, {0, 170, 90}}, {0.75, {245, 235, 40}}, {1.00, {255, 255, 255}}}};
        std::array<Rgb, 256> out{};
        for (int i = 0; i < 256; ++i) {
            const double x = i / 255.0;
            std::size_t k = 0;
            while (k + 2 < stops.size() && x > stops[k + 1].at) ++k;
            const Stop& lo = stops[k];
            const Stop& hi = stops[k + 1];
            const double t = (x - lo.at) / (hi.at - lo.at);
            for (int ch = 0; ch < 3; ++ch)
                out[i][ch] = static_cast<std::uint8_t>(lo.color[ch] + t * (hi.color[ch] - lo.color[ch]));
        }
        return out;
    }();
    return lut;
}

// Hann-windowed magnitude spectrum in dB, scaled by 1/NFFT (SSTVAE's dsp::spectrum_db).
std::vector<double> spectrum_db(const std::vector<double>& block) {
    static const std::vector<double> window = [] {
        std::vector<double> w(NFFT);
        for (int i = 0; i < NFFT; ++i) w[i] = 0.5 - 0.5 * std::cos(2.0 * std::numbers::pi * i / (NFFT - 1));
        return w;
    }();
    std::vector<dsp::cdouble> buf(NFFT);
    for (int i = 0; i < NFFT; ++i) buf[i] = block[i] * window[i];
    const auto spec = dsp::fft(buf, true);
    std::vector<double> out(BINS);
    for (int i = 0; i < BINS; ++i) out[i] = 20.0 * std::log10(std::abs(spec[i]) / NFFT + 1e-12);
    return out;
}

// Onto exactly `width` columns: interpolate when widening, peak-hold when
// shrinking so a narrow carrier can't be sampled away (SSTVAE's dsp::reduce_to_width).
std::vector<double> reduce_to_width(const std::vector<double>& v, int width) {
    const int n = static_cast<int>(v.size());
    if (width <= 0 || n == 0) return {};
    if (width == n) return v;
    std::vector<double> out(width);
    if (width > n) {
        for (int i = 0; i < width; ++i) {
            const double at = width == 1 ? 0.0 : static_cast<double>(i) * (n - 1) / (width - 1);
            const int lo = static_cast<int>(at), hi = std::min(lo + 1, n - 1);
            out[i] = v[lo] + (at - lo) * (v[hi] - v[lo]);
        }
        return out;
    }
    for (int i = 0; i < width; ++i) {
        const int start = static_cast<int>(static_cast<long long>(i) * n / width);
        const int stop = i + 1 == width ? n : static_cast<int>(static_cast<long long>(i + 1) * n / width);
        out[i] = *std::max_element(v.begin() + start, v.begin() + stop);
    }
    return out;
}

const QColor OK(60, 180, 75), CAUTION(230, 160, 30), DANGER(240, 60, 60);

}  // namespace

Waterfall::Waterfall(QWidget* parent, int fps) : QWidget(parent) {
    setMinimumSize(160, 90);
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
    auto* timer = new QTimer(this);
    connect(timer, &QTimer::timeout, this, &Waterfall::tick);
    timer->start(std::max(1, 1000 / std::max(1, fps)));
}

void Waterfall::ensure_image() {
    const qreal dpr = devicePixelRatioF();
    const int w = std::max(1, static_cast<int>(std::lround(width() * dpr)));
    const int h = std::max(1, static_cast<int>(std::lround(height() * dpr)));
    if (image_.width() == w && image_.height() == h && qFuzzyCompare(image_.devicePixelRatio(), dpr)) return;
    // carry the history across a resize: rows kept, columns point-resampled
    QImage grown(w, h, QImage::Format_RGB888);
    grown.setDevicePixelRatio(dpr);
    grown.fill(Qt::black);
    if (!image_.isNull()) {
        const int rows = std::min(h, image_.height()), old_w = image_.width();
        for (int y = 0; y < rows; ++y) {
            const uchar* src = image_.constScanLine(y);
            uchar* dst = grown.scanLine(y);
            for (int x = 0; x < w; ++x) std::copy_n(src + std::min(old_w - 1, x * old_w / w) * 3, 3, dst + x * 3);
        }
    }
    image_ = std::move(grown);
}

void Waterfall::tick() {
    if (!source_) return;
    std::uint64_t total = 0;
    const std::vector<double> block = source_(NFFT, &total);
    if (total == seen_ || block.size() < static_cast<std::size_t>(NFFT)) return;
    seen_ = total;

    peak_ = 0.0;
    for (auto it = block.end() - METER_SAMPLES; it != block.end(); ++it) peak_ = std::max(peak_, std::abs(*it));
    clipping_ = peak_ >= 0.99;
    if (clipping_) clip_latched_ = true;

    ensure_image();
    const std::vector<double> row = reduce_to_width(spectrum_db(block), image_.width());
    if (row.empty()) return;
    // one row = one pixel: scroll down by one, bottom-up in place
    const auto stride = static_cast<std::size_t>(image_.bytesPerLine());
    for (int y = image_.height() - 1; y > 0; --y) std::copy_n(image_.constScanLine(y - 1), stride, image_.scanLine(y));
    const auto& lut = colormap();
    uchar* top = image_.scanLine(0);
    for (int x = 0; x < image_.width(); ++x) {
        const double norm = std::clamp((row[x] - DB_FLOOR) / (DB_CEIL - DB_FLOOR), 0.0, 1.0);
        std::copy_n(lut[static_cast<std::size_t>(norm * 255.0)].data(), 3, top + x * 3);
    }
    update();
}

void Waterfall::resizeEvent(QResizeEvent* event) {
    QWidget::resizeEvent(event);
    ensure_image();
}

void Waterfall::paintEvent(QPaintEvent*) {
    ensure_image();
    QPainter painter(this);
    painter.drawImage(QPointF(0, 0), image_);  // 1:1 by construction: a blit
    draw_grid(painter);
    draw_level_meter(painter);
}

// A 500 Hz grid, labelled every kHz, shadowed for contrast.
void Waterfall::draw_grid(QPainter& painter) {
    const int w = width(), h = height();
    const double scale = w / DISPLAY_HZ;
    for (int hz = 500; hz < static_cast<int>(DISPLAY_HZ); hz += 500) {
        const int x = static_cast<int>(hz * scale);
        const int len = hz % 1000 ? 4 : 8;
        painter.setPen(QColor(0, 0, 0, 150));
        painter.drawLine(x + 1, h - len, x + 1, h);
        painter.setPen(QColor(255, 255, 255, 190));
        painter.drawLine(x, h - len, x, h);
        if (hz % 1000) continue;
        const QString tick = QString::number(hz / 1000) + QStringLiteral("k");
        painter.setPen(QColor(0, 0, 0, 150));
        painter.drawText(x + 4, h - 3, tick);
        painter.setPen(QColor(255, 255, 255, 190));
        painter.drawText(x + 3, h - 4, tick);
    }
}

// dBFS bar down the right edge: enough to set the sound card's gain.
void Waterfall::draw_level_meter(QPainter& painter) {
    const int w = width(), h = height();
    constexpr int bar_w = 8;
    const int x0 = w - bar_w - 2;
    painter.fillRect(x0, 2, bar_w, h - 4, QColor(0, 0, 0, 140));
    const double db = 20.0 * std::log10(std::max(peak_, 1e-6));
    const double frac = std::clamp((db + 60.0) / 60.0, 0.0, 1.0);
    const int filled = static_cast<int>((h - 4) * frac);
    painter.fillRect(x0, h - 2 - filled, bar_w, filled, clipping_ ? DANGER : frac > 0.85 ? CAUTION : OK);
    if (clip_latched_) {
        const QString label = tr("CLIP");
        const int text_x = x0 - painter.fontMetrics().horizontalAdvance(label) - 4;
        painter.setPen(QColor(0, 0, 0, 160));
        painter.drawText(text_x + 1, 15, label);
        painter.setPen(DANGER);
        painter.drawText(text_x, 14, label);
    }
}

void Waterfall::clear_clip() {
    clip_latched_ = false;
    update();
}

void Waterfall::mousePressEvent(QMouseEvent* event) {
    if (clip_latched_ && event->position().x() >= width() - 30) {
        clear_clip();
        return;
    }
    QWidget::mousePressEvent(event);
}

}  // namespace data2g::gui
