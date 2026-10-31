#include "waterfall.hpp"

#include <QMouseEvent>
#include <QPainter>
#include <QTimer>

#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>

#include "app/spectrum.hpp"
#include "generated/config.hpp"

namespace data2g::gui {

namespace {

using namespace app::spectrum;
constexpr std::size_t METER_SAMPLES = config::FS / 10;  // one engine block

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
    bool tx = false;
    const std::vector<double> block = source_(NFFT, &total, &tx);
    if (total == seen_ || block.size() < static_cast<std::size_t>(NFFT)) return;
    seen_ = total;

    if (!tx) {  // the meter is the input's: TX audio would read as clipping
        peak_ = app::spectrum::peak(block, METER_SAMPLES);
        clipping_ = peak_ >= CLIP;
        if (clipping_) clip_latched_ = true;
    }

    ensure_image();
    const std::vector<Rgb> row = rgb_row(block, tx, image_.width());
    if (row.empty()) return;
    // one row = one pixel: scroll down by one, bottom-up in place
    const auto stride = static_cast<std::size_t>(image_.bytesPerLine());
    for (int y = image_.height() - 1; y > 0; --y) std::copy_n(image_.constScanLine(y - 1), stride, image_.scanLine(y));
    std::copy_n(row.front().data(), 3 * row.size(), image_.scanLine(0));
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
