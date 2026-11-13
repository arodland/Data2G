#include "waterfall.hpp"

#include <QPainter>
#include <QQuickWindow>

#include <algorithm>
#include <cmath>
#include <utility>

#include "app/spectrum.hpp"
#include "modem.hpp"

using namespace data2g::app::spectrum;

Waterfall::Waterfall(QQuickItem* parent) : QQuickPaintedItem(parent) {
    timer_.setInterval(100);  // the engine reads 0.1 s blocks: a row each
    connect(&timer_, &QTimer::timeout, this, &Waterfall::tick);
    timer_.start();
}

// The image in device pixels, so painting is a blit; history kept across a resize.
void Waterfall::ensure_image() {
    const qreal dpr = window() ? window()->effectiveDevicePixelRatio() : 1.0;
    const int w = std::max(1, static_cast<int>(std::lround(width() * dpr)));
    const int h = std::max(1, static_cast<int>(std::lround(height() * dpr)));
    if (image_.width() == w && image_.height() == h) return;
    QImage grown(w, h, QImage::Format_RGB888);
    grown.fill(Qt::black);
    if (!image_.isNull()) {
        QPainter p(&grown);
        p.drawImage(QRect(0, 0, w, std::min(h, image_.height())), image_, QRect(0, 0, image_.width(), std::min(h, image_.height())));
    }
    image_ = std::move(grown);
}

void Waterfall::geometryChange(const QRectF& now, const QRectF& before) {
    QQuickPaintedItem::geometryChange(now, before);
    ensure_image();
}

void Waterfall::tick() {
    auto* station = data2g::androidapp::Modem::instance().station();
    if (!station || width() <= 0 || height() <= 0) {
        if (peak_ != 0.0) {
            peak_ = 0.0;
            Q_EMIT peakChanged();
        }
        return;
    }
    std::uint64_t total = 0;
    bool tx = false;
    const std::vector<double> block = station->input_tail(NFFT, &total, &tx);
    if (total == seen_ || block.size() < static_cast<std::size_t>(NFFT)) return;
    seen_ = total;
    if (!tx) {  // the meter is the input's: our own TX audio would read as clipping
        peak_ = data2g::app::spectrum::peak(block, data2g::config::FS / 10);
        Q_EMIT peakChanged();
    }
    ensure_image();
    const auto row = rgb_row(block, tx, image_.width());
    if (row.empty()) return;
    const auto stride = static_cast<std::size_t>(image_.bytesPerLine());
    for (int y = image_.height() - 1; y > 0; --y) std::copy_n(image_.constScanLine(y - 1), stride, image_.scanLine(y));
    std::copy_n(row.front().data(), 3 * row.size(), image_.scanLine(0));
    update();
}

void Waterfall::paint(QPainter* painter) {
    painter->drawImage(boundingRect(), image_);
    // 500 Hz ticks along the bottom, labelled each kHz, shadowed for contrast.
    const double w = width(), h = height();
    for (int hz = 500; hz < static_cast<int>(DISPLAY_HZ); hz += 500) {
        const double x = hz * w / DISPLAY_HZ;
        const double len = hz % 1000 ? 4 : 8;
        painter->setPen(QColor(255, 255, 255, 190));
        painter->drawLine(QPointF(x, h - len), QPointF(x, h));
        if (hz % 1000 == 0) {
            const QString label = QString::number(hz / 1000) + QStringLiteral("k");
            painter->setPen(QColor(0, 0, 0, 150));
            painter->drawText(QPointF(x + 4, h - 3), label);
            painter->setPen(QColor(255, 255, 255, 190));
            painter->drawText(QPointF(x + 3, h - 4), label);
        }
    }
}
