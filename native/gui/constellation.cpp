#include "constellation.hpp"

#include <QPainter>
#include <QResizeEvent>
#include <QShowEvent>

#include <algorithm>
#include <cmath>
#include <utility>

namespace data2g::gui {

ConstellationPlot::ConstellationPlot(QWidget* parent) : QWidget(parent) {
    setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Expanding);
}

void ConstellationPlot::set(app::RxConstellation c) {
    c_ = std::move(c);
    update();
}

constexpr double MAX_WINDOW_SHARE = 0.3;

// The layout sets our height (the row's); the width follows it, up to
// MAX_WINDOW_SHARE of the window's.
void ConstellationPlot::fit() {
    const int cap = static_cast<int>(MAX_WINDOW_SHARE * window()->width());
    const int side = std::max(1, std::min(height(), cap));
    if (side != width()) setFixedWidth(side);
}

void ConstellationPlot::resizeEvent(QResizeEvent*) { fit(); }

void ConstellationPlot::showEvent(QShowEvent*) {
    window()->installEventFilter(this);  // a narrower window, same height: no resize of ours to react to
    fit();
}

bool ConstellationPlot::eventFilter(QObject* watched, QEvent* event) {
    if (watched == window() && event->type() == QEvent::Resize) fit();
    return false;
}

void ConstellationPlot::paintEvent(QPaintEvent*) {
    QPainter p(this);
    const int side = std::min(width(), height());
    const QRect square(0, 0, side, side);  // at the top; below it, the row's spare height stays empty
    p.setClipRect(square);
    p.fillRect(square, Qt::black);
    const QPointF mid = QRectF(square).center();

    // the outermost point (a tone: the unit circle) at 2/3 of the half-width: room for the spread around it
    double reach = 0.0;
    for (const auto& q : c_.points) reach = std::max({reach, double(std::abs(q.real())), double(std::abs(q.imag()))});
    const double scale = side / 2.0 / (reach > 0 ? 1.5 * reach : 1.5);
    const auto at = [=](std::complex<float> z) { return mid + QPointF(z.real() * scale, -z.imag() * scale); };
    p.setPen(QColor(60, 60, 60));
    p.drawLine(QPointF(mid.x(), 0), QPointF(mid.x(), side));
    p.drawLine(QPointF(0, mid.y()), QPointF(side, mid.y()));
    if (c_.tones) p.drawEllipse(mid, scale, scale);  // a tone's share, all of it

    // tones: early blue to late green, so drift and fades show by when
    const std::size_t n = c_.symbols.size();
    for (std::size_t i = 0; i < n; ++i) {
        const double t = c_.tones && n > 1 ? double(i) / double(n - 1) : 1.0;
        p.setPen(QColor(0, static_cast<int>(120 + 135 * t), static_cast<int>(255 - 135 * t), 140));
        p.drawPoint(at(c_.symbols[i]));
    }

    p.setPen(QColor(255, 80, 80));
    if (c_.tones) {  // a tick outside the rim: a perfect symbol, on its tone, stays in sight
        for (const auto& q : c_.points) p.drawLine(at(q * 1.06f), at(q * 1.16f));
    } else {
        for (const auto& q : c_.points) {
            const QPointF c = at(q);
            p.drawLine(c - QPointF(3, 0), c + QPointF(3, 0));
            p.drawLine(c - QPointF(0, 3), c + QPointF(0, 3));
        }
    }

    p.setPen(Qt::gray);
    p.drawText(square.adjusted(4, 2, -4, -2), Qt::AlignTop | Qt::AlignLeft, QString::fromStdString(c_.mode) + (c_.live ? tr(" (live)") : c_.lost ? tr(" (lost)") : QString()));
}

}  // namespace data2g::gui
