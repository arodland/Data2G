#include "constellation.hpp"

#include <QPainter>
#include <QResizeEvent>

#include <algorithm>
#include <cmath>
#include <functional>

namespace data2g::gui {

ConstellationPlot::ConstellationPlot(QWidget* parent) : QWidget(parent) {
    setMinimumSize(80, 80);
    setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Expanding);
}

void ConstellationPlot::set(app::RxConstellation c) {
    c_ = std::move(c);
    update();
}

// The layout sets our height (the row's); the width follows it.
void ConstellationPlot::resizeEvent(QResizeEvent* event) {
    if (event->size().height() != width()) setFixedWidth(std::max(event->size().height(), minimumHeight()));
}

void ConstellationPlot::paintEvent(QPaintEvent*) {
    QPainter p(this);
    p.fillRect(rect(), Qt::black);
    const double side = std::min(width(), height());
    const QPointF mid(width() / 2.0, height() / 2.0);
    std::function<QPointF(std::complex<float>)> at;

    p.setPen(QColor(60, 60, 60));
    if (c_.tones) {
        // shares 0..1 on both axes, a small margin so the ideal (1, 0) stays clear of the edge
        const double margin = side * 0.08, span = side - 2 * margin;
        const QPointF origin = mid + QPointF(-side / 2 + margin, side / 2 - margin);
        at = [=](std::complex<float> z) { return origin + QPointF(z.real() * span, -z.imag() * span); };
        p.drawLine(at({0, 0}), at({1, 0}));
        p.drawLine(at({0, 0}), at({0, 1}));
        p.setPen(QPen(QColor(90, 90, 90), 1, Qt::DashLine));
        p.drawLine(at({0, 0}), at({0.5f, 0.5f}));  // top == runner-up: the tone was a tie
    } else {
        // the constellation's outermost point at 2/3 of the half-width: room for the spread around it
        double reach = 0.0;
        for (const auto& q : c_.points) reach = std::max({reach, double(std::abs(q.real())), double(std::abs(q.imag()))});
        const double scale = side / 2.0 / (reach > 0 ? 1.5 * reach : 1.5);
        at = [=](std::complex<float> z) { return mid + QPointF(z.real() * scale, -z.imag() * scale); };
        p.drawLine(QPointF(mid.x(), 0), QPointF(mid.x(), height()));
        p.drawLine(QPointF(0, mid.y()), QPointF(width(), mid.y()));
    }

    p.setPen(QColor(0, 255, 120, 140));
    for (const auto& z : c_.symbols) p.drawPoint(at(z));

    p.setPen(QColor(255, 80, 80));
    for (const auto& q : c_.points) {
        const QPointF c = at(q);
        p.drawLine(c - QPointF(3, 0), c + QPointF(3, 0));
        p.drawLine(c - QPointF(0, 3), c + QPointF(0, 3));
    }

    p.setPen(Qt::gray);
    p.drawText(rect().adjusted(4, 2, -4, -2), Qt::AlignTop | Qt::AlignLeft, QString::fromStdString(c_.mode));
}

}  // namespace data2g::gui
