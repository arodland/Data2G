// The receive waterfall and the input level, for Qt Quick: data2g-gui's
// waterfall (gui/waterfall.cpp) over the same rows (app/spectrum.hpp), read
// from the running modem's input. 0-4 kHz, newest row at the top, our own
// TX audio in its own colours. `peak` feeds LevelMeter.qml.
#pragma once

#include <QImage>
#include <QQuickPaintedItem>
#include <QTimer>
#include <QtQml/qqmlregistration.h>

#include <cstdint>

class Waterfall : public QQuickPaintedItem {
    Q_OBJECT
    QML_ELEMENT
    Q_PROPERTY(double peak READ peak NOTIFY peakChanged)  // 0..1, the newest 0.1 s of input

public:
    explicit Waterfall(QQuickItem* parent = nullptr);
    void paint(QPainter* painter) override;
    double peak() const { return peak_; }

Q_SIGNALS:
    void peakChanged();

protected:
    void geometryChange(const QRectF& now, const QRectF& before) override;

private:
    void tick();
    void ensure_image();

    QImage image_;
    QTimer timer_;
    std::uint64_t seen_ = 0;
    double peak_ = 0.0;
};
