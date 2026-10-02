// Scrolling spectrum of the 8 kHz RX input, with the input level meter down
// its right edge. Lifted from SSTVAE's gui/waterfall.{hpp,cpp}: the backing
// image is kept exactly the widget's size in device pixels so the painter
// never rescales it (rows would shimmer), spectra are reduced to the width
// when computed (peak-hold when shrinking), and the CLIP marker latches
// until the meter is clicked. Changed for Data2G: 0-4 kHz (the whole 8 kHz
// input), no band caption, and a row only when new audio has arrived (the
// engine reads 0.1 s blocks), read through a callback instead of a ring.
#pragma once

#include <QImage>
#include <QWidget>

#include <cstdint>
#include <functional>
#include <vector>
#include <cstddef>
#include <utility>

namespace data2g::gui {

class Waterfall : public QWidget {
    Q_OBJECT

public:
    // The newest n input samples, and how many have been read in all.
    using Source = std::function<std::vector<double>(std::size_t n, std::uint64_t* total)>;

    explicit Waterfall(QWidget* parent = nullptr, int fps = 10);

    QSize sizeHint() const override { return {560, 160}; }
    void set_source(Source s) { source_ = std::move(s); }
    double peak() const { return peak_; }
    bool clip_latched() const { return clip_latched_; }
    void clear_clip();

public Q_SLOTS:
    void tick();  // a slot so a test can drive a frame directly

protected:
    void paintEvent(QPaintEvent* event) override;
    void resizeEvent(QResizeEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;

private:
    void ensure_image();
    void draw_grid(QPainter& painter);
    void draw_level_meter(QPainter& painter);

    Source source_;
    std::uint64_t seen_ = 0;
    QImage image_;
    double peak_ = 0.0;
    bool clipping_ = false, clip_latched_ = false;
};

}  // namespace data2g::gui
