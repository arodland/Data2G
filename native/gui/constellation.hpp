// The last burst's data symbols (app::RxConstellation), as VARA's
// constellation plot: dots over the ideal points (crosses). OFDM: equalized
// IQ. CPM: each symbol's strongest tone share (x) against the runner-up's (y),
// ideally (1, 0); a dot near the diagonal was nearly another tone.
// Kept until the next burst replaces it, and as wide as it is tall.
#pragma once

#include <QWidget>

#include "app/station.hpp"

namespace data2g::gui {

class ConstellationPlot : public QWidget {
    Q_OBJECT

public:
    explicit ConstellationPlot(QWidget* parent = nullptr);

    QSize sizeHint() const override { return {160, 160}; }
    void set(app::RxConstellation c);
    const app::RxConstellation& shown() const { return c_; }

protected:
    void paintEvent(QPaintEvent* event) override;
    void resizeEvent(QResizeEvent* event) override;

private:
    app::RxConstellation c_;
};

}  // namespace data2g::gui
