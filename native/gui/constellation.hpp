// The last burst's data symbols (app::RxConstellation), as VARA's
// constellation plot: dots over the ideal points (crosses). OFDM: equalized
// IQ. CPM: each symbol's 1 - runner-up tone share (x) against its strongest
// tone's share (y), ideally (1, 1). The shares sum to 1, so every dot lies
// between the diagonal (all the energy in two tones) and the line x + y = 1
// (a tie): the further below the diagonal, the more noise.
// A live view (still arriving) or a lost burst's is dimmer, and says so.
// Kept until the next burst replaces it. Square: as wide as the row is tall,
// but no wider than 30% of the window; then it loses height (the square sits
// at the top, the rest of the row left empty).
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
    void showEvent(QShowEvent* event) override;
    bool eventFilter(QObject* watched, QEvent* event) override;  // the window's resizes

private:
    void fit();

    app::RxConstellation c_;
};

}  // namespace data2g::gui
