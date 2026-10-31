// The station's saved settings (QSettings, so Qt Core only): the persisted
// subset of Args, shared by data2g-gui's settings dialog and the Android app.
// data2g-gui's command line overrides what is loaded for one run.
#pragma once

#include "app/options.hpp"

class QSettings;

namespace data2g::app {

// The persisted subset of `base`'s fields: callsign, devices, sample rate,
// volume, PTT and rig, ports, recording, decode worker, live constellation.
Args load_settings(QSettings& s, Args base = {});
void save_settings(QSettings& s, const Args& a);

}  // namespace data2g::app
