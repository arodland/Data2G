// The saved settings: the persisted subset of app::Args (callsign, interface
// (TCI), devices, sample rate, volume, PTT and rig, ports, recording, decode
// worker, live constellation), keyed by the Args field's name. What is saved
// is overridden by the same flags on a front end's command line.
// data2g-gui keeps them in QSettings; data2g-web too, and sends the same map
// as JSON to its settings tab. QtCore only.
#pragma once

#include <QVariantMap>

#include "app/options.hpp"

class QSettings;

namespace data2g::app {

QVariantMap settings_map(const Args& a);
// `m` over `base`: keys missing from `m` keep base's value.
Args from_settings(const QVariantMap& m, Args base = {});
Args load_settings(QSettings& s, Args base = {});
void save_settings(QSettings& s, const Args& a);

}  // namespace data2g::app
