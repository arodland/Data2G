// The saved settings: the persisted subset of app::Args (callsign, interface
// (TCI), devices, sample rate, volume, PTT and rig, ports, recording, decode
// worker, live constellation), keyed by the Args field's name. What is saved
// is overridden by the same flags on a front end's command line.
// data2g-gui and data2g-web share one QSettings store (settings_store());
// data2g-web also sends the map as JSON to its settings tab. QtCore only.
#pragma once

#include <QVariantMap>

#include <memory>

#include "app/options.hpp"

class QSettings;

namespace data2g::app {

// The one store both front ends use: organization Data2G, application
// data2g-gui (its name from before data2g-web, kept so saved settings stay).
// Each front end's own keys live under a group: gui/, web/.
std::unique_ptr<QSettings> settings_store();

QVariantMap settings_map(const Args& a);
// `m` over `base`: keys missing from `m` keep base's value.
Args from_settings(const QVariantMap& m, Args base = {});
Args load_settings(QSettings& s, Args base = {});
void save_settings(QSettings& s, const Args& a);

}  // namespace data2g::app
