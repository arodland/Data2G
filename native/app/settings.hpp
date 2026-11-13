// The saved settings: the persisted subset of app::Args (callsign, interface
// (TCI), devices, broadcast mode, sample rate, volume, PTT and rig, ports, recording, decode
// worker, live constellation), keyed by the Args field's name. What is saved
// is overridden by the same flags on a front end's command line.
// data2g-gui, data2g-web and data2g-host read one config file (open_config());
// data2g-web also sends the map as JSON to its settings tab. QtCore only.
#pragma once

#include <QStringList>
#include <QVariantMap>

#include <memory>

#include "app/options.hpp"

class QSettings;

namespace data2g::app {

// The config file, an ini: --config's on this command line, else (config
// dir)/Data2G/data2g.conf. Sets the file settings_store() opens from then on.
// `migrate` (the front ends, never with --config): with no data2g.conf yet,
// the store older versions saved (Data2G/data2g-gui) moves to it.
std::unique_ptr<QSettings> open_config(int argc, char** argv, bool migrate);
// The file open_config() chose (the default one before it runs).
// Each front end's own keys live under a group: gui/, web/.
std::unique_ptr<QSettings> settings_store();

QVariantMap settings_map(const Args& a);
// `m` over `base`: keys missing from `m` keep base's value.
Args from_settings(const QVariantMap& m, Args base = {});
Args load_settings(QSettings& s, Args base = {});
void save_settings(QSettings& s, const Args& a);

// The settings a running Station takes without a restart (Station::update).
extern const QStringList LIVE_SETTINGS;
// Going from `from` to `to` restarts the station: something other than LIVE_SETTINGS differs.
bool needs_restart(const Args& from, const Args& to);

}  // namespace data2g::app
