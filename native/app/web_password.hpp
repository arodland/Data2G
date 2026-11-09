// data2g-web's password: one, no user name (one station, one operator).
// Kept as a salted PBKDF2-SHA256 hash in data2g-web's QSettings
// (web_password_store()), where data2g-gui's settings dialog and
// `data2g-web --set-password` both write it. QtCore + QtNetwork.
#pragma once

#include <QString>

#include <memory>

class QSettings;

namespace data2g::app {

// data2g-web's own QSettings (organization Data2G, application data2g-web),
// from any front end.
std::unique_ptr<QSettings> web_password_store();
bool has_web_password(QSettings& s);
// The stored hash, salt and all: it changes whenever the password is set,
// so a session can be tied to it. Empty when none is set.
QString web_password_hash(QSettings& s);
// `pw` must not be empty.
void set_web_password(QSettings& s, const QString& pw);
bool check_web_password(QSettings& s, const QString& pw);

}  // namespace data2g::app
