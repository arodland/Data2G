#include "app/web_password.hpp"

#include <QByteArray>
#include <QCryptographicHash>
#include <QPasswordDigestor>
#include <QRandomGenerator>
#include <QSettings>
#include <QStringList>

#include <stdexcept>

namespace data2g::app {

namespace {

const QString KEY = QStringLiteral("web/password");
// "pbkdf2-sha256$iterations$salt$key", salt and key base64. A login is rare;
// this many iterations is ~0.1 s here and a few times that on a Pi.
constexpr int ITERATIONS = 100000;

QByteArray derive(const QString& pw, const QByteArray& salt, int iterations) {
    return QPasswordDigestor::deriveKeyPbkdf2(QCryptographicHash::Sha256, pw.toUtf8(), salt, iterations, 32);
}

}  // namespace

QString web_password_hash(QSettings& s) { return s.value(KEY).toString(); }

bool has_web_password(QSettings& s) { return !web_password_hash(s).isEmpty(); }

void set_web_password(QSettings& s, const QString& pw) {
    if (pw.isEmpty()) throw std::invalid_argument("an empty password");
    QByteArray salt(16, '\0');
    QRandomGenerator::system()->fillRange(reinterpret_cast<quint32*>(salt.data()), salt.size() / 4);
    s.setValue(KEY, QStringLiteral("pbkdf2-sha256$%1$%2$%3")
                        .arg(ITERATIONS)
                        .arg(QString::fromLatin1(salt.toBase64()), QString::fromLatin1(derive(pw, salt, ITERATIONS).toBase64())));
    s.sync();  // a running data2g-web reads it from the file
}

bool check_web_password(QSettings& s, const QString& pw) {
    const QStringList parts = web_password_hash(s).split(QLatin1Char('$'));
    if (parts.size() != 4 || parts[0] != QLatin1String("pbkdf2-sha256")) return false;
    const int iterations = parts[1].toInt();
    if (iterations < 1) return false;
    const QByteArray want = QByteArray::fromBase64(parts[3].toLatin1());
    const QByteArray got = derive(pw, QByteArray::fromBase64(parts[2].toLatin1()), iterations);
    if (want.size() != got.size()) return false;
    char diff = 0;  // every byte compared: no early exit to time
    for (qsizetype i = 0; i < got.size(); ++i) diff |= char(want[i] ^ got[i]);
    return diff == 0;
}

}  // namespace data2g::app
