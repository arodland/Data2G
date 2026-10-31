package org.cleverdomain.data2g;

import org.qtproject.qt.android.bindings.QtActivity;

/**
 * Qt's activity, stopping the modem cleanly when it goes for good.
 *
 * <p>The modem can't outlive this activity. Qt 6.11's {@code onDestroy()}
 * terminates the Qt application and exits the process, and its
 * {@code onCreate()} restarts the process if Qt is already running, so a modem
 * kept alive past a swipe from Recents (measured on API 36, by marking the
 * destroy as a configuration change) died anyway the moment the app was
 * reopened, mid-over and keyed. Back, Home and the screen going off don't
 * destroy the activity (Main.qml sends Back to the background), so they keep
 * the modem running; a swipe from Recents is a shutdown, done properly: PTT
 * down and ports closed before Qt exits.
 */
public class TncActivity extends QtActivity {
    @Override
    protected void onDestroy() {
        if (!isChangingConfigurations()) TncService.stopNow(this);
        super.onDestroy();
    }
}
