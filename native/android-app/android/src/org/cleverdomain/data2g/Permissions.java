package org.cleverdomain.data2g;

import android.Manifest;
import android.app.Activity;
import android.content.Context;
import android.content.pm.PackageManager;
import android.os.Build;

/**
 * POST_NOTIFICATIONS (API 33+), which Qt has no permission type for.
 * Not awaited: denying it costs the notification, never the modem.
 */
public final class Permissions {
    private static final int REQUEST_NOTIFICATIONS = 1001;

    private Permissions() {}

    public static void requestNotifications(Context context) {
        if (Build.VERSION.SDK_INT < 33 || !(context instanceof Activity)) return;
        Activity activity = (Activity) context;
        if (activity.checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) == PackageManager.PERMISSION_GRANTED) return;
        activity.requestPermissions(new String[] {Manifest.permission.POST_NOTIFICATIONS}, REQUEST_NOTIFICATIONS);
    }
}
