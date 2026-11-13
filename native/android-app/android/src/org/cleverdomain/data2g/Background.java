package org.cleverdomain.data2g;

import android.app.Activity;
import android.content.Context;

/**
 * What Back means while the modem runs: the task goes to the background, as a
 * recorder's or a navigation app's does. Ending the activity would end the
 * process (Qt's onDestroy exits it), and the modem with it.
 */
public final class Background {
    private Background() {}

    public static void moveToBack(Context context) {
        if (context instanceof Activity) ((Activity) context).moveTaskToBack(true);
    }
}
