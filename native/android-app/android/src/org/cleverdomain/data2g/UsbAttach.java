package org.cleverdomain.data2g;

import android.app.Activity;
import android.os.Bundle;

/**
 * Receives {@code USB_DEVICE_ATTACHED} and does nothing with it (SSTVAE's).
 *
 * <p>The side effect is the point: Android drops a USB permission when the
 * device is detached, unless an app that handles the attach intent was told
 * "always open with this app". Its own activity rather than a filter on the
 * main one, because {@code res/xml/device_filter.xml} matches whole vendor
 * ranges, and plugging in an Arduino must not open a radio app.
 */
public final class UsbAttach extends Activity {
    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        finish();
    }
}
