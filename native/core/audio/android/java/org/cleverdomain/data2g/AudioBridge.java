package org.cleverdomain.data2g;

import android.annotation.SuppressLint;
import android.content.Context;
import android.media.AudioAttributes;
import android.media.AudioDeviceInfo;
import android.media.AudioFormat;
import android.media.AudioManager;
import android.media.AudioRecord;
import android.media.AudioTrack;
import android.media.MediaRecorder;
import android.os.Process;
import android.util.Log;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.HashMap;
import java.util.Map;

/**
 * The Java half of {@code core/audio/android/} (androidaudio.hpp has the design).
 *
 * <p>Lifted from SSTVAE's AudioBridge. Each stream owns a thread that calls
 * into C++ on the data path: capture pushes what it read, playback pulls what
 * to write. C++ calls in here only to list, open and close. Call {@link #init}
 * once with a context before anything else.
 */
public final class AudioBridge {
    private static final String TAG = "data2g";

    private static Context context;
    private static final Map<Integer, Thread> streams = new HashMap<>();

    private AudioBridge() {}

    public static void init(Context appContext) {
        context = appContext.getApplicationContext();
    }

    private static AudioManager manager() {
        return (AudioManager) context.getSystemService(Context.AUDIO_SERVICE);
    }

    // --- enumeration: "Product (Type)", since several phones give every
    // input the same product name and the type is what the operator is
    // choosing between.

    static String[] inputDeviceNames() {
        return names(AudioManager.GET_DEVICES_INPUTS);
    }

    static String[] outputDeviceNames() {
        return names(AudioManager.GET_DEVICES_OUTPUTS);
    }

    private static String[] names(int flags) {
        AudioDeviceInfo[] devices = manager().getDevices(flags);
        String[] out = new String[devices.length];
        for (int i = 0; i < devices.length; i++) out[i] = describe(devices[i]);
        return out;
    }

    static String describe(AudioDeviceInfo d) {
        String product = String.valueOf(d.getProductName()).trim();
        String type = typeName(d.getType());
        return product.isEmpty() ? type : product + " (" + type + ")";
    }

    static String typeName(int type) {
        switch (type) {
            case AudioDeviceInfo.TYPE_BUILTIN_MIC: return "Builtin Mic";
            case AudioDeviceInfo.TYPE_BUILTIN_SPEAKER: return "Speaker";
            case AudioDeviceInfo.TYPE_BUILTIN_EARPIECE: return "Earpiece";
            case AudioDeviceInfo.TYPE_USB_DEVICE: return "USB Device";
            case AudioDeviceInfo.TYPE_USB_HEADSET: return "USB Headset";
            case AudioDeviceInfo.TYPE_USB_ACCESSORY: return "USB Accessory";
            case AudioDeviceInfo.TYPE_WIRED_HEADSET: return "Wired Headset";
            case AudioDeviceInfo.TYPE_WIRED_HEADPHONES: return "Wired Headphones";
            case AudioDeviceInfo.TYPE_BLUETOOTH_SCO: return "Bluetooth SCO";
            case AudioDeviceInfo.TYPE_BLUETOOTH_A2DP: return "Bluetooth A2DP";
            case AudioDeviceInfo.TYPE_TELEPHONY: return "Telephony";
            default: return "Type " + type;
        }
    }

    /** The first device with exactly this name (C++ picked it from our list), or null. */
    private static AudioDeviceInfo find(int flags, String name) {
        if (name == null || name.isEmpty()) return null;
        for (AudioDeviceInfo d : manager().getDevices(flags)) {
            if (describe(d).equals(name)) return d;
        }
        return null;
    }

    private static void track(int token, Thread t) {
        synchronized (streams) {
            streams.put(token, t);
        }
    }

    /** Stops a stream's thread and waits for it, so no call reaches C++ after this returns. */
    private static void close(int token) {
        Thread t;
        synchronized (streams) {
            t = streams.remove(token);
        }
        if (t == null) return;
        t.interrupt();
        if (t instanceof Reader) ((Reader) t).shutdown();
        if (t instanceof Writer) ((Writer) t).shutdown();
        try {
            t.join(2000);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    // --- capture

    /**
     * Starts capture and returns the rate it opened at, or -1. {@code rate}
     * first, then the other multiples of {@code fs} the C++ decimator takes.
     */
    @SuppressLint("MissingPermission")  // RECORD_AUDIO is the activity's job
    static int openInput(int token, String deviceName, int rate, int fs) {
        int[] rates = {rate, 48000, 16000, fs};
        // UNPROCESSED first: AGC or noise suppression on a modem signal reads as a
        // bad radio. VOICE_RECOGNITION skips most of the chain where it is missing.
        int[] sources = {MediaRecorder.AudioSource.UNPROCESSED, MediaRecorder.AudioSource.VOICE_RECOGNITION};
        for (int source : sources) {
            for (int r : rates) {
                if (r <= 0 || r % fs != 0) continue;
                int min = AudioRecord.getMinBufferSize(r, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT);
                if (min <= 0) continue;
                // Room for a stalled reader, not latency: we read 20 ms at a time.
                int bytes = Math.max(min * 4, r * 2);
                AudioRecord rec;
                try {
                    rec = new AudioRecord(source, r, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT, bytes);
                } catch (IllegalArgumentException e) {
                    continue;
                }
                if (rec.getState() != AudioRecord.STATE_INITIALIZED) {
                    rec.release();
                    continue;
                }
                AudioDeviceInfo target = find(AudioManager.GET_DEVICES_INPUTS, deviceName);
                // Not checking the result: SSTVAE saw it return false with the routing in
                // effect. The Reader compares getRoutedDevice() once running instead.
                if (target != null) rec.setPreferredDevice(target);
                try {
                    rec.startRecording();
                } catch (IllegalStateException e) {
                    rec.release();
                    continue;
                }
                Reader reader = new Reader(token, rec, r, target);
                track(token, reader);
                reader.start();
                Log.i(TAG, "capture " + r + " Hz, source " + source + ", device " + deviceName);
                return r;
            }
        }
        return -1;
    }

    static void closeInput(int token) {
        close(token);
    }

    private static final class Reader extends Thread {
        private final int token;
        private final AudioRecord record;
        private final int chunkBytes;
        private final AudioDeviceInfo requested;
        private volatile boolean running = true;

        Reader(int token, AudioRecord record, int rate, AudioDeviceInfo requested) {
            super("data2g-capture-" + token);
            this.token = token;
            this.record = record;
            this.chunkBytes = rate / 50 * 2;  // 20 ms
            this.requested = requested;
        }

        void shutdown() {
            running = false;
            try {
                record.stop();  // wakes a blocked read
            } catch (IllegalStateException ignored) {
            }
        }

        @Override
        public void run() {
            Process.setThreadPriority(Process.THREAD_PRIORITY_URGENT_AUDIO);
            ByteBuffer buf = ByteBuffer.allocateDirect(chunkBytes).order(ByteOrder.nativeOrder());
            boolean routeChecked = false;
            while (running) {
                buf.clear();
                int n = record.read(buf, chunkBytes);
                if (n > 0) {
                    nativePush(token, buf, n);
                    if (!routeChecked) {
                        routeChecked = true;
                        AudioDeviceInfo routed = record.getRoutedDevice();
                        if (requested != null && routed != null && routed.getId() != requested.getId()) {
                            nativeError(token, "routed to " + describe(routed) + ", not " + describe(requested));
                        }
                    }
                } else if (n < 0 && running) {
                    // The device went (a USB interface unplugged): stop rather than spin.
                    nativeError(token, "capture read failed (" + n + "); was the device unplugged?");
                    break;
                }
            }
            try {
                record.stop();
            } catch (IllegalStateException ignored) {
            }
            record.release();
        }
    }

    // --- playback

    /** Starts playback at exactly {@code rate}; returns the track's buffer in frames, or -1. */
    static int openOutput(int token, String deviceName, int rate) {
        int min = AudioTrack.getMinBufferSize(rate, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_16BIT);
        if (min <= 0) return -1;
        AudioTrack t;
        try {
            t = new AudioTrack.Builder()
                    .setAudioAttributes(new AudioAttributes.Builder()
                            .setUsage(AudioAttributes.USAGE_MEDIA)
                            .setContentType(AudioAttributes.CONTENT_TYPE_MUSIC)
                            .build())
                    .setAudioFormat(new AudioFormat.Builder()
                            .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                            .setSampleRate(rate)
                            .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                            .build())
                    // Short: it is latency on every key-down, and drain() waits it out.
                    .setBufferSizeInBytes(Math.max(min * 2, rate / 10 * 2))
                    .setTransferMode(AudioTrack.MODE_STREAM)
                    .build();
        } catch (Exception e) {
            Log.e(TAG, "openOutput", e);
            return -1;
        }
        AudioDeviceInfo target = find(AudioManager.GET_DEVICES_OUTPUTS, deviceName);
        if (target != null) t.setPreferredDevice(target);
        try {
            t.play();
        } catch (IllegalStateException e) {
            t.release();
            return -1;
        }
        Writer w = new Writer(token, t, rate);
        track(token, w);
        w.start();
        return t.getBufferSizeInFrames();
    }

    static void closeOutput(int token) {
        close(token);
    }

    /** Always writing (silence when the FIFO is empty), so keying needs no restart. */
    private static final class Writer extends Thread {
        private final int token;
        private final AudioTrack audio;
        private final int frames;
        private volatile boolean running = true;

        Writer(int token, AudioTrack audio, int rate) {
            super("data2g-playback-" + token);
            this.token = token;
            this.audio = audio;
            this.frames = rate / 50;  // 20 ms
        }

        void shutdown() {
            running = false;
        }

        @Override
        public void run() {
            Process.setThreadPriority(Process.THREAD_PRIORITY_URGENT_AUDIO);
            ByteBuffer buf = ByteBuffer.allocateDirect(frames * 2).order(ByteOrder.nativeOrder());
            while (running) {
                nativePull(token, buf, frames);
                buf.position(0);
                int n = audio.write(buf, frames * 2, AudioTrack.WRITE_BLOCKING);
                if (n < 0 && running) {
                    nativeError(token, "playback write failed (" + n + "); was the device unplugged?");
                    break;
                }
            }
            try {
                audio.pause();
                audio.flush();
                audio.stop();
            } catch (IllegalStateException ignored) {
            }
            audio.release();
        }
    }

    private static native void nativePush(int token, ByteBuffer buf, int length);

    private static native void nativePull(int token, ByteBuffer buf, int frames);

    private static native void nativeError(int token, String message);
}
