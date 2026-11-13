// The thin half: JNI marshalling and nothing else (see the header).

#include "audio/android/androidaudio.hpp"

#include <jni.h>

#include <atomic>
#include <cstring>
#include <map>
#include <mutex>
#include <span>
#include <stdexcept>
#include <utility>

#include "generated/config.hpp"

namespace data2g::audio::android {

namespace {

constexpr const char* BRIDGE = "org/cleverdomain/data2g/AudioBridge";

JavaVM* g_vm = nullptr;
// Resolved on the UI thread in set_java_vm: FindClass from a thread we made
// sees only the system class loader (SSTVAE's transmit thread found that out).
jclass g_bridge = nullptr;

// Attaches the calling thread for one control call; never on the data path.
class Env {
public:
    Env() {
        if (g_vm == nullptr || g_bridge == nullptr) throw std::runtime_error("android audio: set_java_vm() was never called");
        const jint rc = g_vm->GetEnv(reinterpret_cast<void**>(&env_), JNI_VERSION_1_6);
        if (rc == JNI_EDETACHED) {
            if (g_vm->AttachCurrentThread(&env_, nullptr) != JNI_OK) throw std::runtime_error("android audio: AttachCurrentThread failed");
            attached_ = true;
        } else if (rc != JNI_OK) {
            throw std::runtime_error("android audio: GetEnv failed");
        }
    }
    ~Env() {
        if (attached_) g_vm->DetachCurrentThread();
    }
    Env(const Env&) = delete;
    Env& operator=(const Env&) = delete;
    JNIEnv* operator->() const { return env_; }
    JNIEnv* get() const { return env_; }

    jmethodID method(const char* name, const char* sig) const {
        jmethodID m = env_->GetStaticMethodID(g_bridge, name, sig);
        if (m == nullptr) {
            env_->ExceptionClear();
            throw std::runtime_error(std::string("android audio: no AudioBridge.") + name);
        }
        return m;
    }
    // A pending Java exception, as a C++ one.
    void check(const char* what) const {
        if (!env_->ExceptionCheck()) return;
        env_->ExceptionDescribe();
        env_->ExceptionClear();
        throw std::runtime_error(std::string("android audio: ") + what + " threw");
    }

private:
    JNIEnv* env_ = nullptr;
    bool attached_ = false;
};

std::string to_std(JNIEnv* env, jstring s) {
    if (s == nullptr) return {};
    const char* c = env->GetStringUTFChars(s, nullptr);
    std::string out(c == nullptr ? "" : c);
    if (c != nullptr) env->ReleaseStringUTFChars(s, c);
    return out;
}

std::vector<DeviceInfo> devices(const char* method) {
    Env env;
    auto arr = static_cast<jobjectArray>(env->CallStaticObjectMethod(g_bridge, env.method(method, "()[Ljava/lang/String;")));
    env.check(method);
    std::vector<DeviceInfo> out;
    if (arr == nullptr) return out;
    for (jsize i = 0, n = env->GetArrayLength(arr); i < n; ++i) {
        auto s = static_cast<jstring>(env->GetObjectArrayElement(arr, i));
        out.push_back({to_std(env.get(), s), 1});
        env->DeleteLocalRef(s);
    }
    return out;
}

std::string name_at(const std::vector<DeviceInfo>& list, std::optional<std::size_t> index) {
    if (!index) return {};
    if (*index >= list.size()) throw std::invalid_argument("no such audio device");
    return list[*index].name;
}

std::atomic<jint> g_next_token{1};

}  // namespace

struct Capture::Impl {
    jint token = 0;
    std::string name;
    Report on_error;
    std::unique_ptr<CapturePipeline> pipeline;
    bool stopped = false;
};

struct Playback::Impl {
    jint token = 0;
    std::string name;
    Report on_error;
    PlaybackFifo* fifo = nullptr;
    std::vector<float> mono;
    std::vector<std::byte> bytes;
    bool stopped = false;
};

namespace {

// Live streams by token. Held across a push or pull, so stop() can't free a
// stream under a Java thread that is still delivering to it (Java joins its
// thread on close, but a late call must still find nothing rather than a
// freed object).
std::mutex g_mu;
std::map<jint, Capture::Impl*> g_captures;
std::map<jint, Playback::Impl*> g_playbacks;

}  // namespace

void set_java_vm(JavaVM_* vm) {
    g_vm = reinterpret_cast<JavaVM*>(vm);
    if (g_vm == nullptr || g_bridge != nullptr) return;
    JNIEnv* env = nullptr;
    if (g_vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) != JNI_OK) return;
    jclass local = env->FindClass(BRIDGE);
    if (local == nullptr) {
        env->ExceptionClear();
        return;
    }
    g_bridge = static_cast<jclass>(env->NewGlobalRef(local));
    env->DeleteLocalRef(local);
}

std::vector<DeviceInfo> input_devices() { return devices("inputDeviceNames"); }
std::vector<DeviceInfo> output_devices() { return devices("outputDeviceNames"); }

// --- capture

Capture::Capture(std::optional<std::size_t> device, int rate, CaptureFifo& fifo, Report on_error)
    : impl_(std::make_unique<Impl>()) {
    impl_->name = name_at(input_devices(), device);
    impl_->on_error = std::move(on_error);
    impl_->token = g_next_token++;
    Env env;
    jstring jname = env->NewStringUTF(impl_->name.c_str());
    // Registered first: the reader thread starts pushing inside openInput.
    {
        std::lock_guard lock(g_mu);
        g_captures[impl_->token] = impl_.get();
    }
    const jint got = env->CallStaticIntMethod(g_bridge, env.method("openInput", "(ILjava/lang/String;II)I"), impl_->token, jname,
                                              rate, data2g::config::FS);
    env->DeleteLocalRef(jname);
    try {
        env.check("openInput");
        if (got <= 0) throw std::runtime_error("could not start capture on \"" + device_name() + "\"");
    } catch (...) {
        std::lock_guard lock(g_mu);
        g_captures.erase(impl_->token);
        throw;
    }
    std::lock_guard lock(g_mu);
    impl_->pipeline = std::make_unique<CapturePipeline>(SampleFormat::Int16, 1, got, fifo);
}

Capture::~Capture() { stop(); }

void Capture::stop() {
    if (impl_->stopped) return;
    impl_->stopped = true;
    try {
        Env env;
        env->CallStaticVoidMethod(g_bridge, env.method("closeInput", "(I)V"), impl_->token);  // joins the reader
        env->ExceptionClear();
    } catch (const std::exception&) {
    }
    std::lock_guard lock(g_mu);
    g_captures.erase(impl_->token);
}

std::string Capture::device_name() const { return impl_->name.empty() ? "(default)" : impl_->name; }
std::uint64_t Capture::frames_in() const {
    std::lock_guard lock(g_mu);
    return impl_->pipeline ? impl_->pipeline->frames_in() : 0;
}

// --- playback

Playback::Playback(std::optional<std::size_t> device, int rate, PlaybackFifo& fifo, Report on_error)
    : impl_(std::make_unique<Impl>()) {
    impl_->name = name_at(output_devices(), device);
    impl_->on_error = std::move(on_error);
    impl_->fifo = &fifo;
    impl_->token = g_next_token++;
    {
        std::lock_guard lock(g_mu);
        g_playbacks[impl_->token] = impl_.get();
    }
    Env env;
    jstring jname = env->NewStringUTF(impl_->name.c_str());
    const jint frames =
        env->CallStaticIntMethod(g_bridge, env.method("openOutput", "(ILjava/lang/String;I)I"), impl_->token, jname, rate);
    env->DeleteLocalRef(jname);
    try {
        env.check("openOutput");
        if (frames <= 0) throw std::runtime_error("could not start playback on \"" + device_name() + "\"");
    } catch (...) {
        std::lock_guard lock(g_mu);
        g_playbacks.erase(impl_->token);
        throw;
    }
    // ponytail: the track's buffer plus a fixed 0.1 s for the mixer and HAL behind it, which
    // AudioTrack doesn't report; measure with getTimestamp() if PTT ever clips a tail.
    fifo.set_output_latency(static_cast<double>(frames) / rate + 0.1);
}

Playback::~Playback() { stop(); }

void Playback::stop() {
    if (impl_->stopped) return;
    impl_->stopped = true;
    try {
        Env env;
        env->CallStaticVoidMethod(g_bridge, env.method("closeOutput", "(I)V"), impl_->token);  // joins the writer
        env->ExceptionClear();
    } catch (const std::exception&) {
    }
    std::lock_guard lock(g_mu);
    g_playbacks.erase(impl_->token);
}

std::string Playback::device_name() const { return impl_->name.empty() ? "(default)" : impl_->name; }

}  // namespace data2g::audio::android

// --- the data path, Java -> C++: on each stream's own Java thread

extern "C" {

JNIEXPORT void JNICALL Java_org_cleverdomain_data2g_AudioBridge_nativePush(JNIEnv* env, jclass, jint token, jobject buf, jint length) {
    using namespace data2g::audio::android;
    auto* data = static_cast<std::byte*>(env->GetDirectBufferAddress(buf));
    if (data == nullptr || length <= 0) return;
    std::lock_guard lock(g_mu);
    auto it = g_captures.find(token);
    if (it == g_captures.end() || !it->second->pipeline) return;  // before the pipeline: openInput hasn't returned
    (*it->second->pipeline)({data, static_cast<std::size_t>(length)});
}

// Fills `frames` int16 mono samples into `buf` (silence past the FIFO's end).
JNIEXPORT void JNICALL Java_org_cleverdomain_data2g_AudioBridge_nativePull(JNIEnv* env, jclass, jint token, jobject buf, jint frames) {
    using namespace data2g::audio;
    auto* data = static_cast<std::byte*>(env->GetDirectBufferAddress(buf));
    if (data == nullptr || frames <= 0) return;
    std::lock_guard lock(android::g_mu);
    auto it = android::g_playbacks.find(token);
    if (it == android::g_playbacks.end()) {
        std::memset(data, 0, static_cast<std::size_t>(frames) * 2);
        return;
    }
    android::Playback::Impl& p = *it->second;
    p.mono.resize(static_cast<std::size_t>(frames));
    p.fifo->pull(p.mono);
    to_device(p.mono, SampleFormat::Int16, 1, p.bytes);
    std::memcpy(data, p.bytes.data(), p.bytes.size());
}

JNIEXPORT void JNICALL Java_org_cleverdomain_data2g_AudioBridge_nativeError(JNIEnv* env, jclass, jint token, jstring jmsg) {
    using namespace data2g::audio::android;
    const std::string msg = to_std(env, jmsg);
    Report report;
    const char* dir = "in";
    {
        std::lock_guard lock(g_mu);
        if (auto it = g_captures.find(token); it != g_captures.end()) {
            report = it->second->on_error;
        } else if (auto pt = g_playbacks.find(token); pt != g_playbacks.end()) {
            report = pt->second->on_error;
            dir = "out";
        }
    }
    if (report) report(std::string("[audio ") + dir + "] " + msg);
}

}  // extern "C"
