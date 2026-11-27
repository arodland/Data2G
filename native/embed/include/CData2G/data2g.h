/*
 * Data2G as a library: the HF modem for an application that owns the audio and
 * the PTT. docs/EMBEDDING.md is the contract; in short:
 *
 *  - Call every function from one thread or serial queue, one call at a time.
 *    Callbacks run on that thread, from inside the call that caused them,
 *    before it returns. Nothing in the library runs between calls.
 *  - Time is the sample count: d2g_rx_audio plus d2g_advance.
 *  - A transmission is handed over whole (tx_audio) after ptt(true). The
 *    application plays it and reports a running total of samples played
 *    (d2g_tx_played); the modem then asks for ptt(false).
 *  - The session is driven with VARA host commands (d2g_command) and answers
 *    with host lines (host_line); data goes in with d2g_send and comes out of
 *    host_data.
 *  - One modem per process: the decoder's settings and the log are process-wide.
 */
#ifndef DATA2G_H
#define DATA2G_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The modem's own sample rate (Hz): D2gConfig.sample_rate is a multiple of it. */
#define D2G_RATE_BASE 8000

/* D2gConfig.sample_rate's default (Hz). */
#define D2G_RATE_DEFAULT 48000

/* Bandwidths, for d2g_max_net_bps: what the host commands BW500, BW1200 and
 * BW2300 select. */
#define D2G_BW_500 0
#define D2G_BW_1200 1
#define D2G_BW_2300 2

/* Opaque modem handle. */
typedef struct D2gModem D2gModem;

/* Configuration for d2g_new. Fill it with d2g_config_default first. */
typedef struct D2gConfig {
  /* Rate of all audio in and out (Hz): a multiple of D2G_RATE_BASE. */
  uint32_t sample_rate;
  /* Silence at the head of each transmission, after ptt(true) (s). */
  float ptt_on_delay_s;
  /* From the last sample played to ptt(false) (s). */
  float ptt_off_delay_s;
  /* On the transmit audio. A burst's peak is 1.0 before it; output is clipped
   * to +-1.0. */
  float tx_gain;
  /* Decode threads, the calling thread included. 0 = the library's default
   * (every core up to 4, more on a larger machine); 1 = decode on the calling
   * thread alone. They work only inside a call. */
  uint8_t decode_threads;
  /* The LDPC decoder's iteration cap. 0 = leave the library's (25). */
  uint8_t decoder_effort;
  /* Seconds of decision-directed decoding per burst, at most. 0 = none. */
  float dd_budget_s;
  /* 0 = pick a random seed; any other value is used as-is. */
  uint32_t seed;
} D2gConfig;

/* A burst sent, reported as it is keyed. */
typedef struct D2gTxReport {
  /* Modem time (s): the sample clock, from d2g_new. */
  double time_s;
  /* The mode's name, e.g. "qpsk-r1/2"; "nod" for a nod (a short keyed reply).
   * Valid during the callback only. */
  const char *mode;
  /* What the mode occupies (Hz). */
  uint32_t width_hz;
  /* Codewords in the burst. */
  uint16_t codewords;
  /* ... of them data, new or sent again. */
  uint16_t data_codewords;
  /* Bytes of the session's stream first sent in this burst: none for what is
   * sent again. On the wire, so compressed where the session compresses. */
  uint32_t bytes;
  /* The burst's time on air (s). */
  float air_s;
  /* bytes over air_s (bit/s). */
  float net_bps;
} D2gTxReport;

/* A burst heard, reported once the session has read it. */
typedef struct D2gRxReport {
  double time_s;
  /* As D2gTxReport's. Valid during the callback only. */
  const char *mode;
  uint32_t width_hz;
  /* Codewords the burst's header announced. */
  uint16_t codewords;
  /* Codewords decoded. */
  uint16_t codewords_ok;
  /* Codewords tried and not decoded. */
  uint16_t codewords_failed;
  /* Payload of the data codewords new to the session's stream with this burst
   * (on the wire). A codeword counts whole, so the last one of a message adds
   * its padding: a little more than the sender's report says. */
  uint32_t bytes;
  /* The header was heard, the burst was not. */
  bool lost;
  /* Estimated S/N (dB); NaN when there is none (a lost burst, a nod). */
  float snr_db;
  float air_s;
  /* bytes over air_s (bit/s). */
  float net_bps;
} D2gRxReport;

/* The application's callbacks. Optional pointers may be null. */
typedef struct D2gCallbacks {
  /* One transmission, whole: mono float samples at D2gConfig.sample_rate.
   * Follows ptt(true), or an earlier transmission that is still playing (play
   * it straight after). Required. */
  void (*tx_audio)(void *ctx, const float *samples, size_t n);
  /* Drop any queued transmit audio. The dropped samples do not count as played
   * (see d2g_tx_played). Required. */
  void (*tx_flush)(void *ctx);
  /* Key or un-key the radio. May block. Required. */
  void (*ptt)(void *ctx, bool on);
  /* One host command-port line (no CR). Valid during the callback only.
   * Required. */
  void (*host_line)(void *ctx, const char *line);
  /* Bytes for the host data port. Required. */
  void (*host_data)(void *ctx, const uint8_t *data, size_t n);
  /* Diagnostic log line: bursts sent and heard, the session, warnings
   * ("warning: ...") and errors ("error: ..."). Optional. */
  void (*log)(void *ctx, const char *line);
  /* Per-burst transmit report. Optional. */
  void (*tx_report)(void *ctx, const D2gTxReport *report);
  /* Per-burst receive report. Optional. */
  void (*rx_report)(void *ctx, const D2gRxReport *report);
} D2gCallbacks;

/* Fill `cfg` with the defaults data2g-host uses. */
void d2g_config_default(D2gConfig *cfg);

/* Create a modem. Returns null if `cfg` or `cbs` is refused: a required
 * callback missing, a sample rate that is not a multiple of D2G_RATE_BASE, a
 * negative or non-finite delay, gain or budget. */
D2gModem *d2g_new(const D2gConfig *cfg, const D2gCallbacks *cbs, void *ctx);

/* Flush audio, un-key, and free the modem: tx_flush and ptt(false) are the
 * last callbacks. Do not use `m` afterwards.
 *
 * May be called from inside a callback. Then no other callback fires any more,
 * and the flush, the un-key and the free happen when the call that fired the
 * callback returns: `ctx` must stay valid until it does. */
void d2g_free(D2gModem *m);

/* Submit one host command line (without CR): MYCALL, LISTEN ON, BW2300,
 * CONNECT, DISCONNECT, ABORT, ... Its reply (OK, WRONG, ...) comes through
 * host_line before this returns. May be called from inside a callback: the
 * command then runs before the call that fired the callback returns. */
void d2g_command(D2gModem *m, const char *line);

/* Bytes for the data port. May be called from inside a callback, like
 * d2g_command. */
void d2g_send(D2gModem *m, const uint8_t *data, size_t n);

/* Feed received audio (mono float at D2gConfig.sample_rate) and advance the
 * sample clock by `n`.
 *
 * Not for a real-time audio callback: this may decode a burst (hundreds of
 * milliseconds), allocate, and wait in ptt. Ignored (and logged) from inside a
 * callback. */
void d2g_rx_audio(D2gModem *m, const float *samples, size_t n);

/* Advance the sample clock by `samples` without audio: silence is heard.
 *
 * Required while keyed when the radio delivers no receive audio, and when the
 * capture stalls, so that the session's timers, the PTT release and the
 * watchdog can fire. d2g_advance(m, 0) is allowed: it only acts on what
 * d2g_tx_played reported. Ignored (and logged) from inside a callback. */
void d2g_advance(D2gModem *m, uint64_t samples);

/* Report how many transmit samples the application has played so far: a
 * running total since d2g_new of samples that really went out, clamped to what
 * was handed over.
 *
 * Samples dropped in tx_flush were not played: never add them. The count the
 * modem holds when tx_flush returns is taken as exact, so an application that
 * reports late should call this from inside tx_flush. May be called from
 * inside any callback. The radio is un-keyed by the next d2g_rx_audio or
 * d2g_advance call, D2gConfig.ptt_off_delay_s of samples after everything was
 * played. */
void d2g_tx_played(D2gModem *m, uint64_t total_samples);

/* The fastest mode's rate within a bandwidth (D2G_BW_*), in bit/s: no report's
 * net_bps passes it, so it is a gauge's full scale. 0 for an unknown bandwidth. */
float d2g_max_net_bps(uint8_t bandwidth);

/* Library version string, e.g. "Data2G 0.5.3": what the host command VERSION
 * answers. Valid for the life of the program. */
const char *d2g_version(void);

#ifdef __cplusplus
}
#endif

#endif /* DATA2G_H */
