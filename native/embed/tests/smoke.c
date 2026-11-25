/* The header as plain C, linked against the library: a modem made, asked its version over the host
 * interface, run through a second of silence, and freed. The C++ test (tests/test_embed.cpp) runs sessions. */
#include <stdio.h>
#include <string.h>

#include "CData2G/data2g.h"

struct app {
  int lines, flushes, ptt_off, version_seen, keyed;
};

static void tx_audio(void *ctx, const float *samples, size_t n) {
  (void)ctx;
  (void)samples;
  (void)n;
}
static void tx_flush(void *ctx) { ((struct app *)ctx)->flushes++; }
static void ptt(void *ctx, bool on) {
  struct app *a = (struct app *)ctx;
  a->keyed = on;
  if (!on) a->ptt_off++;
}
static void host_line(void *ctx, const char *line) {
  struct app *a = (struct app *)ctx;
  a->lines++;
  if (strncmp(line, "VERSION ", 8) == 0 && strcmp(line + 8, d2g_version()) == 0) a->version_seen = 1;
}
static void host_data(void *ctx, const uint8_t *data, size_t n) {
  (void)ctx;
  (void)data;
  (void)n;
}

#define CHECK(ok, what) \
  do { \
    if (!(ok)) { \
      fprintf(stderr, "FAIL %s\n", what); \
      return 1; \
    } \
  } while (0)

int main(void) {
  struct app a = {0, 0, 0, 0, 0};
  D2gConfig cfg;
  D2gCallbacks cbs;
  D2gModem *m;
  static float silence[4800];
  int i;

  CHECK(strncmp(d2g_version(), "Data2G ", 7) == 0, "version");
  d2g_config_default(&cfg);
  CHECK(cfg.sample_rate == D2G_RATE_DEFAULT && cfg.sample_rate % D2G_RATE_BASE == 0, "default rate");
  CHECK(d2g_max_net_bps(D2G_BW_2300) > d2g_max_net_bps(D2G_BW_500) && d2g_max_net_bps(D2G_BW_500) > 0, "max net bps");
  CHECK(d2g_max_net_bps(9) == 0, "max net bps: unknown bandwidth");

  memset(&cbs, 0, sizeof cbs);
  CHECK(d2g_new(&cfg, &cbs, &a) == NULL, "callbacks required");
  cbs.tx_audio = tx_audio;
  cbs.tx_flush = tx_flush;
  cbs.ptt = ptt;
  cbs.host_line = host_line;
  cbs.host_data = host_data;
  cfg.sample_rate = 44100;
  CHECK(d2g_new(&cfg, &cbs, &a) == NULL, "44100 Hz refused");
  cfg.sample_rate = D2G_RATE_DEFAULT;
  cfg.decode_threads = 1;
  cfg.seed = 1;
  m = d2g_new(&cfg, &cbs, &a);
  CHECK(m != NULL, "new");

  d2g_command(m, "VERSION");
  CHECK(a.version_seen, "VERSION answered before d2g_command returns");
  d2g_command(m, "MYCALL N0CALL");
  d2g_command(m, "LISTEN ON");
  for (i = 0; i < 10; i++) d2g_rx_audio(m, silence, sizeof silence / sizeof silence[0]);
  d2g_advance(m, 48000);
  d2g_tx_played(m, 0);
  CHECK(!a.keyed, "silence keys nothing");

  d2g_free(m);
  CHECK(a.flushes == 1 && a.ptt_off == 1, "free: tx_flush and ptt(false) last");
  d2g_free(NULL);
  printf("smoke: ok (%d host lines)\n", a.lines);
  return 0;
}
