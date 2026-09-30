"""speedtest.py over seeds `--seed`, `--seed` + 1, ... until 5 transfers
complete or 10 fail. Reports the success rate and the mean speed: of the
middle three with 5 successes (high and low dropped), else of all of them.

    python scripts/cpu_profile/speedtrials.py <channel> <snr_db> [--bytes 20000] [--bw 2300] [--seed 0] [--half-burst]

--half-burst: data bursts capped at 6 s (OFDM) or 12 s (FSK), about half
the shifter's longest, but never below control plus one data codeword (a
mode stays able to carry data). Patches the shifter's slots_for, so it
plans and prices the capped bursts too; control-only bursts unchanged.
"""
import argparse

import speedtest as T  # first: it sets the thread caps before numpy loads

import trials  # noqa: E402
from data2g.arq import modes as MD  # noqa: E402
from data2g.arq import policy as G  # noqa: E402

HALF_BURST_S = {"ofdm": 6.0, "cpm": 12.0}  # --half-burst's data-burst caps
HALF_BURST_MIN_DATA = 1


def half_burst():
    """Cap data bursts: HALF_BURST_S by family, but at least control + one data codeword."""
    slots_for = G.slots_for

    def capped(spec, seconds, data=True, dup=False):
        n = slots_for(spec, seconds, data, dup)
        if not data:
            return n
        extra = int(MD.is_cpm(spec) and dup)  # a duplicated CPM control codeword rides on top
        limit = HALF_BURST_S["cpm" if MD.is_cpm(spec) else "ofdm"]
        fit = 1
        while fit < 64 and MD.burst_seconds(spec, fit + 1) <= limit:
            fit += 1
        return min(n - extra, max(fit, G.ctl_slots(spec) + HALF_BURST_MIN_DATA)) + extra
    G.slots_for = capped


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("channel", help="awgn | mpg | mpp | mpd | mps | doppler_hz:delay_ms")
    ap.add_argument("snr", type=float, help="dB, PEP over noise in 3000 Hz")
    ap.add_argument("--bytes", type=int, default=20000)
    ap.add_argument("--bw", choices=["500", "1200", "2300", "2750"], default="2300")
    ap.add_argument("--seed", type=int, default=0, help="first seed")
    ap.add_argument("--latency", type=float, default=T.LATENCY_S)
    ap.add_argument("--half-burst", action="store_true", help="cap data bursts: 6 s OFDM, 12 s FSK (at least one data codeword)")
    T.log_arg(ap)
    a = ap.parse_args()
    T.log_setup(a)
    if a.half_burst:
        half_burst()

    def airtime(results):  # the sender's top two submodes, pooled over the results
        pooled = {}
        for r in results:
            for k, v in r["airtime"].items():
                pooled[k] = pooled.get(k, 0.0) + v
        return "; " + T.top2(pooled)

    trials.run(lambda seed: T.run(a.channel, a.snr, a.bytes, seed, cap=T.BW["BW" + a.bw], latency=a.latency),
               a.seed, f"{a.channel} {a.snr:g} dB, {a.bytes} B, BW{a.bw}" + (", half bursts" if a.half_burst else ""), airtime)


if __name__ == "__main__":
    main()
