"""speedtest.py over seeds `--seed`, `--seed` + 1, ... until 5 transfers
complete or 10 fail. Reports the success rate and the mean speed: of the
middle three with 5 successes (high and low dropped), else of all of them.

    python scripts/cpu_profile/speedtrials.py <channel> <snr_db> [--bytes 20000] [--bw 2300] [--seed 0] [--callee-sends]
"""
import argparse

import speedtest as T  # first: it sets the thread caps before numpy loads

import trials  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("channel", help="awgn | mpg | mpp | mpd | mps | doppler_hz:delay_ms")
    ap.add_argument("snr", type=float, help="dB, PEP over noise in 3000 Hz")
    ap.add_argument("--bytes", type=int, default=20000)
    ap.add_argument("--bw", choices=["500", "1200", "2300", "2750"], default="2300")
    ap.add_argument("--seed", type=int, default=0, help="first seed")
    ap.add_argument("--latency", type=float, default=T.LATENCY_S)
    ap.add_argument("--callee-sends", action="store_true", help="B (the callee) sends, A receives")
    T.log_arg(ap)
    a = ap.parse_args()
    T.log_setup(a)

    def airtime(results):  # the sender's top two submodes, pooled over the results
        pooled = {}
        for r in results:
            for k, v in r["airtime"].items():
                pooled[k] = pooled.get(k, 0.0) + v
        return "; " + T.top2(pooled)

    trials.run(lambda seed: T.run(a.channel, a.snr, a.bytes, seed, cap=T.BW["BW" + a.bw], latency=a.latency,
                                    callee_sends=a.callee_sends),
               a.seed, f"{a.channel} {a.snr:g} dB, {a.bytes} B, BW{a.bw}{', callee sends' * a.callee_sends}", airtime)


if __name__ == "__main__":
    main()
