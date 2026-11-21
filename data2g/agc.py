"""Simulation only, not on the live path and not ported to native.

A receiver's AGC for the session studies (issue #257): a power envelope
follower (attack when the power rises, release when it falls) whose gain
holds the audio's power at `thresh_db` over the idle floor, falling at most
`depth_db`. Idle gain is 1, so the floor between bursts is unchanged.

On air (two volunteer nodes, Nov 2026, ~2500 bursts) the out-of-band level
under a Data2G burst fell about as much as the burst's snr_est (-5..-7 dB at
5-10 dB, -9..-11 at 10-15, -13..-16 at 15-20, -18..-21 over 20), already in
the burst's first 0.1 s block, and came back over 1-3 s after it (one node a
smooth ~8 dB/s, the other a ~2 s hang then fast). Nothing before a burst.
draw()'s range runs from none to a bit more than that."""

from dataclasses import dataclass

import numpy as np

from .config import FS

STEP = FS // 100  # the follower runs on 10 ms of power (1 ms: idle noise alone pumped it 3 dB)


@dataclass(frozen=True)
class Spec:
    thresh_db: float  # over the idle floor's power: where the gain starts to fall
    depth_db: float  # the most it falls
    attack_s: float
    release_s: float


def draw(rng: np.random.Generator, none_p: float = 0.25) -> Spec | None:
    """One station's AGC: none `none_p` of the time, else threshold 0-6 dB
    (at 0, idle noise alone pumps it ~1 dB), depth 10-30 dB, attack 5-50 ms
    and release 0.1-3 s (log-uniform). The follower sees the noise over
    FS/2, so its view of a burst is ~2 dB under snr_est (SNR_REF_BW_HZ):
    at threshold 0 the gain falls about as far as on air."""
    if rng.random() < none_p:
        return None
    lu = lambda a, b: float(np.exp(rng.uniform(np.log(a), np.log(b))))  # noqa: E731
    return Spec(float(rng.uniform(0, 6)), float(rng.uniform(10, 30)), lu(0.005, 0.05), lu(0.1, 3.0))


def describe(spec: Spec | None) -> str:
    if spec is None:
        return ""
    return f"t{spec.thresh_db:.1f}d{spec.depth_db:.0f}a{spec.attack_s * 1e3:.0f}r{spec.release_s * 1e3:.0f}"


def gain(spec: Spec, y: np.ndarray, floor_power: float) -> np.ndarray:
    """The AGC's amplitude gain per sample of y, from idle (its follower at
    floor_power, the idle noise's power) at y's start."""
    n = -(-len(y) // STEP)
    p = np.pad(y.astype(np.float64) ** 2, (0, n * STEP - len(y))).reshape(n, STEP).mean(axis=1)
    k_att = 1 - np.exp(-1 / (spec.attack_s * FS / STEP))
    k_rel = 1 - np.exp(-1 / (spec.release_s * FS / STEP))
    env = np.empty(n)
    e = floor_power
    for i, v in enumerate(p.tolist()):
        e += (k_att if v > e else k_rel) * (v - e)
        env[i] = e
    g = np.clip(floor_power * 10 ** (spec.thresh_db / 10) / np.maximum(env, 1e-30), 10 ** (-spec.depth_db / 10), 1.0)
    return np.repeat(np.sqrt(g), STEP)[:len(y)]


def apply(spec: Spec | None, y: np.ndarray, floor_power: float) -> np.ndarray:
    """y through the AGC (unchanged without one)."""
    return y if spec is None else y * gain(spec, y, floor_power)


if __name__ == "__main__":
    # a 10 s, 15 dB-over-floor burst after 2 s of noise: idle gain 1, about -15 dB under it, back after
    rng = np.random.default_rng(0)
    x = rng.normal(size=14 * FS)
    x[2 * FS:12 * FS] += np.sqrt(10 ** 1.5) * np.sin(2 * np.pi * 1500 * np.arange(10 * FS) / FS) * np.sqrt(2)
    g = 20 * np.log10(gain(Spec(2.0, 30.0, 0.005, 0.5), x, 1.0))
    assert abs(g[FS]) < 0.5 and -15 < g[7 * FS] < -12 and abs(g[-1]) < 0.5, (g[FS], g[7 * FS], g[-1])
    print("ok")
