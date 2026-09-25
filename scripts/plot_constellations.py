"""Scatter plot of every learned constellation a submode uses, with its
bit labels, over the square Gray QAM of the same size at the same average
power (both unit average power, as constellation.load returns them).

    uv run python scripts/plot_constellations.py --out runs/plots
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from data2g import codes, constellation  # noqa: E402
from data2g.config import SUBMODES  # noqa: E402

INK, MUTED, GRID, POINT, REF = "#1C2428", "#5D6A70", "#E4E8E7", "#1F5C85", "#B9C2C6"


def plot(name: str, users: list, path: Path):
    pts = constellation.load(name)
    m = constellation.bits_per_symbol(pts)
    ref = constellation.gray_qam(m)
    fig, ax = plt.subplots(figsize=(6.4, 6.8), dpi=150)
    ax.scatter(ref.real, ref.imag, s=10, color=REF, linewidths=0, label=f"square QAM{2**m}", zorder=1)
    ax.scatter(pts.real, pts.imag, s=26 if m <= 6 else 12, color=POINT, edgecolors="white", linewidths=0.6,
               label="learned", zorder=2)
    if m <= 6:
        for i, p in enumerate(pts):
            ax.annotate(format(i, f"0{m}b"), (p.real, p.imag), xytext=(0, 5.5), textcoords="offset points",
                        ha="center", fontsize=5.2, color=MUTED, zorder=3)
    peak = np.abs(pts).max()
    ax.add_patch(plt.Circle((0, 0), peak, fill=False, color=MUTED, lw=0.6, ls=(0, (3, 3))))
    lim = 1.12 * max(peak, np.abs(ref).max())
    ax.set(xlim=(-lim, lim), ylim=(-lim, lim), aspect="equal")
    ax.axhline(0, color=GRID, lw=0.8, zorder=0)
    ax.axvline(0, color=GRID, lw=0.8, zorder=0)
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    papr = 10 * np.log10(peak**2 / np.mean(np.abs(pts) ** 2))
    ax.set_title(f"{name}  ({2**m} points)", color=INK, fontsize=12, loc="left", pad=22)
    ax.text(0, 1.015, "used by " + ", ".join(users) + f"   ·   peak/avg power {papr:.2f} dB (dashed circle)",
            transform=ax.transAxes, fontsize=7.5, color=MUTED)
    ax.legend(loc="lower right", fontsize=7.5, frameon=False, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/plots")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    users = {}
    for s in SUBMODES.values():
        if not s.constellation.startswith("gray-qam"):
            bps = (s.k - codes.crc_bits(s)) / (s.frames_per_cw * 0.144)
            users.setdefault(s.constellation, []).append(f"{s.name} ({bps:.0f} bps)")
    for name, u in sorted(users.items()):
        p = out / f"constellation_{name}.png"
        plot(name, u, p)
        print(p)


if __name__ == "__main__":
    main()
