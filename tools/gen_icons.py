#!/usr/bin/env python3
"""Rasterize native/packaging/data2g.svg into the shipped icon formats.

    tools/gen_icons.py

    data2g.ico     Windows: the .exe resource (data2g.rc.in) and NSIS
    data2g.icns    macOS: Contents/Resources, named by CFBundleIconFile
    icons/*.png    freedesktop hicolor sizes (Linux menus, AppImage)

Committed, so a build needs neither librsvg nor Pillow. Not a CI gate:
rasterizer output is not byte-stable across librsvg versions. Each size is
rendered from the vector rather than downscaled (SSTVAE's lesson: a 16 px
reduction of a large render is a blur).

Needs rsvg-convert and Pillow.
"""

import subprocess
import sys
from io import BytesIO
from pathlib import Path

from PIL import Image

PKG = Path(__file__).resolve().parent.parent / "native" / "packaging"
SVG = PKG / "data2g.svg"
PNG_SIZES = (16, 24, 32, 48, 64, 128, 256, 512)
ICO_SIZES = (16, 24, 32, 48, 64, 128, 256)
ICNS_SIZES = (16, 32, 64, 128, 256, 512, 1024)


def render(size):
    png = subprocess.run(["rsvg-convert", "-w", str(size), "-h", str(size), str(SVG)],
                         check=True, stdout=subprocess.PIPE).stdout
    return Image.open(BytesIO(png)).convert("RGBA")


def main():
    images = {s: render(s) for s in sorted({*PNG_SIZES, *ICO_SIZES, *ICNS_SIZES})}
    (PKG / "icons").mkdir(exist_ok=True)
    for s in PNG_SIZES:
        images[s].save(PKG / "icons" / f"data2g-{s}.png", optimize=True)
    images[256].save(PKG / "data2g.ico", sizes=[(s, s) for s in ICO_SIZES],
                     append_images=[images[s] for s in ICO_SIZES])
    images[1024].save(PKG / "data2g.icns", append_images=[images[s] for s in ICNS_SIZES[:-1]])
    print(f"wrote icons under {PKG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
