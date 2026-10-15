#!/usr/bin/env python3
"""Fourier epicycle animation that traces the border of India.

The outline of India is treated as a closed curve in the complex plane
(x + iy).  Its discrete Fourier transform gives a set of rotating vectors
("epicycles"): each coefficient c_k is a circle of radius |c_k| whose arm
rotates k times per period, starting at phase arg(c_k).  Chaining the
first N circles (largest first) and following the tip of the chain
re-draws the border, and the more circles you keep the closer the trace
gets to the real coastline.

Examples
--------
    # side-by-side comparison of 5, 10, 15, 100 and 200 circles (default)
    python fourier_india.py

    # one big panel with 100 circles, saved as mp4
    python fourier_india.py --circles 100 -o india_100.mp4

    # custom levels, slower / smoother
    python fourier_india.py --circles 3 25 500 --frames 600 --fps 60
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_OUTLINE = HERE / "india_outline.json"
DEFAULT_LEVELS = [5, 10, 15, 100, 200]

# colours
C_TARGET = "#c8c8c8"   # faint real border, drawn underneath
C_CIRCLE = "#6fa8dc"   # epicycle circles
C_ARM = "#1f3f66"      # arms joining circle centres
C_TRACE = "#e4572e"    # curve drawn by the tip
C_BG = "#fbfbf8"


# --------------------------------------------------------------------------- #
# geometry helpers
# --------------------------------------------------------------------------- #
def load_outline(path: pathlib.Path) -> np.ndarray:
    """Load lon/lat pairs and return a centred, unit-scaled complex curve."""
    with open(path) as fh:
        data = json.load(fh)
    pts = np.asarray(data["coordinates"], dtype=float)
    lon, lat = pts[:, 0], pts[:, 1]
    # Equirectangular projection: shrink longitude by cos(latitude) so the
    # map keeps the proportions you are used to seeing.
    lat0 = math.radians(lat.mean())
    x = (lon - lon.mean()) * math.cos(lat0)
    y = lat - lat.mean()
    z = x + 1j * y
    return z / np.abs(z).max()


def resample(z: np.ndarray, m: int) -> np.ndarray:
    """Resample a closed curve to *m* points equally spaced along its length.

    Uniform spacing in arc length is what makes the Fourier coefficients
    behave: it stops densely digitised stretches of coast from dominating
    the transform.
    """
    if abs(z[0] - z[-1]) > 1e-12:
        z = np.append(z, z[0])
    seg = np.abs(np.diff(z))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    t = np.linspace(0.0, s[-1], m, endpoint=False)
    zr = np.interp(t, s, z.real) + 1j * np.interp(t, s, z.imag)
    return zr - zr.mean()  # centre so the k = 0 term vanishes


def fourier_coefficients(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """DFT of the curve, sorted by decreasing circle radius.

    Returns (coefficients, integer frequencies).  The k = 0 term is only a
    translation, so it is dropped: the shape is already centred.
    """
    m = len(z)
    c = np.fft.fft(z) / m
    k = np.rint(np.fft.fftfreq(m, d=1.0 / m)).astype(int)
    order = np.argsort(-np.abs(c), kind="stable")
    order = order[k[order] != 0]
    return c[order], k[order]


def epicycle_centres(c: np.ndarray, k: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Positions of every joint of the epicycle chain for each time in *t*.

    Result has shape (len(t), len(c) + 1).  Column 0 is the origin, column
    j is the tip of the first j arms, and the last column is the pen.
    """
    terms = c[None, :] * np.exp(2j * np.pi * k[None, :] * t[:, None])
    cum = np.cumsum(terms, axis=1)
    return np.concatenate([np.zeros((len(t), 1), dtype=complex), cum], axis=1)


# --------------------------------------------------------------------------- #
# drawing
# --------------------------------------------------------------------------- #
class Panel:
    """One axes showing the chain of the first *n* circles."""

    _theta = np.linspace(0.0, 2.0 * np.pi, 49)

    def __init__(self, ax, c, k, n, target, times):
        self.n = n
        self.c, self.k = c[:n], k[:n]
        self.radii = np.abs(self.c)
        self.centres = epicycle_centres(self.c, self.k, times)
        self.path = self.centres[:, -1]

        ax.set_facecolor(C_BG)
        ax.set_aspect("equal")
        ax.set_xlim(-1.18, 1.18)
        ax.set_ylim(-1.18, 1.18)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(f"{n} circle{'s' if n != 1 else ''}", fontsize=12, pad=6)

        ax.plot(target.real, target.imag, color=C_TARGET, lw=0.8, zorder=1)
        (self.circle_line,) = ax.plot([], [], color=C_CIRCLE, lw=0.6, alpha=0.8, zorder=2)
        (self.arm_line,) = ax.plot([], [], color=C_ARM, lw=0.9, zorder=3)
        (self.trace_line,) = ax.plot([], [], color=C_TRACE, lw=1.8, zorder=4)
        (self.tip,) = ax.plot([], [], "o", color=C_TRACE, ms=4, zorder=5)
        self.artists = (self.circle_line, self.arm_line, self.trace_line, self.tip)

    def draw(self, frame: int):
        cen = self.centres[frame]
        # all circles in a single polyline, separated by NaNs (fast to draw)
        circ = cen[:-1, None] + self.radii[:, None] * np.exp(1j * self._theta)[None, :]
        circ = np.concatenate([circ, np.full((self.n, 1), np.nan + 1j * np.nan)], axis=1).ravel()
        self.circle_line.set_data(circ.real, circ.imag)
        self.arm_line.set_data(cen.real, cen.imag)
        drawn = self.path[: frame + 1]
        self.trace_line.set_data(drawn.real, drawn.imag)
        self.tip.set_data([cen[-1].real], [cen[-1].imag])
        return self.artists


def build_animation(z, levels, frames, fps, hold):
    c, k = fourier_coefficients(z)
    if max(levels) > len(c):
        raise SystemExit(f"--circles cannot exceed {len(c)} (increase --samples)")

    times = np.arange(frames) / frames
    n = len(levels)
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n + 0.4, 4.1), squeeze=False)
    fig.patch.set_facecolor(C_BG)
    fig.suptitle("Tracing India's border with Fourier epicycles", fontsize=14, y=0.98)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.03, wspace=0.03)

    panels = [Panel(ax, c, k, lvl, z, times) for ax, lvl in zip(axes[0], levels)]

    total = frames + hold

    def update(i):
        f = min(i, frames - 1)
        artists = []
        for p in panels:
            artists.extend(p.draw(f))
        return artists

    anim = FuncAnimation(fig, update, frames=total, interval=1000 / fps, blit=True)
    return fig, anim


def save(anim, out: pathlib.Path, fps: int, dpi: int):
    out.parent.mkdir(parents=True, exist_ok=True)
    ext = out.suffix.lower()
    if ext == ".gif":
        writer = PillowWriter(fps=fps)
    elif ext in {".mp4", ".webm", ".mov"}:
        writer = FFMpegWriter(fps=fps, bitrate=-1,
                              extra_args=["-pix_fmt", "yuv420p", "-crf", "22"])
    else:
        raise SystemExit("output must end in .gif, .mp4, .webm or .mov")
    anim.save(str(out), writer=writer, dpi=dpi)


# --------------------------------------------------------------------------- #
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--circles", "-n", type=int, nargs="+", default=DEFAULT_LEVELS,
                    help="number of epicycles per panel (default: %(default)s)")
    ap.add_argument("--outline", type=pathlib.Path, default=DEFAULT_OUTLINE,
                    help="JSON file with the border coordinates")
    ap.add_argument("--samples", "-m", type=int, default=2048,
                    help="points the border is resampled to before the DFT (default: %(default)s)")
    ap.add_argument("--frames", type=int, default=240, help="frames for one full trace (default: %(default)s)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--hold", type=int, default=30,
                    help="extra frames to pause on the finished drawing (default: %(default)s)")
    ap.add_argument("--dpi", type=int, default=100)
    ap.add_argument("--output", "-o", type=pathlib.Path, default=HERE / "fourier_india.gif")
    return ap.parse_args()


def main():
    args = parse_args()
    z = resample(load_outline(args.outline), args.samples)
    fig, anim = build_animation(z, args.circles, args.frames, args.fps, args.hold)
    save(anim, args.output, args.fps, args.dpi)
    plt.close(fig)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
