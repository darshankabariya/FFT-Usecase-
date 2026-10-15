#!/usr/bin/env python3
"""Turn a picture into a CNC engraving toolpath (G-code).

Two ways of reading a picture:

  contours    (default) The dark areas of the image are the design.  Their
              outlines become closed cutting loops, ordered so the tool
              travels as little as possible between them.  Good for logos,
              lettering, silhouettes, line drawings, anything black on white.

  singleline  The whole picture becomes ONE continuous line that never
              lifts: dark areas get many points, light areas few, and the
              points are joined into a single tour ("TSP art").  Good for
              portraits and photos engraved with a V-bit or a pen.

The G-code is plain GRBL-style (G21 / G90 / G0 / G1), which every hobby
controller reads.  Co-ordinates are in millimetres, X right, Y up, Z=0 on
the top of the wood and negative into it.  Nothing here compensates for
the cutter diameter: the tool runs along the drawn line, as an engraver or
V-bit does.  For a cut-out you still need to offset the path by the tool
radius in your CAM program.

Examples
--------
    python image_to_gcode.py logo.png -o logo.nc --width-mm 150 --depth 1.0
    python image_to_gcode.py logo.png -o logo.nc --smooth 40          # Fourier-smoothed outlines
    python image_to_gcode.py photo.jpg -o photo.nc --mode singleline --points 4000
"""
from __future__ import annotations

import argparse
import math
import pathlib
import sys
import time

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# image -> paths (lists of (x, y) in millimetres, machine orientation)
# --------------------------------------------------------------------------- #
def load_gray(path: pathlib.Path, blur: int) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise SystemExit(f"cannot read image {path}")
    if img.ndim == 3 and img.shape[2] == 4:            # flatten transparency onto white
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        rgb = img[:, :, :3].astype(np.float32)
        img = (rgb * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if blur > 0:
        k = 2 * blur + 1
        img = cv2.GaussianBlur(img, (k, k), 0)
    return img


def to_mask(gray: np.ndarray, threshold: str, invert: bool) -> np.ndarray:
    """Binary mask of the design (255 = cut here)."""
    if threshold == "auto":
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        _, mask = cv2.threshold(gray, int(threshold), 255, cv2.THRESH_BINARY_INV)
    if invert:
        mask = 255 - mask
    return mask


def contour_paths(mask: np.ndarray, scale: float, simplify_mm: float, min_len_mm: float,
                  smooth: int, step_mm: float) -> list[np.ndarray]:
    """Outlines of every blob and hole in the mask, as closed loops in mm."""
    h = mask.shape[0]
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    paths = []
    for c in contours:
        if len(c) < 4:
            continue
        if smooth > 0:
            pts = c[:, 0, :].astype(float)
            pts = fourier_smooth(pts, smooth, step_mm / scale)
        else:
            eps = simplify_mm / scale
            pts = cv2.approxPolyDP(c, eps, True)[:, 0, :].astype(float)
        if len(pts) < 3:
            continue
        xy = np.column_stack([pts[:, 0] * scale, (h - 1 - pts[:, 1]) * scale])
        if loop_length(xy) >= min_len_mm:
            paths.append(xy)
    return paths


def fourier_smooth(pts: np.ndarray, harmonics: int, step_px: float) -> np.ndarray:
    """Low-pass a closed pixel contour: keep frequencies |k| <= harmonics.

    This is where the Fourier transform earns its place in a CNC pipeline:
    a traced bitmap outline is jagged at pixel scale, and dropping the high
    frequencies gives a smooth curve the cutter can follow at speed.  The
    curve is then sampled back to straight segments about *step_px* apart,
    because the controller only understands lines.
    """
    z = pts[:, 0] + 1j * pts[:, 1]
    z = np.append(z, z[0])
    seg = np.abs(np.diff(z))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    if total <= 0:
        return pts
    m = max(64, int(2 ** math.ceil(math.log2(max(len(pts), 4 * harmonics)))))
    t = np.linspace(0.0, total, m, endpoint=False)
    zr = np.interp(t, s, z.real) + 1j * np.interp(t, s, z.imag)
    c = np.fft.fft(zr)
    k = np.fft.fftfreq(m, d=1.0 / m)
    c[np.abs(k) > harmonics] = 0.0
    n_out = max(16, int(total / max(step_px, 1e-6)))
    tt = np.arange(n_out) / n_out
    kk = k[np.abs(k) <= harmonics]
    cc = c[np.abs(k) <= harmonics] / m
    zs = (cc[None, :] * np.exp(2j * np.pi * kk[None, :] * tt[:, None])).sum(axis=1)
    return np.column_stack([zs.real, zs.imag])


def singleline_path(gray: np.ndarray, scale: float, n_points: int, gamma: float, invert: bool,
                    seed: int, opt_passes: int) -> np.ndarray:
    """One continuous open path through the picture (TSP art)."""
    h, w = gray.shape
    dark = 1.0 - gray.astype(np.float64) / 255.0
    if invert:
        dark = 1.0 - dark
    weight = np.clip(dark, 0.0, 1.0) ** gamma
    weight = weight.ravel()
    if weight.sum() <= 0:
        raise SystemExit("image is blank after thresholding")
    rng = np.random.default_rng(seed)
    n_points = min(n_points, int((weight > 0).sum()))
    idx = rng.choice(weight.size, size=n_points, replace=False, p=weight / weight.sum())
    py, px = np.divmod(idx, w)
    pts = np.column_stack([px + rng.random(n_points) - 0.5, py + rng.random(n_points) - 0.5])
    pts = lloyd_relax(pts, weight.reshape(h, w), iterations=3)
    order = nearest_neighbour_tour(pts)
    order = two_opt(pts, order, opt_passes)
    pts = pts[order]
    return np.column_stack([pts[:, 0] * scale, (h - 1 - pts[:, 1]) * scale])


def lloyd_relax(pts: np.ndarray, weight: np.ndarray, iterations: int) -> np.ndarray:
    """A few weighted Lloyd steps: spread the dots evenly within dark areas."""
    h, w = weight.shape
    yy, xx = np.mgrid[0:h, 0:w]
    flat_xy = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float64)
    wflat = weight.ravel()
    live = wflat > 0
    flat_xy, wflat = flat_xy[live], wflat[live]
    for _ in range(iterations):
        # nearest site for every pixel, in chunks to bound memory
        owner = np.empty(len(flat_xy), dtype=np.int64)
        s32 = pts.astype(np.float32)
        s_sq = (s32 ** 2).sum(axis=1)
        for a in range(0, len(flat_xy), 4096):
            q = flat_xy[a:a + 4096].astype(np.float32)
            d = (q ** 2).sum(axis=1)[:, None] - 2.0 * (q @ s32.T) + s_sq[None, :]
            owner[a:a + 4096] = d.argmin(axis=1)
        sw = np.bincount(owner, weights=wflat, minlength=len(pts))
        sx = np.bincount(owner, weights=wflat * flat_xy[:, 0], minlength=len(pts))
        sy = np.bincount(owner, weights=wflat * flat_xy[:, 1], minlength=len(pts))
        ok = sw > 0
        pts = pts.copy()
        pts[ok, 0] = sx[ok] / sw[ok]
        pts[ok, 1] = sy[ok] / sw[ok]
    return pts


def nearest_neighbour_tour(pts: np.ndarray) -> np.ndarray:
    n = len(pts)
    visited = np.zeros(n, dtype=bool)
    order = np.empty(n, dtype=np.int64)
    cur = int(np.argmin(pts[:, 0] + pts[:, 1]))     # start near the bottom-left corner
    for i in range(n):
        order[i] = cur
        visited[cur] = True
        d = ((pts - pts[cur]) ** 2).sum(axis=1)
        d[visited] = np.inf
        if i < n - 1:
            cur = int(np.argmin(d))
    return order


def two_opt(pts: np.ndarray, order: np.ndarray, passes: int) -> np.ndarray:
    """2-opt improvement of an OPEN path (no return to start)."""
    p = pts[order]
    n = len(p)
    for _ in range(passes):
        improved = False
        for i in range(1, n - 2):
            a, b = p[i - 1], p[i]
            c = p[i + 1:n - 1]
            d = p[i + 2:n]
            gain = (np.hypot(*(b - a)) + np.hypot(c[:, 0] - d[:, 0], c[:, 1] - d[:, 1])) \
                   - (np.hypot(c[:, 0] - a[0], c[:, 1] - a[1]) + np.hypot(d[:, 0] - b[0], d[:, 1] - b[1]))
            j = int(np.argmax(gain))
            if gain[j] > 1e-9:
                k = i + 1 + j
                p[i:k + 1] = p[i:k + 1][::-1]
                order[i:k + 1] = order[i:k + 1][::-1]
                improved = True
        # also try reversing a tail (open path, free end)
        a = p[:-1]
        gain = np.hypot(a[:, 0] - p[1:, 0], a[:, 1] - p[1:, 1]) - np.hypot(a[:, 0] - p[-1, 0], a[:, 1] - p[-1, 1])
        j = int(np.argmax(gain))
        if gain[j] > 1e-9:
            p[j + 1:] = p[j + 1:][::-1]
            order[j + 1:] = order[j + 1:][::-1]
            improved = True
        if not improved:
            break
    return order


# --------------------------------------------------------------------------- #
# path ordering and G-code
# --------------------------------------------------------------------------- #
def loop_length(xy: np.ndarray) -> float:
    d = np.diff(np.vstack([xy, xy[:1]]), axis=0)
    return float(np.hypot(d[:, 0], d[:, 1]).sum())


def path_length(xy: np.ndarray) -> float:
    d = np.diff(xy, axis=0)
    return float(np.hypot(d[:, 0], d[:, 1]).sum())


def order_loops(loops: list[np.ndarray], start=(0.0, 0.0)) -> list[np.ndarray]:
    """Greedy nearest-loop ordering; each loop is rotated to begin at the vertex nearest the tool."""
    remaining = list(loops)
    pos = np.asarray(start, dtype=float)
    out = []
    while remaining:
        best_i, best_j, best_d = -1, -1, np.inf
        for i, lp in enumerate(remaining):
            d = ((lp - pos) ** 2).sum(axis=1)
            j = int(np.argmin(d))
            if d[j] < best_d:
                best_i, best_j, best_d = i, j, d[j]
        lp = remaining.pop(best_i)
        lp = np.roll(lp, -best_j, axis=0)
        out.append(lp)
        pos = lp[0]
    return out


def fmt(v: float) -> str:
    return f"{v:.3f}".rstrip("0").rstrip(".") if abs(v) > 1e-9 else "0"


def write_gcode(paths: list[np.ndarray], closed: bool, args, out: pathlib.Path, bbox) -> dict:
    depth_steps = [args.depth] if args.stepdown <= 0 else \
        [min(args.depth, args.stepdown * i) for i in range(1, math.ceil(args.depth / args.stepdown) + 1)]
    lines = [
        f"( {out.name} - generated by image_to_gcode.py )",
        f"( source: {args.image.name}  mode: {args.mode}  size: {fmt(bbox[0])} x {fmt(bbox[1])} mm )",
        f"( depth {fmt(args.depth)} mm in {len(depth_steps)} pass(es), feed {args.feed}, plunge {args.plunge}, safe Z {fmt(args.safe_z)} )",
        "( Z0 = top of the stock, X/Y origin = " + ("centre of the design" if args.origin == "center" else "bottom-left corner of the design") + " )",
        "G21 ( mm )",
        "G90 ( absolute )",
        "G17 ( XY plane )",
        "G94 ( feed per minute )",
        f"G0 Z{fmt(args.safe_z)}",
    ]
    if args.spindle > 0:
        lines.append(f"M3 S{args.spindle}")
        lines.append("G4 P2 ( let the spindle spin up )")
    cut_len = 0.0
    rapid_len = 0.0
    pos = np.array([0.0, 0.0])
    for n, xy in enumerate(paths, 1):
        lines.append(f"( path {n} of {len(paths)} )")
        rapid_len += float(np.hypot(*(xy[0] - pos)))
        lines.append(f"G0 X{fmt(xy[0, 0])} Y{fmt(xy[0, 1])}")
        for p, d in enumerate(depth_steps):
            lines.append(f"G1 Z{fmt(-d)} F{args.plunge}")
            if closed:
                seq = np.vstack([xy[1:], xy[:1]])
            else:
                seq = xy[1:] if p % 2 == 0 else xy[::-1][1:]
            first = True
            for x, y in seq:
                lines.append(f"G1 X{fmt(x)} Y{fmt(y)}" + (f" F{args.feed}" if first else ""))
                first = False
            cut_len += (loop_length(xy) if closed else path_length(xy))
            pos = (xy[0] if closed else (xy[-1] if p % 2 == 0 else xy[0]))
            if not closed and p < len(depth_steps) - 1:
                pass  # next pass starts where this one ended
        lines.append(f"G0 Z{fmt(args.safe_z)}")
    if args.spindle > 0:
        lines.append("M5")
    lines.append("G0 X0 Y0")
    lines.append("M30")
    out.write_text("\n".join(lines) + "\n")
    minutes = cut_len / args.feed + len(paths) * len(depth_steps) * (args.depth / args.plunge) + rapid_len / 3000.0
    return {"paths": len(paths), "passes": len(depth_steps), "cut_mm": cut_len, "rapid_mm": rapid_len,
            "plunges": len(paths) * len(depth_steps), "minutes": minutes, "lines": len(lines)}


# --------------------------------------------------------------------------- #
# previews, generated by re-reading the G-code (so they show what the machine will do)
# --------------------------------------------------------------------------- #
def parse_gcode(path: pathlib.Path):
    x = y = z = 0.0
    cuts, rapids = [], []
    for raw in path.read_text().splitlines():
        line = raw.split("(")[0].split(";")[0].strip().upper()
        if not line or not (line.startswith("G0") or line.startswith("G1")):
            continue
        words = line.split()
        nx, ny, nz = x, y, z
        for w in words[1:]:
            if w[0] == "X": nx = float(w[1:])
            elif w[0] == "Y": ny = float(w[1:])
            elif w[0] == "Z": nz = float(w[1:])
        rapid = words[0] in ("G0", "G00")
        if (nx, ny) != (x, y):
            (rapids if rapid or nz > 0 else cuts).append(((x, y), (nx, ny)))
        x, y, z = nx, ny, nz
    return cuts, rapids


def write_preview(gcode: pathlib.Path, png: pathlib.Path, stats: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    cuts, rapids = parse_gcode(gcode)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_facecolor("#f4efe3")
    if rapids:
        ax.add_collection(LineCollection(rapids, colors="#d9534f", linewidths=0.6, linestyles=(0, (3, 3)), alpha=0.7))
    if cuts:
        ax.add_collection(LineCollection(cuts, colors="#2b2b2b", linewidths=0.9))
    ax.autoscale()
    ax.set_aspect("equal")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.grid(True, color="#d6cfbf", lw=0.5)
    ax.set_title(f"{gcode.name}: {stats['paths']} path(s), cut {stats['cut_mm']:.0f} mm, "
                 f"rapid {stats['rapid_mm']:.0f} mm, ~{stats['minutes']:.0f} min", fontsize=10)
    fig.tight_layout()
    fig.savefig(png, dpi=110)
    plt.close(fig)


def write_svg(gcode: pathlib.Path, svg: pathlib.Path):
    cuts, _ = parse_gcode(gcode)
    if not cuts:
        return
    xs = [p for seg in cuts for p in (seg[0][0], seg[1][0])]
    ys = [p for seg in cuts for p in (seg[0][1], seg[1][1])]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    w, h = x1 - x0 + 2, y1 - y0 + 2
    d = []
    last = None
    for a, b in cuts:
        if a != last:
            d.append(f"M{a[0]-x0+1:.3f} {y1-a[1]+1:.3f}")
        d.append(f"L{b[0]-x0+1:.3f} {y1-b[1]+1:.3f}")
        last = b
    svg.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w:.2f}mm" height="{h:.2f}mm" viewBox="0 0 {w:.3f} {h:.3f}">\n'
        f'<path d="{" ".join(d)}" fill="none" stroke="#000" stroke-width="0.2"/>\n</svg>\n')


# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", type=pathlib.Path)
    ap.add_argument("-o", "--output", type=pathlib.Path, help="G-code file (default: <image>.nc)")
    ap.add_argument("--mode", choices=["contours", "singleline"], default="contours")
    ap.add_argument("--width-mm", type=float, default=150.0, help="width of the finished design (default %(default)s)")
    ap.add_argument("--origin", choices=["corner", "center"], default="corner",
                    help="where X0 Y0 is: bottom-left corner or centre of the design")
    ap.add_argument("--blur", type=int, default=1, help="pre-blur radius in px, 0 = off (default %(default)s)")
    ap.add_argument("--threshold", default="auto", help="'auto' (Otsu) or 0-255; darker than this = design")
    ap.add_argument("--invert", action="store_true", help="light areas are the design instead of dark ones")
    # contours
    ap.add_argument("--simplify-mm", type=float, default=0.15, help="polyline tolerance (default %(default)s)")
    ap.add_argument("--min-length-mm", type=float, default=3.0, help="drop outlines shorter than this")
    ap.add_argument("--smooth", type=int, default=0,
                    help="Fourier low-pass: keep this many harmonics per outline (0 = off, 30-80 is a good range)")
    ap.add_argument("--step-mm", type=float, default=0.3, help="segment length after Fourier smoothing")
    # singleline
    ap.add_argument("--points", type=int, default=3000, help="dots for singleline mode (default %(default)s)")
    ap.add_argument("--gamma", type=float, default=1.6, help="contrast of the dot density (default %(default)s)")
    ap.add_argument("--opt-passes", type=int, default=6, help="2-opt improvement passes (default %(default)s)")
    ap.add_argument("--seed", type=int, default=1)
    # machine
    ap.add_argument("--depth", type=float, default=1.0, help="total cut depth in mm (default %(default)s)")
    ap.add_argument("--stepdown", type=float, default=0.0, help="depth per pass, 0 = one pass")
    ap.add_argument("--safe-z", type=float, default=5.0)
    ap.add_argument("--feed", type=int, default=800, help="XY feed, mm/min (default %(default)s)")
    ap.add_argument("--plunge", type=int, default=300, help="Z feed, mm/min (default %(default)s)")
    ap.add_argument("--spindle", type=int, default=10000, help="spindle rpm for M3, 0 = no M3/M5")
    ap.add_argument("--preview", type=pathlib.Path, help="PNG preview of the toolpath (default: <output>_toolpath.png)")
    ap.add_argument("--svg", type=pathlib.Path, help="also write the cut path as SVG")
    ap.add_argument("--no-preview", action="store_true")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    t0 = time.time()
    out = args.output or args.image.with_suffix(".nc")
    gray = load_gray(args.image, args.blur)
    h, w = gray.shape
    if args.mode == "singleline":
        # work at a modest resolution: the dot count sets the detail, not the pixels
        target = 420
        if max(h, w) > target:
            f = target / max(h, w)
            gray = cv2.resize(gray, (max(1, int(w * f)), max(1, int(h * f))), interpolation=cv2.INTER_AREA)
            h, w = gray.shape
    scale = args.width_mm / w

    if args.mode == "contours":
        mask = to_mask(gray, args.threshold, args.invert)
        loops = contour_paths(mask, scale, args.simplify_mm, args.min_length_mm, args.smooth, args.step_mm)
        if not loops:
            raise SystemExit("no outlines found: try --invert, --threshold, or a lower --min-length-mm")
        paths, closed = loops, True
    else:
        if args.threshold != "auto":
            gray = to_mask(gray, args.threshold, False)
            gray = 255 - gray
        paths, closed = [singleline_path(gray, scale, args.points, args.gamma, args.invert, args.seed, args.opt_passes)], False

    # origin
    allpts = np.vstack(paths)
    lo, hi = allpts.min(axis=0), allpts.max(axis=0)
    shift = lo if args.origin == "corner" else (lo + hi) / 2
    paths = [p - shift for p in paths]
    if closed:
        paths = order_loops(paths, start=(0.0, 0.0) if args.origin == "corner" else tuple(-(hi - lo) / 2))

    stats = write_gcode(paths, closed, args, out, hi - lo)
    print(f"{out}: {stats['paths']} path(s) x {stats['passes']} pass(es), {stats['lines']} lines, "
          f"cut {stats['cut_mm']:.0f} mm, rapid {stats['rapid_mm']:.0f} mm, {stats['plunges']} plunge(s), "
          f"about {stats['minutes']:.1f} min at F{args.feed}  [{time.time() - t0:.1f}s]")
    if not args.no_preview:
        png = args.preview or out.with_name(out.stem + "_toolpath.png")
        write_preview(out, png, stats)
        print(f"preview: {png}")
    if args.svg:
        write_svg(out, args.svg)
        print(f"svg: {args.svg}")


if __name__ == "__main__":
    main()
