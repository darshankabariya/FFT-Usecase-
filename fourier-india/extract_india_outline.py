#!/usr/bin/env python3
"""Extract India's mainland border from a Natural Earth GeoJSON file.

    python extract_india_outline.py ne_10m_admin_0_countries_ind.geojson

Natural Earth (public domain) publishes "point of view" country files; the
``*_ind`` variant draws India's borders as India maps them.  Download it from
https://github.com/nvkelso/natural-earth-vector/tree/master/geojson

The largest polygon (the mainland) is kept, the Andaman, Nicobar and
Lakshadweep islands are dropped because a single closed curve is what the
Fourier animation needs, and the ring is simplified with Douglas-Peucker so
the committed JSON stays small.  The output is ``india_outline.json`` next to
this script unless ``-o`` is given.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent


def ring_area(ring: np.ndarray) -> float:
    x, y = ring[:, 0], ring[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def douglas_peucker(pts: np.ndarray, tol: float) -> np.ndarray:
    """Iterative Douglas-Peucker on an open polyline."""
    keep = np.zeros(len(pts), dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        seg = pts[b] - pts[a]
        rel = pts[a + 1:b] - pts[a]
        norm = np.hypot(*seg)
        if norm == 0:
            d = np.hypot(rel[:, 0], rel[:, 1])
        else:
            d = np.abs(rel[:, 0] * seg[1] - rel[:, 1] * seg[0]) / norm
        i = int(np.argmax(d))
        if d[i] > tol:
            idx = a + 1 + i
            keep[idx] = True
            stack.append((a, idx))
            stack.append((idx, b))
    return pts[keep]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("geojson", type=pathlib.Path)
    ap.add_argument("-o", "--output", type=pathlib.Path, default=HERE / "india_outline.json")
    ap.add_argument("--tolerance", type=float, default=0.01,
                    help="Douglas-Peucker tolerance in degrees (default: %(default)s)")
    args = ap.parse_args()

    with open(args.geojson) as fh:
        data = json.load(fh)

    feature = next(f for f in data["features"]
                   if f["properties"].get("ADMIN") == "India" or f["properties"].get("NAME") == "India")
    geom = feature["geometry"]
    polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
    rings = [np.asarray(p[0], dtype=float) for p in polys]
    mainland = max(rings, key=ring_area)
    if np.allclose(mainland[0], mainland[-1]):
        mainland = mainland[:-1]

    # Simplify. Rotate the ring so the DP end-points are not adjacent twice.
    closed = np.vstack([mainland, mainland[:1]])
    simplified = douglas_peucker(closed, args.tolerance)[:-1]

    out = {
        "name": "India (mainland)",
        "source": f"Natural Earth {data.get('name', '')}".strip(),
        "license": "Natural Earth data is in the public domain",
        "crs": "WGS84 lon/lat (degrees)",
        "tolerance_deg": args.tolerance,
        "coordinates": [[round(float(x), 4), round(float(y), 4)] for x, y in simplified],
    }
    with open(args.output, "w") as fh:
        json.dump(out, fh, separators=(",", ":"))
    print(f"{len(mainland)} points -> {len(simplified)} after simplification, saved {args.output}")


if __name__ == "__main__":
    main()
