"""
Surface-texture measures that separate water-carved (fluvial) terrain from lunar-like terrain.

* depression_frac : share of the box inside closed depressions (> 0.5 m deep after pit filling).
                    Cratered surfaces have many closed pits; river-drained surfaces have almost none.
* drainage_frac   : share of cells that collect flow from >= CHANNEL_CELLS upslope cells (D8 on the
                    pit-filled surface), i.e. how much of the box is a channel network.
* hp_ratio        : fine-scale relief (< ~50 m) relative to mid-scale relief (~50-200 m).
                    Gullies put energy in the mid band; small craters and blocks in the fine band.
"""

from __future__ import annotations

import heapq
import math

import numpy as np
from scipy.ndimage import gaussian_filter

CHANNEL_CELLS = 100           # 100 cells x 900 m^2 = 0.09 km^2 contributing area
NB = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def fill_pits(z: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """Priority-flood (+epsilon) pit filling from the box edge."""
    n0, n1 = z.shape
    f = z.copy()
    seen = np.zeros(z.shape, bool)
    pq = []
    for r in range(n0):
        for c in (0, n1 - 1):
            heapq.heappush(pq, (f[r, c], r, c)); seen[r, c] = True
    for c in range(1, n1 - 1):
        for r in (0, n0 - 1):
            heapq.heappush(pq, (f[r, c], r, c)); seen[r, c] = True
    while pq:
        h, r, c = heapq.heappop(pq)
        for dr, dc in NB:
            rr, cc = r + dr, c + dc
            if 0 <= rr < n0 and 0 <= cc < n1 and not seen[rr, cc]:
                seen[rr, cc] = True
                if f[rr, cc] <= h:
                    f[rr, cc] = h + eps
                heapq.heappush(pq, (f[rr, cc], rr, cc))
    return f


def flow_accumulation(f: np.ndarray) -> np.ndarray:
    """D8 flow accumulation (cells) on a pit-filled surface."""
    n0, n1 = f.shape
    acc = np.ones(f.size)
    order = np.argsort(-f, axis=None)
    flat = f.ravel()
    for idx in order:
        r, c = divmod(int(idx), n1)
        best, bj = 0.0, -1
        for dr, dc in NB:
            rr, cc = r + dr, c + dc
            if 0 <= rr < n0 and 0 <= cc < n1:
                drop = (flat[idx] - flat[rr * n1 + cc]) / (1.4142 if dr and dc else 1.0)
                if drop > best:
                    best, bj = drop, rr * n1 + cc
        if bj >= 0:
            acc[bj] += acc[idx]
    return acc.reshape(f.shape)


def texture(z: np.ndarray) -> dict:
    """
    depression_frac : closed pits (craters) share, > 0.5 m deep after filling.
    gully_band      : relief energy at gully scale (~60-300 m) / energy at landform scale (~300-1500 m).
                      Water-carved slopes are rich in the gully band; old lunar slopes are smooth there.
    """
    f = fill_pits(z)
    dep = float(((f - z) > 0.5).mean())
    b1 = np.std(gaussian_filter(z, 1) - gaussian_filter(z, 5))
    b2 = np.std(gaussian_filter(z, 5) - gaussian_filter(z, 25))
    return {"depression_frac": dep, "gully_band": float(b1 / (b2 + 1e-6))}


def texture_distance(e: dict, t: dict) -> float:
    """0 = same texture as the target; about 1 per clearly different term."""
    d_dep = (e["depression_frac"] - t["depression_frac"]) / 0.03
    d_band = math.log((e["gully_band"] + 1e-3) / (t["gully_band"] + 1e-3)) / math.log(1.5)
    return float(math.sqrt(d_dep ** 2 + d_band ** 2))
