"""
3D shape-matching discovery scan: find Earth terrain whose *surface shape* best matches each
target DEM (not only its slope statistics), outside known analog sites.

Stage 1  MODIS land cover (NASA GIBS, IGBP 2024, z7 ~1.2 km). Relaxed cover rule: at least
         MIN_COVER of the 5 x 5 km box is barren (IGBP 16) or open shrubland (IGBP 7).
Stage 2  AWS Terrain Tiles z9 (~250-300 m). Every cell is resampled to a 16 x 16 grid and compared
         with the target (resampled the same way) under the 8 rotations/flips of a square.
         Score = RMSE of mean-removed elevation [m]: low = same shape AND same relief.
Stage 3  Best spaced candidates re-measured at 30 m (169 x 169, same grid as the target) and
         matched over 24 rotation angles x mirror inside the central circle. Reports RMSE [m],
         elevation correlation (shape only, scale-free) and the best rotation.

Usage: python discovery/scan_shape.py   -> discovery/shape_candidates.csv
"""

from __future__ import annotations

import io
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import rotate as nd_rotate
from scipy.ndimage import zoom as nd_zoom

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from scan import (DEM_URL, LAT_MAX, LAT_MIN, LC_URL, Z_DEM, Z_LC, get, haversine, known_mask,  # noqa: E402
                  lat_of_row, lon_of_col, log, row_of_lat)
from terrain_twin import TARGET_DEM_FILES, fetch_earth_dem, load_target_dem  # noqa: E402

COVER_RGB = [(191, 191, 189), (245, 222, 179)]   # IGBP 16 barren, IGBP 7 open shrubland
MIN_COVER = 0.6
G = 16                       # coarse comparison grid
CELL_KM = 5.04               # same box as the target DEM (168 x 30 m)
STAGE3_PER_TARGET = 150
SPACING_KM = 30.0
ANGLES = list(range(0, 360, 15))


def dihedral(a: np.ndarray):
    out = []
    for k in range(4):
        r = np.rot90(a, k)
        out += [r, r[:, ::-1]]
    return out


def to_grid(z: np.ndarray, n: int = G) -> np.ndarray:
    z = nd_zoom(z, n / z.shape[0], order=1)[:n, :n]
    return z - z.mean()


# --------------------------------------------------------------------------- #
def stage1():
    y0, y1 = int(row_of_lat(LAT_MAX, Z_LC)), int(row_of_lat(LAT_MIN, Z_LC))
    keys = [(x, y) for y in range(y0, y1 + 1) for x in range(2 ** Z_LC)]
    log(f"stage 1: {len(keys)} land-cover tiles")

    def one(k):
        raw = get(LC_URL.format(z=Z_LC, x=k[0], y=k[1]))
        if raw is None:
            return k, None
        rgb = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))
        m = np.zeros(rgb.shape[:2], bool)
        for c in COVER_RGB:
            m |= np.all(rgb == c, axis=-1)
        return k, (m if m.mean() > 0.05 else None)

    masks = {}
    with ThreadPoolExecutor(32) as ex:
        for k, m in ex.map(one, keys):
            if m is not None:
                masks[k] = m
    log(f"stage 1: {len(masks)} tiles with barren / open-shrub land")
    return masks


def stage2(masks, sites, targets16):
    f = 2 ** (Z_DEM - Z_LC)
    jobs = []
    for (x7, y7), m in masks.items():
        for dy in range(f):
            for dx in range(f):
                sub = m[dy * 256 // f:(dy + 1) * 256 // f, dx * 256 // f:(dx + 1) * 256 // f]
                if sub.mean() > MIN_COVER * 0.5:
                    jobs.append((x7 * f + dx, y7 * f + dy, sub))
    log(f"stage 2: {len(jobs)} elevation tiles")
    names = list(targets16)
    tstack = {n: np.stack(dihedral(targets16[n])) for n in names}   # (8, G, G)

    def one(job):
        x, y, sub = job
        raw = get(DEM_URL.format(z=Z_DEM, x=x, y=y))
        if raw is None:
            return []
        rgb = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB")).astype(float)
        dem = rgb[..., 0] * 256 + rgb[..., 1] + rgb[..., 2] / 256 - 32768
        lat_c = lat_of_row(y + 0.5, Z_DEM)
        px_m = 156543.03392 / 2 ** Z_DEM * math.cos(math.radians(lat_c))
        cover = np.kron(sub, np.ones((256 // sub.shape[0], 256 // sub.shape[1]))).astype(float)
        b = max(8, int(round(CELL_KM * 1000 / px_m)))
        step = max(4, b // 2)            # half-overlapping boxes so a feature is not cut in two
        rows = []
        for r0 in range(0, 256 - b + 1, step):
            for c0 in range(0, 256 - b + 1, step):
                cov = cover[r0:r0 + b, c0:c0 + b].mean()
                if cov < MIN_COVER:
                    continue
                blk = dem[r0:r0 + b, c0:c0 + b]
                if blk.max() <= 0:
                    continue
                e = to_grid(blk)
                rec = [lat_of_row(y + (r0 + b / 2) / 256, Z_DEM), lon_of_col(x + (c0 + b / 2) / 256, Z_DEM),
                       float(blk.mean()), float(blk.max() - blk.min()), float(cov)]
                for n in names:
                    rec.append(float(np.sqrt(((tstack[n] - e) ** 2).mean(axis=(1, 2))).min()))
                rows.append(rec)
        return rows

    out, done = [], 0
    with ThreadPoolExecutor(32) as ex:
        for res in ex.map(one, jobs):
            out.extend(res)
            done += 1
            if done % 1000 == 0:
                log(f"  {done}/{len(jobs)} tiles, {len(out)} boxes")
    cols = ["lat", "lon", "elevation_m", "relief_coarse_m", "cover_frac"] + [f"rmse16_{n}" for n in names]
    df = pd.DataFrame(out, columns=cols)
    k = known_mask(df["lat"].to_numpy(), df["lon"].to_numpy(), sites)
    log(f"stage 2: {len(df)} boxes, {int(k.sum())} inside known analog sites (removed)")
    return df.loc[~k].reset_index(drop=True)


# --------------------------------------------------------------------------- #
def circle_mask(n: int) -> np.ndarray:
    yy, xx = np.mgrid[0:n, 0:n] - (n - 1) / 2
    return (xx ** 2 + yy ** 2) <= ((n - 1) / 2) ** 2


def best_rotation(target: np.ndarray, earth: np.ndarray):
    """Best (rmse, corr, angle, mirrored) of earth rotated/mirrored onto target, central circle only."""
    m = circle_mask(target.shape[0])
    t = target[m] - target[m].mean()
    best = (1e18, 0.0, 0, False)
    for mirrored in (False, True):
        e0 = earth[:, ::-1] if mirrored else earth
        for ang in ANGLES:
            e = nd_rotate(e0, ang, reshape=False, order=1, mode="nearest")[m]
            e = e - e.mean()
            rmse = float(np.sqrt(((e - t) ** 2).mean()))
            if rmse < best[0]:
                corr = float((e * t).sum() / (np.linalg.norm(e) * np.linalg.norm(t) + 1e-9))
                best = (rmse, corr, ang, mirrored)
    return best


def spaced(df, n, km):
    keep, la, lo = [], [], []
    for i, r in zip(df.index, df[["lat", "lon"]].itertuples(index=False)):
        if la and haversine(r.lat, r.lon, np.array(la), np.array(lo)).min() < km:
            continue
        keep.append(i)
        la.append(r.lat)
        lo.append(r.lon)
        if len(keep) >= n:
            break
    return df.loc[keep]


def stage3(df, targets):
    out = []
    for name, tz in targets.items():
        cand = spaced(df.sort_values(f"rmse16_{name}"), STAGE3_PER_TARGET, SPACING_KM)
        log(f"stage 3: {name}: 30 m shape match on {len(cand)} candidates")

        def one(r):
            try:
                ez = fetch_earth_dem(float(r.lat), float(r.lon))["z"]
            except Exception:
                return None
            return best_rotation(tz, ez), float(ez.max() - ez.min())

        rows = [r for _, r in cand.iterrows()]
        with ThreadPoolExecutor(8) as ex:
            res = list(ex.map(one, rows))
        trel = float(tz.max() - tz.min())
        for r, x in zip(rows, res):
            if x is None:
                continue
            (rmse, corr, ang, mir), rel = x
            out.append({
                "target": name, "lat": round(float(r.lat), 5), "lon": round(float(r.lon), 5),
                "elevation_m": round(float(r.elevation_m)), "cover_frac": round(float(r.cover_frac), 2),
                "rmse_m": round(rmse, 1), "shape_corr": round(corr, 3),
                "relief_m": round(rel), "target_relief_m": round(trel),
                "rotation_deg": ang, "mirrored": mir,
            })
    res = pd.DataFrame(out).sort_values(["target", "rmse_m"])
    res["rank"] = res.groupby("target").cumcount() + 1
    return res


def main():
    t0 = time.time()
    sites = json.loads((HERE / "known_analogs.json").read_text(encoding="utf-8"))["sites"]
    targets = {n: load_target_dem(n)["z"] for n in TARGET_DEM_FILES}
    targets16 = {n: to_grid(z) for n, z in targets.items()}
    masks = stage1()
    boxes = stage2(masks, sites, targets16)
    res = stage3(boxes, targets)
    res.to_csv(HERE / "shape_candidates.csv", index=False)
    log(f"done in {(time.time() - t0) / 60:.1f} min -> {HERE / 'shape_candidates.csv'}")
    for name, g in res.groupby("target"):
        log(name, g.head(5)[["lat", "lon", "rmse_m", "shape_corr", "relief_m", "target_relief_m"]].to_dict("records"))


if __name__ == "__main__":
    main()
