"""
Global discovery scan (open data, no account needed).

Stage 1  MODIS land cover (NASA GIBS, IGBP, 2024, zoom 7 ~1.2 km): keep 5 km cells that are
         entirely "barren" (IGBP 16), i.e. a full 5 x 5 km box fits inside barren land.
         Cells inside known analog sites (known_analogs.json) are removed.
Stage 2  AWS Terrain Tiles (zoom 9, ~250-300 m): slope median / 90th pct and relief per cell,
         compared with each target DEM resampled to the same pixel size.
Stage 3  Best spaced candidates per target re-measured at 30 m with terrain_twin (same code as
         the Terrain Twin panel); final rank uses the 30 m numbers.

Usage:  python discovery/scan.py            (writes discovery/discovery_candidates.csv)
"""

from __future__ import annotations

import io
import json
import math
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import zoom as nd_zoom

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from terrain_twin import TARGET_DEM_FILES, fetch_earth_dem, load_target_dem, summarize  # noqa: E402

LC_URL = ("https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/MODIS_Combined_L3_IGBP_Land_Cover_Type_Annual/"
          "default/2024-01-01/GoogleMapsCompatible_Level8/{z}/{y}/{x}.png")
DEM_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
BARREN_RGB = (191, 191, 189)
Z_LC, Z_DEM = 7, 9
LAT_MIN, LAT_MAX = -60.0, 72.0
CELL_KM = 5.0
TOP_PER_TARGET = 40          # stage-3 candidates per target
SPACING_KM = 60.0            # minimum distance between stage-3 candidates


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def get(url: str, tries: int = 3) -> bytes | None:
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "artemis-shield-discovery"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read()
        except Exception:
            time.sleep(1 + i)
    return None


def lat_of_row(y: float, z: int) -> float:
    n = math.pi - 2 * math.pi * y / 2 ** z
    return math.degrees(math.atan(math.sinh(n)))


def row_of_lat(lat: float, z: int) -> float:
    la = math.radians(lat)
    return (1 - math.log(math.tan(la) + 1 / math.cos(la)) / math.pi) / 2 * 2 ** z


def lon_of_col(x: float, z: int) -> float:
    return x / 2 ** z * 360 - 180


def haversine(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(a))


def known_mask(lat: np.ndarray, lon: np.ndarray, sites) -> np.ndarray:
    hit = np.zeros(lat.shape, bool)
    for s in sites:
        if "box" in s:
            a, b, c, d = s["box"]
            hit |= (lat >= a) & (lat <= b) & (lon >= c) & (lon <= d)
        else:
            hit |= haversine(lat, lon, s["lat"], s["lon"]) <= s["radius_km"]
    return hit


# --------------------------------------------------------------------------- #
# Stage 1: barren mask
# --------------------------------------------------------------------------- #
def stage1():
    y0, y1 = int(row_of_lat(LAT_MAX, Z_LC)), int(row_of_lat(LAT_MIN, Z_LC))
    keys = [(x, y) for y in range(y0, y1 + 1) for x in range(2 ** Z_LC)]
    log(f"stage 1: {len(keys)} land-cover tiles (z{Z_LC})")

    def one(k):
        raw = get(LC_URL.format(z=Z_LC, x=k[0], y=k[1]))
        if raw is None:
            return k, None
        rgb = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))
        m = np.all(rgb == BARREN_RGB, axis=-1)
        return k, (m if m.any() else False)

    masks = {}
    with ThreadPoolExecutor(32) as ex:
        for k, m in ex.map(one, keys):
            if m is not None and m is not False:
                masks[k] = m
    log(f"stage 1: {len(masks)} tiles contain barren land")
    return masks


# --------------------------------------------------------------------------- #
# Stage 2: coarse topography on barren cells
# --------------------------------------------------------------------------- #
def horn(z, cell):
    zp = np.pad(z, 1, mode="edge")
    a, b, c = zp[:-2, :-2], zp[:-2, 1:-1], zp[:-2, 2:]
    d, f = zp[1:-1, :-2], zp[1:-1, 2:]
    g, h, k = zp[2:, :-2], zp[2:, 1:-1], zp[2:, 2:]
    dzdx = (c + 2 * f + k - (a + 2 * d + g)) / (8 * cell)
    dzdy = (g + 2 * h + k - (a + 2 * b + c)) / (8 * cell)
    return np.degrees(np.arctan(np.hypot(dzdx, dzdy)))


_target_cache: dict = {}


def target_coarse(name: str, px_m: float):
    key = (name, int(round(px_m / 10) * 10))
    if key not in _target_cache:
        z = load_target_dem(name)["z"]
        z2 = nd_zoom(z, 30.0 / key[1], order=1)
        s = horn(z2, key[1])[1:-1, 1:-1]
        _target_cache[key] = (float(np.median(s)), float(np.percentile(s, 90)), float(z2.max() - z2.min()))
    return _target_cache[key]


def stage2(masks, sites):
    f = 2 ** (Z_DEM - Z_LC)
    jobs = []
    for (x7, y7), m in masks.items():
        for dy in range(f):
            for dx in range(f):
                sub = m[dy * 256 // f:(dy + 1) * 256 // f, dx * 256 // f:(dx + 1) * 256 // f]
                if sub.mean() > 0.3:
                    jobs.append((x7 * f + dx, y7 * f + dy, sub))
    log(f"stage 2: {len(jobs)} elevation tiles (z{Z_DEM})")

    def one(job):
        x, y, sub = job
        raw = get(DEM_URL.format(z=Z_DEM, x=x, y=y))
        if raw is None:
            return []
        rgb = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB")).astype(float)
        dem = rgb[..., 0] * 256 + rgb[..., 1] + rgb[..., 2] / 256 - 32768
        lat_c = lat_of_row(y + 0.5, Z_DEM)
        px_m = 156543.03392 / 2 ** Z_DEM * math.cos(math.radians(lat_c))
        slope = horn(dem, px_m)
        mask = np.kron(sub, np.ones((256 // sub.shape[0], 256 // sub.shape[1]), bool))
        b = max(6, int(round(CELL_KM * 1000 / px_m)))
        out = []
        for r0 in range(0, 256 - b + 1, b):
            for c0 in range(0, 256 - b + 1, b):
                if not mask[r0:r0 + b, c0:c0 + b].all():
                    continue  # a full 5 x 5 km box must be barren
                blk = dem[r0:r0 + b, c0:c0 + b]
                if blk.min() <= 0 and blk.max() <= 0:
                    continue
                s = slope[r0 + 1:r0 + b - 1, c0 + 1:c0 + b - 1]
                out.append((
                    lat_of_row(y + (r0 + b / 2) / 256, Z_DEM), lon_of_col(x + (c0 + b / 2) / 256, Z_DEM),
                    float(blk.mean()), float(np.median(s)), float(np.percentile(s, 90)),
                    float(blk.max() - blk.min()), px_m,
                ))
        return out

    rows = []
    done = 0
    with ThreadPoolExecutor(32) as ex:
        for res in ex.map(one, jobs):
            rows.extend(res)
            done += 1
            if done % 500 == 0:
                log(f"  {done}/{len(jobs)} tiles, {len(rows)} cells")
    df = pd.DataFrame(rows, columns=["lat", "lon", "elevation_m", "slope_med_c", "slope_p90_c", "relief_c", "px_m"])
    k = known_mask(df["lat"].to_numpy(), df["lon"].to_numpy(), sites)
    log(f"stage 2: {len(df)} fully-barren 5 km cells, {int(k.sum())} inside known analog sites (removed)")
    return df.loc[~k].reset_index(drop=True)


def coarse_score(df, name):
    t = np.array([target_coarse(name, p) for p in df["px_m"]])
    d = np.sqrt(((df["slope_med_c"] - t[:, 0]) / 3.0) ** 2 + ((df["slope_p90_c"] - t[:, 1]) / 3.0) ** 2
                + (np.log((df["relief_c"] + 1) / (t[:, 2] + 1)) / math.log(1.25)) ** 2)
    return d


def spaced(df, n, km):
    keep, la, lo = [], [], []
    for i, r in df.iterrows():
        if la and haversine(r.lat, r.lon, np.array(la), np.array(lo)).min() < km:
            continue
        keep.append(i)
        la.append(r.lat)
        lo.append(r.lon)
        if len(keep) >= n:
            break
    return df.loc[keep]


# --------------------------------------------------------------------------- #
# Stage 3: 30 m re-measure
# --------------------------------------------------------------------------- #
def stage3(df):
    out = []
    for name in TARGET_DEM_FILES:
        ts = summarize(load_target_dem(name)["z"])
        d = df.assign(coarse_dist=coarse_score(df, name)).sort_values("coarse_dist")
        cand = spaced(d, TOP_PER_TARGET, SPACING_KM)
        log(f"stage 3: {name}: measuring {len(cand)} candidates at 30 m")

        def one(r):
            try:
                e = summarize(fetch_earth_dem(float(r.lat), float(r.lon))["z"])
            except Exception:
                return None
            return e

        with ThreadPoolExecutor(8) as ex:
            res = list(ex.map(one, [r for _, r in cand.iterrows()]))
        for (_, r), e in zip(cand.iterrows(), res):
            if e is None:
                continue
            dist = math.sqrt(((e["slope_median"] - ts["slope_median"]) / 3.0) ** 2
                             + ((e["slope_p90"] - ts["slope_p90"]) / 3.0) ** 2
                             + (math.log((e["roughness_median"] + 0.1) / (ts["roughness_median"] + 0.1)) / math.log(1.25)) ** 2)
            out.append({
                "target": name, "lat": round(r.lat, 5), "lon": round(r.lon, 5), "elevation_m": round(r.elevation_m),
                "slope_median": round(e["slope_median"], 2), "slope_p90": round(e["slope_p90"], 2),
                "roughness_median": round(e["roughness_median"], 2), "relief_m": round(e["relief_m"], 1),
                "target_slope_median": round(ts["slope_median"], 2), "target_slope_p90": round(ts["slope_p90"], 2),
                "target_roughness_median": round(ts["roughness_median"], 2),
                "match_distance": round(dist, 3),
                "slope_match": abs(e["slope_median"] - ts["slope_median"]) <= 3 and abs(e["slope_p90"] - ts["slope_p90"]) <= 3,
            })
    res = pd.DataFrame(out).sort_values(["target", "match_distance"])
    res["rank"] = res.groupby("target").cumcount() + 1
    return res


def main():
    t0 = time.time()
    sites = json.loads((HERE / "known_analogs.json").read_text(encoding="utf-8"))["sites"]
    masks = stage1()
    cells = stage2(masks, sites)
    cells.to_csv(HERE / "barren_cells_coarse.csv", index=False)
    res = stage3(cells)
    res.to_csv(HERE / "discovery_candidates.csv", index=False)
    log(f"done in {(time.time() - t0) / 60:.1f} min -> {HERE / 'discovery_candidates.csv'}")
    for name, g in res.groupby("target"):
        log(name, g.head(5)[["lat", "lon", "slope_median", "slope_p90", "roughness_median", "match_distance"]].to_dict("records"))


if __name__ == "__main__":
    main()
