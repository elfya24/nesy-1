"""
Re-rank shape_candidates.csv with surface texture and slope / relief agreement.

final_score = detrended_shape_corr x texture_similarity x slope_similarity
  texture_similarity = exp(-texture_distance^2 / 2)            (crater pits + gully-scale relief)
  slope_similarity   = exp(-slope_distance^2 / 2), slope_distance combines
                       typical slope (3 deg tolerance), steep-part slope (3 deg) and relief (x1.5 tolerance).
The slope term stops flat plains with one cliff from matching a steep ridge just because both look
"smooth" and correlate in shape. Only SRTM latitudes (56 S - 60 N) are kept.
-> discovery/shape_texture_candidates.csv
"""
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import rotate as nd_rotate

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from scan_shape import circle_mask  # noqa: E402
from terrain_twin import fetch_earth_dem, load_target_dem, summarize  # noqa: E402
from texture import texture, texture_distance  # noqa: E402

SLOPE_TOL_DEG = 3.0
RELIEF_TOL = 1.5


def plane_resid(z, m):
    yy, xx = np.mgrid[0:z.shape[0], 0:z.shape[1]]
    A = np.c_[xx[m], yy[m], np.ones(m.sum())]
    coef, *_ = np.linalg.lstsq(A, z[m], rcond=None)
    return z[m] - A @ coef


def slope_distance(e: dict, t: dict) -> float:
    return math.sqrt(((e["slope_median"] - t["slope_median"]) / SLOPE_TOL_DEG) ** 2
                     + ((e["slope_p90"] - t["slope_p90"]) / SLOPE_TOL_DEG) ** 2
                     + (math.log((e["relief_m"] + 1) / (t["relief_m"] + 1)) / math.log(RELIEF_TOL)) ** 2)


d = pd.read_csv(HERE / "shape_candidates.csv")
d = d[d["lat"].between(-56, 60)].reset_index(drop=True)
tz = {t: load_target_dem(t)["z"] for t in d["target"].unique()}
mask = circle_mask(169)
tres = {t: plane_resid(z, mask) for t, z in tz.items()}
ttex = {t: texture(z) for t, z in tz.items()}
tsum = {t: summarize(z) for t, z in tz.items()}


def one(r):
    try:
        ez = fetch_earth_dem(r.lat, r.lon)["z"]
    except Exception:
        return None
    e0 = ez[:, ::-1] if r.mirrored else ez
    e = nd_rotate(e0, r.rotation_deg, reshape=False, order=1, mode="nearest")
    er, tr = plane_resid(e, mask), tres[r.target]
    corr = float((er * tr).sum() / (np.linalg.norm(er) * np.linalg.norm(tr) + 1e-9))
    tx = texture(ez)
    sm = summarize(ez)
    return corr, tx, texture_distance(tx, ttex[r.target]), sm, slope_distance(sm, tsum[r.target])


rows = [r for _, r in d.iterrows()]
with ThreadPoolExecutor(8) as ex:
    res = list(ex.map(one, rows))
out = []
for r, x in zip(rows, res):
    if x is None:
        continue
    corr, tx, tdist, sm, sdist = x
    t, ts = ttex[r.target], tsum[r.target]
    out.append({**r.to_dict(), "detrended_corr": round(corr, 3),
                "slope_median": round(sm["slope_median"], 2), "target_slope_median": round(ts["slope_median"], 2),
                "slope_p90": round(sm["slope_p90"], 2), "target_slope_p90": round(ts["slope_p90"], 2),
                "slope_similarity": round(math.exp(-sdist ** 2 / 2), 3),
                "depression_frac": round(tx["depression_frac"], 3), "target_depression_frac": round(t["depression_frac"], 3),
                "gully_band": round(tx["gully_band"], 3), "target_gully_band": round(t["gully_band"], 3),
                "texture_distance": round(tdist, 2), "texture_similarity": round(math.exp(-tdist ** 2 / 2), 3)})
o = pd.DataFrame(out)
o["final_score"] = (o["detrended_corr"].clip(lower=0) * o["texture_similarity"] * o["slope_similarity"]).round(3)
o = o.sort_values(["target", "final_score"], ascending=[True, False])
o["shape_rank"] = o["rank"]
o["rank"] = o.groupby("target").cumcount() + 1
o.to_csv(HERE / "shape_texture_candidates.csv", index=False)
pd.set_option("display.width", 240)
for t, g in o.groupby("target"):
    ts = tsum[t]
    print(f"\n== {t}: target slope {ts['slope_median']:.1f} / p90 {ts['slope_p90']:.1f} / relief {ts['relief_m']:.0f} m")
    print(g.head(6)[["rank", "lat", "lon", "slope_median", "slope_p90", "relief_m", "detrended_corr",
                     "texture_similarity", "slope_similarity", "final_score"]].to_string(index=False))
