"""Discovery page: Earth terrain that matches the lunar / Martian targets, outside known analog sites."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
from terrain_twin import TARGET_DEM_FILES  # noqa: E402
from twin_ui import _stretch, render_terrain_twin  # noqa: E402

DISC = BASE_DIR / "discovery"
RANKINGS = {
    "3D shape × texture × slope": {
        "file": "shape_texture_candidates.csv", "score": "final_score", "higher_better": True,
        "about": "final_score = shape × texture × slope, each 0–1 (1 = same as the target). "
                 "**Shape** (detrended_corr): the target terrain laid over the Earth site, rotated / mirrored to the best fit "
                 "and with the overall tilt removed: are ridges and hollows in the same places? "
                 "**Texture** (texture_similarity): crater-like closed pits and no water-carved gullies. "
                 "**Slope** (slope_similarity): typical slope and steep-part slope within ±3° and total relief within ×1.5 "
                 "of the target, so a flat plain with one cliff no longer passes as a steep lunar ridge.",
    },
    "3D shape only": {
        "file": "shape_candidates.csv", "score": "shape_corr", "higher_better": True,
        "about": "shape_corr: how well the Earth surface matches the target surface at 30 m (rotations and mirror allowed), "
                 "including the overall tilt. Large-scale form only: ignores texture and does not check that slopes are "
                 "equally steep. This ranking chose the 150 candidates per target that the other 3D ranking re-scores.",
    },
    "Slope statistics": {
        "file": "discovery_candidates.csv", "score": "match_distance", "higher_better": False,
        "about": "match_distance (lower = closer): typical slope, steep-part slope and roughness at 30 m vs. the target, "
                 "fully barren 5 × 5 km boxes only. Says nothing about how the slopes are arranged, so a match here "
                 "can look very different in 3D.",
    },
}

st.title("🧭 Discovery: Analog Terrain Not Yet Pinned")
st.caption(
    "Global scans of open satellite data for Earth terrain that matches each target, with known NASA / ESA / literature "
    "analog sites removed before scanning."
)


@st.cache_data(show_spinner=False)
def load(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def known_sites() -> list:
    return json.loads((DISC / "known_analogs.json").read_text(encoding="utf-8"))["sites"]


sb = st.sidebar
target = sb.selectbox("Planetary Target Zone", list(TARGET_DEM_FILES))
body = "Mars" if "Jezero" in target else "Moon"
available = {k: v for k, v in RANKINGS.items() if (DISC / v["file"]).exists()}
if not available:
    st.warning("No scan results found in discovery/. Run `python discovery/scan.py` and `python discovery/scan_shape.py`.")
    st.stop()
mode = sb.radio("Ranking", list(available))
top_n = sb.slider("Candidates shown", 5, 40, 15, 5)
show_known = sb.toggle("Show excluded known analog sites", value=True)

spec = available[mode]
f = DISC / spec["file"]
df = load(str(f), f.stat().st_mtime)
d = df[(df["target"] == target) & df["lat"].between(-56, 60)].copy()
d = d.sort_values(spec["score"], ascending=not spec["higher_better"]).head(top_n).reset_index(drop=True)
d.insert(0, "#", range(1, len(d) + 1))
if d.empty:
    st.info(f"No candidates for {target} in this ranking.")
    st.stop()

st.info(f"**{mode}:** {spec['about']}")

# ---- Map ------------------------------------------------------------------ #
m = folium.Map(location=[20, 10], zoom_start=2, tiles=None, control_scale=True)
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri World Imagery", name="Satellite",
).add_to(m)
if show_known:
    for s in known_sites():
        tip = f"Excluded: {s['name']} ({s['source']})"
        style = dict(color="#94a3b8", weight=1, fill=True, fill_opacity=0.18)
        if "box" in s:
            a, b, c, e = s["box"]
            folium.Rectangle([[a, c], [b, e]], tooltip=tip, **style).add_to(m)
        else:
            folium.Circle([s["lat"], s["lon"]], radius=s["radius_km"] * 1000, tooltip=tip, **style).add_to(m)
for _, r in d.iterrows():
    folium.CircleMarker(
        [r["lat"], r["lon"]], radius=8, color="#0f172a", weight=1.2, fill=True, fill_opacity=0.95,
        fill_color="#10b981" if r["#"] <= 3 else "#f59e0b",
        tooltip=f"#{r['#']} · {r['lat']:.3f}°, {r['lon']:.3f}° · {spec['score']} {r[spec['score']]:.3f}",
    ).add_to(m)
st_folium(m, height=430, use_container_width=True, key=f"disc_map_{target}_{mode}", returned_objects=[])
st.caption("Green: top 3. Amber: other candidates. Grey: known analog sites removed before the scan.")

# ---- Table ---------------------------------------------------------------- #
cols = ["#", "lat", "lon", "elevation_m"] + [c for c in [
    "final_score", "detrended_corr", "texture_similarity", "slope_similarity", "shape_corr", "rmse_m",
    "relief_m", "target_relief_m",
    "slope_median", "target_slope_median", "slope_p90", "target_slope_p90", "roughness_median",
    "target_roughness_median", "match_distance", "cover_frac"] if c in d.columns]
st.dataframe(d[cols], hide_index=True, **_stretch())

# ---- Candidate inspector -------------------------------------------------- #
st.subheader("Inspect a candidate")
pick = st.selectbox(
    "Candidate",
    list(d.index),
    format_func=lambda i: f"#{d.loc[i, '#']} · {d.loc[i, 'lat']:.3f}°, {d.loc[i, 'lon']:.3f}°",
    key=f"disc_pick_{target}_{mode}",
)
r = d.loc[pick]
lat, lon = float(r["lat"]), float(r["lon"])
l1, l2 = st.columns(2)
l1.link_button(
    "🌍 Open in NASA Worldview",
    f"https://worldview.earthdata.nasa.gov/?v={lon - 0.4:.4f},{lat - 0.3:.4f},{lon + 0.4:.4f},{lat + 0.3:.4f}"
    "&l=MODIS_Terra_CorrectedReflectance_TrueColor,Reference_Labels_15m,Reference_Features_15m,Coastlines_15m",
    **_stretch(),
)
l2.link_button("🗺️ Open in Google Maps (satellite)", f"https://www.google.com/maps/@{lat},{lon},13z/data=!3m1!1e3",
               **_stretch())
render_terrain_twin(
    pd.Series({"lat": lat, "lon": lon, "macro_region": f"Discovery candidate #{r['#']}"}),
    target, body, key="disc",
    rotate_deg=float(r["rotation_deg"]) if "rotation_deg" in r else 0.0,
    mirrored=bool(r["mirrored"]) if "mirrored" in r else False,
)

with st.expander("Method, data and limits", expanded=False):
    st.markdown(
        "**How the 3D candidates were found**\n"
        "1. *Screen:* every 5 × 5 km box (half-overlapping) between 56°S and 60°N that is at least 60% barren or open "
        "shrubland (NASA MODIS IGBP 2024), minus known analog sites. About 4.3 million boxes.\n"
        "2. *Coarse shape:* each box (AWS Terrain Tiles, ~250 m) compared with the target surface under the 8 "
        "rotations / mirrors of a square.\n"
        "3. *Fine shape:* the best 150 per target re-measured at 30 m on the same 169 × 169 grid as the target and "
        "matched over 24 rotation angles × mirror. Result: **3D shape only**.\n"
        "4. *Re-score:* those 150 re-ranked by shape (tilt removed) × texture × slope. Result: **3D shape × texture × slope**.\n\n"
        "**Reading the scores**\n"
        "- final_score above ~0.5: shape, texture and slope all close to the target. Around 0.2: at least one of the "
        "three is clearly off; check `texture_similarity` and `slope_similarity` to see which.\n"
        "- Shackleton and Malapert currently top out near 0.2: the steep Earth candidates with the right shape are "
        "carved by water gullies, which lunar slopes do not have. That is a real limit of the analogy, not a bug.\n"
        "- Jezero matches are weak in shape (detrended_corr below 0.5): the target is a nearly flat crater floor, "
        "so little shape is left after the tilt is removed.\n\n"
        "**Data**\n"
        "- Earth elevation: AWS Terrain Tiles (mostly SRTM 30 m). Above 60°N other, smoother sources are used that "
        "fake a lunar-like texture, so those latitudes are excluded.\n"
        "- Targets: NASA PDS LOLA (Shackleton, Malapert), SLDEM2015 (Marius Hills), USGS CTX DTM (Jezero).\n"
        "- Land cover: NASA MODIS IGBP 2024 via GIBS. The *Slope statistics* scan uses a stricter rule: the whole "
        "5 × 5 km box barren.\n"
        "- Excluded: the known analog sites in `discovery/known_analogs.json`, as whole areas (grey on the map).\n\n"
        "**Limits**\n"
        "- Terrain only: mineralogy, temperature, ice and radiation are not part of these scans.\n"
        "- The 3D re-score only sees the 150 candidates chosen by shape; a site that matches all three criteria but "
        "ranked lower on shape alone is not in the list.\n"
        "- A candidate is a pre-screen, not a confirmed analog. Check each one on satellite imagery: dunes and SRTM "
        "noise on flat ground can mimic crater texture.\n\n"
        "**Re-run:** `python discovery/scan.py` (slope statistics), `python discovery/scan_shape.py` (3D shape), then "
        "`python discovery/rerank_texture.py` (shape × texture × slope)."
    )
