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
        "about": "Shape, texture and slope, each scored 0–1 and multiplied (1 = identical to the target). Shape: same ridges and hollows after rotation and tilt removal. Texture: crater-like pits, no water gullies. Slope: similar steepness and relief.",
    },
    "3D shape only": {
        "file": "shape_candidates.csv", "score": "shape_corr", "higher_better": True,
        "about": "How closely the Earth surface follows the target surface at 30 m (rotation and mirror allowed). Overall form only: texture and steepness are not checked.",
    },
    "Slope statistics": {
        "file": "discovery_candidates.csv", "score": "match_distance", "higher_better": False,
        "about": "Distance between slope and roughness statistics at 30 m (lower = closer). Does not look at how the terrain is laid out, so the 3D shape can differ.",
    },
}

st.title("🧭 Discovery: Analog Terrain Not Yet Pinned")
st.caption(
    "Earth terrain that matches each target, found by scanning open satellite data. Known analog sites are excluded."
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
    show_synthetic=True,
)

with st.expander("Method, data and limits", expanded=False):
    st.markdown(
        "**How it works**\n"
        "1. Keep every 5 × 5 km box between 56°S and 60°N that is mostly barren (NASA MODIS land cover), "
        "minus known analog sites. About 4.3 million boxes.\n"
        "2. Compare each box with the target shape at ~250 m and keep the best 150 per target.\n"
        "3. Re-measure those at 30 m with 24 rotations × mirror → **3D shape only**.\n"
        "4. Re-score by shape × texture × slope → **3D shape × texture × slope**.\n\n"
        "**Reading the scores**\n"
        "- final_score above ~0.5 is a close match; around 0.2 means one part is off "
        "(see `texture_similarity` and `slope_similarity`).\n"
        "- Shackleton and Malapert stay near 0.2: steep Earth slopes have water gullies, lunar ones do not.\n"
        "- Jezero is a nearly flat crater floor, so shape scores stay low.\n\n"
        "**Data:** AWS Terrain Tiles (SRTM 30 m) for Earth; NASA PDS LOLA, SLDEM2015 and USGS CTX DTM for the "
        "targets; NASA MODIS IGBP 2024 for land cover; excluded sites in `discovery/known_analogs.json`.\n\n"
        "**Limits:** terrain only (no mineralogy, temperature, ice or radiation). A candidate is a pre-screen: "
        "check it on satellite imagery, since dunes and SRTM noise can mimic craters.\n\n"
        "**Re-run:** `python discovery/scan.py`, `python discovery/scan_shape.py`, `python discovery/rerank_texture.py`."
    )
