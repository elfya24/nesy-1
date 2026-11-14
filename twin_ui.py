"""Terrain Twin panel (target DEM vs. an Earth site on the same 5 km / 30 m grid), shared by all pages."""

from __future__ import annotations

import html
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.ndimage import rotate as nd_rotate

from terrain_morph import default_sun_elevation, morph_to_target
from terrain_twin import fetch_earth_dem, hillshade, load_target_dem, summarize


def _stretch() -> Dict[str, Any]:
    try:
        from packaging.version import Version
        if Version(st.__version__) >= Version("1.52"):
            return {"width": "stretch"}
    except Exception:
        pass
    return {"use_container_width": True}


@st.cache_data(show_spinner=False)
def cached_target_dem(target_name: str) -> Any:
    return load_target_dem(target_name)


@st.cache_data(show_spinner="Fetching Earth elevation tiles (AWS Terrain Tiles)...", ttl=24 * 3600)
def cached_earth_dem(lat: float, lon: float) -> Dict[str, Any]:
    return fetch_earth_dem(lat, lon)


def _mark(a: float, b: Optional[float], *, abs_tol: Optional[float] = None, rel_tol: Optional[float] = None) -> str:
    if a is None or b is None or pd.isna(a) or pd.isna(b):
        return "?"
    if abs_tol is not None:
        return "=" if abs(a - b) <= abs_tol else "≠"
    hi = max(abs(a), abs(b), 1e-9)
    return "=" if abs(a - b) / hi <= rel_tol else "≠"


def _tint(shade: np.ndarray, rgb: Tuple[float, float, float]) -> np.ndarray:
    return np.clip(np.stack([shade * c for c in rgb], axis=-1), 0, 255).astype(np.uint8)


def render_terrain_twin(row: pd.Series, target_name: str, body: str, key: str = "twin",
                        rotate_deg: float = 0.0, mirrored: bool = False, show_synthetic: bool = False) -> None:
    """rotate_deg / mirrored turn the Earth plate the way the shape matcher aligned it (display only).
    show_synthetic adds the Synthetic Twin panel below (used on the Discovery page only)."""
    if key == "twin":
        st.subheader("🌙 Terrain Twin: Target vs. Selected Earth Site")
        st.caption(
            "Both 5 km × 5 km boxes are sampled on the same 30 m grid and measured with the same code "
            "(Horn slope, 8-neighbour roughness, cast-shadow hillshade). Ported from the Nesy terrain module."
        )
    tdem = cached_target_dem(target_name)
    if tdem is None:
        st.info(f"No terrain model is bundled for {target_name}.")
        return

    polar = abs(float(tdem["meta"]["target"].get("lat", 0))) > 80
    c1, c2 = st.columns(2)
    az = c1.slider("Sun azimuth (°, clockwise from plate top)", 0, 359, 135, 5, key=f"{key}_az")
    el = c2.slider(
        "Sun elevation (°)",
        1, 60, 6 if polar else 25, 1, key=f"{key}_el",
        help="At the lunar south pole the real sun stays around 1.5° above the horizon.",
    )

    lat, lon = float(row["lat"]), float(row["lon"])
    earth, err = None, None
    try:
        earth = cached_earth_dem(round(lat, 5), round(lon, 5))
    except Exception as exc:  # network / tile errors must not break the page
        err = str(exc)

    ts = summarize(tdem["z"])
    es = summarize(earth["z"]) if earth else None
    view = None
    if earth:
        view = earth["z"][:, ::-1] if mirrored else earth["z"]
        if rotate_deg:
            view = nd_rotate(view, rotate_deg, reshape=False, order=1, mode="nearest")
    aligned = bool(rotate_deg or mirrored)
    meta = tdem["meta"]
    tcolor = (1.0, 1.0, 1.0) if body == "Moon" else (1.0, 0.72, 0.52)

    left, right = st.columns(2)
    with left:
        st.markdown(f"**{target_name}** ({body})")
        st.image(_tint(hillshade(tdem["z"], az, el), tcolor), **_stretch())
        st.caption(
            f"{meta.get('source', '')} · {meta.get('product', '')} (native {meta.get('native', '?')}) · "
            f"{meta['target'].get('lat', 0):.2f}°, {meta['target'].get('lon', 0):.2f}°. "
            + ("Plate top = 0° longitude, not north." if polar else "Plate top = north.")
        )
    with right:
        st.markdown(f"**Earth: {html.escape(str(row['macro_region']))}** ({lat:.3f}°, {lon:.3f}°)")
        if earth:
            st.image(_tint(hillshade(view, az, el), (1.0, 0.86, 0.66)), **_stretch())
            orient = (f"rotated {rotate_deg:.0f}°{' and mirrored' if mirrored else ''} to align with the target"
                      if aligned else "plate top = north")
            st.caption(f"{earth['source']} · zoom {earth['zoom']} · {orient}.")
        else:
            st.warning(f"Earth elevation tiles could not be fetched: {err}")

    rows = [
        ("Typical slope (median, °)", ts["slope_median"], es and es["slope_median"], dict(abs_tol=3.0)),
        ("Steep parts (90th pct., °)", ts["slope_p90"], es and es["slope_p90"], dict(abs_tol=3.0)),
        ("Roughness (median step, m)", ts["roughness_median"], es and es["roughness_median"], dict(rel_tol=0.25)),
        ("Relief in box (m)", ts["relief_m"], es and es["relief_m"], dict(rel_tol=0.25)),
    ]
    table = pd.DataFrame(
        [
            {
                "Measure": name,
                target_name: round(t, 2),
                "Earth site": None if e is None else round(e, 2),
                "Match": _mark(t, e, **tol),
            }
            for name, t, e, tol in rows
        ]
    )
    tcol, hcol = st.columns([1, 1.25])
    with tcol:
        st.dataframe(table, hide_index=True, **_stretch())
        csv_slope = row.get("slope_deg", np.nan)
        st.caption(
            "`=` within tolerance (3° for slopes, 25% for roughness/relief), `≠` outside it, `?` not measured."
            + ("" if pd.isna(csv_slope) else f" Grid-cell CSV slope for this site: {csv_slope:.2f}°.")
        )
    with hcol:
        fig = go.Figure()
        bins = dict(start=0, end=46, size=1)
        fig.add_trace(go.Histogram(x=ts["slopes"], xbins=bins, histnorm="percent", name=target_name,
                                   marker_color="#e2e8f0" if body == "Moon" else "#fb923c", opacity=0.75))
        if es:
            fig.add_trace(go.Histogram(x=es["slopes"], xbins=bins, histnorm="percent", name="Earth site",
                                       marker_color="#38bdf8", opacity=0.6))
        fig.update_layout(
            barmode="overlay", template="plotly_dark", height=260,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.45)",
            xaxis_title="Slope (°)", yaxis_title="% of box", legend=dict(orientation="h", y=1.15),
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig, key=f"{key}_hist", **_stretch())

    with st.expander("3D relief (same vertical scale for both)", expanded=False):
        zs = [tdem["z"]] + ([view] if earth else [])
        rel = max(float(np.nanmax(z) - np.nanmin(z)) for z in zs)
        cols = st.columns(len(zs))
        for col, z, name, scale in zip(cols, zs, [target_name, "Earth site"], ["Greys", "Earth"]):
            ax = np.arange(z.shape[0]) * 30.0
            f3 = go.Figure(go.Surface(z=z - np.nanmin(z), x=ax, y=ax[::-1], colorscale=scale, showscale=False))
            f3.update_layout(
                title=name, template="plotly_dark", height=380, margin=dict(l=0, r=0, t=30, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                scene=dict(zaxis=dict(range=[0, rel], title="m"), xaxis_title="m", yaxis_title="m",
                           aspectmode="manual", aspectratio=dict(x=1, y=1, z=0.35)),
            )
            col.plotly_chart(f3, key=f"{key}_3d_{name}", **_stretch())

    if earth and show_synthetic:
        render_synthetic_twin(lat, lon, target_name, tdem, earth["z"], body, az, el, key)


@st.cache_data(show_spinner="Morphing the Earth site toward the target...", max_entries=32)
def cached_morph(lat: float, lon: float, target_name: str, min_share: float, seed: int,
                 _z_earth: np.ndarray, _z_target: np.ndarray, sun_el: float) -> Dict[str, Any]:
    rep = morph_to_target(_z_earth, _z_target, seed=seed, min_earth_share=min_share, sun_el_deg=sun_el)
    rep["stages"] = [{"name": s.name, "z": s.z, "metrics": s.metrics, "distance": s.distance, "info": s.info}
                     for s in rep["stages"]]
    return rep


_STAGE_LABELS = {"original": "Original Earth site", "spectral_shape": "+ spectral shaping",
                 "craters": "+ craters", "repose_cap": "+ angle-of-repose cap"}


def render_synthetic_twin(lat: float, lon: float, target_name: str, tdem: Dict[str, Any], z_earth: np.ndarray,
                          body: str, az: float, el: float, key: str) -> None:
    """Synthetic twin: the Earth DEM morphed toward the target, with per-stage gap closure (terrain_morph)."""
    st.markdown("#### 🧪 Synthetic Twin: Earth site morphed toward the target")
    st.markdown(
        "**What it's for:** no Earth site is an exact copy of the target. This panel reshapes the candidate toward "
        "the target step by step and shows how much of the difference each step removes. Use it to:\n"
        "- see *what* separates the candidate from the target (missing craters, too steep, too smooth...);\n"
        "- judge the candidate: if it gets close while most of it is still the real site, it is a strong analog; "
        "if it only gets close once it is mostly synthetic, the match is weak;\n"
        "- plan test terrain: the steps show what would have to be added or flattened on the site to make it "
        "more target-like for rover or lander tests."
    )
    st.caption(
        "How: keeps the Earth site's layout, gives each spatial scale the target's amplitude, adds lunar-like craters "
        "and caps slopes at the 35° angle of repose. **Earth share** = how much of the final relief is still the "
        "real site; a near-zero value means the twin is mostly synthetic."
    )
    c1, c2, c3 = st.columns([2, 1, 1])
    min_share = c1.slider("Minimum Earth share to keep", 0.0, 0.9, 0.0, 0.05, key=f"{key}_minshare",
                          help="Forbid morphs that replace more of the real site than this.")
    seed = int(c2.number_input("Crater seed", 0, 999, 0, 1, key=f"{key}_seed"))
    run = c3.toggle("Show synthetic twin", value=False, key=f"{key}_morph_on")
    if not run:
        return

    sun_el = default_sun_elevation(tdem["meta"])
    try:
        rep = cached_morph(round(lat, 5), round(lon, 5), target_name, float(min_share), seed,
                           z_earth, tdem["z"], sun_el)
    except ValueError as exc:
        st.warning(str(exc))
        return
    final = rep["stages"][-1]
    share = rep["earth_share"]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Distance", f"{final['distance']:.2f}",
              delta=f"{final['distance'] - rep['stages'][0]['distance']:+.2f}", delta_color="inverse",
              help="Distance to the target (0 = identical); delta is the change from the real Earth site.")
    m2.metric("Earth share", f"{share:.0%}")
    m3.metric("β", f"{rep['beta']:.2f}", help="Spectral strength: 0 = Earth amplitudes, 1 = target amplitudes.")
    m4.metric("Craters", f"{rep['stages'][2]['info'].get('n_craters', 0)}",
              help=f"Crater density {rep['density']:g}× the lunar equilibrium-like baseline.")
    if share < 0.2:
        st.warning("Less than 20% of this twin is the real Earth site: it is closer to a copy of the target than "
                   "to an analog. Raise the minimum Earth share or pick another site.")

    tcolor = (1.0, 1.0, 1.0) if body == "Moon" else (1.0, 0.72, 0.52)
    cols = st.columns(3)
    plates = [(f"{target_name} (target)", tdem["z"], tcolor), ("Earth site (real)", z_earth, (1.0, 0.86, 0.66)),
              ("Synthetic twin", final["z"], (0.75, 0.9, 1.0))]
    for col, (name, z, rgb) in zip(cols, plates):
        col.markdown(f"**{name}**")
        col.image(_tint(hillshade(z, az, el), rgb), **_stretch())
    st.caption("All three plates use the sun sliders above; plates are not rotated.")

    tm = rep["target"]
    metrics = [("Typical slope (median, °)", "slope_median"), ("Steep parts (90th pct., °)", "slope_p90"),
               ("Roughness (median step, m)", "roughness_median"), ("Relief in box (m)", "relief_m"),
               (f"Shadowed at sun {sun_el:g}° (fraction)", "shadow_fraction")]
    records = [{"Stage": "Target", **{lbl: round(tm[k], 3) for lbl, k in metrics}, "Distance": 0.0}]
    for s in rep["stages"]:
        records.append({"Stage": _STAGE_LABELS.get(s["name"], s["name"]),
                        **{lbl: round(s["metrics"][k], 3) for lbl, k in metrics},
                        "Distance": round(s["distance"], 3)})
    st.dataframe(pd.DataFrame(records), hide_index=True, **_stretch())

    gl, gr = st.columns([1, 1.25])
    with gl:
        gap = pd.DataFrame([{"Stage": _STAGE_LABELS.get(n, n), **{lbl: round(g[k], 1) for lbl, k in metrics}}
                            for n, g in rep["gap_closure"].items()])
        st.markdown("**Gap closed (%) — cumulative after each stage**")
        st.dataframe(gap, hide_index=True, **_stretch())
        st.caption("100 = matches the target, 0 = no better than the real site, negative = overshoot.")
    with gr:
        fig = go.Figure()
        bins = dict(start=0, end=46, size=1)
        for name, z, color in [(target_name, tdem["z"], "#e2e8f0" if body == "Moon" else "#fb923c"),
                               ("Earth site", z_earth, "#38bdf8"), ("Synthetic twin", final["z"], "#a78bfa")]:
            fig.add_trace(go.Histogram(x=summarize(z)["slopes"], xbins=bins, histnorm="percent", name=name,
                                       marker_color=color, opacity=0.55))
        fig.update_layout(
            barmode="overlay", template="plotly_dark", height=260,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.45)",
            xaxis_title="Slope (°)", yaxis_title="% of box", legend=dict(orientation="h", y=1.15),
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig, key=f"{key}_morph_hist", **_stretch())
