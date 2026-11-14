"""
Landing Safety Map: classifies every 30 m cell of a target DEM as safe / risky / no-go for a lander,
then finds the spots with the widest safe margin around them.

Criteria (all thresholds adjustable in the sidebar):
* Slope      - Horn 3x3 slope from terrain_twin.slope_deg.
* Bumpiness  - |height - mean of 8 neighbours| in metres: local bumps/pits that a plane does not explain
               (unlike raw roughness, a smooth tilted plane scores 0 here, so it does not double-count slope).
* Sunlight   - polar targets only: share of sun azimuths (every 15 deg, at a fixed low sun elevation) in which
               the cell is lit, using the same cast-shadow hillshade as the Terrain Twin.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import ndimage

from terrain_twin import CELL_M, TARGET_DEM_FILES, hillshade, load_target_dem, slope_deg

SAFE, RISKY, NOGO = 0, 1, 2
CLASS_COLORS = {SAFE: (16, 185, 129), RISKY: (245, 158, 11), NOGO: (239, 68, 68)}
AZIMUTHS = np.arange(0, 360, 15)


def _stretch() -> Dict[str, Any]:
    try:
        from packaging.version import Version
        if Version(st.__version__) >= Version("1.52"):
            return {"width": "stretch"}
    except Exception:
        pass
    return {"use_container_width": True}


@st.cache_data(show_spinner=False)
def terrain_layers(target_name: str) -> Dict[str, Any]:
    dem = load_target_dem(target_name)
    z = dem["z"]
    n0, n1 = z.shape
    zc = z[1:-1, 1:-1]
    neigh = sum(
        z[1 + dr:n0 - 1 + dr, 1 + dc:n1 - 1 + dc] for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc
    ) / 8
    return {
        "z": z,
        "slope": slope_deg(z),
        "bump": np.abs(zc - neigh),
        "meta": dem["meta"],
    }


@st.cache_data(show_spinner="Tracing shadows around the full horizon...")
def sunlight_fraction(target_name: str, sun_el: float) -> np.ndarray:
    z = load_target_dem(target_name)["z"]
    lit = np.zeros(z.shape)
    for az in AZIMUTHS:
        lit += hillshade(z, float(az), sun_el) > 0
    return (lit / len(AZIMUTHS))[1:-1, 1:-1]


def best_spots(safe: np.ndarray, min_radius_cells: float, count: int = 5, spacing_cells: int = 17) -> List[Dict[str, float]]:
    """Centres of the widest safe areas: peaks of the distance-to-nearest-unsafe-cell map."""
    # Outside the box is unknown, so treat it as unsafe.
    dist = ndimage.distance_transform_edt(np.pad(safe, 1, constant_values=False))[1:-1, 1:-1]
    work = dist.copy()
    spots = []
    rr, cc = np.mgrid[0:safe.shape[0], 0:safe.shape[1]]
    for _ in range(count):
        r, c = np.unravel_index(np.argmax(work), work.shape)
        if work[r, c] < min_radius_cells:
            break
        spots.append({"row": int(r), "col": int(c), "margin_cells": float(dist[r, c])})
        work[(rr - r) ** 2 + (cc - c) ** 2 <= spacing_cells ** 2] = 0
    return spots


def render() -> None:
    st.title("🛬 Landing Safety Map")
    st.caption(
        "Every 30 m cell of the 5 km × 5 km target box is graded for a lander: green = safe, amber = risky, red = no-go. "
        "Stars mark the spots with the widest safe margin around them."
    )

    sb = st.sidebar
    sb.header("Landing Site")
    target_name = sb.selectbox("Target", list(TARGET_DEM_FILES.keys()), key="ls_target")
    layers = terrain_layers(target_name)
    meta = layers["meta"]
    tmeta = meta["target"]
    polar = abs(float(tmeta.get("lat", 0))) > 80
    body = "Mars" if tmeta.get("body") == "mars" else "Moon"

    sb.header("Lander Limits")
    slope_safe, slope_nogo = sb.slider(
        "Slope: safe up to / no-go above (°)", 0, 30, (8, 15),
        help="Artemis-class landers are designed for roughly 8-10°; Apollo LM tolerated about 12°.",
    )
    bump_safe, bump_nogo = sb.slider(
        "Bumpiness: safe up to / no-go above (m)", 0.0, 6.0, (1.5, 3.0), 0.1,
        help="Height of local bumps and pits relative to the surrounding 30 m cells (boulders, crater rims).",
    )
    radius_m = sb.slider(
        "Required safe radius around touchdown (m)", 30, 300, 90, 30,
        help="A spot is only offered if every cell within this radius is green.",
    )

    use_sun = False
    lit = None
    if polar:
        sb.header("Sunlight (polar)")
        use_sun = sb.checkbox("Grade by sunlight for solar power", value=True, key="ls_sun")
        sun_el = sb.slider(
            "Sun elevation (°)", 0.5, 5.0, 1.5, 0.5,
            help="At the lunar south pole the sun stays about 1.5° above the horizon and circles all the way round.",
        )
        lit_min = sb.slider("Risky if lit less than (% of the day)", 0, 100, 30, 5)
        lit = sunlight_fraction(target_name, float(sun_el))

    slope, bump = layers["slope"], layers["bump"]
    cls = np.full(slope.shape, SAFE, dtype=int)
    risky = (slope > slope_safe) | (bump > bump_safe)
    nogo = (slope > slope_nogo) | (bump > bump_nogo)
    if use_sun and lit is not None:
        risky |= lit * 100 < lit_min
    cls[risky] = RISKY
    cls[nogo] = NOGO

    radius_cells = radius_m / CELL_M
    spots = best_spots(cls == SAFE, radius_cells)

    # ---- headline numbers ---------------------------------------------------- #
    cell_km2 = (CELL_M / 1000) ** 2
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Safe", f"{(cls == SAFE).mean() * 100:.0f}%")
    m2.metric("Risky", f"{(cls == RISKY).mean() * 100:.0f}%")
    m3.metric("No-go", f"{(cls == NOGO).mean() * 100:.0f}%")
    if (cls == SAFE).any():
        labels, nlab = ndimage.label(cls == SAFE)
        largest = np.bincount(labels.ravel())[1:].max() * cell_km2
        m4.metric("Largest safe patch", f"{largest:.2f} km²")
    else:
        m4.metric("Largest safe patch", "none")

    # ---- map ----------------------------------------------------------------- #
    n = slope.shape[0]
    half_km = n * CELL_M / 2000
    km = (np.arange(n) - (n - 1) / 2) * CELL_M / 1000
    shade = hillshade(layers["z"], 135, 6 if polar else 25)[1:-1, 1:-1].astype(float)
    shade = 0.25 + 0.75 * shade / 255
    rgb = np.zeros((n, n, 3))
    for k, col in CLASS_COLORS.items():
        rgb[cls == k] = col
    alpha = 0.45
    img = np.clip(rgb * alpha + (shade[..., None] * 255) * (1 - alpha), 0, 255) * shade[..., None] ** 0.3
    img = img.astype(np.uint8)

    lit_layer = lit * 100 if lit is not None else np.full(slope.shape, np.nan)
    custom = np.stack([slope, bump, lit_layer, cls], axis=-1)
    names = np.array(["Safe", "Risky", "No-go"])

    fig = go.Figure()
    # Image rows are drawn bottom-up here (y increases upward), so flip so that grid row 0 ends up on top.
    fig.add_trace(go.Image(z=img[::-1], x0=km[0], dx=CELL_M / 1000, y0=km[0], dy=CELL_M / 1000, hoverinfo="skip"))
    fig.add_trace(go.Heatmap(
        z=cls, x=km, y=-km, opacity=0, showscale=False, customdata=custom,
        text=names[cls],
        hovertemplate=(
            "<b>%{text}</b><br>x %{x:.2f} km, y %{y:.2f} km<br>slope %{customdata[0]:.1f}°"
            "<br>bumpiness %{customdata[1]:.2f} m" + ("<br>lit %{customdata[2]:.0f}% of the day" if lit is not None else "")
            + "<extra></extra>"
        ),
    ))
    if spots:
        sx = [km[s["col"]] for s in spots]
        sy = [-km[s["row"]] for s in spots]
        fig.add_trace(go.Scatter(
            x=sx, y=sy, mode="markers+text", text=[f"#{i + 1}" for i in range(len(spots))],
            textposition="top center", textfont=dict(color="white", size=13),
            marker=dict(symbol="star", size=18, color="#facc15", line=dict(color="#0f172a", width=1.5)),
            hovertemplate="Landing spot %{text}<extra></extra>",
        ))
        theta = np.linspace(0, 2 * np.pi, 60)
        for x0, y0 in zip(sx, sy):
            fig.add_trace(go.Scatter(
                x=x0 + radius_m / 1000 * np.cos(theta), y=y0 + radius_m / 1000 * np.sin(theta),
                mode="lines", line=dict(color="#facc15", width=1.5, dash="dot"), hoverinfo="skip", showlegend=False,
            ))
    fig.update_layout(
        template="plotly_dark", height=620, showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(title="km " + ("right" if polar else "east") + " of box centre", range=[-half_km, half_km], scaleanchor="y", constrain="domain"),
        yaxis=dict(title="km " + ("toward 0° longitude" if polar else "north") + " of box centre",
                   range=[-half_km, half_km]),
    )

    left, right = st.columns([1.6, 1])
    with left:
        st.plotly_chart(fig, key="ls_map", **_stretch())
        st.caption(
            f"{target_name} ({body}) · {meta.get('source', '')} · {meta.get('product', '')} (native {meta.get('native', '?')}). "
            + ("Plate top = 0° longitude, not north." if polar else "Plate top = north.")
            + (f" {tmeta['caveat']}" if tmeta.get("caveat") else "")
        )
    with right:
        st.subheader("Best touchdown spots")
        if not spots:
            st.warning(
                f"No spot has {radius_m} m of green all around it with these limits. "
                "Loosen the slope / bumpiness limits or shrink the safe radius."
            )
        else:
            rows = []
            for i, s in enumerate(spots):
                r, c = s["row"], s["col"]
                rows.append({
                    "Spot": f"#{i + 1}",
                    "x (km)": round(float(km[c]), 2),
                    "y (km)": round(float(-km[r]), 2),
                    "Safe margin (m)": int(s["margin_cells"] * CELL_M),
                    "Slope (°)": round(float(slope[r, c]), 1),
                    **({"Lit (%)": int(round(lit[r, c] * 100))} if lit is not None else {}),
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True, **_stretch())
            st.caption("Safe margin = distance from the spot to the nearest non-green cell (or the box edge).")

        st.subheader("How cells are graded")
        lines = [
            f"- **No-go**: slope > {slope_nogo}° or bumpiness > {bump_nogo:.1f} m",
            f"- **Risky**: slope > {slope_safe}° or bumpiness > {bump_safe:.1f} m"
            + (f" or lit < {lit_min}% of the day" if use_sun else ""),
            "- **Safe**: everything else",
        ]
        st.markdown("\n".join(lines))
        with st.expander("Limitations"):
            st.markdown(
                "- 30 m cells cannot see individual boulders smaller than a few metres; real site certification "
                "uses ~1 m LROC NAC DEMs and hazard imagery.\n"
                "- Shadows are cast only by terrain inside this 5 km box; distant ridges are ignored, so polar "
                "sunlight is somewhat optimistic.\n"
                "- Sunlight is a fixed-elevation sweep around the horizon, not a full ephemeris."
            )


render()
