"""
Terrain morph: make the best Earth candidate *synthetically* closer to its target (Moon / Mars),
and report honestly how much of the gap each change closes.

Works on the same 169 x 169 / 30 m grid as ``terrain_twin`` and measures with the same code
(Horn slope, 8-neighbour roughness, cast-shadow hillshade), so "before" and "after" are comparable.

Pipeline (each stage is reported separately):

1. **Spectral shaping** - keep the Earth site's *phase* (where its hills and valleys are) and give each
   spatial scale the target's amplitude (radially averaged power spectrum).  ``beta`` in [0, 1] blends
   from "pure Earth" (0) to "full target spectrum" (1); the gain per scale is capped so SRTM noise is
   not blown up into fake terrain.
2. **Crater injection** - simple bowl craters (depth/diameter ~0.2, raised rim, ejecta falloff) drawn
   from a lunar-like power-law size-frequency distribution, with random degradation.
3. **Angle-of-repose cap** - local smoothing wherever slope exceeds the regolith friction angle.

Surface properties (albedo, thermal, mineralogy, volatiles) are not part of the DEM; ``surface_gaps``
lists them with how (or whether) they could be closed.

Honesty rule: every report carries ``earth_share`` (how much of the final relief is still the real
Earth site) so a "99 % similar" synthetic twin that is really just the Moon is visible as such.

CLI:  python terrain_morph.py --target "Shackleton Ridge" [--lat .. --lon ..] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter

from terrain_twin import CELL_M, hillshade, roughness_m, slope_deg, summarize

DEFAULT_REPOSE_DEG = 35.0
GEOM_KEYS = ("slope_median", "slope_p90", "roughness_median", "relief_m")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _detrend(z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Remove the best-fit plane; return (residual, plane)."""
    n0, n1 = z.shape
    yy, xx = np.mgrid[0:n0, 0:n1].astype(float)
    a = np.column_stack([xx.ravel(), yy.ravel(), np.ones(z.size)])
    coef, *_ = np.linalg.lstsq(a, z.ravel(), rcond=None)
    plane = (a @ coef).reshape(z.shape)
    return z - plane, plane


def _mirror(z: np.ndarray) -> np.ndarray:
    """Symmetric extension to 2n x 2n so the FFT sees no edge jumps."""
    top = np.hstack([z, z[:, ::-1]])
    return np.vstack([top, top[::-1, :]])


def _radial_bins(shape: Tuple[int, int], nbins: int = 40) -> Tuple[np.ndarray, np.ndarray]:
    ky = np.fft.fftfreq(shape[0])
    kx = np.fft.fftfreq(shape[1])
    k = np.hypot(*np.meshgrid(ky, kx, indexing="ij"))
    edges = np.geomspace(1.0 / max(shape), k.max() + 1e-9, nbins + 1)
    idx = np.clip(np.digitize(k, edges) - 1, -1, nbins - 1)
    idx[k < edges[0]] = -1  # DC / lowest frequencies stay untouched
    return idx, k


def radial_psd(z: np.ndarray, nbins: int = 40) -> np.ndarray:
    """Radially averaged power spectrum of the detrended, mirrored DEM (one value per log bin)."""
    r, _ = _detrend(z)
    f = np.fft.fft2(_mirror(r))
    p = np.abs(f) ** 2
    idx, _ = _radial_bins(p.shape, nbins)
    out = np.zeros(nbins)
    for b in range(nbins):
        m = idx == b
        out[b] = p[m].mean() if m.any() else np.nan
    return out


def _full_slope(z: np.ndarray, cell: float = CELL_M) -> np.ndarray:
    return slope_deg(np.pad(z, 1, mode="edge"), cell)


def shadow_fraction(z: np.ndarray, sun_el_deg: float, sun_az_deg: float = 135.0) -> float:
    return float((hillshade(z, sun_az_deg, sun_el_deg) == 0).mean())


def measure(z: np.ndarray, sun_el_deg: Optional[float] = None) -> Dict[str, Any]:
    s = summarize(z)
    out = {k: s[k] for k in GEOM_KEYS}
    out["slope_quantiles"] = np.percentile(s["slopes"], np.linspace(5, 95, 19))
    if sun_el_deg is not None:
        out["shadow_fraction"] = shadow_fraction(z, sun_el_deg)
    return out


def distance(m: Mapping[str, Any], t: Mapping[str, Any]) -> float:
    """0 = identical. Mean |log ratio| over the four headline metrics + slope-distribution term."""
    eps = 0.05
    terms = [abs(math.log((m[k] + eps) / (t[k] + eps))) for k in GEOM_KEYS]
    qm, qt = np.asarray(m["slope_quantiles"]), np.asarray(t["slope_quantiles"])
    terms.append(float(np.mean(np.abs(qm - qt)) / (np.mean(qt) + eps)))
    return float(np.mean(terms))


def earth_share(z_final: np.ndarray, z_earth: np.ndarray) -> float:
    """R^2 of the final (detrended) relief explained by the original Earth relief. 1 = untouched."""
    a, _ = _detrend(z_final)
    b, _ = _detrend(z_earth)
    if a.std() < 1e-9 or b.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1] ** 2)


# --------------------------------------------------------------------------- #
# Stage 1: spectral shaping
# --------------------------------------------------------------------------- #
def spectral_shape(z_earth: np.ndarray, z_target: np.ndarray, beta: float,
                   max_gain: float = 25.0, nbins: int = 40) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Give each spatial scale of the Earth DEM the target's amplitude (keeping Earth's phase)."""
    if beta <= 0:
        return z_earth.copy(), {"gain_capped_bins": 0, "max_gain_used": 1.0}
    n0, n1 = z_earth.shape
    r, plane = _detrend(z_earth)
    f = np.fft.fft2(_mirror(r))
    idx, _ = _radial_bins(f.shape, nbins)
    pe, pt = radial_psd(z_earth, nbins), radial_psd(z_target, nbins)
    with np.errstate(divide="ignore", invalid="ignore"):
        g_raw = np.sqrt(pt / pe)
    g_raw = np.where(np.isfinite(g_raw), g_raw, 1.0)
    g = np.clip(g_raw, 1.0 / max_gain, max_gain) ** beta
    gain = np.ones(f.shape)
    valid = idx >= 0
    gain[valid] = g[idx[valid]]
    shaped = np.real(np.fft.ifft2(f * gain))[:n0, :n1]
    # The plane (regional tilt) is a large-scale feature: scale it with the same blend as the lowest bin.
    z = shaped + plane.mean() + (plane - plane.mean()) * g[0]
    info = {"gain_capped_bins": int(np.sum((g_raw > max_gain) | (g_raw < 1.0 / max_gain))),
            "max_gain_used": float(np.max(g))}
    return z, info


# --------------------------------------------------------------------------- #
# Stage 2: craters
# --------------------------------------------------------------------------- #
def crater_profile(r_norm: np.ndarray, depth: float, rim: float) -> np.ndarray:
    """Height change for a simple crater; r_norm = r / R."""
    inside = -depth + (depth + rim) * r_norm ** 2
    outside = rim * np.clip(r_norm, 1.0, None) ** -3.0
    return np.where(r_norm <= 1.0, inside, outside)


def sample_crater_diameters(rng: np.random.Generator, area_km2: float, density: float,
                            d_min_m: float = 3 * CELL_M, d_max_m: float = 1500.0,
                            slope_b: float = 2.0, c_per_km2: float = 0.1) -> np.ndarray:
    """Cumulative SFD N(>D) = density * c * D_km^-b per km^2 (lunar equilibrium-like for b ~ 2)."""
    n_expected = density * c_per_km2 * area_km2 * (d_min_m / 1000.0) ** -slope_b
    n = int(rng.poisson(max(n_expected, 0.0)))
    if n == 0:
        return np.zeros(0)
    u = rng.random(n)
    lo, hi = d_min_m ** -slope_b, d_max_m ** -slope_b
    return (lo - u * (lo - hi)) ** (-1.0 / slope_b)  # inverse-CDF of truncated power law


def inject_craters(z: np.ndarray, density: float, seed: int = 0, cell: float = CELL_M,
                   depth_ratio: float = 0.2, rim_ratio: float = 0.04) -> Tuple[np.ndarray, Dict[str, Any]]:
    if density <= 0:
        return z.copy(), {"n_craters": 0}
    rng = np.random.default_rng(seed)
    n0, n1 = z.shape
    area_km2 = n0 * n1 * (cell / 1000.0) ** 2
    diams = np.sort(sample_crater_diameters(rng, area_km2, density))[::-1]  # big first, small on top
    out = z.copy()
    yy, xx = np.mgrid[0:n0, 0:n1].astype(float)
    for d in diams:
        rad_px = d / 2 / cell
        cy, cx = rng.uniform(0, n0), rng.uniform(0, n1)
        fresh = rng.uniform(0.3, 1.0)  # degradation: old craters are shallower
        reach = int(math.ceil(rad_px * 3))
        r0, r1 = max(0, int(cy) - reach), min(n0, int(cy) + reach + 1)
        c0, c1 = max(0, int(cx) - reach), min(n1, int(cx) + reach + 1)
        if r0 >= r1 or c0 >= c1:
            continue
        rn = np.hypot(yy[r0:r1, c0:c1] - cy, xx[r0:r1, c0:c1] - cx) / rad_px
        out[r0:r1, c0:c1] += crater_profile(rn, depth_ratio * d * fresh, rim_ratio * d * fresh)
    return out, {"n_craters": int(len(diams)),
                 "largest_m": float(diams[0]) if len(diams) else 0.0}


# --------------------------------------------------------------------------- #
# Stage 3: angle of repose
# --------------------------------------------------------------------------- #
def cap_repose(z: np.ndarray, repose_deg: float = DEFAULT_REPOSE_DEG, iters: int = 30) -> Tuple[np.ndarray, Dict[str, Any]]:
    out = z.copy()
    before = float((_full_slope(out) > repose_deg).mean())
    for _ in range(iters):
        steep = _full_slope(out) > repose_deg
        if not steep.any():
            break
        smooth = gaussian_filter(out, 1.0, mode="nearest")
        mask = gaussian_filter(steep.astype(float), 1.0) > 0.05
        out = np.where(mask, smooth, out)
    return out, {"steep_fraction_before": before,
                 "steep_fraction_after": float((_full_slope(out) > repose_deg).mean())}


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
@dataclass
class Stage:
    name: str
    z: np.ndarray
    metrics: Dict[str, Any]
    distance: float
    info: Dict[str, Any] = field(default_factory=dict)


def default_sun_elevation(target_meta: Optional[Mapping[str, Any]]) -> float:
    """Polar targets: sun hugs the horizon (~1.5 deg). Otherwise a mid-morning 20 deg."""
    lat = None
    if target_meta:
        lat = (target_meta.get("target") or {}).get("lat")
    return 1.5 if lat is not None and abs(lat) > 80 else 20.0


def morph_to_target(z_earth: np.ndarray, z_target: np.ndarray, *, seed: int = 0,
                    betas: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
                    densities: Tuple[float, ...] = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0),
                    repose_deg: float = DEFAULT_REPOSE_DEG, max_gain: float = 25.0,
                    min_earth_share: float = 0.0, sun_el_deg: float = 20.0) -> Dict[str, Any]:
    """
    Grid-search (beta, crater density) for the smallest distance to the target, subject to
    ``earth_share >= min_earth_share``.  Returns the stage-by-stage report.
    """
    tm = measure(z_target)
    best: Optional[Tuple[float, float, float, float]] = None  # (dist, beta, density, share)
    shaped_cache: Dict[float, np.ndarray] = {}
    for b in betas:
        shaped_cache[b], _ = spectral_shape(z_earth, z_target, b, max_gain)
        for d in densities:
            zc, _ = inject_craters(shaped_cache[b], d, seed)
            share = earth_share(zc, z_earth)
            if min_earth_share > 0 and not (share >= min_earth_share):
                continue
            dist = distance(measure(zc), tm)
            if best is None or dist < best[0]:
                best = (dist, b, d, share)
    if best is None:
        raise ValueError(f"No setting keeps earth_share >= {min_earth_share}. Lower the constraint.")
    _, beta, dens, _ = best

    def stage(name: str, z: np.ndarray, info: Optional[Dict[str, Any]] = None) -> Stage:
        m = measure(z, sun_el_deg)
        return Stage(name, z, m, distance(m, tm), info or {})

    stages = [stage("original", z_earth.copy())]
    z1, i1 = spectral_shape(z_earth, z_target, beta, max_gain)
    stages.append(stage("spectral_shape", z1, {"beta": beta, **i1}))
    z2, i2 = inject_craters(z1, dens, seed)
    stages.append(stage("craters", z2, {"density": dens, **i2}))
    z3, i3 = cap_repose(z2, repose_deg)
    stages.append(stage("repose_cap", z3, {"repose_deg": repose_deg, **i3}))

    target_m = measure(z_target, sun_el_deg)
    return {"target": target_m, "stages": stages, "beta": beta, "density": dens,
            "earth_share": earth_share(z3, z_earth), "sun_el_deg": sun_el_deg,
            "gap_closure": gap_closure(stages, target_m)}


def gap_closure(stages: List[Stage], target_m: Mapping[str, Any]) -> Dict[str, Dict[str, float]]:
    """Per metric: % of the original gap closed after each stage (negative = overshoot got worse)."""
    keys = list(GEOM_KEYS) + (["shadow_fraction"] if "shadow_fraction" in target_m else [])
    g0 = {k: abs(stages[0].metrics[k] - target_m[k]) for k in keys}
    out: Dict[str, Dict[str, float]] = {}
    for s in stages[1:]:
        out[s.name] = {k: (100.0 * (1 - abs(s.metrics[k] - target_m[k]) / g0[k]) if g0[k] > 1e-9 else 100.0)
                       for k in keys}
    return out


# --------------------------------------------------------------------------- #
# Surface properties (not in the DEM)
# --------------------------------------------------------------------------- #
SURFACE_HOW = {
    "albedo": ("physical_patch", "Spread a regolith simulant layer on a small test patch; or recolour in simulation."),
    "mafic_index": ("physical_patch", "Choose a simulant with the target mineralogy (e.g. anorthositic vs. basaltic)."),
    "thermal_inertia_proxy": ("physical_patch", "Fine loose simulant lowers it, rocky/blocky cover raises it; only on a small patch."),
    "delta_t_k": ("simulation_only", "Needs vacuum / no atmosphere; only a thermal-vacuum chamber or simulation."),
    "clay_hydrated_index": ("simulation_only", "Site mineralogy; can only be overridden in simulation."),
    "subsurface_ice_moisture": ("simulation_only", "Can be mimicked with buried ice samples locally, otherwise simulation."),
}


def surface_gaps(row: Mapping[str, Any], target_data: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """For each non-DEM parameter: real value, target ideal/range, inside-range flag and how it could be closed."""
    params = target_data.get("parameters", {})
    rows = []
    for k, (kind, how) in SURFACE_HOW.items():
        if k not in params or k not in row:
            continue
        spec, v = params[k], float(row[k])
        rows.append({"feature": k, "earth": v, "ideal": spec["ideal"], "min": spec["min"], "max": spec["max"],
                     "in_range": spec["min"] <= v <= spec["max"],
                     "direction": "ok" if spec["min"] <= v <= spec["max"] else ("raise" if v < spec["min"] else "lower"),
                     "closable": kind, "how": how})
    return rows


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _fmt_report(rep: Dict[str, Any]) -> str:
    keys = list(GEOM_KEYS) + ["shadow_fraction"]
    lines = [f"{'stage':<16}" + "".join(f"{k:>18}" for k in keys) + f"{'distance':>10}"]
    t = rep["target"]
    lines.append(f"{'TARGET':<16}" + "".join(f"{t[k]:>18.3f}" for k in keys) + f"{0:>10.3f}")
    for s in rep["stages"]:
        lines.append(f"{s.name:<16}" + "".join(f"{s.metrics[k]:>18.3f}" for k in keys) + f"{s.distance:>10.3f}")
    lines.append("")
    lines.append("gap closed (%) after each stage:")
    for name, g in rep["gap_closure"].items():
        lines.append(f"  {name:<14}" + "".join(f"{k}={v:6.1f}  " for k, v in g.items()))
    lines.append("")
    lines.append(f"chosen beta={rep['beta']}  crater density={rep['density']}  sun elevation={rep['sun_el_deg']} deg")
    for s in rep["stages"][1:]:
        if s.info:
            lines.append(f"  {s.name}: {json.dumps(s.info, default=float)}")
    lines.append(f"earth_share (R^2 of final relief explained by the real site) = {rep['earth_share']:.3f}")
    return "\n".join(lines)


def main() -> None:
    import pandas as pd
    from terrain_twin import fetch_earth_dem, load_target_dem

    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", default="Shackleton Ridge")
    ap.add_argument("--lat", type=float)
    ap.add_argument("--lon", type=float)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-earth-share", type=float, default=0.0)
    args = ap.parse_args()

    ranked = pd.read_csv(here / "ranked_candidates.csv")
    row = ranked.iloc[0] if args.lat is None else None
    lat = args.lat if args.lat is not None else float(row["lat"])
    lon = args.lon if args.lon is not None else float(row["lon"])

    tdem = load_target_dem(args.target)
    if tdem is None:
        raise SystemExit(f"No DEM for target '{args.target}'.")
    z_earth = fetch_earth_dem(lat, lon)["z"]
    rep = morph_to_target(z_earth, tdem["z"], seed=args.seed, min_earth_share=args.min_earth_share,
                          sun_el_deg=default_sun_elevation(tdem["meta"]))
    print(f"Earth site {lat:.4f}, {lon:.4f}  ->  {args.target}\n")
    print(_fmt_report(rep))

    if row is not None:
        targets = json.loads((here / "lunar_targets.json").read_text(encoding="utf-8"))["targets"]
        if args.target in targets:
            print("\nsurface properties (not changed by the DEM morph):")
            for g in surface_gaps(row, targets[args.target]):
                flag = "ok   " if g["in_range"] else g["direction"]
                print(f"  [{flag}] {g['feature']:<24} earth={g['earth']:.3f}  target={g['ideal']} "
                      f"[{g['min']}, {g['max']}]  -> {g['closable']}: {g['how']}")


if __name__ == "__main__":
    main()
