"""Draw target vs. best 3 shape matches (rotated/mirrored as matched) -> discovery/shape_matches.png"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy.ndimage import rotate as nd_rotate

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from terrain_twin import fetch_earth_dem, hillshade, load_target_dem  # noqa: E402

SRC = sys.argv[1] if len(sys.argv) > 1 else "shape_candidates.csv"
OUT = sys.argv[2] if len(sys.argv) > 2 else "shape_matches.png"
d = pd.read_csv(HERE / SRC)
S, PAD, HEAD = 260, 10, 34
targets = list(d["target"].unique())
canvas = Image.new("RGB", (PAD + 4 * (S + PAD), len(targets) * (S + HEAD + PAD) + PAD), (15, 23, 42))
draw = ImageDraw.Draw(canvas)


def plate(z):
    hs = hillshade(z - z.min(), 135, 20)
    return Image.fromarray(hs).resize((S, S), Image.BILINEAR).convert("RGB")


for i, t in enumerate(targets):
    y = PAD + i * (S + HEAD + PAD)
    tz = load_target_dem(t)["z"]
    canvas.paste(plate(tz), (PAD, y + HEAD))
    draw.text((PAD, y + 4), f"{t} (target)", fill=(255, 255, 255))
    for j, (_, r) in enumerate(d[d["target"] == t].head(3).iterrows(), start=1):
        ez = fetch_earth_dem(r.lat, r.lon)["z"]
        e0 = ez[:, ::-1] if r.mirrored else ez
        e = nd_rotate(e0, r.rotation_deg, reshape=False, order=1, mode="nearest")
        x = PAD + j * (S + PAD)
        canvas.paste(plate(e), (x, y + HEAD))
        draw.text((x, y + 4), f"#{j}  {r.lat:.3f}, {r.lon:.3f}", fill=(125, 211, 252))
        draw.text((x, y + 18), f"score {r.get('final_score', r.shape_corr):.2f}  rot {r.rotation_deg}{' mir' if r.mirrored else ''}",
                  fill=(148, 163, 184))

canvas.save(HERE / OUT)
print(HERE / OUT)
