"""Produce lossless ERP seam metrics and six rectilinear inspection views."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inspect_native_video import perspective


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    pixels = np.array(Image.open(args.image).convert("RGB"))
    h, w, _ = pixels.shape
    if w != 2*h:
        raise ValueError(f"Expected 2:1 ERP; got {w}x{h}")
    f = pixels.astype(np.float32)
    jumps = np.abs(f - np.roll(f, 1, axis=1)).mean((0, 2))
    metrics = {"source": str(args.image.resolve()), "width": w, "height": h,
               "sha256": hashlib.sha256(args.image.read_bytes()).hexdigest(),
               "edge_mae_255": float(jumps[0]),
               "median_column_jump": float(np.median(jumps)),
               "edge_ratio": float(jumps[0]/max(np.median(jumps), 1e-8)),
               "p95_column_jump": float(np.percentile(jumps, 95)),
               "note": "These metrics do not validate spherical geometry. Inspect all six views."}
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    views = [("front", 0, 0), ("right", np.pi/2, 0), ("join", np.pi, 0),
             ("left", -np.pi/2, 0), ("zenith", 0, np.pi/2), ("nadir", 0, -np.pi/2)]
    canvas = Image.new("RGB", (3*384, 2*412), "#222222")
    draw = ImageDraw.Draw(canvas)
    for i, (name, yaw, pitch) in enumerate(views):
        crop = Image.fromarray(perspective(pixels, yaw, pitch, 384))
        crop.save(args.output / f"{name}.png")
        x, y = (i % 3)*384, (i // 3)*412
        canvas.paste(crop, (x, y+28))
        draw.text((x+8, y+7), name, fill="white")
    canvas.save(args.output / "six_views.jpg", quality=95)
    Image.fromarray(np.roll(pixels, w//2, axis=1)).save(args.output / "join_centered.png")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
