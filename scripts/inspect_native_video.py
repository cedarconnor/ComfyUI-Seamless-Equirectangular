"""Measure lossless frames and make rectilinear join/pole review sheets."""
import argparse
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageDraw


def perspective(image, yaw, pitch, size=384):
    p = (np.arange(size, dtype=np.float32) + 0.5) / size * 2 - 1
    x, y = np.meshgrid(p, -p)
    rays = np.stack((x, y, np.ones_like(x)), -1)
    cp, sp, cy, sy = np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    rays = rays @ np.array([[1, 0, 0], [0, cp, sp], [0, -sp, cp]], np.float32).T
    rays = rays @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], np.float32).T
    lon = np.arctan2(rays[..., 0], rays[..., 2])
    lat = np.arctan2(rays[..., 1], np.hypot(rays[..., 0], rays[..., 2]))
    mx = ((lon / (2 * np.pi) + 0.5) * image.shape[1] - 0.5).astype(np.float32)
    my = ((0.5 - lat / np.pi) * image.shape[0] - 0.5).astype(np.float32)
    # Longitude wraps; latitude clamps rather than wrapping north to south.
    wrapped = np.concatenate((image[:, -1:], image, image[:, :1]), axis=1)
    return cv2.remap(wrapped, (mx % image.shape[1]) + 1, my,
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--prefix", default="full")
    args = parser.parse_args()
    files = sorted(args.folder.glob(args.prefix + "_[0-9][0-9][0-9].png"))
    if not files:
        raise FileNotFoundError("No lossless frames found")
    rows, previous = [], None
    for file in files:
        image = np.array(Image.open(file).convert("RGB"), dtype=np.float32)
        jumps = np.abs(image - np.roll(image, 1, axis=1)).mean((0, 2))
        rows.append({"frame": file.name, "edge_mae_255": float(jumps[0]),
                     "edge_ratio": float(jumps[0] / max(float(np.median(jumps)), 1e-8)),
                     "temporal_mae_255": float(np.abs(image - previous).mean()) if previous is not None else None})
        previous = image
    report = {"frames": rows, "edge_ratio_mean": float(np.mean([r["edge_ratio"] for r in rows])),
              "edge_ratio_max": max(r["edge_ratio"] for r in rows),
              "edge_mae_255_mean": float(np.mean([r["edge_mae_255"] for r in rows]))}
    (args.folder / (args.prefix + "_metrics.json")).write_text(json.dumps(report, indent=2))
    indices = sorted({0, len(files) // 2, len(files) - 1})
    canvas = Image.new("RGB", (1152, len(indices) * 412), "#222222")
    draw = ImageDraw.Draw(canvas)
    for row, index in enumerate(indices):
        image = np.array(Image.open(files[index]).convert("RGB"))
        for col, (label, yaw, pitch) in enumerate((("join", np.pi, 0), ("zenith", 0, np.pi / 2), ("nadir", 0, -np.pi / 2))):
            canvas.paste(Image.fromarray(perspective(image, yaw, pitch)), (col * 384, row * 412 + 28))
            draw.text((col * 384 + 8, row * 412 + 6), f"{args.prefix} / frame {index} / {label}", fill="white")
    canvas.save(args.folder / (args.prefix + "_inspection.jpg"))
    import imageio_ffmpeg
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-framerate", "24",
                    "-i", str(args.folder / (args.prefix + "_%03d.png")), "-c:v", "libx264", "-crf", "18",
                    "-pix_fmt", "yuv420p", str(args.folder / (args.prefix + ".mp4"))], check=True)
    print(json.dumps({k: v for k, v in report.items() if k != "frames"}, indent=2))


if __name__ == "__main__":
    main()
