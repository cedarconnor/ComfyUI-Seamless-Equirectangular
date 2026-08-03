"""Manual CLI metrics tool for measuring equirectangular video seam and quality metrics.

Usage:
    python tests/measure_h3_output.py /path/to/png_frames_dir [--guide /path/to/first_frame_guide.png] [--output /path/to/report.json]
"""

import argparse
import glob
import json
import math
import os
import sys

try:
    from PIL import Image
    import numpy as np
except ImportError:
    Image = None
    np = None


def measure_frame_metrics(img_np):
    # img_np shape: (H, W, C) float [0, 1]
    H, W, C = img_np.shape

    # Edge MAE between left column (0) and right column (W-1)
    edge_mae = float(np.mean(np.abs(img_np[:, 0, :] - img_np[:, W - 1, :])))

    # Adjacent column MAE for all x in 0..W-2
    adj_mae = np.mean(np.abs(img_np[:, :-1, :] - img_np[:, 1:, :]), axis=(0, 2))

    # Baseline median from columns outside the outer 5% bands
    b_margin = max(1, int(0.05 * W))
    interior_adj_mae = adj_mae[b_margin : (W - 1) - b_margin]
    if len(interior_adj_mae) == 0:
        interior_adj_mae = adj_mae

    baseline = float(np.median(interior_adj_mae))
    seam_ratio = float(edge_mae / max(baseline, 1e-8))

    return edge_mae, seam_ratio, baseline


def calculate_psnr(img1, img2):
    mse = float(np.mean((img1 - img2) ** 2))
    if mse == 0:
        return float("inf")
    return float(10 * math.log10(1.0 / mse))


def main():
    parser = argparse.ArgumentParser(description="Measure seam and quality metrics for equirectangular frames.")
    parser.add_argument("frames_dir", help="Directory containing PNG frame sequence")
    parser.add_argument("--guide", help="Optional path to first panorama keyframe guide image")
    parser.add_argument("--output", help="Optional path to save output JSON report")
    args = parser.parse_args()

    if Image is None or np is None:
        print("Error: PIL and numpy required for measure_h3_output.py", file=sys.stderr)
        sys.exit(1)

    frame_paths = sorted(
        glob.glob(os.path.join(args.frames_dir, "*.png"))
        + glob.glob(os.path.join(args.frames_dir, "*.jpg"))
    )

    if not frame_paths:
        print(f"No PNG/JPG frames found in {args.frames_dir}", file=sys.stderr)
        sys.exit(1)

    frames = []
    edge_maes = []
    seam_ratios = []

    for path in frame_paths:
        with Image.open(path) as img:
            img_np = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
        frames.append(img_np)
        edge_mae, seam_ratio, _ = measure_frame_metrics(img_np)
        edge_maes.append(edge_mae)
        seam_ratios.append(seam_ratio)

    edge_maes_np = np.array(edge_maes)
    seam_ratios_np = np.array(seam_ratios)
    max_idx = int(np.argmax(seam_ratios_np))

    # Calculate temporal deltas between consecutive frames
    temp_maes = []
    for i in range(len(frames) - 1):
        delta = float(np.mean(np.abs(frames[i + 1] - frames[i])))
        temp_maes.append(delta)

    median_later_temp_mae = float(np.median(temp_maes[3:])) if len(temp_maes) > 3 else (float(np.median(temp_maes)) if temp_maes else 0.0)

    temporal_metrics = {
        "frame_0_1_mae": temp_maes[0] if len(temp_maes) > 0 else None,
        "frame_1_2_mae": temp_maes[1] if len(temp_maes) > 1 else None,
        "frame_2_3_mae": temp_maes[2] if len(temp_maes) > 2 else None,
        "median_later_temporal_mae": median_later_temp_mae,
        "frame_0_1_ratio_to_later": (temp_maes[0] / max(median_later_temp_mae, 1e-8)) if len(temp_maes) > 0 else None,
    }

    # Opening guide PSNR if guide image provided
    guide_psnr = {}
    if args.guide and os.path.exists(args.guide):
        with Image.open(args.guide) as g_img:
            H, W, _ = frames[0].shape
            g_resized = g_img.convert("RGB").resize((W, H), Image.Resampling.LANCZOS)
            g_np = np.array(g_resized, dtype=np.float32) / 255.0
            for idx in range(min(5, len(frames))):
                guide_psnr[f"frame_{idx}_guide_psnr"] = calculate_psnr(g_np, frames[idx])

    report = {
        "num_frames": len(frame_paths),
        "frame_resolution": [int(frames[0].shape[0]), int(frames[0].shape[1])],
        "median_edge_mae": float(np.median(edge_maes_np)),
        "median_seam_ratio": float(np.median(seam_ratios_np)),
        "p95_seam_ratio": float(np.percentile(seam_ratios_np, 95)),
        "max_seam_ratio": float(seam_ratios_np[max_idx]),
        "max_seam_ratio_frame_index": max_idx,
        "frame_0_seam_ratio": float(seam_ratios_np[0]) if len(seam_ratios_np) > 0 else None,
        "frame_1_seam_ratio": float(seam_ratios_np[1]) if len(seam_ratios_np) > 1 else None,
        "frame_2_seam_ratio": float(seam_ratios_np[2]) if len(seam_ratios_np) > 2 else None,
        "frame_3_seam_ratio": float(seam_ratios_np[3]) if len(seam_ratios_np) > 3 else None,
        "temporal_metrics": temporal_metrics,
        "guide_psnr": guide_psnr,
    }

    report_json = json.dumps(report, indent=2)
    print(report_json)

    if args.output:
        with open(args.output, "w") as f:
            f.write(report_json + "\n")
        print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
