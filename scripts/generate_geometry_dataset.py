#!/usr/bin/env python3
"""
Generate synthetic equirectangular video clips for geometry-anchor LoRA training.

Each clip is a short MP4 showing a procedural equirectangular pattern (gradient
sphere, grid, checkerboard, etc.) that encodes the spherical topology of the
360° projection. The dataset is fully procedural — no real 360° footage needed.

Output format (musubi-tuner compatible):
    <out_dir>/
        clip_000000.mp4   +  clip_000000.txt  (caption)
        clip_000001.mp4   +  clip_000001.txt  (caption)
        ...

Usage:
    # Generate 300 clips with default settings (gradient sphere + grid patterns)
    python generate_geometry_dataset.py --out_dir ./geo_data --num_clips 300

    # Custom resolution, slower rotation, all pattern types
    python generate_geometry_dataset.py --out_dir ./geo_data \\
        --num_clips 500 --height 384 --width 768 --num_frames 25 \\
        --pattern all --rotation_speed 6.0 --fps 24

Training cmdline (musubi-tuner):
    python ltx2_train_network.py \\
        --dataset_config '{"datasets": [{"video_directory": "./geo_data", \\
          "caption_extension": ".txt", "resolution": 512, "batch_size": 1, \\
          "cache_directory": "./geo_cache"}]}' \\
        --periodic_w_rope --roll_w_aug \\
        --ltx2_checkpoint path/to/model.safetensors \\
        --network_module networks.lora_ltx2 --network_dim 8 \\
        --learning_rate 2e-4 --max_train_steps 300 \\
        --output_dir ./geo_lora_output
"""

import argparse
import math
import os
import random
import struct
import sys
import time
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PI = math.pi
TAU = 2 * PI
DEG2RAD = PI / 180.0

CAPTION = "equirectangular projection, spherical panoramic view, 360-degree sphere"

# ---------------------------------------------------------------------------
# Colour utilities (vectorised numpy)
# ---------------------------------------------------------------------------


def hsv_to_rgb(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Vectorised HSV→RGB. All inputs [0,1]. Returns uint8 [H,W,3]."""
    h = h % 1.0
    hi = (h * 6).astype(np.int32) % 6
    f = h * 6.0 - np.floor(h * 6.0)
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)

    r = np.select(
        [hi == 0, hi == 1, hi == 2, hi == 3, hi == 4, hi == 5],
        [v, q, p, p, t, v],
    )
    g = np.select(
        [hi == 0, hi == 1, hi == 2, hi == 3, hi == 4, hi == 5],
        [t, v, v, q, p, p],
    )
    b = np.select(
        [hi == 0, hi == 1, hi == 2, hi == 3, hi == 4, hi == 5],
        [p, p, t, v, v, q],
    )
    rgb = np.stack([r, g, b], axis=-1)
    return (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    """Smooth Hermite interpolation between two edges."""
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# ---------------------------------------------------------------------------
# Spherical coordinate grids
# ---------------------------------------------------------------------------


def spherical_grids(H: int, W: int, rotation_rad: float = 0.0):
    """Compute latitude and longitude grids for an equirectangular frame.

    Returns
    -------
    lat : ndarray [H, W]  latitude in [-π/2, +π/2]  (north→south)
    lon : ndarray [H, W]  longitude in [0, 2π)
    """
    rows = np.arange(H, dtype=np.float32)[:, None]  # [H, 1]
    cols = np.arange(W, dtype=np.float32)[None, :]  # [1, W]
    lat = (rows / float(H)) * PI - PI / 2.0
    lon = (cols / float(W)) * TAU + rotation_rad
    lon = lon % TAU
    return lat, lon


# ---------------------------------------------------------------------------
# Noise helpers for colour palette variety
# ---------------------------------------------------------------------------

_PRNG = random.Random()


def _rand(a: float = 0.0, b: float = 1.0) -> float:
    return _PRNG.uniform(a, b)


def _randint(a: int, b: int) -> int:
    return _PRNG.randint(a, b)


def _choose(seq):
    return _PRNG.choice(seq)


# ---------------------------------------------------------------------------
# Pattern generators
# Each returns uint8 [H, W, 3]
# ---------------------------------------------------------------------------


def pattern_gradient_sphere(lat: np.ndarray, lon: np.ndarray, **kw) -> np.ndarray:
    """Latitude gradient sphere: hue=longitude, value=latitude (north bright).

    The canonical equirectangular encoding — cyclic hue demonstrates the
    longitude wrap, and brightness converging at the poles demonstrates
    polar convergence.
    """
    h = lon / TAU                            # [0, 1)
    s = np.full_like(h, kw.get("saturation", 0.85))
    # North pole bright → south pole dark
    v = (lat + PI / 2.0) / PI                # [0, 1]
    v = v * kw.get("v_range", 0.8) + (1.0 - kw.get("v_range", 0.8)) * 0.5
    return hsv_to_rgb(h, s, v)


def pattern_lat_bands(lat: np.ndarray, lon: np.ndarray, **kw) -> np.ndarray:
    """Horizontal latitude bands that visibly compress near the poles.

    Each band sits at a fixed latitude. In equirectangular projection the
    bands are straight horizontal lines, but their vertical spacing maps to
    equal angular intervals on the sphere.
    """
    n_bands = kw.get("n_bands", 8)
    band_half = PI / (n_bands * 6.0)
    bg = np.array(kw.get("bg", (0.05, 0.05, 0.08)), dtype=np.float32)
    fg_h = _rand(0.0, 1.0)

    rgb = np.broadcast_to(bg, (*lat.shape, 3)).copy()

    for i in range(1, n_bands):
        lat_center = -PI / 2.0 + i * (PI / (n_bands + 1))
        inside = np.abs(lat - lat_center) < band_half
        band_hue = fg_h + lon / TAU * 0.15
        # Build band colour in float [0,1] to match bg
        band_h = band_hue % 1.0
        band_s = np.full_like(band_h, 0.7)
        band_v = np.full_like(band_h, 0.9)
        band_rgb = hsv_to_rgb(band_h, band_s, band_v).astype(np.float32) / 255.0
        rgb = np.where(inside[..., None], band_rgb, rgb)

    return (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)


def pattern_lon_lines(lat: np.ndarray, lon: np.ndarray, **kw) -> np.ndarray:
    """Vertical longitude lines, continuous across the ±π seam.

    Each line spans the full height of the frame and corresponds to a fixed
    meridian. Teaches that column 0 and column W-1 are the same meridian.
    """
    n_lines = kw.get("n_lines", 12)
    line_half = TAU / (n_lines * 8.0)
    bg = np.array(kw.get("bg", (0.02, 0.02, 0.05)), dtype=np.float32)
    fg_h = _rand(0.0, 1.0)

    rgb = np.broadcast_to(bg, (*lat.shape, 3)).copy()

    for i in range(n_lines):
        lon_center = i * TAU / n_lines
        d = np.minimum(np.abs(lon - lon_center), np.abs(lon - (lon_center + TAU)))
        d = np.minimum(d, np.abs(lon - (lon_center - TAU)))
        inside = d < line_half
        # Build line colour as arrays matching spatial dims
        line_h = np.full_like(lat, (fg_h + lon_center / TAU * 0.1) % 1.0)
        line_s = np.full_like(lat, 0.6)
        line_v = np.full_like(lat, 0.85)
        line_rgb = hsv_to_rgb(line_h, line_s, line_v).astype(np.float32) / 255.0
        rgb = np.where(inside[..., None], line_rgb, rgb)

    return (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)


def pattern_lat_lon_grid(lat: np.ndarray, lon: np.ndarray, **kw) -> np.ndarray:
    """Combined latitude + longitude grid (wireframe sphere)."""
    # Generate band and line patterns separately, then overlay
    rgb1 = pattern_lat_bands(lat, lon, **{**kw, "bg": (0.0, 0.0, 0.0)})
    rgb2 = pattern_lon_lines(lat, lon, **{**kw, "bg": (0.0, 0.0, 0.0)})
    # Additive blend
    rgb = rgb1.astype(np.float32) * 0.5 + rgb2.astype(np.float32) * 0.5
    return np.clip(rgb, 0, 255).astype(np.uint8)


def pattern_checkerboard(lat: np.ndarray, lon: np.ndarray, **kw) -> np.ndarray:
    """UV checkerboard projected onto the sphere and rendered equirectangular.

    The squares are uniform in UV space so they appear regular at the equator
    and pinch visibly toward the poles — a strong polar-convergence signal.
    """
    n_u = kw.get("n_u", 16)
    n_v = kw.get("n_v", 8)

    # Normalised UV: u in [0,1] along longitude, v in [0,1] along latitude
    u = lon / TAU                                              # [0, 1)
    v = (lat + PI / 2.0) / PI                                  # [0, 1]

    ui = (u * n_u).astype(np.int32) % 2
    vi = (v * n_v).astype(np.int32) % 2
    checker = (ui ^ vi).astype(np.float32)

    # Alternate colour pairs
    c1 = np.array(kw.get("c1", (0.15, 0.40, 0.70)))
    c2 = np.array(kw.get("c2", (0.85, 0.75, 0.55)))
    img = np.where(checker[..., None], c1, c2)
    return (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)


def pattern_meridian_convergence(lat: np.ndarray, lon: np.ndarray, **kw) -> np.ndarray:
    """Concentric conic rings from the north to south pole.

    At the north pole all rings collapse to a dot; at the south pole they
    are at their widest. Each ring corresponds to a line of latitude.
    """
    n_rings = kw.get("n_rings", 12)
    # Normalised latitude [0, 1], 0 = north pole
    v = 1.0 - (lat + PI / 2.0) / PI
    ring_idx = (v * n_rings).astype(np.int32) % 2

    bg = np.array(kw.get("bg", (0.02, 0.02, 0.06)))
    fg_h = _rand(0.0, 1.0)
    fg_rgb = hsv_to_rgb(np.full_like(lon, fg_h), 0.7, 0.9)

    img = np.where(ring_idx[..., None], fg_rgb.astype(np.float32), bg)
    return np.clip(img, 0, 255).astype(np.uint8)


def pattern_equator_fade(lat: np.ndarray, lon: np.ndarray, **kw) -> np.ndarray:
    """Colourful at the equator, fading to neutral gray at the poles.

    Teaches that the equator is where the most visual information lives
    in equirectangular projection (least distortion).
    """
    # Blend factor: 1 at equator, 0 at poles
    lat_norm = np.abs(lat) / (PI / 2.0)  # [0, 1]
    fade = 1.0 - lat_norm ** kw.get("fade_power", 1.5)

    # Colourful pattern at equator
    hue = (lon / TAU + _rand(0.0, 1.0)) % 1.0
    sat = np.full_like(hue, kw.get("saturation", 0.8))
    val = np.full_like(hue, 0.9)

    colour = hsv_to_rgb(hue, sat, val).astype(np.float32)
    grey = np.full_like(colour, 0.4)

    img = colour * fade[..., None] + grey * (1.0 - fade[..., None])
    return np.clip(img, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Palette customisation
# ---------------------------------------------------------------------------

PATTERNS = {
    "gradient_sphere": pattern_gradient_sphere,
    "lat_bands": pattern_lat_bands,
    "lon_lines": pattern_lon_lines,
    "lat_lon_grid": pattern_lat_lon_grid,
    "checkerboard": pattern_checkerboard,
    "meridian_convergence": pattern_meridian_convergence,
    "equator_fade": pattern_equator_fade,
}

PATTERN_NAMES = sorted(PATTERNS.keys())

# Weights for random pattern selection when ``--pattern all``
# Gradient sphere is the primary signal so it gets higher weight.
_PATTERN_WEIGHTS = {
    "gradient_sphere": 3,
    "lat_bands": 1,
    "lon_lines": 1,
    "lat_lon_grid": 2,
    "checkerboard": 2,
    "meridian_convergence": 1,
    "equator_fade": 1,
}


def _random_palette_kw() -> dict:
    """Return random keyword arguments for pattern generators.

    Each clip gets its own random palette for visual diversity.
    """
    return {
        "saturation": _rand(0.5, 1.0),
        "v_range": _rand(0.6, 1.0),
        "n_bands": _randint(6, 16),
        "n_lines": _randint(8, 24),
        "n_u": _randint(8, 24),
        "n_v": _randint(4, 12),
        "n_rings": _randint(8, 20),
        "fade_power": _rand(1.0, 3.0),
    }


# ---------------------------------------------------------------------------
# Frame → PNG bytes  (used to build MP4 via pipe to ffmpeg)
# ---------------------------------------------------------------------------


def _frame_to_png_bytes(frame: np.ndarray) -> bytes:
    """Encode a uint8 [H,W,3] frame as PNG bytes."""
    from io import BytesIO
    buf = BytesIO()
    Image.fromarray(frame, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


def _write_mp4(
    frames: list[np.ndarray],
    path: str,
    fps: float = 8,
    preset: str = "ultrafast",
    ffmpeg_cmd: str = "ffmpeg",
):
    """Write frames as an MP4 via a pipe to ffmpeg.

    This avoids any Python video-encoding dependency — ffmpeg is assumed
    available on the system (it ships with ComfyUI's Python env or is
    installed globally).
    """
    import subprocess

    H, W = frames[0].shape[:2]

    cmd = [
        ffmpeg_cmd,
        "-y",
        "-f", "image2pipe",
        "-vcodec", "png",
        "-r", str(fps),
        "-i", "-",
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", preset,
        "-crf", "17",
        "-vf", f"scale={W}:{H}:flags=lanczos",
        "-an",
        str(path),
    ]

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        for frame in frames:
            proc.stdin.write(_frame_to_png_bytes(frame))
        proc.stdin.close()
        proc.wait()
    except Exception:
        proc.kill()
        raise

    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg exited with code {proc.returncode} for {path}"
        )


# ---------------------------------------------------------------------------
# Clip generation
# ---------------------------------------------------------------------------


def generate_clip(
    H: int,
    W: int,
    num_frames: int,
    pattern: str,
    rotation_speed: float = 2.0,
    fps: float = 8,
) -> list[np.ndarray]:
    """Generate a single video clip.

    Parameters
    ----------
    H, W : int
        Frame dimensions.
    num_frames : int
        Number of frames.
    pattern : str
        One of PATTERN_NAMES or "all" (random per clip).
    rotation_speed : float
        Degrees of Y-axis rotation per second.  2.0 ≈ one full rotation per 3 min.
    fps : float
        Frames per second (determines rotation per frame).

    Returns
    -------
    list[ndarray]
        List of uint8 [H,W,3] frames.
    """
    if pattern == "all":
        names = PATTERN_NAMES
        w = [_PATTERN_WEIGHTS.get(n, 1) for n in names]
        pattern = _PRNG.choices(names, weights=w, k=1)[0]

    gen_fn = PATTERNS[pattern]
    pal_kw = _random_palette_kw()

    # Pre-generate a stationary seed frame for the base lat/lon.
    # Rotation is applied per-frame on top.
    deg_per_frame = rotation_speed / fps
    frames: list[np.ndarray] = []

    for fi in range(num_frames):
        rotation_deg = fi * deg_per_frame
        lat, lon = spherical_grids(H, W, rotation_rad=rotation_deg * DEG2RAD)
        frame = gen_fn(lat, lon, **pal_kw)
        frames.append(frame)

    return frames


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------


def generate_dataset(
    out_dir: str,
    num_clips: int,
    H: int,
    W: int,
    num_frames: int,
    pattern: str,
    rotation_speed: float,
    fps: float,
    ffmpeg_cmd: str = "ffmpeg",
    seed: int = 0,
):
    """Generate the full dataset directory."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _PRNG.seed(seed + 1)  # separate from global seed

    t0 = time.time()

    for i in range(num_clips):
        clip_t0 = time.time()

        frames = generate_clip(
            H=H,
            W=W,
            num_frames=num_frames,
            pattern=pattern,
            rotation_speed=rotation_speed,
            fps=fps,
        )

        clip_id = f"clip_{i:06d}"
        mp4_path = str(out_dir / f"{clip_id}.mp4")
        txt_path = str(out_dir / f"{clip_id}.txt")

        _write_mp4(frames, mp4_path, fps=fps, ffmpeg_cmd=ffmpeg_cmd)

        with open(txt_path, "w") as f:
            f.write(CAPTION + "\n")

        elapsed = time.time() - clip_t0
        fps_actual = num_frames / elapsed if elapsed > 0 else 0
        eta = (time.time() - t0) / (i + 1) * (num_clips - i) if i > 0 else 0

        # Single-line status update
        pattern_name = pattern if pattern != "all" else frames.shape if hasattr(frames, 'shape') else pattern
        sys.stdout.write(
            f"\r{i + 1:>6d}/{num_clips}  |  "
            f"{pattern if pattern != 'all' else 'random'}  |  "
            f"{elapsed:.1f}s ({fps_actual:.1f} f/s)  |  "
            f"ETA {eta:.0f}s  "
        )
        sys.stdout.flush()

    total = time.time() - t0
    sys.stdout.write(f"\nDone.  {num_clips} clips in {total:.0f}s "
                     f"({total / num_clips:.1f}s/clip).\n")
    print(f"Output: {out_dir.resolve()}")
    print(f"Caption: \"{CAPTION}\"")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic equirectangular video clips for geometry LoRA training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory for generated clips and captions",
    )
    parser.add_argument(
        "--num_clips",
        type=int,
        default=300,
        help="Number of video clips to generate (default: 300)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=384,
        help="Frame height (default: 384, use 512 for square-ish or 480 for taller)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=768,
        help="Frame width (default: 768, use 512 for square or 848 for wider)",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=17,
        help="Frames per clip (default: 17 — odd number avoids odd-length bucket issues)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="all",
        choices=["all"] + PATTERN_NAMES,
        help="Pattern type: 'all' = random per clip (default), or a specific pattern name",
    )
    parser.add_argument(
        "--rotation_speed",
        type=float,
        default=6.0,
        help="Y-axis rotation speed in degrees/second (default: 6.0, "
             "~4° visible drift across a 17-frame clip at 24 fps)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=24,
        help="Frames per second for output MP4 (default: 24 — match LTX-2 training distribution)",
    )
    parser.add_argument(
        "--ffmpeg",
        type=str,
        default="ffmpeg",
        help="Path to ffmpeg executable (default: ffmpeg on PATH)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for dataset reproducibility (default: 42)",
    )

    args = parser.parse_args()
    kwargs = vars(args)
    kwargs["H"] = kwargs.pop("height")
    kwargs["W"] = kwargs.pop("width")
    kwargs["ffmpeg_cmd"] = kwargs.pop("ffmpeg")
    generate_dataset(**kwargs)


if __name__ == "__main__":
    main()
