"""CPU FLUX.1 VAE boundary controls; no denoiser or prompt is involved."""
import argparse
import json
import os
from pathlib import Path
import time

os.environ.update(USE_TF="0", USE_FLAX="0", HF_HUB_OFFLINE="1")

import numpy as np
from PIL import Image
import torch
from diffusers import AutoencoderKL


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    vae = AutoencoderKL.from_pretrained(
        str(args.model_dir / "vae"), torch_dtype=torch.float32,
        local_files_only=True).eval()
    h, w, pad = 512, 1024, 18
    lon, lat = np.meshgrid(
        ((np.arange(w) + .5) / w - .5) * 2 * np.pi,
        (.5 - (np.arange(h) + .5) / h) * np.pi)
    x, y, z = np.cos(lat)*np.sin(lon), np.sin(lat), np.cos(lat)*np.cos(lon)
    rgb = np.stack((.5 + .22*x + .12*np.sin(13*z + 4*y),
                    .5 + .22*y + .12*np.sin(11*x - 3*z),
                    .5 + .22*z + .12*np.sin(9*y + 5*x)), -1)
    sphere = (rgb.clip(0, 1)*255).round().astype(np.uint8)
    photo = np.array(Image.open(args.image).convert("RGB").resize(
        (w, h), Image.Resampling.LANCZOS))
    rows = []
    with torch.no_grad():
        for name, source in (("sphere", sphere), ("source_photo", photo)):
            Image.fromarray(source).save(args.output / f"{name}_source.png")
            pixels = torch.from_numpy(source.copy()).permute(2, 0, 1)[None].float()/127.5 - 1
            started = time.perf_counter()
            latent = vae.encode(pixels).latent_dist.mode()
            # Both outputs share exactly the same deterministic encoded latent.
            for mode in ("stock", "circular_decode"):
                value = latent
                if mode == "circular_decode":
                    value = torch.cat((latent[..., -pad:], latent, latent[..., :pad]), -1)
                decoded = vae.decode(value).sample
                if mode == "circular_decode":
                    decoded = decoded[..., pad*8:-pad*8]
                output = ((decoded[0].permute(1, 2, 0).clamp(-1, 1)+1)*127.5).round().byte().numpy()
                Image.fromarray(output).save(args.output / f"{name}_{mode}.png")
                f, src = output.astype(np.float32), source.astype(np.float32)
                residual = f-src
                polar_rows = np.r_[0:28, h-28:h]
                row = {
                    "input": name, "mode": mode,
                    "edge_mae_255": float(np.abs(f[:, 0]-f[:, -1]).mean()),
                    "source_edge_mae_255": float(np.abs(src[:, 0]-src[:, -1]).mean()),
                    "boundary_residual_mae_255": float(np.abs(residual[:, 0]-residual[:, -1]).mean()),
                    "polar_band_residual_mae_255": float(np.abs(residual[polar_rows]).mean()),
                    "image_residual_mae_255": float(np.abs(residual).mean()),
                    "elapsed_seconds_since_encode": time.perf_counter()-started,
                }
                rows.append(row)
                print(json.dumps(row), flush=True)
    report = {
        "scope": "CPU FP32 FLUX.1 VAE, deterministic posterior mode, 1024x512; not BF16 2K generation parity",
        "model_dir": str(args.model_dir.resolve()), "circular_pad_latent_columns": pad,
        "results": rows,
    }
    (args.output / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
