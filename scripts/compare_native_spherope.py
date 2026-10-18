"""Run controlled FLUX.1 ablations through an external, unmodified SpheRoPE checkout.

Run this with the standalone environment's Python, not ComfyUI's embedded Python.
The model must already exist in a local Diffusers directory; no downloads occur.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-repo", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--seed", type=int, default=80084)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--cases", nargs="+", default=["stock", "circular", "rope", "cfg_only", "full"],
                        choices=["stock", "circular", "rope", "cfg_only", "full"])
    args = parser.parse_args()
    if not (args.reference_repo / "panorama_diffusers").is_dir():
        parser.error("--reference-repo must contain the original panorama_diffusers package")
    if not (args.model_dir / "model_index.json").is_file():
        parser.error("--model-dir must be an existing local Diffusers model")
    if args.width != 2 * args.height:
        parser.error("Use a 2:1 equirectangular resolution")

    os.environ.update(USE_TF="0", USE_FLAX="0", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    sys.path.insert(0, str(args.reference_repo.resolve()))
    import numpy as np
    import torch
    import diffusers
    import transformers
    from panorama_diffusers import ERPFluxPipeline

    args.output_dir.mkdir(parents=True, exist_ok=True)
    environment = {
        "reference_commit": subprocess.check_output(
            ["git", "-C", str(args.reference_repo), "rev-parse", "HEAD"], text=True).strip(),
        "python": sys.version, "torch": torch.__version__, "diffusers": diffusers.__version__,
        "transformers": transformers.__version__, "model_dir": str(args.model_dir.resolve()),
        "prompt": args.prompt, "seed": args.seed, "steps": args.steps,
        "width": args.width, "height": args.height,
        "embedded_guidance": 3.5, "real_cfg": 3.5, "geometry_gamma": 6.0,
        "geometry_start_step": 2,
    }
    (args.output_dir / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    pipe = ERPFluxPipeline.from_pretrained(
        str(args.model_dir), torch_dtype=torch.bfloat16, local_files_only=True)
    pipe.enable_model_cpu_offload()
    results = []
    for case in args.cases:
        started = time.perf_counter()
        settings = {
            "sfc_rope": case in ("rope", "cfg_only", "full"),
            "enable_circular_encoding": case != "stock",
            "semantic_distortion_cfg": case in ("cfg_only", "full"),
            "semantic_distortion_gamma": 0.0 if case == "cfg_only" else 6.0,
            "no_cfg_until_timestep": 2,
        }
        image = pipe(
            prompt=args.prompt, height=args.height, width=args.width,
            num_inference_steps=args.steps, guidance_scale=3.5,
            generator=torch.Generator("cpu").manual_seed(args.seed), **settings).images[0]
        image.save(args.output_dir / f"{case}.png")
        pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
        differences = np.abs(pixels - np.roll(pixels, 1, axis=1)).mean(axis=(0, 2))
        results.append({
            "case": case, "seconds": time.perf_counter() - started, **settings,
            "edge_mae_255": float(differences[0]),
            "edge_ratio": float(differences[0] / max(float(np.median(differences)), 1e-8)),
        })
        (args.output_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(json.dumps(results[-1]), flush=True)


if __name__ == "__main__":
    main()
