"""Run controlled FLUX.1 ablations through an external, unmodified SpheRoPE checkout.

Run this with the standalone environment's Python, not ComfyUI's embedded Python.
The model must already exist in a local Diffusers directory; no downloads occur.
"""

import argparse
import ast
import hashlib
import inspect
import json
import math
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
    parser.add_argument("--radius-multiplier", type=float, default=1.0,
                        help="Diagnostic only: multiply the released FLUX.1 spherical radius; pi matches the paper's W/2.")
    parser.add_argument("--stock-time", action="store_true",
                        help="Diagnostic only: retain stock FLUX image time coordinate zero.")
    parser.add_argument("--paper-offset", action="store_true",
                        help="Diagnostic only: add R to spherical X and Y, as in paper equation 2.")
    parser.add_argument("--save-latents", action="store_true")
    parser.add_argument("--real-cfg", type=float, default=3.5)
    parser.add_argument("--geometry-start", type=int, default=2)
    parser.add_argument("--probe-only", action="store_true",
                        help="Check the diagnostic builder's axis isolation without loading model weights.")
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
    import panorama_diffusers.erp_utils as erp_module

    args.output_dir.mkdir(parents=True, exist_ok=True)
    builder = erp_module.build_sfc_spherical_rope
    builder_source = inspect.getsource(builder)
    original_source_hash = hashlib.sha256(builder_source.encode("utf-8")).hexdigest()
    if args.radius_multiplier != 1.0 or args.stock_time or args.paper_offset:
        # Execute the released function body with only the named assignments
        # changed. Preserve the exact diagnostic source beside every result.
        tree = ast.parse(builder_source)
        changed = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            if not isinstance(target, ast.Name):
                continue
            if target.id == "R_width" and args.radius_multiplier != 1.0:
                node.value = ast.BinOp(left=node.value, op=ast.Mult(),
                                       right=ast.Constant(value=args.radius_multiplier))
                changed.append("R_width")
            if target.id == "time_angles" and args.stock_time:
                node.value = ast.BinOp(left=node.value, op=ast.Mult(), right=ast.Constant(value=0.0))
                changed.append("time_angles")
            if target.id in ("X_spherical", "Y_spherical") and args.paper_offset:
                node.value = ast.BinOp(left=node.value, op=ast.Add(),
                                       right=ast.Name(id="R_width", ctx=ast.Load()))
                changed.append(target.id)
        expected = int(args.radius_multiplier != 1.0) + int(args.stock_time) + 2 * int(args.paper_offset)
        if len(changed) != expected:
            raise RuntimeError(f"Unexpected upstream builder structure: {changed}")
        ast.fix_missing_locations(tree)
        variant_source = ast.unparse(tree)
        (args.output_dir / "diagnostic_builder.py").write_text(
            "# Derived from orhir/SpheRoPE erp_utils.py; CC BY-NC 4.0.\n" + variant_source,
            encoding="utf-8")
        scope = dict(builder.__globals__)
        exec(compile(tree, str(args.output_dir / "diagnostic_builder.py"), "exec"), scope)
        erp_module.build_sfc_spherical_rope = scope[builder.__name__]
    environment = {
        "reference_commit": subprocess.check_output(
            ["git", "-C", str(args.reference_repo), "rev-parse", "HEAD"], text=True).strip(),
        "python": sys.version, "torch": torch.__version__, "diffusers": diffusers.__version__,
        "transformers": transformers.__version__, "model_dir": str(args.model_dir.resolve()),
        "prompt": args.prompt, "seed": args.seed, "steps": args.steps,
        "width": args.width, "height": args.height,
        "embedded_guidance": 3.5, "real_cfg": args.real_cfg, "geometry_gamma": 6.0,
        "geometry_start_step": args.geometry_start,
        "radius_multiplier": args.radius_multiplier, "stock_time": args.stock_time,
        "paper_offset": args.paper_offset,
        "released_builder_sha256": original_source_hash,
        "modified_reference_behavior": (args.radius_multiplier != 1.0 or args.stock_time
                                        or args.paper_offset or args.real_cfg != 3.5
                                        or args.geometry_start != 2),
    }
    (args.output_dir / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    if args.probe_only:
        original = builder(args.height // 16, args.width // 16)
        actual = erp_module.build_sfc_spherical_rope(args.height // 16, args.width // 16)
        for before, after in zip(original, actual):
            assert torch.equal(before[..., 16:72], after[..., 16:72]), "Height axis changed"
            if not args.stock_time:
                assert torch.equal(before[..., :16], after[..., :16]), "Time axis changed"
            if args.radius_multiplier == 1.0 and not args.paper_offset:
                assert torch.equal(before[..., 72:], after[..., 72:]), "Width axis changed"
        if args.stock_time:
            assert torch.equal(actual[0][..., :16], torch.ones_like(actual[0][..., :16]))
            assert torch.count_nonzero(actual[1][..., :16]) == 0
        if args.paper_offset:
            # Independent check at the pole: low-frequency coordinates must
            # converge to +R (paper eq. 2), not zero. This also verifies that
            # offset injection uses the selected radius and leaves harmonics.
            width_tokens = args.width // 16
            freqs = 10000.0 ** (-2 * torch.arange(28, dtype=torch.float64) / 56)
            k = freqs / (2 * math.pi / width_tokens)
            valid = (k >= 1) & ((k-k.round()).abs() / (k+1e-8) <= 0.06)
            invalid = (~valid).nonzero().flatten()
            split = int(invalid[0]) if invalid.numel() else 28
            radius = width_tokens / (2 * math.pi) * args.radius_multiplier
            expected_angle = radius * freqs[split:]
            assert torch.allclose(actual[0][0, 0, 72+split*2::2],
                                  expected_angle.cos().float(), atol=1e-6)
            assert torch.allclose(actual[1][0, 0, 72+split*2::2],
                                  expected_angle.sin().float(), atol=1e-6)
            for before, after in zip(original, actual):
                assert torch.equal(before[..., 72:72+split*2], after[..., 72:72+split*2])
        probe = {"axis_isolation_passed": True,
                 "time_max_change": max(float((a[..., :16]-b[..., :16]).abs().max())
                                        for a, b in zip(original, actual)),
                 "width_max_change": max(float((a[..., 72:]-b[..., 72:]).abs().max())
                                         for a, b in zip(original, actual))}
        (args.output_dir / "builder_probe.json").write_text(json.dumps(probe, indent=2), encoding="utf-8")
        print(json.dumps(probe))
        return
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
            "no_cfg_until_timestep": args.geometry_start,
        }
        def capture_latent(pipeline, step_index, timestep, callback_kwargs):
            if step_index == args.steps - 1:
                torch.save(callback_kwargs["latents"].detach().cpu(),
                           args.output_dir / f"{case}_packed_latents.pt")
            return callback_kwargs

        extra = {"callback_on_step_end": capture_latent} if args.save_latents else {}
        image = pipe(
            prompt=args.prompt, height=args.height, width=args.width,
            num_inference_steps=args.steps, guidance_scale=3.5,
            guidance_scale_real=args.real_cfg,
            generator=torch.Generator("cpu").manual_seed(args.seed), **settings, **extra).images[0]
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
