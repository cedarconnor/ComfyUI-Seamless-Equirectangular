"""Compare actual native/Comfy rotary application without loading model weights.

Loads the two upstream function definitions verbatim through AST to avoid the
native pipeline's unrelated model-loading imports. Does not modify either repo.
This checks positional preprocessing and Q/K rotation, not whole-model parity.
"""
import argparse
import ast
import json
import logging
import math
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Callable

import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-repo", type=Path, required=True)
    p.add_argument("--ltx-repo", type=Path, required=True)
    p.add_argument("--comfy", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    sys.path[:0] = [str(args.comfy), str(Path(__file__).resolve().parents[1]),
                    str(args.ltx_repo / "packages/ltx-core/src")]
    from comfy.ldm.lightricks import model as comfy_rope
    from comfy.model_patcher import ModelPatcher
    from ltx_core.model.transformer import rope as native_rope
    from seamless_equirect.nodes import EquirectSphericalRoPE

    source = args.reference_repo / "panorama_diffusers/pipelines/pipeline_ltx23_erp.py"
    names = {"_build_ltx23_width_angles", "patch_native_ltx23_rope"}
    tree = ast.parse(source.read_text(encoding="utf-8"))
    subset = ast.Module(body=[x for x in tree.body if isinstance(x, ast.FunctionDef)
                             and x.name in names], type_ignores=[])
    scope = dict(torch=torch, math=math, Callable=Callable,
                 logger=logging.getLogger("upstream_rope"))
    exec(compile(subset, str(source), "exec"), scope)
    torch.manual_seed(80084)
    rows = []
    for h, w in ((8, 16), (16, 32), (32, 64)):
        for dtype in (torch.float32, torch.bfloat16):
            dim, heads = 4096, 32
            t, y, x = torch.meshgrid(torch.arange(2), torch.arange(h),
                                     torch.arange(w), indexing="ij")
            start = torch.stack((t / 3, y * 32, x * 32)).reshape(1, 3, -1).float()
            end = start + torch.tensor([1/3, 32, 32]).reshape(1, 3, 1)
            positions = torch.stack((start, end), -1)
            def original(pos, inner_dim, max_pos, middle, nheads, out_dtype):
                return native_rope.precompute_freqs_cis(
                    pos, inner_dim, out_dtype, max_pos=max_pos,
                    use_middle_indices_grid=middle, num_attention_heads=nheads,
                    freq_grid_generator=native_rope.generate_freq_grid_np)
            pre = SimpleNamespace(inner_dim=dim, positional_embedding_theta=10000.,
                                  double_precision_rope=True,
                                  rope_type=native_rope.LTXRopeType.SPLIT,
                                  _prepare_positional_embeddings=original)
            native = SimpleNamespace(velocity_model=SimpleNamespace(video_args_preprocessor=pre))
            scope["patch_native_ltx23_rope"](None, h, w, 10., r_scale=1.)(native)
            native_pe = pre._prepare_positional_embeddings(
                positions, dim, [20, 2048, 2048], True, heads, dtype)

            class Tiny(torch.nn.Module):
                _precompute_freqs_cis = comfy_rope.LTXBaseModel._precompute_freqs_cis
                freq_grid_generator = staticmethod(comfy_rope.generate_freq_grid_np)
                split_positional_embedding = comfy_rope.LTXRopeType.SPLIT
                vae_scale_factors = (8, 32, 32)
            model = torch.nn.Module()
            model.diffusion_model = Tiny()
            patcher = ModelPatcher(model, torch.device("cpu"), torch.device("cpu"))
            patched = EquirectSphericalRoPE().patch(patcher, True, 1.)[0]
            fn = patched.get_model_object("diffusion_model._precompute_freqs_cis")
            comfy_pe = fn(positions, dim, dtype, use_middle_indices_grid=True,
                          num_attention_heads=heads)
            matrix = comfy_pe[0]
            cos = matrix[..., 0, 0].transpose(1, 2)
            sin = matrix[..., 1, 0].transpose(1, 2)
            q = torch.randn(1, start.shape[-1], dim, dtype=dtype)
            k = torch.randn_like(q)
            nq = native_rope.apply_rotary_emb(q, native_pe)
            nk = native_rope.apply_rotary_emb(k, native_pe)
            cq, ck = comfy_rope.apply_rotary_emb_qk(q, k, comfy_pe)
            rows.append({"h": h, "w": w, "frames": 2, "dtype": str(dtype),
                         "cos_max_abs": float((cos-native_pe[0]).abs().max()),
                         "sin_max_abs": float((sin-native_pe[1]).abs().max()),
                         "q_max_abs": float((cq-nq).abs().max()),
                         "k_max_abs": float((ck-nk).abs().max()),
                         "q_relative_rms": float((cq.float()-nq.float()).square().mean().sqrt()
                                                / nq.float().square().mean().sqrt())})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
