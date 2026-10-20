"""Check native FLUX spherical embedding integration against installed Diffusers."""
import argparse
import json
import math
import os
from pathlib import Path
import sys

os.environ.update(USE_TF="0", USE_FLAX="0", HF_HUB_OFFLINE="1")

import torch
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.transformers.transformer_flux import FluxPosEmbed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-repo", type=Path, required=True)
    parser.add_argument("--variant-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.reference_repo))
    from panorama_diffusers.erp_utils import build_sfc_spherical_rope, SFCFluxPosEmbed

    scope = {"torch": torch, "math": math}
    exec(compile(args.variant_source.read_text(encoding="utf-8"),
                 str(args.variant_source), "exec"), scope)
    variant = scope["build_sfc_spherical_rope"]
    height, width, text_length = 64, 128, 512
    ids = torch.zeros(height, width, 3)
    ids[..., 1] = torch.arange(height)[:, None]
    ids[..., 2] = torch.arange(width)[None, :]
    ids = torch.cat((torch.zeros(text_length, 3), ids.reshape(-1, 3)))
    stock = FluxPosEmbed(theta=10000, axes_dim=[16, 56, 56])
    stock_table = stock(ids)
    results = []
    for name, builder in (("released", build_sfc_spherical_rope), ("paper_coordinates", variant)):
        table = SFCFluxPosEmbed(*builder(height, width), stock)(ids)
        text_error = max(float((a[:text_length] - b[:text_length]).abs().max())
                         for a, b in zip(table, stock_table))
        time_error = max(float((a[text_length:, :16] - b[text_length:, :16]).abs().max())
                         for a, b in zip(table, stock_table))
        height_error = max(float((a[text_length:, 16:72] - b[text_length:, 16:72]).abs().max())
                           for a, b in zip(table, stock_table))
        assert text_error == 0
        assert height_error < 1e-5
        if name == "paper_coordinates":
            assert time_error == 0
        rotations = {}
        for dtype in (torch.float32, torch.bfloat16):
            q = torch.randn(1, 1, ids.shape[0], 128, generator=torch.Generator().manual_seed(17)).to(dtype)
            a = apply_rotary_emb(q, table)
            b = apply_rotary_emb(q, stock_table)
            error = float((a[:, :, text_length:, :72] - b[:, :, text_length:, :72]).abs().max())
            rotations[str(dtype)] = error
            if name == "paper_coordinates":
                torch.testing.assert_close(a[..., :text_length, :], b[..., :text_length, :])
                torch.testing.assert_close(a[:, :, text_length:, :72], b[:, :, text_length:, :72],
                                           atol=1e-5, rtol=1e-5)
        results.append({"builder": name, "text_table_max_error": text_error,
                        "image_time_table_max_error": time_error,
                        "image_height_table_max_error": height_error,
                        "rotated_image_time_height_max_error": rotations})
    report = {"grid": [height, width], "text_tokens": text_length,
              "scope": "CPU stock embedding, native wrapper and installed Diffusers rotary application; not GPU attention or generation parity",
              "results": results}
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
