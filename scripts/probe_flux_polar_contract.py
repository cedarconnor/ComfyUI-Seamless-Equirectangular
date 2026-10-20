"""Measure pole convergence of the released hybrid RoPE, without model weights.

This tests positional embeddings, not generated pixels. The paper explicitly
allows its harmonic subspace to violate polar convergence (section 3.2).
"""
import argparse
import importlib.util
import json
from pathlib import Path

import torch


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference = load_file(
        "reference_erp_utils", args.reference_repo / "panorama_diffusers/erp_utils.py")
    local = load_file(
        "local_flux2_angles",
        Path(__file__).resolve().parents[1] / "seamless_equirect/flux2_angles.py")
    results = []
    for height, width in ((32, 64), (64, 128), (128, 256)):
        for name, builder, start, end in (
            ("native_flux1", reference.build_sfc_spherical_rope, 72, 128),
            ("native_flux2", reference.build_sfc_spherical_rope_flux2, 64, 96),
        ):
            cos, sin = builder(height, width)
            # Each rotary pair is duplicated in the native output. Retain one.
            c = cos[0, :, start:end:2].reshape(height, width, -1)
            s = sin[0, :, start:end:2].reshape(height, width, -1)
            # Compare every longitude with column zero at the same pole.
            pole_distance = torch.sqrt(
                (c[[0, -1]] - c[[0, -1], :1]).square()
                + (s[[0, -1]] - s[[0, -1], :1]).square())
            per_pair = pole_distance.amax(dim=(0, 1))
            nonconverging = (per_pair > 1e-5).nonzero().flatten().tolist()
            # A unit complex rotation's chord distance cannot exceed 2.
            assert float(per_pair.max()) <= 2.00001
            assert nonconverging, "Recheck this result if upstream changes"
            row = {
                "builder": name, "token_grid": [height, width],
                "width_rotation_pairs": c.shape[-1],
                "nonconverging_pair_indices": nonconverging,
                "max_pole_rotation_chord_distance": float(per_pair.max()),
                "converging_pairs_max_distance": float(
                    per_pair[per_pair <= 1e-5].max()),
            }
            if name == "native_flux2":
                angles = local.build_flux2_width_angles(height, width)
                errors = [
                    (angles.cos().float() - c.reshape(-1, c.shape[-1])).abs().max(),
                    (angles.sin().float() - s.reshape(-1, s.shape[-1])).abs().max(),
                ]
                row["comfy_port_width_max_error"] = float(max(errors))
                assert row["comfy_port_width_max_error"] == 0
            results.append(row)
    report = {
        "scope": "CPU positional tables only; does not prove the cause of rendered pole artifacts",
        "interpretation": (
            "The spherical subspace converges at each pole; the harmonic "
            "subspace does not. This is an acknowledged paper tradeoff, not "
            "a ComfyUI port discrepancy or a full-sphere guarantee."),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
