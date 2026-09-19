"""Tests for Ripple guide seam masks and per-step mask synchronization."""

import pathlib
import sys

import torch


repo_dir = pathlib.Path(__file__).resolve().parents[1]
comfy_dir = repo_dir.parents[1]
for directory in (str(comfy_dir), str(repo_dir)):
    if directory not in sys.path:
        sys.path.insert(0, directory)

from seamless_equirect import nodes
from seamless_equirect import ripple


def test_guide_mask_shape_edges_interior_and_symmetry():
    mask, = ripple.EquirectGuideAttentionMask().create(
        width=16,
        height=4,
        frames=3,
        seam_width=8,
        edge_strength=0.0,
    )

    assert mask.shape == (3, 4, 16)
    assert torch.all(mask[:, :, 0] == 0)
    assert torch.all(mask[:, :, -1] == 0)
    assert torch.all(mask[:, :, 4:12] == 1)
    assert torch.allclose(mask, torch.flip(mask, dims=[-1]))
    assert torch.all(mask[0, 0, 1:5].diff() > 0)


def test_zero_width_is_full_strength():
    mask, = ripple.EquirectGuideAttentionMask().create(
        width=8,
        height=2,
        frames=1,
        seam_width=0,
        edge_strength=0.25,
    )
    assert torch.all(mask == 1)


def test_conditioning_mask_roll_is_proportional_and_immutable():
    mask = torch.arange(8, dtype=torch.float32).view(1, 1, 1, 1, 8)
    entry = {
        "pre_filter_count": 8,
        "strength": 1.0,
        "pixel_mask": mask,
        "latent_shape": [1, 1, 4],
    }
    conditioning = {"guide_attention_entries": [entry], "other": object()}

    result = ripple.roll_ltx_guide_attention_masks(
        conditioning,
        latent_shift=2,
        latent_width=4,
    )

    assert result is not conditioning
    assert result["guide_attention_entries"] is not conditioning["guide_attention_entries"]
    assert result["guide_attention_entries"][0] is not entry
    assert result["other"] is conditioning["other"]
    assert torch.equal(
        result["guide_attention_entries"][0]["pixel_mask"],
        torch.roll(mask, shifts=4, dims=-1),
    )
    assert torch.equal(entry["pixel_mask"], mask)


def test_conditioning_mask_roll_noops_without_work():
    plain = {"guide_attention_entries": [{"pixel_mask": None}]}
    assert ripple.roll_ltx_guide_attention_masks(plain, 1, 4) is plain
    assert ripple.roll_ltx_guide_attention_masks(plain, 0, 4) is plain
    assert ripple.roll_ltx_guide_attention_masks({}, 1, 4) == {}


def test_node_is_registered():
    assert (
        nodes.NODE_CLASS_MAPPINGS["EquirectGuideAttentionMask"]
        is ripple.EquirectGuideAttentionMask
    )
    assert "Guide Attention Mask" in nodes.NODE_DISPLAY_NAME_MAPPINGS[
        "EquirectGuideAttentionMask"
    ]


if __name__ == "__main__":
    test_guide_mask_shape_edges_interior_and_symmetry()
    test_zero_width_is_full_strength()
    test_conditioning_mask_roll_is_proportional_and_immutable()
    test_conditioning_mask_roll_noops_without_work()
    test_node_is_registered()
    print("ok: synchronized Ripple guide mask tests")
