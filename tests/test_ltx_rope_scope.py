"""Exercise cached model branches with ComfyUI's actual object-patch resolver."""
import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))
from comfy.ldm.lightricks import model as ltx
from comfy.model_patcher import ModelPatcher
from seamless_equirect import equirope, nodes


class TinyLTX(torch.nn.Module):
    _precompute_freqs_cis = ltx.LTXBaseModel._precompute_freqs_cis
    freq_grid_generator = staticmethod(ltx.generate_freq_grid_pytorch)
    split_positional_embedding = ltx.LTXRopeType.SPLIT
    vae_scale_factors = (8, 32, 32)


@pytest.mark.parametrize("node_type", [equirope.EquiRoPE, nodes.EquirectSphericalRoPE])
@pytest.mark.parametrize("rope_type", [ltx.LTXRopeType.SPLIT, ltx.LTXRopeType.INTERLEAVED])
def test_cached_disabled_model_and_other_patch_instances_stay_independent(node_type, rope_type):
    base = torch.nn.Module()
    base.diffusion_model = TinyLTX()
    base.diffusion_model.split_positional_embedding = rope_type
    model = ModelPatcher(base, torch.device("cpu"), torch.device("cpu"))
    t, h, w = torch.meshgrid(torch.arange(2), torch.arange(4), torch.arange(8), indexing="ij")
    grid = torch.stack((t * 8, h * 32, w * 32)).reshape(1, 3, -1).float()
    key = "diffusion_model._precompute_freqs_cis"

    def phases(patcher):
        return patcher.get_model_object(key)(grid, 48, torch.float32, num_attention_heads=2)

    original_generate = ltx.generate_freqs
    stock = phases(model)
    node = node_type()
    disabled = node.patch(model, False, 1)[0]
    full = node.patch(model, True, 1)[0]
    full_phases = phases(full)
    try:
        assert ltx.generate_freqs is original_generate, "Node modified process-wide frequency generation"
        assert not torch.equal(full_phases[0], stock[0])
        half = node.patch(model, True, .5)[0]
        assert not torch.equal(phases(half)[0], full_phases[0])
        # Reusing disabled/full outputs models a ComfyUI cache hit (no node execution).
        for actual, expected in zip(phases(disabled), stock):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for actual, expected in zip(phases(full), full_phases):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        audio = grid[:, :1]
        for actual, expected in zip(
            full.get_model_object(key)(audio, 48, torch.float32, max_pos=(20,), num_attention_heads=2),
            model.get_model_object(key)(audio, 48, torch.float32, max_pos=(20,), num_attention_heads=2),
        ):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        full.patch_model(load_weights=False)
        for actual, expected in zip(base.diffusion_model._precompute_freqs_cis(
            grid, 48, torch.float32, num_attention_heads=2
        ), full_phases):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        full.unpatch_model(unpatch_weights=False)
        for actual, expected in zip(phases(model), stock):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert not model.object_patches
    finally:
        # Clean up old implementation on the initial red run.
        nodes._periodic_rope.uninstall(ltx)
        equirope.uninstall(ltx)
