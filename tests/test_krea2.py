"""Tests for Krea2 seamless equirectangular support."""

import importlib
import pathlib
import sys

import torch


repo_dir = pathlib.Path(__file__).resolve().parents[1]
comfy_dir = repo_dir.parents[1]
for path in (str(comfy_dir), str(repo_dir)):
    if path not in sys.path:
        sys.path.insert(0, path)

k2 = importlib.import_module("seamless_equirect.krea2")
sn = importlib.import_module("seamless_equirect.nodes")
cc = importlib.import_module("seamless_equirect.circular_conv")

from comfy.ldm.flux.layers import EmbedND
from comfy.ldm.flux.math import rope as flux_rope
from comfy.ldm.wan.vae import CausalConv3d


torch.manual_seed(7)


def test_krea2_periodic_width_rope():
    pos = torch.tensor([[0.0, 8.0]])
    periods = torch.tensor([[8.0]])
    mask = torch.ones_like(pos, dtype=torch.bool)

    stock = flux_rope(pos, 48, 1000)
    disabled = k2.periodic_width_rope(pos, 48, 1000, periods, mask, blend=0.0)
    assert torch.equal(disabled, stock)

    periodic = k2.periodic_width_rope(pos, 48, 1000, periods, mask, blend=1.0)
    assert periodic.shape == stock.shape
    assert torch.allclose(periodic[:, 0], periodic[:, 1], atol=1e-5)


def test_krea2_spherical_width_rope_closes_seam_and_converges_at_poles():
    widths = torch.tensor([[8.0]])
    heights = torch.tensor([[3.0]])
    mask = torch.ones(1, 2, dtype=torch.bool)

    seam = k2.spherical_width_rope(
        torch.tensor([[0.0, 8.0]]),
        torch.tensor([[1.0, 1.0]]),
        48,
        1000,
        widths,
        heights,
        mask,
        seam_blend=1.0,
        geometry_blend=1.0,
        low_frequency_fraction=1.0,
        sphere_scale=1.0,
    )
    assert torch.allclose(seam[:, 0], seam[:, 1], atol=1e-5)

    north_pole = k2.spherical_width_rope(
        torch.tensor([[0.0, 4.0]]),
        torch.tensor([[0.0, 0.0]]),
        48,
        1000,
        widths,
        heights,
        mask,
        seam_blend=1.0,
        geometry_blend=1.0,
        low_frequency_fraction=1.0,
        sphere_scale=1.0,
    )
    equator = k2.spherical_width_rope(
        torch.tensor([[0.0, 4.0]]),
        torch.tensor([[1.0, 1.0]]),
        48,
        1000,
        widths,
        heights,
        mask,
        seam_blend=1.0,
        geometry_blend=1.0,
        low_frequency_fraction=1.0,
        sphere_scale=1.0,
    )
    assert torch.allclose(north_pole[:, 0], north_pole[:, 1], atol=1e-5)
    assert not torch.allclose(equator[:, 0], equator[:, 1])


def test_krea2_embedder_only_patches_target_width():
    text_ids = torch.zeros(1, 3, 3)

    target_ids = torch.zeros(1, 2, 8, 3)
    target_ids[..., 1] = torch.arange(2).view(1, 2, 1)
    target_ids[..., 2] = torch.arange(8).view(1, 1, 8)
    target_ids = target_ids.reshape(1, 16, 3)

    ref_ids = torch.zeros(1, 1, 4, 3)
    ref_ids[..., 0] = 1
    ref_ids[..., 2] = torch.arange(4).view(1, 1, 4)
    ref_ids = ref_ids.reshape(1, 4, 3)

    ids = torch.cat((text_ids, target_ids, ref_ids), dim=1)
    stock = EmbedND(dim=128, theta=1000, axes_dim=[32, 48, 48])(ids)
    periodic = k2.Krea2PeriodicEmbedND(128, 1000, [32, 48, 48], blend=1.0)(ids)

    assert periodic.shape == stock.shape
    width_start = (32 + 48) // 2
    target_start = text_ids.shape[1]
    ref_start = target_start + target_ids.shape[1]

    assert torch.equal(periodic[..., :width_start, :, :], stock[..., :width_start, :, :])
    assert not torch.allclose(
        periodic[:, :, target_start + 1, width_start:],
        stock[:, :, target_start + 1, width_start:],
    )
    assert torch.allclose(
        periodic[:, :, ref_start:, width_start:],
        stock[:, :, ref_start:, width_start:],
        atol=1e-6,
    )


def test_krea2_sphe_embedder_only_patches_main_image():
    text_ids = torch.zeros(1, 3, 3)

    target_ids = torch.zeros(1, 3, 8, 3)
    target_ids[..., 1] = torch.arange(3).view(1, 3, 1)
    target_ids[..., 2] = torch.arange(8).view(1, 1, 8)
    target_ids = target_ids.reshape(1, 24, 3)

    ref_ids = torch.zeros(1, 2, 4, 3)
    ref_ids[..., 0] = 1
    ref_ids[..., 1] = torch.arange(2).view(1, 2, 1)
    ref_ids[..., 2] = torch.arange(4).view(1, 1, 4)
    ref_ids = ref_ids.reshape(1, 8, 3)

    ids = torch.cat((text_ids, target_ids, ref_ids), dim=1)
    mask, heights, widths = k2._target_grid_info(ids)
    target_start = text_ids.shape[1]
    ref_start = target_start + target_ids.shape[1]

    assert heights.item() == 3
    assert widths.item() == 8
    assert not mask[:, :target_start].any()
    assert mask[:, target_start:ref_start].all()
    assert not mask[:, ref_start:].any()

    stock = EmbedND(dim=128, theta=1000, axes_dim=[32, 48, 48])(ids)
    spherical = k2.Krea2SpheEmbedND(
        128,
        1000,
        [32, 48, 48],
        blend=0.5,
        geometry_blend=0.5,
        low_frequency_fraction=0.5,
        sphere_scale=1.0,
    )(ids)

    assert spherical.shape == stock.shape
    assert torch.equal(spherical[:, :, :target_start], stock[:, :, :target_start])
    assert not torch.allclose(
        spherical[:, :, target_start + 1:ref_start],
        stock[:, :, target_start + 1:ref_start],
    )
    assert torch.allclose(spherical[:, :, ref_start:], stock[:, :, ref_start:], atol=1e-6)


class FakeKrea2Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.is_krea2 = True
        self.patch = 2
        self.pe_embedder = EmbedND(dim=128, theta=1000, axes_dim=[32, 48, 48])


class FakePatcher:
    def __init__(self, diffusion_model):
        self.diffusion_model = diffusion_model
        self.object_patches = {}
        self.wrapper = None

    def clone(self):
        out = FakePatcher(self.diffusion_model)
        out.object_patches = dict(self.object_patches)
        out.wrapper = self.wrapper
        return out

    def get_model_object(self, name):
        if name == "diffusion_model":
            return self.diffusion_model
        return self.object_patches.get(name)

    def add_object_patch(self, name, obj):
        self.object_patches[name] = obj

    def set_model_unet_function_wrapper(self, wrapper):
        self.wrapper = wrapper


def test_krea2_rope_node_lifecycle():
    patcher = FakePatcher(FakeKrea2Model())

    disabled, = k2.EquirectKrea2RoPE().patch(patcher, enabled=False, blend=0.5)
    assert disabled is patcher

    patched, = k2.EquirectKrea2RoPE().patch(patcher, enabled=True, blend=0.5)
    embedder = patched.object_patches["diffusion_model.pe_embedder"]
    assert isinstance(embedder, k2.Krea2SpheEmbedND)
    assert embedder.blend == 0.5
    assert embedder.geometry_blend == 0.4
    assert embedder.low_frequency_fraction == 0.5
    assert embedder.sphere_scale == 1.0

    non_krea = FakePatcher(torch.nn.Module())
    try:
        k2.EquirectKrea2RoPE().patch(non_krea, enabled=True, blend=0.5)
        assert False, "non-Krea2 model should be rejected"
    except ValueError as exc:
        assert "Krea2" in str(exc)


def test_krea2_per_step_roll_is_patch_aligned():
    width = 16
    seed = 11
    sigma = next(
        value / 100.0
        for value in range(1, 100)
        if sn.shift_for_sigma(value / 100.0, width, seed, "random_per_step") % 2 == 1
    )

    patcher = FakePatcher(FakeKrea2Model())
    wrapped, = sn.EquirectPerStepRoll().patch(
        patcher, enabled=True, mode="random_per_step", seed=seed
    )
    latent = torch.randn(1, 16, 8, width)
    seen = {}

    def identity(x, timestep, **conditioning):
        seen["input"] = x.clone()
        return x

    out = wrapped.wrapper(identity, {
        "input": latent,
        "timestep": torch.tensor([sigma]),
        "c": {},
    })
    shift = sn.shifts_for_sigma(sigma, width, seed, "random_per_step", multiple=2)[0]

    assert shift % 2 == 0
    assert torch.equal(seen["input"], torch.roll(latent, shifts=shift, dims=-1))
    assert torch.equal(out, latent)


def test_qwen_style_vae_circular_decoder():
    class FakeQwenImageVAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Sequential(CausalConv3d(2, 2, 3, padding=1))
            self.decoder = torch.nn.Sequential(CausalConv3d(2, 2, 3, padding=1))

    class FakeVAEWrapper:
        def __init__(self, model):
            self.first_stage_model = model

    model = FakeQwenImageVAE()
    for conv in (model.encoder[0], model.decoder[0]):
        torch.nn.init.normal_(conv.weight)
        torch.nn.init.zeros_(conv.bias)

    vae = FakeVAEWrapper(model)
    result, = sn.EquirectCircularVAE().patch(
        vae, patch_encoder=False, patch_decoder=True, enabled=True
    )
    assert result is vae
    assert not getattr(model.encoder[0], cc._SENTINEL, False)
    assert getattr(model.decoder[0], cc._SENTINEL, False)

    x = torch.randn(1, 2, 1, 6, 10)
    decoded = model.decoder(x)
    shifted = model.decoder(torch.roll(x, shifts=2, dims=-1))
    assert decoded.shape == x.shape
    assert torch.allclose(shifted, torch.roll(decoded, shifts=2, dims=-1), atol=1e-5)


def test_krea2_4d_noise_and_blend_nodes():
    class DummyNoise:
        def generate_noise(self, latent):
            return torch.randn_like(latent["samples"])

    latent = {"samples": torch.randn(1, 16, 8, 16)}
    wrapped, = sn.EquirectWrappedNoise().wrap(DummyNoise(), enabled=True, radius=4)
    noise = wrapped.generate_noise(latent)
    assert noise.shape == latent["samples"].shape

    blended, = sn.EquirectLatentSeamBlend().blend(latent, blend_columns=2)
    samples = blended["samples"]
    assert samples.shape == latent["samples"].shape
    assert torch.allclose(samples[..., 0], samples[..., -1], atol=1e-6)
    assert torch.equal(samples[..., 3:-3], latent["samples"][..., 3:-3])


def test_krea2_node_registration():
    assert "EquirectKrea2RoPE" in sn.NODE_CLASS_MAPPINGS
    assert "EquirectKrea2RoPE" in sn.NODE_DISPLAY_NAME_MAPPINGS
