"""Circular upscale/decode contracts, including Comfy's managed upscaler."""
import pytest
import torch

pytest.importorskip("comfy.model_management")

from seamless_equirect.erp_nodes import (  # noqa: E402
    EquirectCircularUpscale, EquirectCircularVAEDecodeLTX, EquirectCircularVAEDecodeFlux,
)


class _FakeUpscaleModel(torch.nn.Module):
    """LATENT_UPSCALE_MODEL stand-in: an nn.Module CALLED DIRECTLY (no .upsample).
    Doubles the two trailing spatial dims (H, W) of a (B,C,F,H,W) latent."""
    def __init__(self):
        super().__init__()
        self.marker = torch.nn.Parameter(torch.zeros(1))  # gives .parameters()/.dtype/.to()
        self.called = False

    def forward(self, x):
        self.called = True
        return x.repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2)


class _FakeUpscalePatcher:
    """Current loader returns a ModelPatcher, not an nn.Module."""
    def __init__(self):
        self.model = _FakeUpscaleModel()
        self.load_device = torch.device("cpu")

    def model_dtype(self):
        return torch.float32


@pytest.fixture
def managed_loads(monkeypatch):
    from comfy import model_management
    calls = []
    monkeypatch.setattr(model_management, "load_models_gpu", lambda models, memory_required: calls.append((models, memory_required)))
    monkeypatch.setattr(model_management, "intermediate_device", lambda: torch.device("cpu"))
    return calls


class _FakeStats:
    def __init__(self):
        self.un_norm_calls = 0
        self.norm_calls = 0

    def un_normalize(self, x):
        self.un_norm_calls += 1
        return x

    def normalize(self, x):
        self.norm_calls += 1
        return x


class _FakeInner:
    def __init__(self):
        self.per_channel_statistics = _FakeStats()


class _FakeVAE:
    def __init__(self):
        self.first_stage_model = _FakeInner()

    def decode(self, latent):
        # LTX video VAE returns 5-D (B, F, H, W, C); width = latent_width * 32.
        b, c, f, h, w = latent.shape
        return torch.zeros(b, f, h * 32, w * 32, 3)


def test_upscale_uses_managed_model_and_vae_normalization(managed_loads):
    node = EquirectCircularUpscale()
    model = _FakeUpscalePatcher()
    vae = _FakeVAE()
    lat = torch.randn(1, 4, 2, 8, 16)          # (B,C,F,H,W), W=16
    (out,) = node.upscale({"samples": lat}, model, vae, pad_w=4)
    up = out["samples"]
    assert model.model.called
    assert managed_loads == [([model], 1 * 4 * 2 * 8 * 24 * 3000.0)]
    assert up.shape[-1] == 32 and up.shape[-2] == 16   # W doubled+cropped, H doubled
    assert vae.first_stage_model.per_channel_statistics.un_norm_calls == 1
    assert vae.first_stage_model.per_channel_statistics.norm_calls == 1
    torch.testing.assert_close(up, lat.repeat_interleave(2, -1).repeat_interleave(2, -2))


def test_upscale_model_has_no_upsample_method():
    # Guards the original bug: the real LATENT_UPSCALE_MODEL has no .upsample().
    assert not hasattr(_FakeUpscaleModel(), "upsample")


def test_upscale_preserves_noise_mask_for_conditioning(managed_loads):
    # Two-stage img2video: the conditioning noise_mask (B,1,T,1,1) must survive the
    # spatial upscale so stage-2 keeps the pinned first frame.
    node = EquirectCircularUpscale()
    lat = torch.randn(1, 4, 2, 8, 16)               # (B,C,T,H,W)
    mask = torch.ones(1, 1, 2, 1, 1)
    mask[:, :, 0] = 0.0                             # frame 0 fully pinned
    (out,) = node.upscale({"samples": lat, "noise_mask": mask},
                          _FakeUpscalePatcher(), _FakeVAE(), pad_w=4)
    assert "noise_mask" in out
    assert torch.equal(out["noise_mask"], mask)     # unchanged (T is spatial-agnostic)


def test_circular_vae_decode_flattens_5d_and_crops_width():
    node = EquirectCircularVAEDecodeLTX()
    vae = _FakeVAE()
    lat = torch.randn(1, 4, 2, 8, 16)          # decodes to (1, 2, 8*32, 16*32, 3)
    (imgs,) = node.decode({"samples": lat}, vae, pad_w=4, target_width=16 * 32)
    # 5-D (B,F,H,W,C) must be flattened to 4-D (B*F, H, W, C) like core VAEDecode.
    assert imgs.dim() == 4
    assert imgs.shape == (2, 8 * 32, 16 * 32, 3)


class _FakeFluxVAE:
    """Flux image VAE stand-in: decodes a 4-D (B,C,H,W) latent to (B,H,W,C) at 8x."""
    def decode(self, latent):
        b, c, h, w = latent.shape
        return torch.zeros(b, h * 8, w * 8, 3)


def test_flux_circular_vae_decode_crops_width_via_ratio():
    node = EquirectCircularVAEDecodeFlux()
    vae = _FakeFluxVAE()
    lat = torch.randn(1, 16, 32, 64)           # Flux latent (B,C,H,W), W=64
    (imgs,) = node.decode({"samples": lat}, vae, pad_w=8)
    # pad 8 latent cols -> 80; decode 8x -> 640; crop 8*8=64 each side -> 512 (=64*8).
    assert imgs.dim() == 4
    assert imgs.shape == (1, 32 * 8, 64 * 8, 3)


def test_flux_circular_vae_decode_pad0_is_plain_decode():
    node = EquirectCircularVAEDecodeFlux()
    lat = torch.randn(1, 16, 32, 64)
    (imgs,) = node.decode({"samples": lat}, _FakeFluxVAE(), pad_w=0)
    assert imgs.shape == (1, 32 * 8, 64 * 8, 3)
