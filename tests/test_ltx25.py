"""Tests for the LTX 2.5 dual-rotation seamless VAE decoder."""

import pathlib
import sys

import torch


repo_dir = pathlib.Path(__file__).resolve().parents[1]
comfy_dir = repo_dir.parents[1]
for directory in (str(comfy_dir), str(repo_dir)):
    if directory not in sys.path:
        sys.path.insert(0, directory)

from seamless_equirect import ltx25
from seamless_equirect import nodes


class CausalDiffusionVAE:
    pass


class FakeVAE:
    """Nearest-neighbor decoder with a deterministic boundary artifact."""

    def __init__(self):
        self.first_stage_model = CausalDiffusionVAE()
        self.calls = 0

    def decode(self, latent):
        self.calls += 1
        # Native LTX 2.5 returns [B,T,H,W,C]; the node must flatten B*T.
        image = latent[:, :1].permute(0, 2, 3, 4, 1)
        image = image.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)
        image = image.repeat(1, 1, 1, 1, 3).clone()
        image[:, :, :, 0] = -100
        image[:, :, :, -1] = 100
        return image


def test_dual_decode_uses_shifted_interior_at_boundary():
    latent = torch.arange(8, dtype=torch.float32).view(1, 1, 1, 1, 8)
    vae = FakeVAE()

    result, = ltx25.EquirectLTX25VAEDecode().decode(
        {"samples": latent}, vae, blend_columns=1
    )

    assert vae.calls == 2
    assert result.shape == (1, 2, 16, 3)
    assert torch.all(result[:, :, 0] == 0)
    assert torch.all(result[:, :, -1] == 7)
    assert torch.all(result[:, :, 8] == 4)


def test_zero_blend_uses_exact_stock_decode():
    latent = torch.arange(8, dtype=torch.float32).view(1, 1, 1, 1, 8)
    vae = FakeVAE()

    result, = ltx25.EquirectLTX25VAEDecode().decode(
        {"samples": latent}, vae, blend_columns=0
    )

    assert vae.calls == 1
    assert torch.all(result[:, :, 0] == -100)
    assert torch.all(result[:, :, -1] == 100)


def test_rejects_wrong_vae_and_latent_shape():
    class WrongVAE(FakeVAE):
        def __init__(self):
            self.first_stage_model = object()
            self.calls = 0

    latent = torch.zeros(1, 1, 1, 1, 8)
    try:
        ltx25.EquirectLTX25VAEDecode().decode(
            {"samples": latent}, WrongVAE(), blend_columns=1
        )
        assert False, "wrong VAE should be rejected"
    except ValueError as error:
        assert "CausalDiffusionVAE" in str(error)

    try:
        ltx25.EquirectLTX25VAEDecode().decode(
            {"samples": torch.zeros(1, 1, 8)}, FakeVAE(), blend_columns=1
        )
        assert False, "non-5D latent should be rejected"
    except ValueError as error:
        assert "5D" in str(error)


def test_node_is_registered():
    assert nodes.NODE_CLASS_MAPPINGS["EquirectLTX25VAEDecode"] is ltx25.EquirectLTX25VAEDecode
    assert "LTX 2.5" in nodes.NODE_DISPLAY_NAME_MAPPINGS["EquirectLTX25VAEDecode"]


if __name__ == "__main__":
    test_dual_decode_uses_shifted_interior_at_boundary()
    test_zero_blend_uses_exact_stock_decode()
    test_rejects_wrong_vae_and_latent_shape()
    test_node_is_registered()
    print("ok: LTX 2.5 seamless decode tests")
