import pytest
import torch
from seamless_equirect.circular_video import (
    circular_pad_video_width,
    circular_crop_video_width,
    circular_upsample_width,
    crop_decoded_frame_width,
    circular_pad_frame_width,
    circular_encode_width,
    assert_width_dims_match,
)


def test_assert_width_dims_match_ok():
    assert_width_dims_match(got=2048, want=2048)          # no raise


def test_assert_width_dims_match_raises():
    with pytest.raises(ValueError) as e:
        assert_width_dims_match(got=2040, want=2048)
    assert "2040" in str(e.value) and "2048" in str(e.value)


def _fake_2x_upsampler(t):
    # doubles the last two spatial dims (H,W) via nearest-neighbour repeat
    return t.repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2)


def test_circular_upsample_restores_2x_width():
    x = torch.randn(1, 4, 2, 8, 16)          # W = 16
    up = circular_upsample_width(x, _fake_2x_upsampler, pad_w=4)
    assert up.shape[-1] == 32                 # 2 * 16, pad removed
    assert up.shape[-2] == 16                 # H doubled, untouched by crop


def test_circular_upsample_zero_pad_matches_plain():
    x = torch.randn(1, 4, 1, 4, 8)
    assert torch.equal(
        circular_upsample_width(x, _fake_2x_upsampler, pad_w=0),
        _fake_2x_upsampler(x),
    )


def test_crop_decoded_removes_width_pixels_dim_minus_2():
    frames = torch.randn(5, 64, 64 + 8, 3)   # (F,H,W,C); W padded by 8 px
    out = crop_decoded_frame_width(frames, pixel_pad=4)
    assert out.shape == (5, 64, 64, 3)        # width (dim -2) back to 64


def test_crop_decoded_zero_is_identity():
    frames = torch.randn(2, 8, 8, 3)
    assert torch.equal(crop_decoded_frame_width(frames, 0), frames)


def test_pad_then_crop_is_identity_5d():
    x = torch.randn(2, 4, 3, 8, 16)          # [B,C,T,H,W]
    padded = circular_pad_video_width(x, 4)
    assert padded.shape[-1] == 16 + 8
    out = circular_crop_video_width(padded, 4)
    assert torch.equal(out, x)


def test_pad_then_crop_is_identity_4d():
    x = torch.randn(1, 4, 8, 16)             # [B,C,H,W]
    out = circular_crop_video_width(circular_pad_video_width(x, 3), 3)
    assert torch.equal(out, x)


def test_pad_wraps_edges():
    x = torch.arange(4, dtype=torch.float32).reshape(1, 1, 1, 1, 4)  # W = [0,1,2,3]
    padded = circular_pad_video_width(x, 1)                          # -> [3,0,1,2,3,0]
    assert padded[..., 0].item() == 3.0
    assert padded[..., -1].item() == 0.0


def test_zero_pad_is_identity():
    x = torch.randn(1, 2, 2, 4, 4)
    assert torch.equal(circular_pad_video_width(x, 0), x)
    assert torch.equal(circular_crop_video_width(x, 0), x)


def test_preserves_dtype_and_device():
    x = torch.randn(1, 1, 1, 2, 4, dtype=torch.float16)
    padded = circular_pad_video_width(x, 2)
    assert padded.dtype == torch.float16 and padded.device == x.device


def test_pad_frame_width_wraps_dim_minus_2():
    # (B, H, W, C) with W=8; pad 2 px each side on the width axis (dim -2).
    x = torch.arange(8).float().view(1, 1, 8, 1).expand(1, 3, 8, 2).contiguous()
    out = circular_pad_frame_width(x, 2)
    assert out.shape == (1, 3, 12, 2)
    # leading pad == last 2 columns of the original; trailing pad == first 2 columns
    assert torch.equal(out[:, :, :2, :], x[:, :, -2:, :])
    assert torch.equal(out[:, :, -2:, :], x[:, :, :2, :])
    # middle is the untouched original
    assert torch.equal(out[:, :, 2:-2, :], x)


def test_pad_frame_width_identity_for_nonpositive():
    x = torch.randn(1, 2, 8, 3)
    assert torch.equal(circular_pad_frame_width(x, 0), x)
    assert torch.equal(circular_pad_frame_width(x, -3), x)


def test_encode_width_round_trips_width():
    # Fake VAE encoder: (B,H,W,C) pixels -> (B, 128, 1, H//spatial, W//spatial) latent.
    spatial = 4

    def fake_encode(pixels):
        b, h, w, c = pixels.shape
        return torch.zeros(b, 128, 1, h // spatial, w // spatial)

    pixels = torch.randn(1, 16, 32, 3)          # W=32 -> latent w = 8
    latent = circular_encode_width(pixels, fake_encode, pad_w=2, spatial=spatial)
    # padded width = 32 + 2*(2*4) = 48 -> latent w = 12 -> crop 2 each side -> 8
    assert latent.shape == (1, 128, 1, 4, 8)


def test_encode_width_passes_wrapped_pixels_to_encoder():
    spatial = 4
    seen = {}

    def fake_encode(pixels):
        seen["w"] = pixels.shape[2]              # width after circular pad
        b, h, w, c = pixels.shape
        return torch.zeros(b, 128, 1, h // spatial, w // spatial)

    pixels = torch.randn(1, 16, 32, 3)
    circular_encode_width(pixels, fake_encode, pad_w=2, spatial=spatial)
    assert seen["w"] == 32 + 2 * (2 * spatial)   # encoder sees the wrap-padded width
