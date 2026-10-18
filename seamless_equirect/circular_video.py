"""Seam-safe width wrap for LTX video latents and decoded frames."""
import torch


def circular_pad_video_width(x: torch.Tensor, pad_w: int) -> torch.Tensor:
    """Wrap-pad the width axis (dim=-1) by pad_w columns on each side.
    Works for 4D [B,C,H,W] and 5D [B,C,T,H,W]. pad_w<=0 is identity."""
    if pad_w <= 0:
        return x
    return torch.cat([x[..., -pad_w:], x, x[..., :pad_w]], dim=-1)


def circular_crop_video_width(x: torch.Tensor, pad_w: int) -> torch.Tensor:
    """Inverse of circular_pad_video_width. pad_w<=0 is identity."""
    if pad_w <= 0:
        return x
    return x[..., pad_w:-pad_w]


def circular_upsample_width(latent, upsample_fn, pad_w: int):
    """Pad width, run a 2x-width spatial upsampler, crop 2*pad_w back off the width."""
    padded = circular_pad_video_width(latent, pad_w)
    upsampled = upsample_fn(padded)
    return circular_crop_video_width(upsampled, pad_w * 2)


def crop_decoded_frame_width(frames: torch.Tensor, pixel_pad: int) -> torch.Tensor:
    """Crop pixel_pad px from each side of the width axis (dim=-2) of (F,H,W,C) frames."""
    if pixel_pad <= 0:
        return frames
    return frames[..., pixel_pad:-pixel_pad, :]


def circular_pad_frame_width(frames: torch.Tensor, pixel_pad: int) -> torch.Tensor:
    """Wrap-pad the width axis (dim=-2) of (...,H,W,C) frames by pixel_pad px on
    each side. The mirror of crop_decoded_frame_width. pixel_pad<=0 is identity."""
    if pixel_pad <= 0:
        return frames
    return torch.cat(
        [frames[..., -pixel_pad:, :], frames, frames[..., :pixel_pad, :]], dim=-2
    )


def circular_encode_width(pixels, encode_fn, pad_w: int, spatial: int = 32):
    """Seam-safe VAE encode: wrap-pad the image width (dim=-2) by pad_w*spatial px,
    run encode_fn -> latent (B,C',T,h,w) with width last, crop pad_w latent columns.
    The inverse of the circular decode path so the conditioning latent is seamless."""
    padded = circular_pad_frame_width(pixels, pad_w * spatial)
    latent = encode_fn(padded)
    return circular_crop_video_width(latent, pad_w)


def assert_width_dims_match(got: int, want: int) -> None:
    """Fail loudly if a circular crop did not restore the requested width."""
    if got != want:
        raise ValueError(
            f"[SpheRoPE LTX] Circular crop width mismatch: got {got}, want {want}. "
            "Check pad_w and the upsampler/decoder scale factor."
        )
