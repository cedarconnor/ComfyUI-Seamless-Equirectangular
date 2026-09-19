"""Seamless decode support for LTX 2.5's neighborhood-attention video VAE."""

import logging
import math

import torch


logger = logging.getLogger("SeamlessEquirectangular.ltx25")


def flatten_video_images(images):
    """Convert native [B,T,H,W,C] decoder output to Comfy's [B*T,H,W,C]."""
    if images.ndim == 5:
        return images.reshape(
            -1, images.shape[-3], images.shape[-2], images.shape[-1]
        )
    return images


def dual_rotation_composite(
    stock_images,
    shifted_images,
    shift_lat,
    latent_width,
    blend_columns,
):
    """Unroll a shifted decode and use it around the stock decode's W boundary."""
    image_width = stock_images.shape[-2]
    pixel_shift = round(shift_lat * image_width / latent_width)
    unrolled_images = torch.roll(shifted_images, shifts=-pixel_shift, dims=-2)

    band = round(blend_columns * image_width / latent_width)
    band = max(1, min(band, image_width // 2))
    weights = torch.zeros(
        image_width, device=stock_images.device, dtype=stock_images.dtype
    )
    for x in range(band):
        value = 1.0 if band <= 1 else 0.5 * (
            1.0 + math.cos(math.pi * x / (band - 1))
        )
        weights[x] = value
        weights[image_width - 1 - x] = value

    shape = [1] * stock_images.ndim
    shape[-2] = image_width
    weights = weights.view(*shape)
    return stock_images * (1.0 - weights) + unrolled_images * weights


class EquirectLTX25VAEDecode:
    """Composite stock and half-rotated LTX 2.5 decodes near the true seam."""

    CATEGORY = "latent/video/equirect"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "vae": ("VAE",),
                "blend_columns": ("INT", {"default": 2, "min": 0, "max": 16}),
            }
        }

    def decode(self, samples, vae, blend_columns=2):
        first_stage = vae.first_stage_model
        if first_stage.__class__.__name__ != "CausalDiffusionVAE":
            raise ValueError(
                "EquirectLTX25VAEDecode requires an LTX 2.5 CausalDiffusionVAE."
            )

        video = samples["samples"]
        if video.ndim != 5:
            raise ValueError(
                f"Expected 5D video latent [B, C, T, H, W], got shape {video.shape}"
            )

        stock_images = vae.decode(video)
        if blend_columns == 0:
            return (flatten_video_images(stock_images),)

        latent_width = video.shape[-1]
        shift_lat = latent_width // 2
        shifted_images = vae.decode(
            torch.roll(video, shifts=shift_lat, dims=-1)
        )
        composite = dual_rotation_composite(
            stock_images,
            shifted_images,
            shift_lat,
            latent_width,
            blend_columns,
        )

        logger.info(
            "EquirectLTX25VAEDecode: dual-rotation decode active "
            "(blend_columns=%d, latent_width=%d, image_width=%d)",
            blend_columns,
            latent_width,
            stock_images.shape[-2],
        )
        return (flatten_video_images(composite),)


NODE_CLASS_MAPPINGS = {
    "EquirectLTX25VAEDecode": EquirectLTX25VAEDecode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EquirectLTX25VAEDecode": "Equirect LTX 2.5 Seamless VAE Decode",
}
