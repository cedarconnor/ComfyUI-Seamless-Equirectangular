"""Seam-aware spatial conditioning helpers for LTX Ripple."""

import math

import torch


def roll_ltx_guide_attention_masks(conditioning, latent_shift, latent_width):
    """Roll Ripple pixel masks with the latent without mutating conditioning."""
    if latent_shift == 0 or latent_width <= 0:
        return conditioning

    entries = conditioning.get("guide_attention_entries")
    if not entries:
        return conditioning

    changed = False
    updated_entries = []
    for entry in entries:
        pixel_mask = entry.get("pixel_mask")
        if not isinstance(pixel_mask, torch.Tensor) or pixel_mask.shape[-1] <= 1:
            updated_entries.append(entry)
            continue

        pixel_shift = round(
            int(latent_shift) * int(pixel_mask.shape[-1]) / int(latent_width)
        )
        if pixel_shift == 0:
            updated_entries.append(entry)
            continue

        updated_entry = dict(entry)
        updated_entry["pixel_mask"] = torch.roll(
            pixel_mask,
            shifts=pixel_shift,
            dims=-1,
        )
        updated_entries.append(updated_entry)
        changed = True

    if not changed:
        return conditioning

    updated = dict(conditioning)
    updated["guide_attention_entries"] = updated_entries
    return updated


class EquirectGuideAttentionMask:
    """Create a periodic guide mask that releases Ripple at the longitude seam."""

    CATEGORY = "conditioning/video/equirect"
    RETURN_TYPES = ("MASK",)
    FUNCTION = "create"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 1024, "min": 1, "max": 16384}),
                "height": ("INT", {"default": 512, "min": 1, "max": 16384}),
                "frames": ("INT", {"default": 33, "min": 1, "max": 10000}),
                "seam_width": (
                    "INT",
                    {
                        "default": 64,
                        "min": 0,
                        "max": 4096,
                        "tooltip": "Total released band split equally across both longitude edges.",
                    },
                ),
                "edge_strength": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                    },
                ),
            }
        }

    def create(self, width, height, frames, seam_width, edge_strength):
        width = int(width)
        half_width = min(int(math.ceil(seam_width / 2)), width // 2)
        weights = torch.ones(width, dtype=torch.float32)

        if half_width > 0:
            for x in range(half_width):
                phase = x / half_width
                ramp = 0.5 - 0.5 * math.cos(math.pi * phase)
                value = edge_strength + (1.0 - edge_strength) * ramp
                weights[x] = value
                weights[width - 1 - x] = value

        mask = weights.view(1, 1, width).expand(
            int(frames),
            int(height),
            width,
        ).clone()
        return (mask,)


NODE_CLASS_MAPPINGS = {
    "EquirectGuideAttentionMask": EquirectGuideAttentionMask,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "EquirectGuideAttentionMask": "Equirect Ripple Guide Attention Mask",
}
