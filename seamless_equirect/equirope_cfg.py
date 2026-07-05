"""Latitudinal Geometric CFG for EquiRoPE.

EquiRoPE collapses position encoding at the poles, but the model still outputs
a flat grid. This node applies a latitude-dependent horizontal blur to the CFG
delta — strong at the poles (smoothing the starburst pinch), zero at the equator
(preserving full detail).

Handles all latent formats:
  - 4D spatial (B, C, H, W)  — standard image models
  - 5D spatial (B, C, T, H, W) — video models
  - 3D flat-packed (B, 1, video_numel + audio_numel) — LTX AV, video shape
    recovered from the conditioning's latent_shapes; audio passes through
  - 3D token-packed (B, N, K) — Flux-style token sequences
"""

import math
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger("SeamlessEquirectangular.equirope_cfg")

# Module-level state for the packed-format width detector
_LATENT_W = 0  # set by the node's patch, used during inference


def _infer_hw(N: int, hint_W: int = 0) -> tuple:
    """Given N tokens and an optional W hint, return (H, W)."""
    if hint_W > 0 and N % hint_W == 0:
        return N // hint_W, hint_W
    # Common latent aspect ratios for LTX / Flux
    for W in [40, 30, 24, 20, 16, 15, 12, 10, 8]:
        if N % W == 0:
            H = N // W
            # Reject unrealistic aspect ratios (LTX ~20×40, Flux ~24×30)
            if 0.4 <= H / W <= 4.0:
                return H, W
    # Last resort — treat as roughly square
    W = int(math.isqrt(N))
    while N % W != 0 and W > 1:
        W -= 1
    return N // W, max(W, 1)


def _video_shape_from_cfg_args(args) -> list:
    """Recover the video latent shape from sampler_cfg_function args.

    LTX AV runs pack video+audio into a flat (B, 1, N) latent; the original
    per-latent shapes travel in each cond's model_conds as a CONDConstant.
    """
    for key in ("input_cond", "input_uncond"):
        conds = args.get(key) or []
        for c in conds:
            if not isinstance(c, dict):
                continue
            ls = c.get("model_conds", {}).get("latent_shapes")
            if ls is None:
                continue
            shapes = getattr(ls, "cond", ls)
            if shapes:
                return list(shapes[0])
    return None


def _circular_horizontal_blur(x: torch.Tensor, radius: int) -> torch.Tensor:
    """1D horizontal blur with circular W padding. Handles 4D and 5D spatial."""
    if radius <= 0 or x.shape[-1] == 1:
        return x

    orig_shape = x.shape
    orig_ndim = x.ndim

    # ── Reshape to 4D spatial (B, C, H, W) ──
    if orig_ndim == 5:
        B, C, T, H, W = orig_shape
        x_in = x.reshape(B * T, C, H, W)
    elif orig_ndim == 4:
        B, C, H, W = orig_shape
        x_in = x
    else:
        return x  # don't know how to handle

    # ── Apply blur ──
    _, C, H, W = x_in.shape
    # Clamp radius so padding never exceeds W (circular pad wraps at most once)
    radius = max(0, min(radius, W // 2 - 1))
    if radius == 0:
        return x
    kernel_size = radius * 2 + 1
    kernel = torch.ones((C, 1, 1, kernel_size), device=x.device, dtype=x.dtype) / kernel_size
    x_padded = F.pad(x_in, (radius, radius, 0, 0), mode="circular")
    x_blurred = F.conv2d(x_padded, kernel, groups=C)

    # ── Reshape back ──
    if orig_ndim == 5:
        x_blurred = x_blurred.reshape(B, C, T, H, W)

    return x_blurred


def _latitudinal_blur_mix(delta: torch.Tensor, radius: int, strength: float,
                          side: str = "both") -> torch.Tensor:
    """Blend delta with its horizontally blurred version by latitude.

    delta must be 4D (B, C, H, W) or 5D (B, C, T, H, W). Returns the same shape.
    side: "both" | "bottom" (nadir rows only) | "top" (zenith rows only).
    """
    H = delta.shape[-2]
    rows = torch.arange(H, device=delta.device, dtype=delta.dtype)
    lat = (rows / max(H - 1, 1)) * math.pi - (math.pi / 2.0)
    pole_mask = (1.0 - torch.abs(torch.cos(lat))) ** strength
    if side != "both":
        # lat is -pi/2 at row 0 (zenith), +pi/2 at the last row (nadir)
        keep = (lat > 0.0) if side == "bottom" else (lat < 0.0)
        pole_mask = torch.where(keep, pole_mask, torch.zeros_like(pole_mask))
    mask_shape = [1] * delta.ndim
    mask_shape[-2] = H
    pole_mask = pole_mask.view(mask_shape)

    delta_blurred = _circular_horizontal_blur(delta, radius)
    return delta * (1.0 - pole_mask) + delta_blurred * pole_mask


class EquiRoPEGeometricCFG:
    """Wraps the model's CFG step with latitudinal horizontal blur.
    At the equator: standard CFG.
    At the poles: CFG delta is horizontally blurred.
    """

    CATEGORY = "model_patches/equirect"
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "enabled": ("BOOLEAN", {"default": True}),
                "pole_blur_radius": ("INT", {
                    "default": 8, "min": 0, "max": 64,
                    "tooltip": "Blur kernel radius at the poles. "
                    "0 = no blur.",
                }),
                "pole_strength": ("FLOAT", {
                    "default": 3.0, "min": 0.5, "max": 8.0, "step": 0.1,
                    "tooltip": "Gamma for latitude mask. "
                    "3.0 keeps rows 0-2 affected.",
                }),
            },
            "optional": {
                "latent_width": ("INT", {
                    "default": 0, "min": 0, "max": 512,
                    "tooltip": "Latent width in columns (e.g. 30 for Flux 2). "
                    "0 = auto-detect from token count. Only needed for "
                    "packed-format models (LTX/Flux) when auto-detection fails.",
                }),
                "pole_side": (["both", "bottom", "top"], {
                    "default": "both",
                    "tooltip": "Which pole(s) to blur. 'bottom' = nadir only: "
                    "sky/cloud detail at the zenith is left untouched.",
                }),
            },
        }

    def patch(self, model, enabled, pole_blur_radius, pole_strength,
              latent_width=0, pole_side="both"):
        global _LATENT_W
        _LATENT_W = latent_width

        if not enabled:
            return (model,)

        m = model.clone()
        logger.info(
            "EquiRoPEGeometricCFG: enabled — blur_radius=%d, strength=%.1f, "
            "latent_width=%d, pole_side=%s",
            pole_blur_radius, pole_strength, latent_width, pole_side,
        )

        def cfg_wrapper(args):
            cond_denoised = args["cond_denoised"]
            uncond_denoised = args["uncond_denoised"]
            cond_scale = args["cond_scale"]
            x = args["input"]

            cond = x - cond_denoised

            # At cfg=1 comfy skips the uncond pass (uncond_denoised is zeros),
            # so there is no CFG delta to blur — reproduce stock behavior.
            if math.isclose(float(cond_scale), 1.0):
                return cond

            uncond = x - uncond_denoised
            delta = cond - uncond

            if pole_blur_radius <= 0:
                return uncond + delta * cond_scale

            if delta.ndim in (4, 5):
                # Plain spatial latent
                delta_mixed = _latitudinal_blur_mix(
                    delta, pole_blur_radius, pole_strength, pole_side
                )
            elif delta.ndim == 3 and delta.shape[1] == 1:
                # LTX AV flat-packed (B, 1, video_numel + audio_numel):
                # unflatten the video part, blur it, pass audio through.
                video_shape = _video_shape_from_cfg_args(args)
                if video_shape is None or len(video_shape) != 5:
                    return uncond + delta * cond_scale
                n = int(math.prod(int(s) for s in video_shape[1:]))
                if n > delta.shape[-1]:
                    return uncond + delta * cond_scale
                b = delta.shape[0]
                vid = delta[:, :, :n].reshape(
                    [b] + [int(s) for s in video_shape[1:]]
                )
                vid = _latitudinal_blur_mix(
                    vid, pole_blur_radius, pole_strength, pole_side
                )
                delta_mixed = torch.cat(
                    [vid.reshape(b, 1, n), delta[:, :, n:]], dim=-1
                )
            elif delta.ndim == 3:
                # Token-packed (B, N, K): unfold tokens to a (B, K, H, W) grid.
                B, N, K = delta.shape
                H, W = _infer_hw(N, _LATENT_W if _LATENT_W else 0)
                grid = delta.permute(0, 2, 1).reshape(B, K, H, W)
                grid = _latitudinal_blur_mix(
                    grid, pole_blur_radius, pole_strength, pole_side
                )
                delta_mixed = grid.reshape(B, K, N).permute(0, 2, 1)
            else:
                return uncond + delta * cond_scale

            return uncond + delta_mixed * cond_scale

        m.set_model_sampler_cfg_function(cfg_wrapper)
        return (m,)


NODE_CLASS_MAPPINGS = {
    "EquiRoPEGeometricCFG": EquiRoPEGeometricCFG,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EquiRoPEGeometricCFG": "EquiRoPE Geometric CFG",
}
