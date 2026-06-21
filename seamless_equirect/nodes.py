"""360 equirect seam nodes for LTX-2.x and compatible latent layouts.

Three tools against the longitude-wrap seam:

- EquirectPerStepRoll: training-free per-step circular roll of the video latent during
  sampling (Diffusion360 / PanoDiffusion-style "circular shift denoising"). At every model
  call the latent is rolled along W by a per-sigma deterministic shift and the prediction
  is unrolled, so each longitude is generated as *interior* at most steps and content is
  forced coherent across the wrap. Works with the frozen base + production LoRA.

- EquirectCircularVAE: circular-W conv padding on the loaded VAE (the cosmetic decode-side
  fix; rebuilt from the vendored circular_conv.py). NOTE: ComfyUI caches the VAE by file,
  so this patches the *shared* object in place -- every consumer of the same VAE file in
  the run sees the same state. The node always resets to stock first, then re-patches per
  its flags, so its own output is deterministic; true stock-vs-circular A/B in ONE graph
  is not possible with a single loaded VAE file.

- EquirectLatentSeamBlend: thin cross-fade of the W-edge latent columns (off-by-default
  fallback; smooths the boundary band only, does not make content wrap).

This module intentionally imports no comfy internals so its helpers are unit-testable
standalone.
"""

import logging
import math

import torch

from .circular_conv import patch_vae_circular, seam_blend_w, unpatch_vae_circular
from .latent_layouts import find_layout, roll_model_io, video_shape_from_conditioning

logger = logging.getLogger("SeamlessEquirectangular.seam")


# ---------------------------------------------------------------------------
# Per-step roll helpers (pure functions; unit-tested)
# ---------------------------------------------------------------------------
def shift_for_sigma(sigma_val: float, w: int, seed: int, mode: str) -> int:
    """Deterministic W-shift for a denoising step.

    Keyed only on (sigma, seed) so the cond and uncond model calls of the same step --
    whether batched together or issued separately -- see the identical shift.
    """
    if w <= 1:
        return 0
    if mode == "half_turn":
        return w // 2
    h = math.sin(float(sigma_val) * 12.9898 + (seed % 100000) * 0.6180339887) * 43758.5453
    frac = h - math.floor(h)
    return int(frac * w) % w


def shifts_for_sigma(sigma_val: float, w: int, seed: int, mode: str):
    """All shifts to evaluate for one step.

    dual_average pairs the base shift with a roughly-opposite second shift. The
    half-turn offset is JITTERED (+/-W/8, per-sigma deterministic) rather than exact:
    a fixed +W/2 pairing imposes the model's compositional biases at two longitudes
    exactly 180 deg apart on every step, which the iterative process amplifies into
    twin subjects / mirrored architecture. Jitter keeps the two boundaries far apart
    (separation within [3W/8, 5W/8], so banding is still cancelled) while preventing
    any fixed cross-longitude correlation from accumulating.
    """
    base_mode = "random_per_step" if mode == "dual_average" else mode
    s = shift_for_sigma(sigma_val, w, seed, base_mode)
    if mode != "dual_average" or w <= 1:
        return [s]
    jitter = shift_for_sigma(sigma_val, max(w // 4, 1), seed + 101, "random_per_step") - w // 8
    return [s, (s + w // 2 + jitter) % w]


def feather_model_io(x: torch.Tensor, video_shape, columns: int):
    """Cross-fade the video part's W-edge columns of a rolled model prediction."""
    if columns <= 0:
        return x
    if x.ndim == 5:
        return seam_blend_w(x, columns)
    if x.ndim == 3 and video_shape is not None and len(video_shape) == 5:
        n = int(math.prod([int(s) for s in video_shape[1:]]))
        if n <= x.shape[-1]:
            b = x.shape[0]
            vid = x[:, :, :n].reshape([b] + [int(s) for s in video_shape[1:]])
            vid = seam_blend_w(vid, columns).reshape(b, x.shape[1], n)
            return torch.cat([vid, x[:, :, n:]], dim=-1)
    return x


class EquirectPerStepRoll:
    """Wrap the diffusion model so every denoising step runs on a W-rolled latent."""

    CATEGORY = "model_patches/equirect"
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "enabled": ("BOOLEAN", {"default": True}),
                "mode": (
                    ["random_per_step", "dual_average", "half_turn"],
                    {
                        "default": "random_per_step",
                        "tooltip": "random_per_step: per-sigma pseudo-random shift (fast; boundary "
                        "artifacts scatter as faint bands). "
                        "dual_average: evaluate every step at two opposite shifts and average -- "
                        "cancels the per-step boundary/vignette banding (~2x sampling time). "
                        "half_turn: constant W/2 shift (diagnostic; just relocates the boundary).",
                    },
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
            },
            "optional": {
                "feather_columns": ("INT", {
                    "default": 0, "min": 0, "max": 16,
                    "tooltip": "Cross-fade this many latent columns of each step's prediction at "
                    "that step's processing boundary (free banding mitigation; 2-4 is sensible).",
                }),
            },
        }

    def patch(self, model, enabled, mode, seed, feather_columns=0):
        if not enabled:
            return (model,)

        m = model.clone()
        logged = {"done": False}

        def wrapper(apply_model, args):
            x = args["input"]
            timestep = args["timestep"]
            c = args["c"]

            video_shape = video_shape_from_conditioning(c)
            layout = find_layout(x, video_shape)
            w = layout.width(x, video_shape) if layout is not None else 0

            sigma_val = float(timestep.flatten()[0])
            shifts = shifts_for_sigma(sigma_val, w, seed, mode)

            if w <= 1 or layout is None:
                if not logged["done"]:
                    logger.warning(
                        "EquirectPerStepRoll: unrecognized latent layout (ndim=%d, shapes=%s); passing through.",
                        x.ndim, c.get("latent_shapes", None),
                    )
                    logged["done"] = True
                return apply_model(x, timestep, **c)

            if not logged["done"]:
                logger.info(
                    "EquirectPerStepRoll active: W=%d, mode=%s, seed=%d, feather=%d",
                    w, mode, seed, feather_columns,
                )
                logged["done"] = True

            mask = c.get("denoise_mask", None)
            out_sum = None
            for s in shifts:
                cs = c
                if isinstance(mask, torch.Tensor) and mask.ndim == 5 and s != 0:
                    cs = dict(c)
                    cs["denoise_mask"] = torch.roll(mask, shifts=s, dims=-1)
                out = apply_model(roll_model_io(x, video_shape, s), timestep, **cs)
                out = feather_model_io(out, video_shape, feather_columns)
                out = roll_model_io(out, video_shape, -s)
                out_sum = out if out_sum is None else out_sum + out
            return out_sum if len(shifts) == 1 else out_sum / len(shifts)

        m.set_model_unet_function_wrapper(wrapper)
        return (m,)


class EquirectCircularVAE:
    """Toggle circular-W conv padding on the loaded VAE (in place, reversible)."""

    CATEGORY = "latent/video/equirect"
    RETURN_TYPES = ("VAE",)
    FUNCTION = "patch"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "patch_encoder": ("BOOLEAN", {"default": False}),
                "patch_decoder": ("BOOLEAN", {"default": True}),
                "enabled": ("BOOLEAN", {"default": True}),
            }
        }

    def patch(self, vae, patch_encoder, patch_decoder, enabled):
        model = vae.first_stage_model
        restored = unpatch_vae_circular(model)
        if enabled and (patch_encoder or patch_decoder):
            patched = patch_vae_circular(model, encoder=patch_encoder, decoder=patch_decoder)
            logger.info(
                "EquirectCircularVAE: circular-W ENABLED (encoder=%s, decoder=%s) -- patched %d convs "
                "(in place: affects every node using this loaded VAE file this run).",
                patch_encoder, patch_decoder, patched,
            )
        else:
            logger.info("EquirectCircularVAE: stock padding (restored %d convs).", restored)
        return (vae,)


class EquirectLatentSeamBlend:
    """Cross-fade the latent's W-edge columns (thin-band cosmetic fallback)."""

    CATEGORY = "latent/video/equirect"
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "blend"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "blend_columns": ("INT", {"default": 2, "min": 0, "max": 16}),
            }
        }

    def blend(self, latent, blend_columns):
        out = dict(latent)
        samples = latent["samples"]
        if getattr(samples, "is_nested", False):
            parts = list(samples.unbind())
            parts[0] = seam_blend_w(parts[0], blend_columns)
            out["samples"] = type(samples)(parts)
        else:
            out["samples"] = seam_blend_w(samples, blend_columns)
        return (out,)


NODE_CLASS_MAPPINGS = {
    "EquirectPerStepRoll": EquirectPerStepRoll,
    "EquirectCircularVAE": EquirectCircularVAE,
    "EquirectLatentSeamBlend": EquirectLatentSeamBlend,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EquirectPerStepRoll": "Equirect Per-Step Latent Roll (360 seam)",
    "EquirectCircularVAE": "Equirect Circular VAE (360 seam)",
    "EquirectLatentSeamBlend": "Equirect Latent Seam Blend (fallback)",
}

