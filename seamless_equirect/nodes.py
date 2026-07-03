"""360 equirect seam nodes for LTX-2.x and compatible latent layouts.
Copyright (C) 2026  Burgstall Labs — Licensed under CC BY-NC 4.0.
https://creativecommons.org/licenses/by-nc/4.0/

Tools for equirectangular 360 video generation:

- EquirectCircularVAE: circular-W conv padding on the loaded VAE decoder.
  Fixes the seam that zero-padded VAE convs create at the longitude wrap.

- EquirectPerStepRoll: per-step circular latent roll during denoising.
  Every model call sees a W-rolled latent so no column is always at the
  wrap boundary — forces content coherent across the seam.

- EquirectLatentSeamBlend: thin cross-fade of the W-edge latent columns
  (off-by-default cosmetic fallback).

- EquirectWrappedNoise: SpheRoPE-style spherical phase remaster.
  Modifies the initial noise latent's low-frequency phase to encode
  polar convergence, reducing stretching at the top and bottom.

This module intentionally imports no comfy internals so its helpers are
unit-testable standalone.
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


# ---------------------------------------------------------------------------
# Spherical Phase Remaster (SpheRoPE wrapped noise)
# ---------------------------------------------------------------------------

def _spherical_phase_remaster(x_T: torch.Tensor, radius: int = 12) -> torch.Tensor:
    """Force low-frequency phase of noise to respect spherical pole convergence.

    Args:
        x_T: (B, C, H, W) or (H, W) noise tensor.
        radius: Circular frequency mask radius.

    Returns: Same shape and dtype as input.
    """
    orig_ndim = x_T.ndim
    orig_dtype = x_T.dtype
    if orig_ndim == 2:
        x_T = x_T.unsqueeze(0).unsqueeze(0)
    elif orig_ndim == 3:
        x_T = x_T.unsqueeze(0)
    B, C, H, W = x_T.shape
    device = x_T.device
    dtype = x_T.dtype

    # Step 1: Create macro-structure by pinching poles toward row means
    rows = torch.arange(H, dtype=torch.float32, device=device)
    theta = ((rows + 0.5) / H) * math.pi - (math.pi / 2.0)
    alpha = (1.0 - torch.cos(theta)).view(1, 1, H, 1)  # 0 at equator, ~1 at poles
    row_means = x_T.mean(dim=-1, keepdim=True)           # [B, C, H, 1]
    x_macro = x_T * (1.0 - alpha) + row_means * alpha

    # Step 2: 2D FFT in float32 for precision
    F_orig = torch.fft.fftshift(torch.fft.fft2(x_T.float(), norm="ortho"), dim=(-2, -1))
    F_macro = torch.fft.fftshift(torch.fft.fft2(x_macro.float(), norm="ortho"), dim=(-2, -1))

    Mag_orig = torch.abs(F_orig)
    Phase_macro = torch.angle(F_macro)
    Phase_orig = torch.angle(F_orig)

    # Step 3: Circular low-pass mask centered at DC
    Y, X_grid = torch.meshgrid(
        torch.arange(H, device=device) - H // 2,
        torch.arange(W, device=device) - W // 2,
        indexing="ij",
    )
    mask = (torch.sqrt(Y.float() ** 2 + X_grid.float() ** 2) <= radius).float().view(1, 1, H, W)

    # Step 4: Replace low-frequency phase with macro version
    Phase_final = mask * Phase_macro + (1.0 - mask) * Phase_orig

    # Step 5: Reconstruct with pristine magnitude
    F_final = Mag_orig * torch.exp(1j * Phase_final)
    F_final_unshifted = torch.fft.ifftshift(F_final, dim=(-2, -1))
    x_final = torch.fft.ifft2(F_final_unshifted, norm="ortho").real

    # Restore original dimensionality
    if orig_ndim == 2:
        x_final = x_final.squeeze(0).squeeze(0)
    elif orig_ndim == 3:
        x_final = x_final.squeeze(0)
    return x_final.to(orig_dtype)


class EquirectWrappedNoise:
    """Wrap the NOISE generator so initial noise has spherically coherent low-frequency phase.

    Intercepts the generate_noise() call and applies spherical phase remaster
    per video frame. The magnitude spectrum and high-frequency details are
    preserved; only the low-frequency phase is modified to encode polar
    convergence.

    Place this node between your RandomNoise and SamplerCustomAdvanced.
    """

    CATEGORY = "noise/equirect"
    RETURN_TYPES = ("NOISE",)
    FUNCTION = "wrap"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "noise": ("NOISE",),
                "enabled": ("BOOLEAN", {"default": True}),
                "radius": ("INT", {
                    "default": 12, "min": 1, "max": 64,
                    "tooltip": "Frequency mask radius. Low-frequency bins inside this radius "
                               "get spherical phase. 12 is the SpheRoPE default for 64x128 latents.",
                }),
            },
        }

    def wrap(self, noise, enabled, radius=12):
        if not enabled:
            return (noise,)

        orig_generate = noise.generate_noise
        import functools

        def wrapped_generate(latent):
            n = orig_generate(latent)

            if hasattr(n, 'tensors') and isinstance(n.tensors, list):
                for i in range(len(n.tensors)):
                    t = n.tensors[i]
                    if t.ndim == 2:
                        # (H, W) single channel — add B/C dims, remaster, remove them
                        n.tensors[i] = _spherical_phase_remaster(
                            t.unsqueeze(0).unsqueeze(0), radius=radius
                        ).reshape(t.shape)
                    elif t.ndim == 4:
                        # (B, C, H, W) — direct
                        n.tensors[i] = _spherical_phase_remaster(t, radius=radius)
                return n
            if n.ndim == 5:
                for t in range(n.shape[2]):
                    n[:, :, t, :, :] = _spherical_phase_remaster(n[:, :, t, :, :], radius=radius)
            elif n.ndim == 4:
                n = _spherical_phase_remaster(n, radius=radius)
            else:
                print(f"[EquirectWrappedNoise] skipped (ndim={n.ndim})")
            return n

        noise.generate_noise = wrapped_generate
        return (noise,)


NODE_CLASS_MAPPINGS = {
    "EquirectPerStepRoll": EquirectPerStepRoll,
    "EquirectCircularVAE": EquirectCircularVAE,
    "EquirectLatentSeamBlend": EquirectLatentSeamBlend,
    "EquirectWrappedNoise": EquirectWrappedNoise,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EquirectPerStepRoll": "Equirect Per-Step Latent Roll (360 seam)",
    "EquirectCircularVAE": "Equirect Circular VAE (360 seam)",
    "EquirectLatentSeamBlend": "Equirect Latent Seam Blend (fallback)",
    "EquirectWrappedNoise": "Equirect Wrapped Noise (SpheRoPE spherical phase remaster)",
}

