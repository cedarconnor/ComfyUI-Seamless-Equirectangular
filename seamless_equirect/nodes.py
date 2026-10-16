"""360 equirect seam nodes for Krea2, LTX-2.x, and compatible latent layouts.
Copyright (C) 2026  Burgstall Labs
Licensed under the PolyForm Noncommercial License 1.0.0.
https://polyformproject.org/licenses/noncommercial/1.0.0

Tools for equirectangular 360 video generation:

- EquirectCircularVAE: circular-W conv padding on the loaded VAE decoder.
  Fixes the seam that zero-padded convolutional VAE decoders create at the longitude wrap.

- EquirectPerStepRoll: per-step circular latent roll during denoising.
  Every model call sees a W-rolled latent so no column is always at the
  wrap boundary — forces content coherent across the seam.

- EquirectLatentSeamBlend: thin cross-fade of the W-edge latent columns
  (off-by-default cosmetic fallback).

- EquirectWrappedNoise: spherical phase remaster on the initial noise
  latent, encoding polar convergence to reduce stretching at the poles.

- EquirectSphericalRoPE: monkeypatches the LTX-2.3 transformer's position
  encoding with a two-path spectral split (harmonic snapping for seam,
  spherical coordinates for poles). Based on ideas from the SpheRoPE
  paper (https://arxiv.org/abs/2606.32033).

This module intentionally imports no comfy internals so its helpers are
unit-testable standalone.
"""

import importlib.util
import logging
import math
import os

import torch
import comfy.ldm.lightricks.model as _ltx_model

from .circular_conv import patch_vae_circular, seam_blend_w, unpatch_vae_circular
from . import krea2
from .latent_layouts import find_layout, roll_model_io, video_shape_from_conditioning
from .ripple import roll_ltx_guide_attention_masks

# Load the vendored periodic_rope module
_spec = importlib.util.spec_from_file_location(
    "equirect_periodic_rope", os.path.join(os.path.dirname(__file__), "periodic_rope.py")
)
_periodic_rope = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_periodic_rope)

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


def _quantize_roll_shift(shift: int, w: int, multiple: int) -> int:
    if w <= 1:
        return 0
    shift = int(shift) % w
    if multiple <= 1:
        return shift
    return int((round(shift / multiple) * multiple) % w)


def shifts_for_sigma(sigma_val: float, w: int, seed: int, mode: str, multiple: int = 1):
    """All shifts to evaluate for one step.

    dual_average pairs the base shift with a roughly-opposite second shift. The
    half-turn offset is JITTERED (+/-W/8, per-sigma deterministic) rather than exact:
    a fixed +W/2 pairing imposes the model's compositional biases at two longitudes
    exactly 180 degrees apart on every step, which the iterative process amplifies into
    twin subjects or mirrored architecture. Jitter keeps the boundaries far apart
    without a fixed cross-longitude correlation.
    """
    base_mode = "random_per_step" if mode == "dual_average" else mode
    raw_s = shift_for_sigma(sigma_val, w, seed, base_mode)
    s = _quantize_roll_shift(raw_s, w, multiple)
    if mode != "dual_average" or w <= 1:
        return [s]
    jitter = shift_for_sigma(sigma_val, max(w // 4, 1), seed + 101, "random_per_step") - w // 8
    raw_s2 = (raw_s + w // 2 + jitter) % w
    s2 = _quantize_roll_shift(raw_s2, w, multiple)
    if multiple > 1 and s2 == s:
        s2 = (s + multiple) % w
    return [s, s2]


def run_percent(c, sigma_val: float):
    """Fraction of the current sampler run completed, from the sigma schedule.

    Returns None when the schedule isn't available (e.g. a sampler that doesn't
    populate transformer_options["sample_sigmas"]).
    """
    sched = c.get("transformer_options", {}).get("sample_sigmas", None)
    if sched is None or len(sched) < 2:
        return None
    sched = sched.detach().float().cpu()
    idx = int((sched - sigma_val).abs().argmin().item())
    return idx / (len(sched) - 1)


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
                "start_percent": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Hold off rolling for the first fraction of the run. Scene layout "
                    "forms in the early steps; keeping the boundary at the true wrap edge there "
                    "(where WrappedNoise correlates content) lets the panorama close structurally, "
                    "then rolling kicks in to heal seam tone/texture. 0 = roll every step. "
                    "0.2-0.3 is a sensible starting point against layout cuts in fast-motion scenes.",
                }),
                "end_percent": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Stop rolling after this fraction of the run. Values below 1 let "
                    "early denoise steps establish wrap geometry while late color/detail steps "
                    "settle in stable screen coordinates.",
                }),
            },
        }

    def patch(
        self,
        model,
        enabled,
        mode,
        seed,
        feather_columns=0,
        start_percent=0.0,
        end_percent=1.0,
    ):
        if not enabled:
            return (model,)

        m = model.clone()
        logged = {"done": False}

        diffusion_model = None
        if hasattr(model, "get_model_object"):
            diffusion_model = model.get_model_object("diffusion_model")
        if diffusion_model is None and hasattr(model, "model"):
            diffusion_model = model.model
        if diffusion_model is None and hasattr(model, "diffusion_model"):
            diffusion_model = model.diffusion_model

        is_minimax_h3 = False
        is_krea2 = False
        width_patch_multiple = 1
        if diffusion_model is not None:
            if (
                diffusion_model.__class__.__name__ == "MiniMaxH3Model"
                or getattr(diffusion_model, "is_minimax_h3", False)
            ):
                is_minimax_h3 = True
            patch_size = getattr(diffusion_model, "patch_size", None)
            if patch_size is not None and len(patch_size) >= 3:
                width_patch_multiple = int(patch_size[-1])
            if krea2.is_krea2_model(diffusion_model):
                is_krea2 = True
                width_patch_multiple = max(1, int(diffusion_model.patch))

        def wrapper(apply_model, args):
            x = args["input"]
            timestep = args["timestep"]
            c = args["c"]

            video_shape = video_shape_from_conditioning(c)
            layout = find_layout(x, video_shape)
            w = layout.width(x, video_shape) if layout is not None else 0

            sigma_val = float(timestep.flatten()[0])
            if is_minimax_h3:
                from .minimax_h3 import shifts_for_sigma as h3_shifts_for_sigma
                shifts = h3_shifts_for_sigma(sigma_val, w, seed, mode, width_patch_multiple)
            elif is_krea2:
                shifts = shifts_for_sigma(
                    sigma_val, w, seed, mode, multiple=width_patch_multiple
                )
            else:
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
                    "EquirectPerStepRoll active: W=%d, mode=%s, seed=%d, feather=%d, "
                    "window=%.2f-%.2f",
                    w, mode, seed, feather_columns, start_percent, end_percent,
                )
                logged["done"] = True

            if start_percent > 0.0 or end_percent < 1.0:
                pct = run_percent(c, sigma_val)
                if pct is not None:
                    if pct < start_percent or pct > end_percent:
                        return apply_model(x, timestep, **c)

            mask = c.get("denoise_mask", None)
            out_sum = None
            for s in shifts:
                cs = c
                if is_minimax_h3:
                    from .minimax_h3 import roll_minimax_h3_fl2va_conditioning
                    cs = roll_minimax_h3_fl2va_conditioning(c, s, video_shape)

                cs = roll_ltx_guide_attention_masks(cs, s, w)

                if isinstance(mask, torch.Tensor) and s != 0:
                    rolled_mask = roll_model_io(mask, video_shape, s)
                    if rolled_mask is not None:
                        if cs is c:
                            cs = dict(c)
                        cs["denoise_mask"] = rolled_mask

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
        is_h3_vae = (
            model.__class__.__name__ == "MiniMaxH3VideoVAE"
            or getattr(model, "is_minimax_h3_vae", False)
        )
        if enabled and (patch_encoder or patch_decoder):
            patched = patch_vae_circular(model, encoder=patch_encoder, decoder=patch_decoder)
            if patched == 0 and is_h3_vae:
                logger.warning(
                    "EquirectCircularVAE: H3 uses a tiled ViT decoder; use Equirect MiniMax H3 Seamless VAE Decode."
                )
            else:
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


class EquirectImageSeamBlend:
    """Optional final pixel-space seam blend for the last one-pixel/shallow color discontinuity."""

    CATEGORY = "image/equirect"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "blend"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "blend_pixels": ("INT", {"default": 24, "min": 0, "max": 512}),
            }
        }

    def blend(self, images, blend_pixels=24):
        if blend_pixels <= 0:
            return (images,)
        x = images.movedim(-2, -1)
        out = seam_blend_w(x, blend_pixels).movedim(-1, -2)
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

            if getattr(n, "is_nested", False):
                parts = list(n.unbind())
                if len(parts) > 0 and parts[0].ndim == 5:
                    vid = parts[0]
                    B, C, T, H, W = vid.shape
                    vid_flat = vid.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
                    vid_remastered = _spherical_phase_remaster(vid_flat, radius=radius)
                    parts[0] = vid_remastered.reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4)
                return type(n)(parts)

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
                B, C, T, H, W = n.shape
                vid_flat = n.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
                vid_remastered = _spherical_phase_remaster(vid_flat, radius=radius)
                n = vid_remastered.reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4)
            elif n.ndim == 4:
                n = _spherical_phase_remaster(n, radius=radius)
            else:
                print(f"[EquirectWrappedNoise] skipped (ndim={n.ndim})")
            return n

        noise.generate_noise = wrapped_generate
        return (noise,)


# ---------------------------------------------------------------------------
# Spherical RoPE (SpheRoPE two-path spectral split)
# ---------------------------------------------------------------------------

def _resolve_w_scale(model):
    for attr_chain in (("model", "diffusion_model"), ("model",)):
        obj = model
        ok = True
        for a in attr_chain:
            obj = getattr(obj, a, None)
            if obj is None:
                ok = False
                break
        if ok:
            vsf = getattr(obj, "vae_scale_factors", None)
            if vsf is not None and len(vsf) >= 3:
                return float(vsf[2])
    return 32.0


class EquirectSphericalRoPE:
    """Monkeypatches the LTX-2.3 transformer's RoPE with a spherical-aware
    two-path spectral split for the width (longitude) axis.

    Path A (high-frequency channels, >= threshold cycles): harmonic snapping
      for seam continuity.
    Path B (low-frequency channels, < threshold cycles): spherical Cartesian
      (X, Y) coordinates for polar convergence.

    Based on the SpheRoPE paper (https://arxiv.org/abs/2606.32033).
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
                "blend": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                     "tooltip": "0 = original linear, 1 = fully spherical."}),
            },
            "optional": {
                "path_b_threshold": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 100.0, "step": 0.5,
                                                "tooltip": "Cycle threshold for Path A vs Path B. "
                                                "Lower = more spherical channels."}),
                "r_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1,
                                       "tooltip": "Spherical coordinate magnitude."}),
            },
        }

    def patch(self, model, enabled, blend, path_b_threshold=10.0, r_scale=1.0):
        w_scale = _resolve_w_scale(model)
        _periodic_rope.install(_ltx_model, w_scale=w_scale)
        _periodic_rope.set_params(
            enabled=bool(enabled), alpha=float(blend), w_scale=w_scale,
            path_b_threshold=float(path_b_threshold), r_scale=float(r_scale),
        )
        mode = "spherical Path A+B" if enabled else "linear"
        print(f"[EquirectSphericalRoPE] enabled={enabled} blend={blend} "
              f"threshold={path_b_threshold} r_scale={r_scale} (W-axis: {mode})")
        return (model,)


from . import minimax_h3

NODE_CLASS_MAPPINGS = {
    "EquirectPerStepRoll": EquirectPerStepRoll,
    "EquirectCircularVAE": EquirectCircularVAE,
    "EquirectLatentSeamBlend": EquirectLatentSeamBlend,
    "EquirectWrappedNoise": EquirectWrappedNoise,
    "EquirectSphericalRoPE": EquirectSphericalRoPE,
    "EquirectImageSeamBlend": EquirectImageSeamBlend,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EquirectPerStepRoll": "Equirect Per-Step Latent Roll (360 seam)",
    "EquirectCircularVAE": "Equirect Circular VAE (360 seam)",
    "EquirectLatentSeamBlend": "Equirect Latent Seam Blend (fallback)",
    "EquirectWrappedNoise": "Equirect Wrapped Noise (spherical phase remaster)",
    "EquirectSphericalRoPE": "Equirect Spherical RoPE (SpheRoPE)",
    "EquirectImageSeamBlend": "Equirect Image Seam Blend (final fallback)",
}

if hasattr(minimax_h3, "NODE_CLASS_MAPPINGS"):
    NODE_CLASS_MAPPINGS.update(minimax_h3.NODE_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(minimax_h3.NODE_DISPLAY_NAME_MAPPINGS)

if hasattr(krea2, "NODE_CLASS_MAPPINGS"):
    NODE_CLASS_MAPPINGS.update(krea2.NODE_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(krea2.NODE_DISPLAY_NAME_MAPPINGS)

# Merge in nodes from submodule files that ComfyUI's loader doesn't auto-discover
for _module_name in ("soft_pole", "equirope", "equirope_cfg", "ltx25", "ripple"):
    try:
        _mod = importlib.import_module(f".{_module_name}", __package__)
        if hasattr(_mod, "NODE_CLASS_MAPPINGS"):
            NODE_CLASS_MAPPINGS.update(getattr(_mod, "NODE_CLASS_MAPPINGS", {}))
            NODE_DISPLAY_NAME_MAPPINGS.update(getattr(_mod, "NODE_DISPLAY_NAME_MAPPINGS", {}))
    except Exception:
        pass
