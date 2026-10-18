"""MiniMax H3 Equirectangular Seamless Helpers and Nodes.

This module provides H3-aware per-step latent roll synchronization, periodic
DiT RoPE position encoding, and dual-rotation VAE decoding.
"""

import logging
import math
import torch

try:
    import comfy.model_management
    import comfy.patcher_extension
    from comfy.ldm.minimax.model import (
        MiniMaxH3Model,
        PackedLayout,
        rope_rotation_table,
    )
    from comfy.ldm.minimax.vae import MiniMaxH3VideoVAE
except ImportError:
    MiniMaxH3Model = None
    PackedLayout = None
    rope_rotation_table = None
    MiniMaxH3VideoVAE = None

logger = logging.getLogger("SeamlessEquirectangular.minimax_h3")


def quantize_roll_shift(shift: int, width: int, multiple: int) -> int:
    """Quantize a roll shift to a multiple of patch width within [0, width)."""
    if width <= 1 or multiple <= 0:
        return 0
    shift = int(shift) % width
    if multiple == 1:
        return shift
    q = int((round(shift / multiple) * multiple) % width)
    return q


def shifts_for_sigma(sigma_val: float, w: int, seed: int, mode: str, multiple: int = 1):
    """Calculate patch-quantized shifts for a given denoising step."""
    if w <= 1:
        return [0]

    from .nodes import shift_for_sigma as _base_shift_for_sigma

    base_mode = "random_per_step" if mode == "dual_average" else mode
    raw_s = _base_shift_for_sigma(sigma_val, w, seed, base_mode)
    s1 = quantize_roll_shift(raw_s, w, multiple)

    if mode != "dual_average":
        return [s1]

    jitter = _base_shift_for_sigma(sigma_val, max(w // 4, 1), seed + 101, "random_per_step") - w // 8
    raw_s2 = (raw_s + w // 2 + jitter) % w
    s2 = quantize_roll_shift(raw_s2, w, multiple)

    if s2 == s1:
        # Choose nearest distinct legal shift if quantized shifts collapsed
        step = multiple if multiple > 0 else 1
        cand1 = (s1 + step) % w
        cand2 = (s1 - step) % w
        if abs(raw_s2 - cand1) <= abs(raw_s2 - cand2):
            s2 = cand1 if cand1 != s1 else cand2
        else:
            s2 = cand2 if cand2 != s1 else cand1

    return [s1, s2]


def roll_minimax_h3_fl2va_conditioning(c, shift: int, target_video_shape):
    """Return original c or a shallow copy with H3 FL2VA video guides rolled."""
    if shift == 0:
        return c

    payload = c.get("minimax_payload", None)
    if not isinstance(payload, dict):
        return c

    keyframes = payload.get("keyframes", None)
    if not keyframes:
        return c

    refs = payload.get("refs", None)
    if refs:
        # Reference mode, do not roll
        return c

    cond_video_latents = payload.get("cond_video_latents", None)
    if not cond_video_latents:
        return c

    target_h, target_w = target_video_shape[-2], target_video_shape[-1]
    rolled_latents = []

    for guide in cond_video_latents:
        if not isinstance(guide, torch.Tensor) or guide.ndim != 5:
            raise ValueError(f"H3 FL2VA guide latent must be a 5D tensor, got {type(guide)}")
        if guide.shape[-2] != target_h or guide.shape[-1] != target_w:
            raise ValueError(
                f"H3 FL2VA guide spatial dimensions ({guide.shape[-2]}, {guide.shape[-1]}) "
                f"do not match target spatial dimensions ({target_h}, {target_w})"
            )
        rolled_latents.append(torch.roll(guide, shifts=shift, dims=-1))

    c_copy = dict(c)
    payload_copy = dict(payload)
    payload_copy["cond_video_latents"] = rolled_latents
    c_copy["minimax_payload"] = payload_copy
    return c_copy


def build_h3_rope_rotation_table(diffusion_model, layout, device, dtype, blend: float, W_patches: int = None):
    """Build periodic width RoPE rotation table for H3 DiT."""
    stock_angles = diffusion_model.rope_freqs(layout.position_ids, device)
    angles = stock_angles.clone()

    inv_freq = comfy.model_management.cast_to(diffusion_model.rope.inv_freq, device=device).to(torch.float64)
    K = inv_freq.numel()

    for a, b, kind in layout.segments:
        if kind in ("cond", "video"):
            pos_seg = layout.position_ids[a:b].to(torch.float64)
            n_rows = b - a

            w_patches_seg = W_patches
            if w_patches_seg is None or w_patches_seg <= 0:
                w_diffs = (pos_seg[1:, 2] - pos_seg[:-1, 2])
                resets = (w_diffs < 0).nonzero(as_tuple=True)[0]
                if len(resets) > 0:
                    w_patches_seg = int(resets[0].item()) + 1
                else:
                    w_patches_seg = n_rows

            if w_patches_seg <= 1:
                continue

            w_grid = pos_seg[:w_patches_seg, 2]
            width_step = float((w_grid[1] - w_grid[0]).item())
            w_0 = float(w_grid[0].item())

            final_w_all = torch.zeros(n_rows, K, device=device, dtype=torch.float64)
            j_indices = torch.arange(w_patches_seg, device=device, dtype=torch.float64)

            for k in range(K):
                inv_f = float(inv_freq[k].item())
                cycles = width_step * w_patches_seg * inv_f / (2.0 * math.pi)
                harmonic = round(cycles)
                harmonic_mag = max(1, min(abs(harmonic), w_patches_seg // 2))
                harmonic = harmonic_mag if cycles >= 0 else -harmonic_mag

                d_theta = 2.0 * math.pi * harmonic / w_patches_seg
                stock_w = (w_0 + j_indices * width_step) * inv_f
                periodic_w = w_0 * inv_f + j_indices * d_theta
                final_w_col = stock_w + blend * (periodic_w - stock_w)

                # Repeat column angles across all spatial rows and frames in segment
                final_w_all[:, k] = final_w_col.repeat(n_rows // w_patches_seg)

            angles[a:b, 2 * K : 3 * K] = final_w_all.to(angles.dtype)
            angles[a:b, 5 * K : 6 * K] = final_w_all.to(angles.dtype)

    return rope_rotation_table(angles, dtype)


class EquirectMiniMaxH3RoPE:
    """Supply periodic width rotations to H3 DiT attention without touching text/audio/reference rows."""

    CATEGORY = "model_patches/equirect"
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "enabled": ("BOOLEAN", {"default": True}),
                "blend": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }

    def patch(self, model, enabled, blend):
        if not enabled or blend == 0.0:
            return (model,)

        diffusion_model = None
        if hasattr(model, "get_model_object"):
            diffusion_model = model.get_model_object("diffusion_model")
        if diffusion_model is None and hasattr(model, "model"):
            diffusion_model = model.model
        if diffusion_model is None and hasattr(model, "diffusion_model"):
            diffusion_model = model.diffusion_model

        is_h3 = False
        if diffusion_model is not None:
            if (
                (MiniMaxH3Model is not None and isinstance(diffusion_model, MiniMaxH3Model))
                or diffusion_model.__class__.__name__ == "MiniMaxH3Model"
                or getattr(diffusion_model, "is_minimax_h3", False)
            ):
                is_h3 = True

        if not is_h3:
            raise ValueError("EquirectMiniMaxH3RoPE is designed specifically for MiniMax H3 models.")

        m = model.clone()

        def rope_wrapper(executor, x, timestep, context, transformer_options={}, minimax_payload=None, **kwargs):
            tx_opts = dict(transformer_options)
            video_x, audio_x = x[0], x[1]
            device = video_x.device
            # H3 explicitly uses context.dtype as compute dtype
            dtype = context[0].dtype if isinstance(context, (list, tuple)) else context.dtype

            layout = minimax_payload.get("layout", None) if minimax_payload else None
            W_patches = video_x.shape[4] // 2

            if layout is None or not hasattr(layout, "position_ids"):
                text_len = context[0].shape[0] if isinstance(context, (list, tuple)) else context.shape[1]
                latent_t, lat_h, lat_w = video_x.shape[2], video_x.shape[3], video_x.shape[4]
                audio_t = audio_x.shape[-1]
                layout = PackedLayout(
                    text_len,
                    latent_t,
                    lat_h,
                    lat_w,
                    audio_t,
                    keyframes=minimax_payload.get("keyframes") if minimax_payload else None,
                    refs=minimax_payload.get("refs") if minimax_payload else None,
                )

            table = build_h3_rope_rotation_table(diffusion_model, layout, device, dtype, blend, W_patches=W_patches)
            tx_opts["equirect_minimax_h3_rope_table"] = table
            kwargs["transformer_options"] = tx_opts
            return executor(x, timestep, context, minimax_payload=minimax_payload, **kwargs)

        wrapper_type = getattr(comfy.patcher_extension.WrappersMP, "DIFFUSION_MODEL", "diffusion_model")
        if hasattr(m, "remove_wrappers_with_key"):
            m.remove_wrappers_with_key(wrapper_type, "equirect_minimax_h3_rope")
        if hasattr(m, "add_wrapper_with_key"):
            m.add_wrapper_with_key(wrapper_type, "equirect_minimax_h3_rope", rope_wrapper)
        elif hasattr(m, "add_wrapper"):
            m.add_wrapper(wrapper_type, rope_wrapper)

        def make_attn_forward(original_forward):
            def patched_attn_forward(x, rope_freqs=None, transformer_options={}):
                table = transformer_options.get("equirect_minimax_h3_rope_table", None)
                if table is not None:
                    rope_freqs = table
                return original_forward(x, rope_freqs=rope_freqs, transformer_options=transformer_options)

            return patched_attn_forward

        for i, block in enumerate(diffusion_model.blocks):
            orig_forward = None
            if hasattr(m, "get_model_object"):
                orig_forward = m.get_model_object(f"diffusion_model.blocks.{i}.attn.forward")
            if orig_forward is None:
                orig_forward = block.attn.forward
            m.add_object_patch(f"diffusion_model.blocks.{i}.attn.forward", make_attn_forward(orig_forward))

        return (m,)


class EquirectMiniMaxH3VAEDecode:
    """Composite stock and half-rotated H3 decodes near the true seam."""

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
        is_h3_vae = False
        if (
            (MiniMaxH3VideoVAE is not None and isinstance(first_stage, MiniMaxH3VideoVAE))
            or first_stage.__class__.__name__ == "MiniMaxH3VideoVAE"
            or getattr(first_stage, "is_minimax_h3_vae", False)
        ):
            is_h3_vae = True

        if not is_h3_vae:
            raise ValueError("EquirectMiniMaxH3VAEDecode requires a MiniMaxH3VideoVAE model.")

        s_tensor = samples["samples"]
        if getattr(s_tensor, "is_nested", False):
            video = s_tensor.unbind()[0]
        else:
            video = s_tensor

        if video.ndim != 5:
            raise ValueError(f"Expected 5D video latent [B, C, T, H, W], got shape {video.shape}")

        if blend_columns == 0:
            images = vae.decode(video)
            if images.ndim == 5:
                images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
            return (images,)

        W_lat = video.shape[-1]
        patch_w = getattr(first_stage, "patch_size", (1, 2, 2))[-1]
        shift_lat = quantize_roll_shift(W_lat // 2, W_lat, patch_w)

        stock_images = vae.decode(video)
        rolled_video = torch.roll(video, shifts=shift_lat, dims=-1)
        shifted_images = vae.decode(rolled_video)

        W_img = stock_images.shape[-2]
        pixel_shift = round(shift_lat * W_img / W_lat)
        unrolled_images = torch.roll(shifted_images, shifts=-pixel_shift, dims=-2)

        band = round(blend_columns * W_img / W_lat)
        band = max(1, min(band, W_img // 2))

        w_vec = torch.zeros(W_img, device=stock_images.device, dtype=stock_images.dtype)
        for x in range(band):
            if band <= 1:
                val = 0.5
            else:
                val = 0.5 * (1.0 + math.cos(math.pi * x / (band - 1)))
            w_vec[x] = val
            w_vec[W_img - 1 - x] = val

        view_shape = [1] * stock_images.ndim
        view_shape[-2] = W_img
        w_tensor = w_vec.view(*view_shape)

        composite = stock_images * (1.0 - w_tensor) + unrolled_images * w_tensor

        logger.info(
            "EquirectMiniMaxH3VAEDecode: dual-rotation decode active (blend_columns=%d, band_pixels=%d)",
            blend_columns,
            band,
        )

        if composite.ndim == 5:
            composite = composite.reshape(-1, composite.shape[-3], composite.shape[-2], composite.shape[-1])

        return (composite,)


NODE_CLASS_MAPPINGS = {
    "EquirectMiniMaxH3RoPE": EquirectMiniMaxH3RoPE,
    "EquirectMiniMaxH3VAEDecode": EquirectMiniMaxH3VAEDecode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EquirectMiniMaxH3RoPE": "Equirect MiniMax H3 Periodic RoPE (360 seam)",
    "EquirectMiniMaxH3VAEDecode": "Equirect MiniMax H3 Seamless VAE Decode",
}
