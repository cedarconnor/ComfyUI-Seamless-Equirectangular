"""Per-step soft pole constraint for equirectangular 360 generation.

The top and bottom rows of an equirectangular latent represent the poles — single
physical points mapped to W-pixel-wide scanlines. The diffusion model sees them as
W independent tokens with no structural pressure to converge.

This node wraps each denoising step to gently blend each polar row toward its own
longitude mean, with a cosine-decaying schedule so the constraint tapers off as
structure sets in. The effect is to guide the model toward polar convergence
without monkeypatching position encodings.

The schedule is the key design: early steps get strong blending (the model learns
"these W tokens are actually one point"), late steps get none (fine detail
unconstrained).
"""

import math
import torch
import logging

from .latent_layouts import find_layout, video_shape_from_conditioning

logger = logging.getLogger("SeamlessEquirectangular.soft_pole")


# ---------------------------------------------------------------------------
# Constraint function
# ---------------------------------------------------------------------------

def _apply_pole_constraint(x: torch.Tensor, blend: float,
                           top_rows: int, bot_rows: int) -> torch.Tensor:
    """Force the top/bottom `N` rows toward their longitude-wise means.

    Each row's W values are blended toward that row's mean, fading linearly
    inward so the outermost row gets the strongest push.

    Args:
        x: (B, C, H, W) or (B, C, F, H, W) latent
        blend: 0.0 (no effect) to 1.0 (full mean clamp)
        top_rows, bot_rows: how many rows from each pole to constrain

    Returns:
        Modified tensor (same shape). Does NOT modify the input in place.
    """
    if blend <= 0.0 or (top_rows <= 0 and bot_rows <= 0):
        return x

    x = x.clone()

    # Top rows — row index r where 0 = topmost
    H = x.shape[-2]  # spatial height
    for r in range(top_rows):
        fade = blend * (1.0 - r / max(top_rows, 1.0))
        if fade <= 0:
            break
        row = x[..., r:r+1, :]       # (B, C, [F], 1, W)
        mean = row.mean(dim=-1, keepdim=True)
        x[..., r:r+1, :] = row * (1.0 - fade) + mean * fade

    # Bottom rows — positive index from start to avoid -1:0 slicing edge case
    for r in range(bot_rows):
        fade = blend * (1.0 - r / max(bot_rows, 1.0))
        if fade <= 0:
            break
        row_idx = H - r - 1          # positive index: H-1, H-2, ...
        row = x[..., row_idx:row_idx+1, :]
        mean = row.mean(dim=-1, keepdim=True)
        x[..., row_idx:row_idx+1, :] = row * (1.0 - fade) + mean * fade

    return x


# ---------------------------------------------------------------------------
# Schedule helpers
# ---------------------------------------------------------------------------

def _cosine_schedule(progress: float, steps_fraction: float) -> float:
    """Cosine decay: 1.0 at progress=0, 0 at progress=steps_fraction, 0 after.

    Uses cos(π/2 * t) where t goes 0→1 over the active window.
    """
    if progress >= steps_fraction or steps_fraction <= 0:
        return 0.0
    return math.cos(progress / steps_fraction * math.pi / 2.0)


# ---------------------------------------------------------------------------
# Layout-aware height helper
# ---------------------------------------------------------------------------

def _get_h(x: torch.Tensor, c: dict) -> int:
    """Extract the spatial height from the model input, handling all layouts."""
    if x.ndim >= 4:
        return int(x.shape[-2])
    # Packed layout — get from conditioning
    video_shape = video_shape_from_conditioning(c)
    if video_shape and len(video_shape) >= 4:
        return int(video_shape[-2])
    return 0


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class EquirectSoftPoleConstraint:
    """Wrap the diffusion model so every denoising step softly constrains
    the pole rows toward longitude convergence.

    Place after your UNETLoader and before the sampler. Composes with
    EquirectPerStepRoll (rolls W, this constrains H).
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
                "max_blend": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Peak blend strength at step 0. 1.0 = full row mean, "
                    "0.5 = half toward mean.",
                }),
                "steps_fraction": ("FLOAT", {
                    "default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Fraction of steps where constraint is active. "
                    "0.6 = first 60 % of denoising. Cosine decay to 0 by end of window.",
                }),
                "top_rows": ("INT", {
                    "default": 1, "min": 0, "max": 8,
                    "tooltip": "Number of top (north pole) rows to constrain. "
                    "Values > 1 apply progressively weaker blending inward.",
                }),
                "bot_rows": ("INT", {
                    "default": 1, "min": 0, "max": 8,
                    "tooltip": "Number of bottom (south pole) rows to constrain.",
                }),
            },
        }

    def patch(self, model, enabled, max_blend, steps_fraction,
              top_rows, bot_rows):
        if not enabled:
            return (model,)

        m = model.clone()
        state = {"sigma_first": None}
        logged = {"done": False}

        def wrapper(apply_model, args):
            x = args["input"]
            timestep = args["timestep"]
            c = args["c"]

            sigma_val = float(timestep.flatten()[0])

            if state["sigma_first"] is None:
                state["sigma_first"] = max(sigma_val, 1e-6)

            # Progress: 0 at first step (noisy), 1 at last step (clean)
            progress = max(0.0, min(1.0, 1.0 - sigma_val / state["sigma_first"]))

            blend = max_blend * _cosine_schedule(progress, steps_fraction)

            h = _get_h(x, c)

            # Clamp rows to available height
            top = min(top_rows, h)
            bot = min(bot_rows, h)

            if not logged["done"]:
                logger.info(
                    "EquirectSoftPoleConstraint: H=%d, blend=%.3f at σ=%.4f "
                    "(progress %.2f), top=%d bot=%d, active for ~%d%% of steps",
                    h, blend, sigma_val, progress,
                    top, bot, int(steps_fraction * 100),
                )
                logged["done"] = True

            if blend > 0 and (top > 0 or bot > 0):
                x = _apply_pole_constraint(x, blend, top, bot)

            return apply_model(x, timestep, **c)

        m.set_model_unet_function_wrapper(wrapper)
        return (m,)


NODE_CLASS_MAPPINGS = {
    "EquirectSoftPoleConstraint": EquirectSoftPoleConstraint,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EquirectSoftPoleConstraint": "Equirect Soft Pole Constraint (per-step)",
}


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Test schedule
    for pct in [0, 0.1, 0.3, 0.5, 0.6, 0.7, 1.0]:
        print(f"  progress={pct:.1f}  blend={_cosine_schedule(pct, 0.6):.4f}")

    # Test constraint
    x = torch.zeros(1, 4, 20, 40)
    x[0, 0, 0, :] = torch.rand(40)  # random vals in top row
    x[0, 0, -1, :] = torch.rand(40)  # random vals in bottom row
    y = _apply_pole_constraint(x, 1.0, 2, 2)
    top_std = y[0, 0, 0, :].std().item()
    bot_std = y[0, 0, -1, :].std().item()
    print(f"  After 100% blend: top-row std={top_std:.6f} (should be ~0), "
          f"bot-row std={bot_std:.6f}")
    assert top_std < 1e-6, f"top row not collapsed: {top_std}"
    assert bot_std < 1e-6, f"bot row not collapsed: {bot_std}"
    print("Soft pole: all checks passed.")
