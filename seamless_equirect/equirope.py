"""EquiRoPE: latitude-scaled equatorial RoPE for equirectangular 360 generation.

EquiRoPE applies a spectral-split latitude scaling to the width position encoding:
  - Low-frequency channels (below the cycle threshold): width position is scaled
    by 1-(1-|cos(latitude)|)^pole_gamma, collapsing polar tokens toward
    co-location. pole_gamma=1 is the classic |cos| profile (acts at all
    latitudes); higher gamma keeps mid-latitudes untouched so long horizontal
    structures (rooflines, horizons) stay straight, and only the rows near the
    poles collapse.
  - High-frequency channels (at or above threshold): stock position encoding preserved,
    retaining fine-grained position discrimination at all latitudes.

This spectral split is CRITICAL for LoRA compatibility — the LoRA's attention
weights depend on high-frequency position relationships. Only collapsing the
low-frequency structure channels at the poles preserves the LoRA's learned
attention patterns while geometrically correcting the pole convergence problem.

Inspired by the SpheRoPE paper (https://arxiv.org/abs/2606.32033) but
architecturally distinct — EquiRoPE scales existing positions rather than
replacing the encoding function with spherical Cartesian coordinates.
"""

import logging
import math
import time

import torch

logger = logging.getLogger("SeamlessEquirectangular.equirope")

# ---- module state ------------------------------------------------------------
_ORIG = {}
_ENABLED = False
_BLEND = 1.0        # 0 = stock, 1 = full latitude scaling
_EPS = 0.01         # minimum latitude scaling (prevents complete collapse)
_THRESHOLD = 10.0   # cycles threshold (same default as SpheRoPE Path B)
_POLE_GAMMA = 3.0   # pole gating exponent: scale = 1-(1-|cos lat|)^gamma.
                    # 1.0 = classic |cos lat| scaling (acts at all latitudes);
                    # higher = scaling stays ~1 at mid-latitudes and only
                    # engages near the poles.
_POLE_SIDE = "both"  # "both" | "bottom" (nadir only) | "top" (zenith only).
                     # Bottom-only keeps stock positions for the whole upper
                     # hemisphere (skylines/rooftops) while still collapsing
                     # the nadir rows.
_SENTINEL = "_equirope_installed"

# The monkeypatch is global module state: it survives node removal/bypass. Warn
# on every sampling run so silent carry-over can't skew A/B comparisons.
_WARN_INTERVAL = 60.0
_last_active_warn = 0.0


def set_params(enabled: bool = True, blend: float = 1.0, epsilon: float = 0.01,
               threshold: float = 10.0, pole_gamma: float = 3.0,
               pole_side: str = "both"):
    global _ENABLED, _BLEND, _EPS, _THRESHOLD, _POLE_GAMMA, _POLE_SIDE
    _ENABLED = bool(enabled)
    _BLEND = float(blend)
    _EPS = float(epsilon)
    _THRESHOLD = float(threshold)
    _POLE_GAMMA = max(float(pole_gamma), 1e-3)
    _POLE_SIDE = str(pole_side)


def _warn_active():
    global _last_active_warn
    now = time.monotonic()
    if now - _last_active_warn >= _WARN_INTERVAL:
        _last_active_warn = now
        logger.warning(
            "EquiRoPE global RoPE patch is ACTIVE (blend=%.2f, threshold=%.1f, "
            "pole_gamma=%.1f, pole_side=%s). It applies to every LTX run in "
            "this ComfyUI session — bypassing or deleting the node does NOT "
            "disable it. Run the node with enabled=False or restart ComfyUI to "
            "restore stock RoPE.",
            _BLEND, _THRESHOLD, _POLE_GAMMA, _POLE_SIDE,
        )


def _collapse_grid(indices_grid, use_middle_indices_grid):
    """Same reduction as generate_freqs does internally."""
    if use_middle_indices_grid:
        return (indices_grid[..., 0] + indices_grid[..., 1]) / 2.0
    if indices_grid.dim() == 4:
        return indices_grid[..., 0]
    return indices_grid


# ---------------------------------------------------------------------------
# Core EquiRoPE
# ---------------------------------------------------------------------------

def _normalize_positions(pos):
    """Min-max normalize positions to [0, 1]. pos is [B, N]."""
    min_val = pos.amin(dim=-1, keepdim=True)
    max_val = pos.amax(dim=-1, keepdim=True)
    denom = max_val - min_val
    denom = denom + (denom <= 0).to(denom.dtype) * 1e-8
    return (pos - min_val) / denom  # [B, N], [0, 1]


def build_freqs_equirope(indices, indices_grid, max_pos, use_middle_indices_grid):
    """Replacement for generate_freqs with spectral-split EquiRoPE latitude scaling.

    Low-frequency channels (cycles < threshold) get |cos(lat)|-scaled positions.
    High-frequency channels (cycles >= threshold) keep stock position encoding.

    Args:
        indices: [K] frequency channel values (theta^(k/(K-1)) * pi/2).
        indices_grid: [B, 3, N, 2] or [B, 3, N] grid positions.
        max_pos: [3] max grid ranges.
        use_middle_indices_grid: bool.

    Returns:
        [B, N, 3K] flattened tensor, same as stock generate_freqs.
    """
    g = _collapse_grid(indices_grid, use_middle_indices_grid)  # [B, 3, N]
    indices = indices.to(device=g.device)
    K = indices.shape[-1]

    # Stock linear phases (axis 0 = time, 1 = height, 2 = width)
    frac = torch.stack([g[:, i] / max_pos[i] for i in range(3)], dim=-1)  # [B, N, 3]
    A = indices * (frac.unsqueeze(-1) * 2.0 - 1.0)  # [B, N, 3, K]

    if _BLEND == 0.0:
        return A.transpose(-1, -2).flatten(2)

    # ---- Compute latitude from normalized latent row index ----
    h_norm = _normalize_positions(g[:, 1])  # [B, N], [0, 1]

    lat = (h_norm * 2.0 - 1.0) * (math.pi / 2.0)  # [B, N], [-π/2, π/2]
    # Pole-gated scaling: gamma=1 is the classic |cos(lat)| profile; higher
    # gamma keeps mid-latitudes at ~1 (straight long horizontals) and only
    # collapses the rows near the poles.
    cos_lat = torch.abs(torch.cos(lat))
    scale = (1.0 - (1.0 - cos_lat) ** _POLE_GAMMA).clamp(min=_EPS)  # [B, N]

    # Hemisphere gating: lat is -pi/2 at the top rows (zenith), +pi/2 at the
    # bottom rows (nadir). The excluded side keeps stock positions (scale 1).
    if _POLE_SIDE != "both":
        keep = (lat > 0.0) if _POLE_SIDE == "bottom" else (lat < 0.0)
        scale = torch.where(keep, scale, torch.ones_like(scale))

    # ---- Spectral split via dynamic phase span ----
    # indices ARE the per-channel frequencies (phase = indices * (frac*2-1)),
    # so cycles across the width follow directly from the phase span.
    w_frac = frac[:, :, 2]  # [B, N], stock model-scaled width fractions
    width_span = (w_frac.amax() - w_frac.amin()) * 2.0  # scalar, resolution-agnostic
    cycles = (width_span * indices) / (2.0 * math.pi)   # [K]

    # Channel mask: True = low-frequency channels to EquiRoPE-scale
    scale_mask = (cycles < _THRESHOLD)  # [K]

    # ---- EquiRoPE symmetric scaling on low-frequency channels ----
    w_center = (w_frac.amax(dim=-1, keepdim=True) + w_frac.amin(dim=-1, keepdim=True)) / 2.0  # [B, 1]
    w_frac_scaled = (w_frac - w_center) * scale + w_center  # [B, N], symmetric toward center

    A_scaled = indices * (w_frac_scaled.unsqueeze(-1) * 2.0 - 1.0)  # [B, N, K]

    # Blend
    if _BLEND < 1.0:
        A_lin = A[:, :, 2, :]  # stock linear width angles
        A_scaled = (1.0 - _BLEND) * A_lin + _BLEND * A_scaled

    # Apply mask: latitude-scaled for low-freq, stock for high-freq
    A_hat = A.clone()
    A_hat[:, :, 2, scale_mask] = A_scaled[:, :, scale_mask]

    result = A_hat.transpose(-1, -2).flatten(2)

    if not torch.isfinite(result).all():
        print("[EquiRoPE] NaN detected, falling back to stock")
        return A.transpose(-1, -2).flatten(2)

    return result


# ---------------------------------------------------------------------------
# Install / uninstall
# ---------------------------------------------------------------------------

def make_wrapper(orig_generate_freqs):
    def generate_freqs_equirope(indices, indices_grid, max_pos, use_middle_indices_grid):
        if not _ENABLED or indices_grid.shape[1] != 3:
            return orig_generate_freqs(indices, indices_grid, max_pos, use_middle_indices_grid)
        _warn_active()
        return build_freqs_equirope(indices, indices_grid, max_pos, use_middle_indices_grid)
    generate_freqs_equirope._equirope_wrapper = True
    return generate_freqs_equirope


def install(model_module):
    """Monkeypatch model_module.generate_freqs. Idempotent."""
    if getattr(model_module, _SENTINEL, False):
        return 0
    orig = model_module.generate_freqs
    _ORIG[id(model_module)] = orig
    model_module.generate_freqs = make_wrapper(orig)
    setattr(model_module, _SENTINEL, True)
    return 1


def uninstall(model_module):
    if not getattr(model_module, _SENTINEL, False):
        return 0
    orig = _ORIG.pop(id(model_module), None)
    if orig is not None:
        model_module.generate_freqs = orig
    try:
        delattr(model_module, _SENTINEL)
    except AttributeError:
        pass
    return 1


# ---------------------------------------------------------------------------
# ComfyUI Node
# ---------------------------------------------------------------------------


class EquiRoPE:
    """Monkeypatches the LTX-2.3 transformer's RoPE with EquiRoPE
    latitude scaling for the width (longitude) axis.

    EquiRoPE scales each token's width position by a pole-gated latitude
    profile (1-(1-|cos lat|)^pole_gamma) on low-frequency channels
    (spectral split), collapsing polar tokens toward co-location so
    self-attention treats them as adjacent, while high-frequency channels
    keep stock positions for LoRA compatibility.

    blend=0: stock LTX RoPE
    blend=1: full EquiRoPE latitude scaling
    epsilon: minimum scale factor (prevents complete collapse at exact poles;
             0.01 means polar tokens retain 1% of width discrimination)
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
                "blend": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "0 = stock RoPE, 1 = full EquiRoPE latitude scaling.",
                }),
            },
            "optional": {
                "epsilon": ("FLOAT", {
                    "default": 0.01, "min": 0.001, "max": 0.5, "step": 0.001,
                    "tooltip": "Minimum latitude scaling factor. 0.01 = polar tokens "
                    "retain 1% of width discrimination. Higher = weaker collapse.",
                }),
                "threshold": ("FLOAT", {
                    "default": 10.0, "min": 0.0, "max": 100.0, "step": 0.5,
                    "tooltip": "Cycle threshold for spectral split. Channels with "
                    "fewer cycles across width get EquiRoPE (structure). Higher = "
                    "more channels affected. 10 = SpheRoPE-equivalent default.",
                }),
                "pole_gamma": ("FLOAT", {
                    "default": 3.0, "min": 1.0, "max": 8.0, "step": 0.5,
                    "tooltip": "Pole gating: scale = 1-(1-|cos lat|)^gamma. "
                    "1.0 = classic |cos| scaling, compresses width positions at "
                    "ALL latitudes (can bend long horizontals like rooftops). "
                    "3.0 = scaling stays ~1 until near the poles.",
                }),
                "pole_side": (["both", "bottom", "top"], {
                    "default": "both",
                    "tooltip": "Which pole(s) to collapse. 'bottom' = nadir "
                    "only: the whole upper hemisphere keeps stock positions "
                    "(straight skylines/rooftops), ground rows still collapse.",
                }),
            },
        }

    def patch(self, model, enabled, blend, epsilon=0.01, threshold=10.0,
              pole_gamma=3.0, pole_side="both"):
        from comfy.ldm.lightricks import model as _ltx_model

        install(_ltx_model)
        set_params(
            enabled=bool(enabled),
            blend=float(blend),
            epsilon=float(epsilon),
            threshold=float(threshold),
            pole_gamma=float(pole_gamma),
            pole_side=str(pole_side),
        )
        mode = "EquiRoPE" if enabled else "linear"
        logger.info(
            "EquiRoPE: enabled=%s blend=%s epsilon=%s cycle_thr=%s pole_gamma=%s "
            "pole_side=%s (W-axis: %s latitude scaling)",
            enabled, blend, epsilon, threshold, pole_gamma, pole_side, mode,
        )
        return (model,)


NODE_CLASS_MAPPINGS = {
    "EquiRoPE": EquiRoPE,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EquiRoPE": "EquiRoPE (latitude-scaled equatorial RoPE)",
}


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    B, N, K = 2, 20 * 40, 128 // 6
    vae_scale = 32.0
    _THRESHOLD = 10.0

    # Realistic grid: pixel positions for 20x40 latent with VAE scale 32
    h_pix = (torch.arange(20) * vae_scale).float().view(1, 20, 1).expand(1, 20, 40).reshape(1, N)
    w_pix = (torch.arange(40) * vae_scale).float().view(1, 1, 40).expand(1, 20, 40).reshape(1, N)
    max_pos = torch.tensor([8.0, h_pix.max().item(), w_pix.max().item()])
    indices_grid = torch.stack([torch.zeros_like(h_pix), h_pix, w_pix], dim=1).expand(B, -1, -1)

    # Same frequency ladder as comfy's generate_freq_grid: theta^(k/(K-1)) * pi/2
    indices = (10000.0 ** torch.linspace(0.0, 1.0, K)) * (math.pi / 2.0)
    use_middle = False

    for blend in [0.0, 0.5, 1.0]:
        _BLEND = blend
        result = build_freqs_equirope(indices, indices_grid, max_pos, use_middle)
        print(f"  blend={blend:.1f}: shape={tuple(result.shape)}, finite={torch.isfinite(result).all()}, "
              f"mean={result.mean():.4f}, std={result.std():.4f}")

    # ---- Verify spectral split ----
    _BLEND = 1.0
    _EPS = 0.01
    r = build_freqs_equirope(indices, indices_grid, max_pos, False)
    w_vals = r[0, :, 2::3]  # [N, K] — width axis: every 3rd entry
    w_img = w_vals.view(20, 40, K).permute(2, 0, 1)  # [K, 20, 40]

    # Compute how many channels got EquiRoPE at the poles
    top_std_per_ch = w_img[:, 0, :].std(dim=-1)  # [K] — std across W per channel
    eq_std_per_ch = w_img[:, 10, :].std(dim=-1)  # [K]
    collapsed = (top_std_per_ch < 0.5)            # collapsed at pole
    preserved = ~collapsed
    print(f"\n{int(collapsed.sum())}/{K} width-freq channels collapsed at poles")
    print(f"  Collapsed channels: {[int(i) for i in torch.where(collapsed)[0]]}")
    print(f"  Preserved channels (high-freq): {int(preserved.sum())}/{K}")

    # Verify equator has full range
    eq_std_mean = eq_std_per_ch[~collapsed].mean().item() if preserved.any() else 0
    print(f"  Equator, preserved channels: mean std={eq_std_mean:.4f} (should be > 2)")
    assert collapsed.any(), "No channels collapsed at pole — threshold too low"
    assert eq_std_mean > 2, "Preserved channels should still have wide range at equator"

    print("\nEquiRoPE: spectral split verified.")
    print("EquiRoPE: all checks passed.")
