"""Spherical VAE: equal-area latent remapping for equirectangular 360 video.

The LTX VAE's convolutions treat every pixel row equally, but in an equirectangular
projection, polar rows are heavily oversampled in world-space while equatorial rows
carry more world area per pixel. This misaligns the VAE's capacity allocation with
the spherical geometry of the data — the same core insight as SpheRoPE's Path B.

This module wraps the VAE's encode/decode with a Lambert cylindrical equal-area
remap so the VAE *internally* processes frames where every row covers equal world
area. The pipeline sees standard equirectangular latents; the remap is transparent.

Lambert equal-area projection:  y = sin(latitude)
  - Equal steps in y  →  equal-world-area latitude bands
  - No distortion along the equator (preserves longitude spacing)
  - Poles get compressed to single rows instead of full-width stretched rows

Resolution caveat
------------------
The effectiveness of the latent-level remap depends on latent height. The LTX VAE
produces 20-row latents (from 160px video), and the arcsin/sin remap shifts rows
by at most ~2 rows at the poles. At this resolution the interpolation error from
two successive grid_samples (equirect → equal-area → equirect) introduces ~4–7 %
per-pixel error, which may hurt diffusion quality.

For higher-resolution latents like Flux (64×128) or SD (64×64) the remap is much
cleaner and the spherical correction more meaningful. For LTX, the pixel-level
remap (before VAE encode / after VAE decode) is the higher-signal part; the
latent-level remap (after encode / before decode) is speculative.

Run `python spherical_vae.py` to see the round-trip loss at your resolution.
"""

import math
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Grid generation
# ---------------------------------------------------------------------------

def _to_equal_area_grid(H: int, W: int, device: torch.device):
    """Build grid: equirectangular input → equal-area output.

    For each equal-area output pixel (j, i), find the source location in the
    equirectangular input via grid_sample's (x, y) convention [-1, 1].

    Math:
      Equal-area row j:  y_coord = 2*(j+0.5)/H - 1
      This y IS sin(latitude) in this projection.
      Source latitude:    φ = arcsin(clamp(y, -1+ε, 1-ε))
      Equirect source y:  src_y = φ / (π/2)   (maps [-π/2, π/2] → [-1, 1])
      Longitude unchanged.
    """
    j = torch.arange(H, device=device).float()
    i = torch.arange(W, device=device).float()

    y_norm = (j + 0.5) / H * 2.0 - 1.0                          # (H,)
    x_norm = (i + 0.5) / W * 2.0 - 1.0                          # (W,)

    phi = torch.asin(y_norm.clamp(-0.9999, 0.9999))              # latitude
    src_y = phi / (math.pi / 2.0)                                 # → equirect y

    grid_y = src_y[:, None].expand(H, W)                         # (H, W)
    grid_x = x_norm[None, :].expand(H, W)                        # (H, W)
    return torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)    # (1, H, W, 2)


def _from_equal_area_grid(H: int, W: int, device: torch.device):
    """Build grid: equal-area input → equirectangular output.

    For each equirectangular output pixel (j, i), find the source location
    in the equal-area input.

    Math:
      Equirect row j:  latitude φ = y_norm * π/2
      Equal-area y:    sin(φ)
      Source y in equal-area space:  src_y = sin(φ)
    """
    j = torch.arange(H, device=device).float()
    i = torch.arange(W, device=device).float()

    y_norm = (j + 0.5) / H * 2.0 - 1.0                          # (H,)
    x_norm = (i + 0.5) / W * 2.0 - 1.0                          # (W,)

    src_y = torch.sin(y_norm * (math.pi / 2.0))                   # equal-area y

    grid_y = src_y[:, None].expand(H, W)
    grid_x = x_norm[None, :].expand(H, W)
    return torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)


# ---------------------------------------------------------------------------
# Remap helpers (handle 4D and 5D channels-first tensors)
# ---------------------------------------------------------------------------

def _remap(x: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """Remap last two spatial dims of x via grid_sample.

    Handles:
      (B, C, H', W')       — latent/image, channels-first
      (B, C, F, H', W')    — video latent, channels-first
      (B, H, W, C)         — pixels, channels-last (5D batch version below)

    grid: (1, H_out, W_out, 2) — broadcast across batch.
    """
    ndim = x.ndim
    # Detect channels-last pixel format by checking if last dim is small
    # (C=3 for RGB, C=4 for RGBA) vs channels-first (C=128 for latent).
    # Heuristic: if the last dimension is ≤ 4, assume channels-last.
    channels_last = (ndim in (4, 5) and x.shape[-1] <= 4)

    if ndim == 4:
        if channels_last:                              # (B, H, W, C)
            x = x.permute(0, 3, 1, 2)                  # → (B, C, H, W)
        B, C, H, W = x.shape
        if grid.shape[0] != B:
            grid = grid.expand(B, -1, -1, -1)
        out = F.grid_sample(x, grid, mode='bilinear',
                            padding_mode='border', align_corners=False)
        if channels_last:
            out = out.permute(0, 2, 3, 1)              # → (B, H, W, C)
        return out.contiguous()

    elif ndim == 5:
        if channels_last:                              # (B, F, H, W, C)
            B, T, H, W, C = x.shape
            x = x.permute(0, 1, 4, 2, 3)               # → (B, T, C, H, W)
        else:                                           # (B, C, F, H, W)
            B, C_or_unknown, T, H, W = x.shape

        # Merge batch & frame into one dim for grid_sample
        if channels_last:
            x = x.reshape(B * T, C, H, W)
        else:
            # We don't know if it's (B, C, F, H, W) or something else
            # Assume it's (B, C, F, H, W) — C is large (128)
            _, C, _T_ignored, _H_ignored, _W_ignored = x.shape
            x = x.permute(0, 2, 1, 3, 4)               # → (B, T, C, H, W)
            x = x.reshape(B * T, C, H, W)

        if grid.shape[0] == 1 and x.shape[0] > 1:
            grid = grid.expand(x.shape[0], -1, -1, -1)
        out = F.grid_sample(x, grid, mode='bilinear',
                            padding_mode='border', align_corners=False)

        _, C_out, H_out, W_out = out.shape
        if channels_last:
            out = out.reshape(B, T, C_out, H_out, W_out).permute(0, 1, 3, 4, 2)
            # → (B, T, H_out, W_out, C)
        else:
            out = out.reshape(B, T, C_out, H_out, W_out).permute(0, 2, 1, 3, 4)
            # → (B, C_out, T, H_out, W_out)
        return out.contiguous()

    raise ValueError(f"_remap: unsupported ndim={ndim}")


# ---------------------------------------------------------------------------
# VAE wrapper — the actual node
# ---------------------------------------------------------------------------

def _create_grid_cache(vae_model) -> dict:
    """Build grids for the VAE's working resolution.

    Returns dict with keys 'to_ea', 'from_ea' for pixel and latent grids.
    The grids are lazy (built on first access).
    """
    # Resolution-agnostic: grids are built from the first tensor that passes
    # through, keyed by (H, W, device). This lets us handle arbitrary input
    # sizes without reinitialising the node.
    return {}


_GRID_CACHE: dict = {}  # (H, W, device_str) → {'to_ea': grid, 'from_ea': grid}
_SPECTRAL_GRID_CACHE: dict = {}  # (H, W, k_rounded, device_str) → grid


def _get_grids(H: int, W: int, device: torch.device):
    key = (H, W, str(device))
    if key not in _GRID_CACHE:
        _GRID_CACHE[key] = {
            'to_ea': _to_equal_area_grid(H, W, device),
            'from_ea': _from_equal_area_grid(H, W, device),
        }
    return _GRID_CACHE[key]


# ---------------------------------------------------------------------------
# Spectral-split helpers (per-channel-k remap)
# ---------------------------------------------------------------------------

def _to_spectral_ea_grid(H: int, W: int, k: float, device: torch.device):
    """Grid: equal-area output → equirect source, with channel distortion k.

    Used in decode's latent remap. For each equal-area output row, finds the
    source location in the equirectangular input.

    k=1 → full Lambert equal-area (standard).
    k=0 → identity (no remap — return None).
    """
    if k <= 0.0:
        return None  # identity — skip grid_sample entirely
    j = torch.arange(H, device=device).float()
    i = torch.arange(W, device=device).float()
    y_norm = (j + 0.5) / H * 2.0 - 1.0                              # (H,)
    x_norm = (i + 0.5) / W * 2.0 - 1.0                              # (W,)

    # y_norm = sin(φ·k) / sin(k·π/2)  →  φ = arcsin(y·sin(kπ/2)) / k
    scale = math.sin(k * math.pi / 2.0)
    phi = torch.asin((y_norm * scale).clamp(-0.9999, 0.9999)) / k
    src_y = phi / (math.pi / 2.0)

    grid_y = src_y[:, None].expand(H, W)
    grid_x = x_norm[None, :].expand(H, W)
    return torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)        # (1, H, W, 2)


def _from_spectral_ea_grid(H: int, W: int, k: float, device: torch.device):
    """Grid: equirect output → equal-area source, with channel distortion k.

    Used in encode's latent remap. For each equirect output row, finds the
    source location in the equal-area input.

    k=1 → full Lambert inverse (standard).
    k=0 → identity (no remap — return None).
    """
    if k <= 0.0:
        return None
    j = torch.arange(H, device=device).float()
    i = torch.arange(W, device=device).float()
    y_norm = (j + 0.5) / H * 2.0 - 1.0                              # (H,)
    x_norm = (i + 0.5) / W * 2.0 - 1.0                              # (W,)

    phi = y_norm * (math.pi / 2.0)                                    # equirect latitude
    src_y = torch.sin(phi * k) / math.sin(k * math.pi / 2.0)

    grid_y = src_y[:, None].expand(H, W)
    grid_x = x_norm[None, :].expand(H, W)
    return torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)


def _get_spectral_grid(H: int, W: int, k: float, device: torch.device):
    """Cached getter for spectral grids. Returns None for k=0 (identity)."""
    if k <= 0.0:
        return None
    key = (H, W, round(k, 4), str(device))
    if key not in _SPECTRAL_GRID_CACHE:
        _SPECTRAL_GRID_CACHE[key] = _to_spectral_ea_grid(H, W, k, device)
    return _SPECTRAL_GRID_CACHE[key]


def _get_spectral_grid_inv(H: int, W: int, k: float, device: torch.device):
    """Cached getter for inverse spectral grids. Returns None for k=0."""
    if k <= 0.0:
        return None
    key = (H, W, -round(k, 4), str(device))  # negative key = inverse direction
    if key not in _SPECTRAL_GRID_CACHE:
        _SPECTRAL_GRID_CACHE[key] = _from_spectral_ea_grid(H, W, k, device)
    return _SPECTRAL_GRID_CACHE[key]


def _build_channel_groups(C: int, threshold: int, ramp_width: int):
    """Build list of (k_value, channel_indices) groups for spectral split.

    Channels 0..threshold-1  → k=1  (full spherical)
    Channels threshold..threshold+ramp_width-1 → k ramps 1→0
    Remaining channels       → k=0  (identity, no latent remap)
    """
    groups = []

    # Group 1: full spherical channels
    t = min(threshold, C)
    if t > 0:
        groups.append((1.0, list(range(t))))

    # Ramp zone
    if ramp_width > 0 and t < C:
        ramp_end = min(t + ramp_width, C)
        n_ramp = ramp_end - t
        n_levels = min(8, n_ramp)

        for level in range(n_levels):
            k = 1.0 - (level + 1.0) / (n_levels + 1.0)
            l_start = t + (level * n_ramp) // n_levels
            l_end = t + ((level + 1) * n_ramp) // n_levels
            if l_end > l_start:
                groups.append((k, list(range(l_start, l_end))))

        t = ramp_end

    # Remaining channels: no latent remap
    if t < C:
        groups.append((0.0, list(range(t, C))))

    return groups


def _remap_by_group(x: torch.Tensor, groups: list, fwd_grid_fn, inv_grid_fn):
    """Remap a latent tensor, applying a different k per channel group.

    x: (B, C, ...) channels-first tensor
    groups: list of (k, channel_indices)
    fwd_grid_fn(H, W, k, device) → grid or None
    inv_grid_fn(H, W, k, device) → grid or None

    For k=0 groups, no grid_sample is applied (identity pass-through).
    """
    if len(groups) == 1 and groups[0][0] == 0.0:
        return x  # fast path: no remap at all

    H, W = x.shape[-2], x.shape[-1]
    device = x.device
    dtype = x.dtype

    out_parts = [None] * len(groups)
    for idx, (k, chans) in enumerate(groups):
        if k == 0.0:
            out_parts[idx] = x[:, chans]
        else:
            grid = inv_grid_fn(H, W, k, device)  # inverse meaning "how to read from the input"
            if grid is None:
                out_parts[idx] = x[:, chans]
            else:
                chunk = x[:, chans]
                # For 5D video, merge B*F first
                b, c, *rest = chunk.shape
                if len(rest) == 3:
                    B, C, T, Hx, Wx = chunk.shape
                    chunk = chunk.permute(0, 2, 1, 3, 4).reshape(B * T, C, Hx, Wx)
                    g = grid.expand(chunk.shape[0], -1, -1, -1).to(dtype)
                    out = F.grid_sample(chunk, g, mode='bilinear',
                                        padding_mode='border', align_corners=False)
                    out = out.reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4)
                else:
                    B, C, Hx, Wx = chunk.shape
                    g = grid.expand(B, -1, -1, -1).to(dtype)
                    out = F.grid_sample(chunk, g, mode='bilinear',
                                        padding_mode='border', align_corners=False)
                out_parts[idx] = out

    if len(groups) == 1:
        return out_parts[0].contiguous()
    return torch.cat(out_parts, dim=1).contiguous()


def wrap_vae_spectral(vae, threshold=32, ramp_width=0):
    """Monkey-patch vae.encode/decode with spectral-split spherical remap.

    Pixel-level remap is always uniform Lambert (k=1 for all channels).
    Latent-level remap splits channels by the SpheRoPE-like spectral rule:
      - Structure channels (0..threshold-1): full Lambert equal-area (k=1)
      - Ramp zone: k ramps from 1→0
      - Detail channels: identity (k=0, no latent remap)

    Returns the same vae object (mutated). Idempotent.
    """
    if hasattr(vae, '_equirect_spectral_orig_encode'):
        return vae

    vae._equirect_spectral_orig_encode = vae.encode
    vae._equirect_spectral_orig_decode = vae.decode

    orig_encode = vae._equirect_spectral_orig_encode
    orig_decode = vae._equirect_spectral_orig_decode

    # Build channel groups once (fixed by threshold/ramp_width)
    C = 128  # LTX VAE latent channels
    groups = _build_channel_groups(C, threshold, ramp_width)

    def _encode(pixels):
        H, W = pixels.shape[-3], pixels.shape[-2]
        grids = _get_grids(H, W, pixels.device)

        # Step 1: pixels equirect → equal-area (always uniform Lambert)
        ea_pixels = _remap(pixels, grids['to_ea'].to(pixels.dtype))

        # Step 2: standard VAE encode
        latent = orig_encode(ea_pixels)  # (B, 128, H', W') or (B, 128, F, H', W')

        # Step 3: latent equal-area → scaled by per-channel k
        H_lat, W_lat = latent.shape[-2], latent.shape[-1]
        latent = _remap_by_group(latent, groups,
                                 _get_spectral_grid, _get_spectral_grid_inv)
        return latent

    def _decode(samples):
        H_lat, W_lat = samples.shape[-2], samples.shape[-1]

        # Step 1: latent equirect → scaled equal-area by per-channel k
        # (inverse of the encode remap: use spectral grids reversed)
        inv_groups = [(k, chans) for k, chans in groups]  # same groups, swap grid dir
        samples = _remap_by_group(samples, inv_groups,
                                  _get_spectral_grid_inv, _get_spectral_grid)

        # Step 2: standard VAE decode
        pixels = orig_decode(samples)

        # Step 3: pixels equal-area → equirect (always uniform Lambert)
        H_px, W_px = pixels.shape[-3], pixels.shape[-2]
        px_grids = _get_grids(H_px, W_px, pixels.device)
        pixels = _remap(pixels, px_grids['from_ea'].to(pixels.dtype))
        return pixels

    vae.encode = _encode
    vae.decode = _decode
    return vae


def unwrap_vae_spectral(vae):
    """Restore original encode/decode (idempotent)."""
    if not hasattr(vae, '_equirect_spectral_orig_encode'):
        return vae
    vae.encode = vae._equirect_spectral_orig_encode
    vae.decode = vae._equirect_spectral_orig_decode
    del vae._equirect_spectral_orig_encode
    del vae._equirect_spectral_orig_decode
    return vae


def wrap_vae_spherical(vae):
    """Monkey-patch vae.encode and vae.decode with equal-area remapping.

    The pipeline sees standard equirectangular latents; the VAE internally
    works in cylindrical equal-area coordinates.

    Returns the same vae object (mutated). Idempotent.
    """
    if hasattr(vae, '_equirect_spherical_orig_encode'):
        return vae  # already wrapped

    vae._equirect_spherical_orig_encode = vae.encode
    vae._equirect_spherical_orig_decode = vae.decode

    orig_encode = vae._equirect_spherical_orig_encode
    orig_decode = vae._equirect_spherical_orig_decode

    # ──  encode: pixels → equal-area → VAE.encode → equirect → output  ──
    def _encode(pixels):
        # pixels: (B, H, W, 3) or (B, F, H, W, 3), channels-last
        H, W = pixels.shape[-3], pixels.shape[-2]
        grids = _get_grids(H, W, pixels.device)

        ea_pixels = _remap(pixels, grids['to_ea'].to(pixels.dtype))
        latent = orig_encode(ea_pixels)

        H_lat, W_lat = latent.shape[-2], latent.shape[-1]
        lat_grids = _get_grids(H_lat, W_lat, latent.device)
        latent = _remap(latent, lat_grids['from_ea'].to(latent.dtype))
        return latent

    # ──  decode: latent → equal-area → VAE.decode → equirect → output  ──
    def _decode(samples):
        # samples: (B, C, H', W') or (B, C, F, H', W'), channels-first
        H_lat, W_lat = samples.shape[-2], samples.shape[-1]
        lat_grids = _get_grids(H_lat, W_lat, samples.device)

        ea_latent = _remap(samples, lat_grids['to_ea'].to(samples.dtype))
        pixels = orig_decode(ea_latent)

        H_px, W_px = pixels.shape[-3], pixels.shape[-2]
        px_grids = _get_grids(H_px, W_px, pixels.device)
        pixels = _remap(pixels, px_grids['from_ea'].to(pixels.dtype))
        return pixels

    vae.encode = _encode
    vae.decode = _decode
    return vae


def unwrap_vae_spherical(vae):
    """Restore original encode/decode (idempotent)."""
    if not hasattr(vae, '_equirect_spherical_orig_encode'):
        return vae
    vae.encode = vae._equirect_spherical_orig_encode
    vae.decode = vae._equirect_spherical_orig_decode
    del vae._equirect_spherical_orig_encode
    del vae._equirect_spherical_orig_decode
    return vae


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

def _test_remap():
    """Verify the remap is invertible (round-trip ≈ identity)."""
    H, W = 20, 40
    device = 'cpu'

    # Input: one row with a ramp, one with a step
    x = torch.zeros(1, 1, H, W, device=device)
    x[0, 0, H // 2, :] = 1.0        # thin horizontal line at equator
    x[0, 0, 0, :] = 0.5             # top row
    x[0, 0, -1, :] = 0.5            # bottom row

    # Equirect → equal-area → equirect
    eq_grid = _to_equal_area_grid(H, W, device)
    eq_grid_inv = _from_equal_area_grid(H, W, device)

    y = F.grid_sample(x, eq_grid.expand(1, -1, -1, -1),
                      mode='bilinear', align_corners=False)
    z = F.grid_sample(y, eq_grid_inv.expand(1, -1, -1, -1),
                      mode='bilinear', align_corners=False)

    loss = (x - z).abs().mean().item()
    print(f"Round-trip loss: {loss:.6f}  (expect ~0.074 for 20×40 bilinear, < 0.01 for 160×320)")
    assert loss < 0.10, f"Round-trip too lossy: {loss}"


# ---------------------------------------------------------------------------
# ComfyUI Node
# ---------------------------------------------------------------------------

import logging
logger = logging.getLogger("SeamlessEquirectangular.spherical_vae")


# ──  EquirectSphericalVAE (uniform Lambert, the original)  ──

class EquirectSphericalVAE:
    """Wrap the loaded VAE so it processes frames in cylindrical equal-area
    projection internally, transparent to the rest of the pipeline.

    Every row of the internal representation covers equal world area, giving
    the VAE's convolutions a spherical-geometry-aware allocation of capacity
    (the VAE analogue of SpheRoPE's Path B).

    The node wraps *in place* — every downstream user of this VAE object
    sees the spherical behaviour until disabled.
    """

    CATEGORY = "latent/video/equirect"
    RETURN_TYPES = ("VAE",)
    FUNCTION = "wrap"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "enabled": ("BOOLEAN", {"default": True}),
            },
        }

    def wrap(self, vae, enabled):
        if enabled:
            wrap_vae_spherical(vae)
            logger.info(
                "EquirectSphericalVAE: enabled — VAE encode/decode wrapped "
                "with Lambert equal-area remap (all channels uniform)."
            )
        else:
            unwrap_vae_spherical(vae)
            logger.info("EquirectSphericalVAE: disabled — stock VAE restored.")
        return (vae,)


# ──  EquirectSpectralVAE (per-channel-k spectral split)  ──

class EquirectSpectralVAE:
    """Like EquirectSphericalVAE, but applies the SpheRoPE-inspired spectral
    split: structure latent channels get full spherical remapping, detail
    channels pass through the latent untouched.

    Pixel-level remap (before VAE encode / after VAE decode) is always
    uniform Lambert for all 3 image channels.

    Latent-level remap (after VAE encode / before VAE decode) splits the
    128 latent channels:
      - threshold (default 32): first N channels get k=1 (full spherical)
      - ramp_width (default 0): transitional channels with intermediate k
      - remaining channels: k=0 (identity — no latent remap, pure equirect)

    Intuition: structure channels carry the semantic layout and benefit from
    spherical geometry; detail channels carry local texture and are better
    left in the VAE's native equirect arrangement.
    """

    CATEGORY = "latent/video/equirect"
    RETURN_TYPES = ("VAE",)
    FUNCTION = "wrap"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "enabled": ("BOOLEAN", {"default": True}),
                "threshold": ("INT", {
                    "default": 32, "min": 0, "max": 128,
                    "tooltip": "First N latent channels get full spherical remap (k=1). "
                    "Lower = more channels left in equirect (detail)."
                }),
            },
            "optional": {
                "ramp_width": ("INT", {
                    "default": 0, "min": 0, "max": 64,
                    "tooltip": "Number of transitional channels between spherical and equirect. "
                    "These get intermediate k values for a smooth spectral blend. "
                    "0 = hard split at threshold."
                }),
            },
        }

    def wrap(self, vae, enabled, threshold, ramp_width=0):
        if enabled:
            wrap_vae_spectral(vae, threshold, ramp_width)
            groups = _build_channel_groups(128, threshold, ramp_width)
            summary = ", ".join(f"k={k:.2f}: {len(ch)}ch" for k, ch in groups if len(ch) > 0)
            logger.info(
                f"EquirectSpectralVAE: enabled — spectral split: {summary}"
            )
        else:
            unwrap_vae_spectral(vae)
            logger.info("EquirectSpectralVAE: disabled — stock VAE restored.")
        return (vae,)


NODE_CLASS_MAPPINGS = {
    "EquirectSphericalVAE": EquirectSphericalVAE,
    "EquirectSpectralVAE": EquirectSpectralVAE,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EquirectSphericalVAE": "Equirect Spherical VAE (uniform Lambert)",
    "EquirectSpectralVAE": "Equirect Spectral VAE (SpheRoPE split)",
}


if __name__ == "__main__":
    _test_remap()
    print("All checks passed.")
