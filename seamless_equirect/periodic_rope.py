"""Spherical Width-axis RoPE for LTX-2.3 — SpheRoPE two-path spectral split.
Copyright (C) 2026  Burgstall Labs

Licensed under the PolyForm Noncommercial License 1.0.0.
https://polyformproject.org/licenses/noncommercial/1.0.0

SpheRoPE-style two-path spectral split for the width (longitude) axis,
based on the ideas described in:
  SpheRoPE: Spherical Positional Encoding for 360° Panorama Generation
  https://arxiv.org/abs/2606.32033

SpheRoPE-style two-path spectral split for the width (longitude) axis:
  Path A (high-freq channels with >= path_b_threshold cycles): harmonic snapping
  Path B (low-freq channels): spherical Cartesian (X, Y) coordinates

The Path B uses a SEPARATE frequency vector from the original RoPE indices:
  freq_vec[k] = theta^(k/(D-1)) * pi/2
This is NOT the same as the RoPE indices (theta^(-k/3D)) — using indices
would produce wrong magnitudes and crash the GPU.

Pure torch, no comfy/musubi imports.
"""

import logging
import math
import time

import torch

logger = logging.getLogger("SeamlessEquirectangular.periodic_rope")

# ---- module state (set by install/set_params) -------------------------------
_ORIG = {}          # module id -> original generate_freqs (for uninstall)
_ENABLED = False
_ALPHA = 1.0        # 0 = linear (identity), 1 = fully spherical
_W_SCALE = 32.0     # W VAE spatial scale factor
_SENTINEL = "_periodic_w_rope_installed"
_HARM_CACHE = {}    # (K, W_count) -> cpu float tensor [K]

# The monkeypatch is global module state: it survives node removal/bypass. Warn
# on every sampling run so silent carry-over can't skew A/B comparisons.
_WARN_INTERVAL = 60.0
_last_active_warn = 0.0

# SpheRoPE parameters
_PATH_B_THRESHOLD = 10.0   # cycles threshold for Path B vs Path A
_R_SCALE = 1.0             # spherical coordinate magnitude scaling
_THETA = 10000.0           # RoPE theta


def set_params(enabled: bool = True, alpha: float = 1.0, w_scale: float = 32.0,
               path_b_threshold: float = 10.0, r_scale: float = 1.0):
    global _ENABLED, _ALPHA, _W_SCALE, _PATH_B_THRESHOLD, _R_SCALE
    _ENABLED = bool(enabled)
    _ALPHA = float(alpha)
    _W_SCALE = float(w_scale)
    _PATH_B_THRESHOLD = float(path_b_threshold)
    _R_SCALE = float(r_scale)


def periodic_harmonics(K: int, W_count: int, device, dtype) -> torch.Tensor:
    """Log-spaced integer harmonic ladder over [1, floor(W_count/2)] (Nyquist)."""
    M = max(1, W_count // 2)
    key = (int(K), int(W_count))
    h = _HARM_CACHE.get(key)
    if h is None:
        if M == 1:
            h = torch.ones(K, dtype=torch.float64)
        else:
            ladder = torch.exp(torch.linspace(math.log(1.0), math.log(float(M)), K, dtype=torch.float64))
            h = ladder.round().clamp(1.0, float(M))
        _HARM_CACHE[key] = h
    return h.to(device=device, dtype=dtype)


def _collapse_grid(indices_grid: torch.Tensor, use_middle_indices_grid: bool) -> torch.Tensor:
    """Mirror generate_freqs' grid reduction to a 3D [B,3,N] tensor."""
    if use_middle_indices_grid:
        return (indices_grid[..., 0] + indices_grid[..., 1]) / 2.0
    if indices_grid.dim() == 4:
        return indices_grid[..., 0]
    return indices_grid


def _recover_w(p: torch.Tensor, w_scale=None):
    """Recover integer latent column index w and W_count from the W pixel coord p."""
    wf = p / (_W_SCALE if w_scale is None else w_scale)
    wf = wf - wf.amin(dim=-1, keepdim=True)
    w = torch.round(wf)
    W_count = int(w.amax().item()) + 1
    w = w.clamp(0, max(W_count - 1, 0))
    return w, W_count


# ---------------------------------------------------------------------------
# SpheRoPE: build 2-path width angles on the full (H, W) grid
# ---------------------------------------------------------------------------

def _build_width_angles(W_count: int, H_count: int, freqs_per_dim: int,
                        device, dtype, *, w_scale=None, r_scale=None,
                        path_b_threshold=None, base_width=2048.0) -> torch.Tensor:
    """Build spherical-aware width angles for the full (H, W) latent grid.

    Follows SpheRoPE's _build_ltx23_width_angles exactly.

    Args:
        W_count, H_count: Latent grid dimensions.
        freqs_per_dim: K = inner_dim // 6 (number of frequency channels per axis).
        device, dtype: Output device and dtype.

    Returns:
        (H*W, freqs_per_dim) float32 tensor of width RoPE angles.
        Path A (high-freq): harmonic-snapped angles for seam continuity.
        Path B (low-freq): spherical Cartesian (X,Y) for pole convergence.
    """
    # Use float64 for precision (SpheRoPE does this)
    fd = torch.float64
    w_scale = _W_SCALE if w_scale is None else w_scale
    r_scale = _R_SCALE if r_scale is None else r_scale
    path_b_threshold = _PATH_B_THRESHOLD if path_b_threshold is None else path_b_threshold
    theta = _THETA  # 10000.0

    # ---- Frequency vector (SpheRoPE's formula, NOT the same as RoPE indices) ----
    # freq_vec[k] = theta^(k/(D-1)) * pi/2   (ascending, exponential)
    pow_indices = theta ** torch.linspace(0.0, 1.0, freqs_per_dim, dtype=fd, device=device)
    freq_vec = pow_indices * (math.pi / 2.0)  # [K]

    # ---- width_span ----
    width_span = (W_count * w_scale / base_width) * 2.0  # scalar

    # ---- Stock positions (1D per column) ----
    rows = torch.arange(H_count, dtype=fd, device=device)
    cols = torch.arange(W_count, dtype=fd, device=device)

    # Normalized width coordinate [-1, 1] for each column
    col_grid = ((cols + 0.5) * w_scale / base_width) * 2.0 - 1.0  # [W]
    col_frac = cols / W_count  # [W], fraction of width in [0, 1)
    # Tile across all H*W tokens
    col_grid_flat = col_grid.unsqueeze(0).expand(H_count, -1).reshape(-1)  # [H*W]
    col_frac_flat = col_frac.unsqueeze(0).expand(H_count, -1).reshape(-1)  # [H*W]

    # ---- Path A: corrected (harmonic-snapped) angles ----
    # Stock linear angles
    stock_angles = col_grid_flat.unsqueeze(-1) * freq_vec.unsqueeze(0)  # [H*W, K]

    # Snap to integer cycles
    total_angle = width_span * freq_vec  # [K] — total phase across full width
    cycles = total_angle / (2.0 * math.pi)  # how many 2pi cycles
    rounded_cycles = torch.round(cycles)
    residual = total_angle - rounded_cycles * (2.0 * math.pi)  # correction per channel
    corrected_angles = stock_angles - col_frac_flat.unsqueeze(-1) * residual.unsqueeze(0)

    # ---- Path B: spherical Cartesian (X, Y) coordinates ----
    lat = (rows / max(H_count - 1, 1)) * math.pi - (math.pi / 2.0)  # [H], [-pi/2, pi/2]
    lon = (cols / W_count) * 2.0 * math.pi - math.pi  # [W], [-pi, pi]

    # SpheRoPE uses indexing="xy" for meshgrid
    lon_grid, lat_grid = torch.meshgrid(lon, lat, indexing="xy")  # both [H, W]

    R_width = width_span / 2.0 * r_scale  # scalar
    X_sph = (torch.cos(lat_grid) * torch.cos(lon_grid) * R_width).reshape(-1)  # [H*W]
    Y_sph = (torch.cos(lat_grid) * torch.sin(lon_grid) * R_width).reshape(-1)  # [H*W]

    spherical_x = X_sph.unsqueeze(-1) * freq_vec[0::2].unsqueeze(0)  # [H*W, even_K]
    spherical_y = Y_sph.unsqueeze(-1) * freq_vec[1::2].unsqueeze(0)  # [H*W, odd_K]
    spherical_angles = torch.empty((H_count * W_count, freqs_per_dim), dtype=fd, device=device)
    spherical_angles[:, 0::2] = spherical_x
    spherical_angles[:, 1::2] = spherical_y

    # ---- Selection: Path A for high-freq, Path B for low-freq ----
    use_corrected = (cycles >= path_b_threshold)  # [K], bool
    width_angles = torch.where(use_corrected.unsqueeze(0), corrected_angles, spherical_angles)

    return width_angles.to(dtype=dtype)


# ---------------------------------------------------------------------------
# Main construction
# ---------------------------------------------------------------------------

def build_freqs_periodic(indices, indices_grid, max_pos, use_middle_indices_grid,
                         *, alpha=None, w_scale=None, path_b_threshold=None, r_scale=None):
    """Reimplementation of generate_freqs with SpheRoPE spherical W.

    alpha=0 reproduces original generate_freqs byte-exact.
    alpha=1 applies SpheRoPE two-path spectral split.
    """
    alpha = _ALPHA if alpha is None else alpha
    w_scale = _W_SCALE if w_scale is None else w_scale
    g = _collapse_grid(indices_grid, use_middle_indices_grid)            # [B,3,N]
    indices = indices.to(device=g.device)
    K = indices.shape[-1]

    # linear phases for all 3 axes (exact original formula)
    frac = torch.stack([g[:, i] / max_pos[i] for i in range(3)], dim=-1)  # [B,N,3]
    A = indices * (frac.unsqueeze(-1) * 2.0 - 1.0)                         # [B,N,3,K]

    if alpha == 0.0:
        return A.transpose(-1, -2).flatten(2)

    # ---- Recover grid dimensions ----
    w, W_count = _recover_w(g[:, 2], w_scale)                              # [B,N]
    # Time and appended guide frames must not become extra latitude rows.
    h, H_count = _recover_w(g[:, 1], w_scale)

    phi_linear = A[:, :, 2, :]                                            # [B,N,K]

    # ---- Pre-compute SpheRoPE width angles for this grid size ----
    try:
        width_angles = _build_width_angles(W_count, H_count, K,
            device=g.device, dtype=A.dtype, w_scale=w_scale, r_scale=r_scale,
            path_b_threshold=path_b_threshold, base_width=max_pos[2])
        width_angles_tiled = width_angles[(h * W_count + w).long()]

        # Blend between linear and spherical
        A_hat = A.clone()
        A_hat[:, :, 2, :] = (1.0 - alpha) * phi_linear + alpha * width_angles_tiled

        result = A_hat.transpose(-1, -2).flatten(2)
        if not torch.isfinite(result).all():
            raise RuntimeError("NaN/Inf in spherical RoPE output, falling back to harmonic")
        return result
    except Exception as e:
        print(f"[periodic_rope] SpheRoPE Path B failed ({e}), using harmonic Path A only")
        # Fall back to Path A only
        w = _recover_w(g[:, 2], w_scale)[0]
        m = periodic_harmonics(K, W_count, device=g.device, dtype=A.dtype)
        phi_periodic = m * (2.0 * math.pi / W_count) * w.unsqueeze(-1)
        A_hat = A.clone()
        A_hat[:, :, 2, :] = (1.0 - alpha) * phi_linear + alpha * phi_periodic
        return A_hat.transpose(-1, -2).flatten(2)


# ---------------------------------------------------------------------------
# Wrapper and install
# ---------------------------------------------------------------------------

def _warn_active():
    global _last_active_warn
    now = time.monotonic()
    if now - _last_active_warn >= _WARN_INTERVAL:
        _last_active_warn = now
        logger.warning(
            "SpheRoPE global RoPE patch is ACTIVE (alpha=%.2f). It applies to "
            "every LTX run in this ComfyUI session — bypassing or deleting the "
            "node does NOT disable it. Run the node with enabled=False or "
            "restart ComfyUI to restore stock RoPE.",
            _ALPHA,
        )


def make_wrapper(orig_generate_freqs):
    def generate_freqs_periodic(indices, indices_grid, max_pos, use_middle_indices_grid):
        if not _ENABLED or indices_grid.shape[1] != 3:
            return orig_generate_freqs(indices, indices_grid, max_pos, use_middle_indices_grid)
        _warn_active()
        return build_freqs_periodic(indices, indices_grid, max_pos, use_middle_indices_grid)
    generate_freqs_periodic._periodic_wrapper = True
    return generate_freqs_periodic


def install(model_module, w_scale: float = 32.0):
    """Monkeypatch model_module.generate_freqs (comfy or musubi rope module). Idempotent."""
    global _W_SCALE
    _W_SCALE = float(w_scale)
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
