"""Spherical width RoPE angle table for FLUX.2 (comfy-native).

Faithful port of the width computation in the reference
`build_sfc_spherical_rope_flux2` (orhir/SpheRoPE @ a40a86b, erp_utils.py). Returns the
width **half-angles** only — the caller builds comfy-format (2,2) rotation matrices and
overwrites the width block of `EmbedND`'s output. Mirrors `ltx23_width_angles.py`,
which likewise returns angles rather than cos/sin.

FLUX.2 uses a 4-axis RoPE ``[T, H, W, L]`` (axes_dim=[32,32,32,32], theta=2000). Only
the width axis is replaced, via a two-path spectral split:
  * Path A (channels < split_idx): NTK-aware requantized harmonic frequencies whose
    period divides the width, preserving exact horizontal periodicity.
  * Path B (channels >= split_idx): spherical Cartesian X/Y on the unit sphere scaled by
    W/scale, in even/odd slots. At the poles cos(lat)=0, so polar tokens share
    identical embeddings in this low-frequency subspace. Path A retains longitude
    dependence at the poles; the complete hybrid table does not guarantee polar
    convergence (an explicit tradeoff in the paper, section 3.2).
Computation is float64; the returned tensor is float64 (the patch casts).
"""
import math

import torch


def build_flux2_width_angles(H, W, w_dim=32, theta=2000.0, scale=2.0, max_error=0.06):
    """Return width RoPE half-angles of shape (H*W, w_dim//2), token order (h,w)."""
    H = int(H)
    W = int(W)
    w_half = w_dim // 2

    rows = torch.arange(H, dtype=torch.float64)
    cols = torch.arange(W, dtype=torch.float64)
    rows_grid, cols_grid = torch.meshgrid(rows, cols, indexing="ij")
    W_flat = cols_grid.reshape(-1)                       # (H*W,) width position per token

    # Spherical coordinates (latitude from row, longitude from column).
    theta_lat = (rows / max(H - 1, 1)) * math.pi - (math.pi / 2.0)
    phi_lon = (cols / W) * 2.0 * math.pi - math.pi
    phi_grid, theta_grid = torch.meshgrid(phi_lon, theta_lat, indexing="xy")
    r_width = W / scale
    x_spherical = (torch.cos(theta_grid) * torch.cos(phi_grid) + 1.0) * r_width
    y_spherical = (torch.cos(theta_grid) * torch.sin(phi_grid) + 1.0) * r_width
    x_flat = x_spherical.reshape(-1)
    y_flat = y_spherical.reshape(-1)

    # Width frequencies (same base as comfy's rope() omega for this axis).
    w_freqs = 1.0 / (theta ** (2.0 * torch.arange(w_half, dtype=torch.float64) / w_dim))
    fundamental_freq = (2.0 * math.pi) / W
    k_values = w_freqs / fundamental_freq
    rounded_k = torch.round(k_values)
    quantization_error = torch.abs(k_values - rounded_k) / (k_values + 1e-8)
    valid_mask = (k_values >= 1.0) & (quantization_error <= max_error)
    invalid_indices = torch.where(~valid_mask)[0]
    split_idx = invalid_indices[0].item() if len(invalid_indices) > 0 else len(w_freqs)
    step_mask = torch.arange(len(w_freqs)) < split_idx

    # Path A: NTK-aware requantized harmonic angles.
    harmonic_freqs = rounded_k * fundamental_freq
    n_path_a = split_idx
    if n_path_a > 1:
        a_indices = torch.arange(n_path_a, dtype=torch.float64)
        ntk_scale = scale ** (a_indices / (n_path_a - 1))
    else:
        ntk_scale = torch.ones(max(n_path_a, 1), dtype=torch.float64)
    scaled_freqs = harmonic_freqs[:split_idx] / ntk_scale[:split_idx]
    scaled_k = scaled_freqs / fundamental_freq
    requantized_k = torch.round(scaled_k).clamp(min=1.0)
    requantized_freqs = requantized_k * fundamental_freq

    linear_angles = W_flat.unsqueeze(1) * harmonic_freqs.unsqueeze(0)
    linear_angles[:, :split_idx] = W_flat.unsqueeze(1) * requantized_freqs.unsqueeze(0)

    # Path B: spherical 3-D angles (X on even slots, Y on odd slots).
    spherical_x_angles = x_flat.unsqueeze(1) * w_freqs[0::2].unsqueeze(0)
    spherical_y_angles = y_flat.unsqueeze(1) * w_freqs[1::2].unsqueeze(0)
    spherical_angles = torch.empty((H * W, w_half), dtype=torch.float64)
    spherical_angles[:, 0::2] = spherical_x_angles
    spherical_angles[:, 1::2] = spherical_y_angles

    w_angles = torch.where(step_mask.unsqueeze(0), linear_angles, spherical_angles)
    return w_angles
