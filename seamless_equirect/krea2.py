"""Krea2 equirectangular position encoding helpers and nodes."""

import logging
import math

import torch

try:
    import comfy.model_management
    from comfy.ldm.flux.math import rope as flux_rope
    from comfy.ldm.krea2.model import SingleStreamDiT
except ImportError:
    comfy = None
    flux_rope = None
    SingleStreamDiT = None


logger = logging.getLogger("SeamlessEquirectangular.krea2")


def is_krea2_model(diffusion_model) -> bool:
    if diffusion_model is None:
        return False
    if SingleStreamDiT is not None and isinstance(diffusion_model, SingleStreamDiT):
        return True
    cls = diffusion_model.__class__
    return (
        (cls.__name__ == "SingleStreamDiT" and cls.__module__ == "comfy.ldm.krea2.model")
        or getattr(diffusion_model, "is_krea2", False)
    )


def periodic_width_rope(pos, dim, theta, periods, periodic_mask, blend):
    """Flux-style RoPE with width frequencies snapped to whole cycles."""
    if blend == 0.0:
        return flux_rope(pos, dim, theta)

    out_device = pos.device
    device = out_device if comfy.model_management.supports_fp64(out_device) else torch.device("cpu")
    scale = torch.linspace(0, (dim - 2) / dim, steps=dim // 2, dtype=torch.float64, device=device)
    stock_omega = 1.0 / (theta ** scale)

    periods = periods.to(device=device, dtype=torch.float64).clamp_min(1.0)
    cycles = periods.unsqueeze(-1) * stock_omega.view(1, 1, -1) / (2.0 * math.pi)
    max_harmonic = torch.floor(periods.unsqueeze(-1) / 2.0).clamp_min(1.0)
    harmonic = torch.minimum(torch.round(cycles).clamp_min(1.0), max_harmonic)
    periodic_omega = 2.0 * math.pi * harmonic / periods.unsqueeze(-1)

    pos_calc = pos.to(device=device, dtype=torch.float64).unsqueeze(-1)
    stock_angles = pos_calc * stock_omega.view(1, 1, -1)
    periodic_angles = pos_calc * periodic_omega
    mix = periodic_mask.to(device=device, dtype=torch.float64).unsqueeze(-1) * float(blend)
    angles = stock_angles + mix * (periodic_angles - stock_angles)

    rotations = torch.stack(
        (torch.cos(angles), -torch.sin(angles), torch.sin(angles), torch.cos(angles)),
        dim=-1,
    ).reshape(*angles.shape, 2, 2)
    return rotations.to(device=out_device, dtype=torch.float32)


def _target_grid_info(ids):
    """Find Krea2's main image token span without mistaking zero-ID text for image."""
    if ids.ndim != 3 or ids.shape[-1] < 3:
        raise ValueError("Krea2 position IDs must have shape [batch, tokens, 3].")

    batch, token_count, _ = ids.shape
    positions = torch.arange(token_count, device=ids.device).view(1, -1).expand(batch, -1)
    main_index = ids[..., 0] == 0

    # Krea2 concatenates text, main-image, then reference-image tokens. Reference
    # image IDs use index > 0, so their first occurrence marks the main grid end.
    grid_ends = torch.where(
        ids[..., 0] > 0,
        positions,
        torch.full_like(positions, token_count),
    ).amin(dim=-1, keepdim=True)
    before_refs = main_index & (positions < grid_ends)

    minus_one = torch.full_like(ids[..., 1], -1.0)
    heights = torch.where(before_refs, ids[..., 1], minus_one).amax(dim=-1, keepdim=True) + 1.0
    widths = torch.where(before_refs, ids[..., 2], minus_one).amax(dim=-1, keepdim=True) + 1.0
    heights = heights.clamp_min(1.0)
    widths = widths.clamp_min(1.0)

    grid_sizes = torch.round(heights * widths).to(dtype=torch.long)
    grid_starts = grid_ends - grid_sizes
    valid = (grid_starts >= 0) & (grid_sizes > 0)
    target_mask = (
        valid
        & main_index
        & (positions >= grid_starts)
        & (positions < grid_ends)
    )
    return target_mask, heights, widths


def spherical_width_rope(
    pos,
    latitude,
    dim,
    theta,
    widths,
    heights,
    target_mask,
    seam_blend,
    geometry_blend,
    low_frequency_fraction,
    sphere_scale,
):
    """Krea2/Flux SpheRoPE split: periodic detail plus spherical structure."""
    if seam_blend == 0.0 and geometry_blend == 0.0:
        return flux_rope(pos, dim, theta)

    out_device = pos.device
    device = out_device if comfy.model_management.supports_fp64(out_device) else torch.device("cpu")
    pair_count = dim // 2
    scale = torch.linspace(
        0, (dim - 2) / dim, steps=pair_count, dtype=torch.float64, device=device
    )
    # Krea2 passes absolute grid IDs straight to Flux RoPE, so its native
    # descending frequency ladder is the stable scale for spherical positions.
    # LTX's normalized generate_freqs path needs a different ladder.
    omega = 1.0 / (theta ** scale)

    widths = widths.to(device=device, dtype=torch.float64).clamp_min(1.0)
    heights = heights.to(device=device, dtype=torch.float64).clamp_min(1.0)
    pos_calc = pos.to(device=device, dtype=torch.float64)
    lat_calc = latitude.to(device=device, dtype=torch.float64)
    mask = target_mask.to(device=device).unsqueeze(-1)

    stock_angles = pos_calc.unsqueeze(-1) * omega.view(1, 1, -1)

    cycles = widths.unsqueeze(-1) * omega.view(1, 1, -1) / (2.0 * math.pi)
    max_harmonic = torch.floor(widths.unsqueeze(-1) / 2.0).clamp_min(1.0)
    harmonic = torch.minimum(torch.round(cycles).clamp_min(1.0), max_harmonic)
    periodic_omega = 2.0 * math.pi * harmonic / widths.unsqueeze(-1)
    periodic_angles = pos_calc.unsqueeze(-1) * periodic_omega
    seam_mix = mask.to(dtype=torch.float64) * float(seam_blend)
    angles = stock_angles + seam_mix * (periodic_angles - stock_angles)

    # Radius W/(2*pi) makes motion along the equator locally match Krea2's
    # original unit-spaced width coordinate. At the poles cos(latitude) sends
    # every longitude to the same point.
    height_denominator = (heights - 1.0).clamp_min(1.0)
    lat = lat_calc / height_denominator * math.pi - math.pi / 2.0
    lon = pos_calc / widths * (2.0 * math.pi) - math.pi
    radius = widths / (2.0 * math.pi) * float(sphere_scale)
    radial = torch.cos(lat) * radius
    sphere_x = radial * torch.cos(lon)
    sphere_y = radial * torch.sin(lon)

    channel_indices = torch.arange(pair_count, device=device)
    spherical_positions = torch.where(
        (channel_indices % 2).view(1, 1, -1) == 0,
        sphere_x.unsqueeze(-1),
        sphere_y.unsqueeze(-1),
    )
    spherical_angles = spherical_positions * omega.view(1, 1, -1)

    low_count = int(round(pair_count * float(low_frequency_fraction)))
    low_count = max(0, min(pair_count, low_count))
    low_frequency = channel_indices >= pair_count - low_count
    geometry_mix = (mask & low_frequency.view(1, 1, -1)).to(
        dtype=torch.float64
    ) * float(geometry_blend)
    angles = angles + geometry_mix * (spherical_angles - angles)

    rotations = torch.stack(
        (torch.cos(angles), -torch.sin(angles), torch.sin(angles), torch.cos(angles)),
        dim=-1,
    ).reshape(*angles.shape, 2, 2)
    return rotations.to(device=out_device, dtype=torch.float32)


class Krea2PeriodicEmbedND(torch.nn.Module):
    def __init__(self, dim, theta, axes_dim, blend):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = list(axes_dim)
        self.blend = float(blend)

    def forward(self, ids):
        target_mask, _, periods = _target_grid_info(ids)

        embeddings = []
        for axis, axis_dim in enumerate(self.axes_dim):
            if axis == 2:
                emb = periodic_width_rope(
                    ids[..., axis], axis_dim, self.theta, periods, target_mask, self.blend
                )
            else:
                emb = flux_rope(ids[..., axis], axis_dim, self.theta)
            embeddings.append(emb)
        return torch.cat(embeddings, dim=-3).unsqueeze(1)


class Krea2SpheEmbedND(torch.nn.Module):
    """Flux-compatible SpheRoPE for Krea2's main image grid only."""

    def __init__(
        self,
        dim,
        theta,
        axes_dim,
        blend,
        geometry_blend,
        low_frequency_fraction,
        sphere_scale,
    ):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = list(axes_dim)
        self.blend = float(blend)
        self.geometry_blend = float(geometry_blend)
        self.low_frequency_fraction = float(low_frequency_fraction)
        self.sphere_scale = float(sphere_scale)

    def forward(self, ids):
        target_mask, heights, widths = _target_grid_info(ids)

        embeddings = []
        for axis, axis_dim in enumerate(self.axes_dim):
            if axis == 2:
                emb = spherical_width_rope(
                    ids[..., axis],
                    ids[..., 1],
                    axis_dim,
                    self.theta,
                    widths,
                    heights,
                    target_mask,
                    self.blend,
                    self.geometry_blend,
                    self.low_frequency_fraction,
                    self.sphere_scale,
                )
            else:
                emb = flux_rope(ids[..., axis], axis_dim, self.theta)
            embeddings.append(emb)
        return torch.cat(embeddings, dim=-3).unsqueeze(1)


class EquirectKrea2RoPE:
    """Patch Krea2 with periodic detail RoPE and spherical structural RoPE."""

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
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Periodic seam strength for the width axis. Start at 0.5.",
                }),
            },
            "optional": {
                "geometry_blend": ("FLOAT", {
                    "default": 0.4,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Spherical low-frequency geometry strength. Raise gradually for straighter architecture.",
                }),
                "low_frequency_fraction": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Fraction of the lowest-frequency width channels assigned to spherical geometry.",
                }),
                "sphere_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.25,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "Spherical coordinate radius. 1.0 preserves Krea2's local equatorial position scale.",
                }),
            },
        }

    def patch(
        self,
        model,
        enabled,
        blend,
        geometry_blend=0.4,
        low_frequency_fraction=0.5,
        sphere_scale=1.0,
    ):
        if not enabled or (blend == 0.0 and geometry_blend == 0.0):
            return (model,)

        diffusion_model = None
        if hasattr(model, "get_model_object"):
            diffusion_model = model.get_model_object("diffusion_model")
        if diffusion_model is None and hasattr(model, "model"):
            diffusion_model = getattr(model.model, "diffusion_model", model.model)
        if diffusion_model is None and hasattr(model, "diffusion_model"):
            diffusion_model = model.diffusion_model

        if not is_krea2_model(diffusion_model):
            raise ValueError("EquirectKrea2RoPE requires a Krea2 diffusion model.")

        embedder = diffusion_model.pe_embedder
        patched_embedder = Krea2SpheEmbedND(
            embedder.dim,
            embedder.theta,
            embedder.axes_dim,
            blend,
            geometry_blend,
            low_frequency_fraction,
            sphere_scale,
        )
        m = model.clone()
        m.add_object_patch("diffusion_model.pe_embedder", patched_embedder)
        logger.info(
            "EquirectKrea2RoPE: SpheRoPE enabled "
            "(seam=%.2f, geometry=%.2f, low_freq=%.2f, scale=%.2f)",
            blend,
            geometry_blend,
            low_frequency_fraction,
            sphere_scale,
        )
        return (m,)


NODE_CLASS_MAPPINGS = {
    "EquirectKrea2RoPE": EquirectKrea2RoPE,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EquirectKrea2RoPE": "Equirect Krea2 SpheRoPE (360 geometry + seam)",
}
