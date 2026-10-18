"""Spherical-width RoPE patch for comfy-native FLUX.2.

Object-patches `diffusion_model.pe_embedder` on a cloned ModelPatcher. The wrapper calls
comfy's stock `EmbedND.forward(ids)` to get the rotation-matrix embeddings, then
overwrites ONLY the width block (rotation pairs [32:48] for axes_dim=[32,32,32,32]) of the
generation-image rows with spherical angles from `build_flux2_width_angles`. T/H/L blocks
and all non-image rows stay bit-identical to stock. Analogue of the LTX width-only patch.

v1 = text-to-panorama only: the wrapper delegates unchanged whenever the call is not a
4-axis Flux.2 call or reference/conditioning tokens (T>0) are present.
"""
import torch

from .flux2_angles import build_flux2_width_angles

FLUX_PE_KEY = "diffusion_model.pe_embedder"


class _SphericalFluxEmbedND(torch.nn.Module):
    """Wraps a comfy `EmbedND`, overriding the width rotation block with spherical RoPE."""

    def __init__(self, orig, scale=2.0, max_error=0.06):
        super().__init__()
        self.orig = orig
        self.axes_dim = list(orig.axes_dim)
        self.theta = float(orig.theta)
        self.scale = float(scale)
        self.max_error = float(max_error)
        self.w_start = sum(self.axes_dim[:2]) // 2
        self.w_end = self.w_start + self.axes_dim[2] // 2
        self._cache: dict = {}

    def _width_block(self, H, W, device, dtype):
        key = (H, W)
        angles = self._cache.get(key)
        if angles is None:
            angles = build_flux2_width_angles(
                H, W, w_dim=self.axes_dim[2], theta=self.theta,
                scale=self.scale, max_error=self.max_error,
            )
            self._cache[key] = angles
        a = angles.to(device=device, dtype=torch.float32)
        cos, sin = torch.cos(a), torch.sin(a)
        block = torch.stack([cos, -sin, sin, cos], dim=-1).reshape(*a.shape, 2, 2)
        return block.to(dtype)

    def forward(self, ids):
        emb = self.orig(ids)
        # Guard: only 4-axis Flux.2, text-to-panorama (no reference tokens).
        if len(self.axes_dim) != 4 or ids.shape[-1] != 4:
            return emb
        if float(ids[..., 0].max()) > 0.0:
            return emb
        H = int(round(float(ids[..., 1].max()))) + 1
        W = int(round(float(ids[..., 2].max()))) + 1
        n_img = H * W
        n_tokens = ids.shape[-2]
        if n_img <= 0 or n_img > n_tokens:
            return emb
        # Verify the trailing n_img rows really are a 0-based integral, row-major HxW
        # image grid before overwriting. comfy's process_img adds rope_options
        # shift/scale (from transformer_options) that slide the image ids off 0 or make
        # them non-integral; recovering H/W from the global max would then inflate n_img
        # and slide this write onto text-token width blocks. Delegate rather than
        # mis-write. (Matches build_flux2_width_angles' t = row*W + col token order.)
        img_ids = ids[..., -n_img:, :]
        exp_h = torch.arange(H, device=ids.device, dtype=ids.dtype).repeat_interleave(W)
        exp_w = torch.arange(W, device=ids.device, dtype=ids.dtype).repeat(H)
        if not (torch.allclose(img_ids[..., 1], exp_h.expand_as(img_ids[..., 1]), atol=1e-3)
                and torch.allclose(img_ids[..., 2], exp_w.expand_as(img_ids[..., 2]), atol=1e-3)):
            return emb
        block = self._width_block(H, W, emb.device, emb.dtype)     # (n_img, w_half, 2, 2)
        emb = emb.clone()
        emb[..., -n_img:, self.w_start:self.w_end, :, :] = block
        return emb


def install_flux_spherical_rope(patcher, scale=2.0, max_error=0.06):
    """Clone `patcher`, object-patch the Flux.2 `pe_embedder` with the spherical wrapper,
    and return the clone. Restore is automatic via ModelPatcher unpatch."""
    patched = patcher.clone()
    dm = patched.get_model_object("diffusion_model")
    pe = getattr(dm, "pe_embedder", None)
    if pe is None or len(getattr(pe, "axes_dim", [])) != 4:
        raise ValueError(
            "Equirect Flux: this MODEL is not a 4-axis FLUX.2 transformer "
            "(expected pe_embedder.axes_dim=[32,32,32,32]). "
            "EquirectSphericalRoPEFlux targets FLUX.2 only."
        )
    wrapper = _SphericalFluxEmbedND(pe, scale=scale, max_error=max_error)
    patched.add_object_patch(FLUX_PE_KEY, wrapper)
    return patched
