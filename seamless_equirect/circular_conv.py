"""Circular (cylindrical) latent encoding for equirectangular 360 panoramas.

Implements the "Circular Latent Encoding" idea from 360Anything (arXiv 2601.16192):
the convolution-based VAE uses zero-padding, which creates boundary artifacts at the
longitude wrap and produces a visible seam in the generated panorama. The fix is to
pad the VAE convolutions *circularly along width (longitude)* while keeping height
(the poles -- not periodic) and time padded as before.

We do NOT modify comfy core. Instead this module copies the relevant padding logic
from comfy's LTX VAE convs (comfy/ldm/lightricks/vae/{causal_conv3d,conv_nd_factory,
dual_conv3d}.py) and applies it to an already-loaded VAE at runtime by wrapping each
conv leaf's forward so the input is circular-padded on W before the (zero-padded)
convolution runs. Wrapping forward -- rather than replacing it -- preserves comfy's
weight-casting / fp8 / offload paths in comfy.ops.disable_weight_init.Conv{2,3}d.

The width amount we pre-pad equals the conv's original W padding, and we zero that
conv's W padding so the height/time padding it does internally is unchanged. Net
effect: identical output size, circular continuity across the longitude seam.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

_SENTINEL = "_equirect_circular_patched"


# ---------------------------------------------------------------------------
# Conv2d / Conv3d leaf patching (covers CausalConv3d.conv and standalone convs)
# ---------------------------------------------------------------------------
# torch padding_mode -> F.pad mode
_FPAD_MODE = {"zeros": "constant", "reflect": "reflect", "replicate": "replicate", "circular": "circular"}


def _patch_conv_leaf(conv: nn.Module) -> bool:
    """Wrap a Conv2d/Conv3d instance to circular-pad its width axis only.

    Conv3d padding is (pt, ph, pw); Conv2d padding is (ph, pw). We do ALL padding
    manually: W circularly, and H/T with the conv's ORIGINAL padding_mode, then set
    the conv to do no padding (padding=0, padding_mode="zeros"). This is critical for
    convs whose padding_mode is not "zeros" (e.g. the LTX *decoder* uses "reflect"):
    for those, nn.Conv ignores self.padding and uses a frozen _reversed_padding from
    __init__, so merely zeroing self.padding[W] does NOT stop it padding W -> the input
    would get double-padded on W (size mismatch). Neutralizing padding_mode avoids that
    and keeps W the ONLY axis whose behavior changes (orig mode -> circular).
    """
    if getattr(conv, _SENTINEL, False):
        return False

    padding = tuple(int(p) for p in conv.padding)
    pw = padding[-1]
    if pw == 0:
        return False  # 1x1 projection or no width padding -- nothing to wrap

    orig_mode = getattr(conv, "padding_mode", "zeros")
    hmode = _FPAD_MODE.get(orig_mode, "constant")
    orig_forward = conv.forward  # bound method (preserves any cast-weights path)

    if len(padding) == 3:
        pt, ph, _pw = padding

        def circular_forward(input, *args, **kwargs):
            # F.pad order is (W_l, W_r, H_l, H_r, T_l, T_r)
            input = F.pad(input, (_pw, _pw, 0, 0, 0, 0), mode="circular")
            if ph:
                input = F.pad(input, (0, 0, ph, ph, 0, 0), mode=hmode)
            if pt:
                input = F.pad(input, (0, 0, 0, 0, pt, pt), mode=hmode)
            return orig_forward(input, *args, **kwargs)
    else:
        ph, _pw = padding

        def circular_forward(input, *args, **kwargs):
            input = F.pad(input, (_pw, _pw, 0, 0), mode="circular")
            if ph:
                input = F.pad(input, (0, 0, ph, ph), mode=hmode)
            return orig_forward(input, *args, **kwargs)

    conv._equirect_orig_padding = padding
    conv._equirect_orig_padding_mode = orig_mode
    conv._equirect_orig_forward = orig_forward
    conv.padding = tuple(0 for _ in padding)  # conv itself pads nothing now
    conv.padding_mode = "zeros"               # stop the frozen reflect/replicate double-pad
    conv.forward = circular_forward
    setattr(conv, _SENTINEL, True)
    return True


def _unpatch_conv_leaf(conv: nn.Module) -> bool:
    if not getattr(conv, _SENTINEL, False):
        return False
    conv.padding = conv._equirect_orig_padding
    conv.padding_mode = conv._equirect_orig_padding_mode
    # Remove the instance-level forward override so the class method takes over.
    if "forward" in conv.__dict__:
        del conv.__dict__["forward"]
    else:  # defensive fallback
        conv.forward = conv._equirect_orig_forward
    del conv._equirect_orig_padding
    del conv._equirect_orig_padding_mode
    del conv._equirect_orig_forward
    delattr(conv, _SENTINEL)
    return True


# ---------------------------------------------------------------------------
# DualConv3d patching (the dims==(2,1) factorized-conv VAE variant)
# ---------------------------------------------------------------------------
def _is_dualconv3d(module: nn.Module) -> bool:
    # Match by structure to avoid importing comfy internals.
    return type(module).__name__ == "DualConv3d" and hasattr(module, "padding1")


def _patch_dualconv3d(dc: nn.Module) -> bool:
    if getattr(dc, _SENTINEL, False):
        return False
    p1 = tuple(int(p) for p in dc.padding1)  # (0, ph, pw) -- spatial conv padding
    pw = p1[2]
    if pw == 0:
        return False
    orig_forward = dc.forward

    def circular_forward(x, use_conv3d=False, skip_time_conv=False):
        x = F.pad(x, (pw, pw, 0, 0, 0, 0), mode="circular")
        return orig_forward(x, use_conv3d=use_conv3d, skip_time_conv=skip_time_conv)

    dc._equirect_orig_padding1 = dc.padding1
    dc._equirect_orig_forward = orig_forward
    dc.padding1 = (p1[0], p1[1], 0)  # drop W pad from the spatial conv
    dc.forward = circular_forward
    setattr(dc, _SENTINEL, True)
    return True


def _unpatch_dualconv3d(dc: nn.Module) -> bool:
    if not getattr(dc, _SENTINEL, False):
        return False
    dc.padding1 = dc._equirect_orig_padding1
    if "forward" in dc.__dict__:
        del dc.__dict__["forward"]
    else:
        dc.forward = dc._equirect_orig_forward
    del dc._equirect_orig_padding1
    del dc._equirect_orig_forward
    delattr(dc, _SENTINEL)
    return True


# ---------------------------------------------------------------------------
# VAE-level walk
# ---------------------------------------------------------------------------
def _in_scope(name: str, encoder: bool, decoder: bool) -> bool:
    """Decide whether a module path should be patched given encoder/decoder flags."""
    in_enc = name == "encoder" or name.startswith("encoder.")
    in_dec = name == "decoder" or name.startswith("decoder.")
    if in_enc:
        return encoder
    if in_dec:
        return decoder
    # Top-level convs (e.g. quant_conv / post_quant_conv) -- patch if either side on.
    return encoder or decoder


def patch_vae_circular(vae_model: nn.Module, encoder: bool = True, decoder: bool = True) -> int:
    """Patch a live LTX VAE nn.Module for circular-W padding. Returns #convs patched.

    Idempotent: already-patched convs are skipped. Pass the underlying module
    (vae.first_stage_model), not the comfy VAE wrapper.
    """
    patched = 0
    for name, module in vae_model.named_modules():
        if not _in_scope(name, encoder, decoder):
            continue
        if isinstance(module, (nn.Conv2d, nn.Conv3d)):
            patched += _patch_conv_leaf(module)
        elif _is_dualconv3d(module):
            patched += _patch_dualconv3d(module)
    return patched


def unpatch_vae_circular(vae_model: nn.Module) -> int:
    """Reverse patch_vae_circular. Returns #convs restored. Safe to call always."""
    restored = 0
    for _, module in vae_model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Conv3d)):
            restored += _unpatch_conv_leaf(module)
        elif _is_dualconv3d(module):
            restored += _unpatch_dualconv3d(module)
    return restored


# ---------------------------------------------------------------------------
# Optional latent-space seam blend (fallback belt-and-suspenders)
# ---------------------------------------------------------------------------
def seam_blend_w(out: torch.Tensor, b: int) -> torch.Tensor:
    """Nudge the two longitude edge columns toward their mean, fading inward over
    `b` columns, to enforce C0 continuity across the W wrap. Operates on the last
    (width) axis of an arbitrary-rank tensor. Returns a new tensor."""
    if b <= 0:
        return out
    W = out.shape[-1]
    b = min(int(b), W // 2)
    if b <= 0:
        return out
    out = out.clone()
    left_edge = out[..., :1]
    right_edge = out[..., -1:]
    target = 0.5 * (left_edge + right_edge)
    ramp = torch.linspace(1.0, 0.0, steps=b + 1, device=out.device, dtype=out.dtype)[:b]
    shape = [1] * (out.dim() - 1) + [b]
    ramp_l = ramp.view(*shape)            # weight high near column 0
    ramp_r = ramp.flip(-1).view(*shape)   # weight high near column W-1
    out[..., :b] = out[..., :b] + (target - left_edge) * ramp_l
    out[..., -b:] = out[..., -b:] + (target - right_edge) * ramp_r
    return out
