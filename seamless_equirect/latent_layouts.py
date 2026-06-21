"""Latent layout adapters for seamless equirectangular model IO.

The current tested behavior supports the LTX sampling layouts:
- plain video latent: [B, C, T, H, W]
- packed AV latent: [B, 1, video_numel + audio_numel], with video_shape from
  c["latent_shapes"][0]

This module keeps those semantics intact while making the model-specific part
explicit for future adapters.
"""

import math

import torch


def video_shape_from_conditioning(c):
    shapes = c.get("latent_shapes", None)
    if shapes:
        return list(shapes[0])
    return None


class PlainVideo5DLayout:
    name = "plain_video_5d"

    def detect(self, x: torch.Tensor, video_shape=None) -> bool:
        return x.ndim == 5

    def width(self, x: torch.Tensor, video_shape=None) -> int:
        return int(x.shape[-1])

    def roll(self, x: torch.Tensor, video_shape, shift: int):
        return torch.roll(x, shifts=shift, dims=-1)


class LTXPackedAVLayout:
    name = "ltx_packed_av"

    def detect(self, x: torch.Tensor, video_shape=None) -> bool:
        if x.ndim != 3 or video_shape is None or len(video_shape) != 5:
            return False
        n = int(math.prod([int(s) for s in video_shape[1:]]))
        return n <= x.shape[-1]

    def width(self, x: torch.Tensor, video_shape=None) -> int:
        return int(video_shape[-1])

    def roll(self, x: torch.Tensor, video_shape, shift: int):
        n = int(math.prod([int(s) for s in video_shape[1:]]))
        b = x.shape[0]
        vid = x[:, :, :n].reshape([b] + [int(s) for s in video_shape[1:]])
        vid = torch.roll(vid, shifts=shift, dims=-1).reshape(b, x.shape[1], n)
        return torch.cat([vid, x[:, :, n:]], dim=-1)


_LAYOUTS = (PlainVideo5DLayout(), LTXPackedAVLayout())


def find_layout(x: torch.Tensor, video_shape=None):
    for layout in _LAYOUTS:
        if layout.detect(x, video_shape):
            return layout
    return None


def roll_model_io(x: torch.Tensor, video_shape, shift: int):
    """Circularly roll the video part of a model input/output along W.

    This preserves the original helper behavior: shift 0 returns the exact input
    object, recognized layouts return a rolled tensor, and unknown layouts return
    None so callers can pass through unmodified.
    """
    if shift == 0:
        return x
    layout = find_layout(x, video_shape)
    if layout is None:
        return None
    return layout.roll(x, video_shape, shift)

