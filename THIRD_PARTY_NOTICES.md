# SpheRoPE integration

The ERP guidance, circular processing, FLUX.2 spherical width encoding and
diagnostics were adapted from [cedarconnor/ComfyUI-SpheRoPE](https://github.com/cedarconnor/ComfyUI-SpheRoPE)
at commit `c8148da13c1284df99df6a7b1f5ac0fdd737c8ba` and are now included directly
in `seamless_equirect`. The companion package is not a runtime dependency.

That implementation credits the [original SpheRoPE research code](https://github.com/orhir/SpheRoPE)
(CC BY-NC 4.0). In particular, `flux2_angles.py` follows its spherical width
angle calculation. This integration does not remove upstream attribution or
noncommercial restrictions. This repository's PolyForm Noncommercial license
and the model weights' own licenses still apply.
