# Seamless-Equirectangular

ComfyUI custom nodes for seamless equirectangular / 360-degree inference.

This repo contains the seam-fix nodes originally tested for LTX2.3 VR-Outpaint,
with the LTX latent layout kept behind an adapter boundary so the same approach
can be extended to other models whose latent width maps cleanly to equirectangular
longitude.

## Nodes

- **Equirect Per-Step Latent Roll (360 seam)** (`MODEL -> MODEL`) circularly rolls
  the video latent along width around every model call, then unrolls the prediction.
  Modes and defaults are unchanged from the tested LTX2.3 implementation:
  `random_per_step` is the default, `dual_average` evaluates two roughly opposite
  shifts per step, and `half_turn` is diagnostic. `feather_columns` defaults to `0`.
- **Equirect Circular VAE (360 seam)** (`VAE -> VAE`) patches VAE convolutions to
  use circular padding along longitude. It patches the loaded VAE object in place.
- **Equirect Latent Seam Blend (fallback)** (`LATENT -> LATENT`) cross-fades a thin
  band at the latent width edge. `blend_columns` defaults to `2`.

## LTX2.3 Notes

The current tested layouts are:

- plain video latent: `[B, C, T, H, W]`
- LTX packed audio/video latent: `[B, 1, video_numel + audio_numel]`, using
  `c["latent_shapes"][0]` as the video shape. Audio tokens are passed through
  untouched, and `denoise_mask` is rolled in sync when present.

Recommended starting point for LTX2.3:

- Use **Equirect Per-Step Latent Roll** before sampling.
- Keep `mode=random_per_step` first.
- Use `dual_average` when the per-step boundary/vignette banding matters more
  than the roughly 2x sampling cost.
- Use **Equirect Circular VAE** on decode, usually with decoder patching enabled.

## VAE Patch Warning

ComfyUI caches VAEs by file. **Equirect Circular VAE** modifies the loaded VAE
object in place, so every consumer of that same loaded VAE file in a run sees
the same patched or restored state. The node resets to stock before applying its
selected state, but true stock-vs-circular A/B testing needs separate runs or
separate loaded VAE objects.

## Extending To Other Models

Model-specific latent packing lives in `seamless_equirect/latent_layouts.py`.
To support another model, add an adapter that can:

- detect the model IO tensor layout,
- report the equirectangular longitude width,
- roll only the video/image part along longitude,
- leave non-video tokens unchanged.

Only enable the per-step roll for models where latent width is a real longitude
axis. Packed token layouts that do not preserve spatial width need a dedicated
adapter before they are safe to use.

## Tests

Run with the ComfyUI Python environment:

```bash
/home/juho/ComfyUI/venv/bin/python tests/test_seam_nodes.py
```

