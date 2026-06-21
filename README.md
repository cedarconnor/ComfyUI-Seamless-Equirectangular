# Seamless-Equirectangular

**Seamless-Equirectangular** is a ComfyUI custom node pack by **Burgstall Labs** for seamless equirectangular, 360-degree, VR, and panoramic video inference. It provides training-free seam mitigation for LTX2.3 VR-Outpaint and a reusable adapter structure for applying the same circular-longitude inference method to other diffusion and video generation models.

This repository focuses on one specific problem: equirectangular panoramas wrap horizontally, but most diffusion model inference treats the left and right image edges as unrelated frame boundaries. That mismatch creates visible seams, discontinuities, dark edge bands, and broken 360-degree video loops. Seamless-Equirectangular makes the model see different longitudes as interior content during denoising, then restores the original orientation.

## Authorship And Provenance

This is Burgstall Labs' open-source implementation of seamless equirectangular inference for LTX2.3 VR-Outpaint workflows. To the author's knowledge, this is the first public open-source ComfyUI implementation of this specific per-step circular latent rolling method for seamless 360-degree video generation with LTX2.3.

It follows Burgstall Labs' earlier VR-outpainting work: first practical ComfyUI tooling and LoRA workflows for video VR-outpainting, now extended with a dedicated seam-removal method for equirectangular inference.

Keywords: ComfyUI seamless equirectangular, LTX2.3 VR-Outpaint, 360 video diffusion, VR outpainting, equirectangular seam fix, panoramic video generation, circular latent rolling, circular VAE padding, Burgstall Labs.

## What It Does

Seamless-Equirectangular provides three nodes:

- **Equirect Per-Step Latent Roll (360 seam)** (`MODEL -> MODEL`) circularly rolls the video latent along width around every model call, then unrolls the prediction. This is the core seamless equirectangular inference method.
- **Equirect Circular VAE (360 seam)** (`VAE -> VAE`) patches VAE convolutions to use circular padding along longitude. This reduces decode-side seam artifacts.
- **Equirect Latent Seam Blend (fallback)** (`LATENT -> LATENT`) cross-fades a thin latent band at the width edge. This is a conservative fallback for cosmetic edge smoothing.

Node IDs, categories, behavior, and defaults are intentionally stable for existing tested workflows:

- `EquirectPerStepRoll`
- `EquirectCircularVAE`
- `EquirectLatentSeamBlend`

## Why Equirectangular Inference Needs This

An equirectangular panorama is periodic along longitude: column `0` and the final column represent adjacent directions in the real spherical scene. Standard image and video diffusion inference does not know that. The model sees the panorama as a flat rectangle, so its convolutional and attention behavior can treat the seam as a hard image border.

The result is familiar in 360-degree AI video work:

- a visible vertical seam at the longitude wrap,
- mismatched objects or lighting across the left/right edge,
- dark bands or vignette-like artifacts at processing boundaries,
- temporal instability when the camera pans through the seam,
- decoded VAE artifacts caused by non-periodic padding.

The per-step latent roll node attacks the source of the problem during sampling. Instead of only blending the final image, it changes where the model's artificial boundary lands at each denoising step.

## Core Method: Per-Step Circular Latent Rolling

`EquirectPerStepRoll` wraps the ComfyUI model object and intercepts each model call during sampling:

1. Detect the video latent layout.
2. Roll the video latent along the width / longitude axis.
3. Roll `denoise_mask` in sync when present.
4. Run the model on the shifted latent.
5. Optionally feather the known processing boundary in the prediction.
6. Unroll the prediction back to the original longitude.

For LTX2.3, the node supports both plain 5D video latents and packed audio/video latents. Audio tokens are left untouched.

Modes and defaults are unchanged from the tested implementation:

- `random_per_step` is the default. It uses deterministic per-sigma shifts keyed by sigma and seed.
- `dual_average` evaluates two roughly opposite shifted views per step and averages them. This can reduce boundary/vignette banding at roughly 2x sampling cost.
- `half_turn` is a diagnostic mode that uses a constant half-width shift.
- `feather_columns` defaults to `0` and can be set to a small value such as `2-4` for boundary-local mitigation.

## LTX2.3 VR-Outpaint Support

The current tested layouts are:

- plain video latent: `[B, C, T, H, W]`
- LTX packed audio/video latent: `[B, 1, video_numel + audio_numel]`, using `c["latent_shapes"][0]` as the video shape

For LTX2.3 VR-Outpaint workflows, start with:

- **Equirect Per-Step Latent Roll** before sampling.
- `mode=random_per_step` first.
- `dual_average` when seam and banding quality matter more than sampling speed.
- **Equirect Circular VAE** before decode, usually with decoder patching enabled.

This node pack is intentionally separate from projection and camera-calibration tooling. Use it alongside VR-outpainting prep workflows when the model is already operating on an equirectangular canvas or latent.

## Circular VAE Padding

`EquirectCircularVAE` patches loaded VAE convolution modules so width padding is circular while height and time remain non-periodic. This is important because longitude wraps, but poles and time do not.

The implementation is designed to preserve ComfyUI's model loading and weight-casting behavior by wrapping convolution forwards rather than replacing model classes.

Important warning: ComfyUI caches VAEs by file. `EquirectCircularVAE` modifies the loaded VAE object in place, so every consumer of that same loaded VAE file in a run sees the same patched or restored state. The node resets to stock before applying its selected state, but true stock-vs-circular A/B testing needs separate runs or separate loaded VAE objects.

## Extending To Other Models

Model-specific latent packing lives in `seamless_equirect/latent_layouts.py`. The current adapter boundary was added so the LTX2.3 method can be extended without changing the tested node defaults.

To support another model, add an adapter that can:

- detect the model IO tensor layout,
- report the equirectangular longitude width,
- roll only the video or image portion along longitude,
- leave audio, text, metadata, or other non-video tokens unchanged.

Only enable per-step rolling for models where latent width is a real longitude axis. Packed token layouts that do not preserve spatial width need a dedicated adapter before they are safe to use.

Potential targets include other ComfyUI video diffusion wrappers, panoramic image diffusion workflows, VR180-to-360 expansion workflows, and future native equirectangular video models.

## Installation

Clone into `ComfyUI/custom_nodes`:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Burgstall-labs/Seamless-Equirectangular
```

Restart ComfyUI. The nodes appear under existing equirectangular / model patch categories:

- `model_patches/equirect`
- `latent/video/equirect`

No extra runtime dependency is required beyond the torch runtime already used by ComfyUI.

## Tests

Run with the ComfyUI Python environment:

```bash
/home/juho/ComfyUI/venv/bin/python tests/test_seam_nodes.py
```

The tests cover:

- circular Conv2d / Conv3d patch and unpatch behavior,
- VAE encoder/decoder scoping,
- deterministic per-sigma shift generation,
- LTX packed audio/video latent rolling,
- per-step model wrapper round trips,
- `dual_average` behavior,
- latent seam blending.

## Related Burgstall Labs Work

Burgstall Labs develops practical open-source tools for 360-degree AI video, VR-outpainting, panoramic generation, and ComfyUI-based production workflows. Seamless-Equirectangular is intended to be the reusable seam-inference layer that complements projection, outpainting, and LoRA workflows for immersive video.

