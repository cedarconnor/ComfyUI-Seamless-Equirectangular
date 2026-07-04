# ComfyUI-Seamless-Equirectangular

**ComfyUI-Seamless-Equirectangular** is a ComfyUI custom node pack by **Burgstall Labs** for seamless equirectangular, 360-degree, VR, and panoramic video generation with LTX-2.3. It provides training-free seam mitigation and geometry enhancement for equirectangular inference.

This is the result of extensive experimentation: training a dedicated equirectangular LoRA, then developing circular VAE padding and per-step latent rolling to handle the seam — then implementing the spherical positional encoding and wrapped noise ideas from the SpheRoPE paper (https://arxiv.org/abs/2606.32033) on top of that foundation. The combined pipeline produces seamless 360° video that neither approach achieves alone.

The LoRA is available on Hugging Face: [TheBurgstall/Seamless-Equirectangular-LTX2.3-LoRA](https://huggingface.co/TheBurgstall/Seamless-Equirectangular-LTX2.3-LoRA)

**Important:** This LoRA was trained with the default zero-padded VAE. **Do not use `--circular_vae`** at inference if using musubi-tuner. In ComfyUI, the `EquirectCircularVAE` node is fine — it patches the VAE decoder only, which doesn't conflict with the LoRA's training.

**Keywords:** ComfyUI seamless equirectangular, LTX-2.3 360 video, SpheRoPE, circular VAE padding, per-step latent roll, panoramic video generation, Burgstall Labs.

## Nodes

| Node | Type | What it does |
|---|---|---|
| **EquirectCircularVAE** | `VAE → VAE` | Patches VAE decoder convolutions to use circular padding along the width (longitude) axis. Fixes the decode-side seam caused by zero-padded convolutions at the 360° wrap boundary. |
| **EquirectPerStepRoll** | `MODEL → MODEL` | Wraps the diffusion model so every denoising step runs on a W-rolled latent. Each step sees a different longitudinal shift, so no column is always at the model's processing boundary — forces content coherent across the wrap. Based on the Diffusion360 / PanoDiffusion circular shift method. |
| **EquirectLatentSeamBlend** | `LATENT → LATENT` | Thin cross-fade of the W-edge latent columns. Cosmetic fallback — smooths the boundary band but does not make content wrap. Off by default. |
| **EquirectWrappedNoise** | `NOISE → NOISE` | SpheRoPE-style spherical phase remaster. Modifies the initial noise latent's low-frequency phase to encode polar convergence, reducing stretching at the top and bottom poles. |

## History

### Pre-SpheRoPE development

I started with training a dedicated equirectangular LoRA — a rank-128 LoRA for 15k steps (April 2026). The model cannot produce text-to-video equirectangular output without it; the frozen 22B base model's rectilinear bias is too strong even with inference-time tricks.

With the LoRA producing equirectangular video, two problems remained:

1. **The longitude seam**: column 0 and column W-1 represent adjacent directions in the real spherical scene, but the model treats them as unrelated edges. Standard zero-padded VAE convolutions amplify this with a visible seam at decode time — even WITH the LoRA.

2. **Polar stretching**: the equirectangular projection compresses the poles into single rows, but the model has no inductive bias for this. Sometimes this would result in an "egg-shaped" projection and void zones at the top and bottom — again, even with the LoRA (the base model has clearly seen equirectangular footage during training, but not enough to overcome the problem consistently).

The first solutions were:

- **Circular VAE padding** (`EquirectCircularVAE`): wrapping each VAE convolution's forward so width padding is circular while height and time remain non-periodic. This eliminates the decode-side seam.

- **Per-step latent rolling** (`EquirectPerStepRoll`): rolling the latent by a per-sigma deterministic shift before each model forward, then unrolling the prediction. This forces every longitude to be generated as "interior" at some step and proved critical for seam continuity. **Note:** the `dual_average` mode (evaluating two opposite shifts per step) can introduce repeated geometry — twin subjects or mirrored buildings appearing roughly 180° apart. If you see this, try `random_per_step` mode instead, or disable the roll node entirely (the SpheRoPE spherical RoPE alone may suffice for your use case).

The LoRA, circular VAE, and per-step roll together produced seamless equirectangular video that was already working in production. In a batch of 20 evaluation prompts, this setup produced good results but would occasionally (roughly 2 out of 20) lose the equirectangular projection shape. This instability is what SpheRoPE's method for spherical positional encoding helps resolve — it provides a persistent geometric signal that keeps the model's output structurally consistent across diverse prompts.

### Post-SpheRoPE integration

After the SpheRoPE paper (https://arxiv.org/abs/2606.32033) was released, I implemented its spherical positional encoding ideas — the two-path spectral split (Path A for harmonic snapping at the seam, Path B for spherical Cartesian coordinates at the poles) and wrapped noise (spherical phase remaster on the initial noise latent).

In my testing, a partial implementation of SpheRoPE's methods on their own was not sufficient to produce fully seamless equirectangular output. However, combining their spherical positional encoding ideas with my existing equirectangular LoRA and pre-SpheRoPE findings produced excellent results:

- My **LoRA** provides the essential foundation — the model cannot produce text-to-video equirectangular output without it.
- My **per-step roll** handles the seam boundary more effectively than Path A alone, by ensuring no column always sits at the processing boundary.
- SpheRoPE's **Path B (spherical coordinates)** straightens building geometry and reduces polar stretching, improving overall structural quality.
- My **circular VAE** ensures the decode doesn't reintroduce seam artifacts.
- SpheRoPE's **wrapped noise** gives a subtle but consistent improvement to polar coherence.

The combination of my LoRA-based pipeline with SpheRoPE's positional encoding ideas produces seamless equirectangular 360° video that neither approach achieves alone.

I was unable to get SpheRoPE's **semantic distortion CFG** (the three-way guidance term) working reliably due to conflicts with TiledDiffusion's monkeypatching of ComfyUI's `calc_cond_batch`. The rest of their ideas integrate cleanly.

## Why Equirectangular Inference Needs These Tools

An equirectangular panorama is periodic along longitude: column 0 and the final column represent adjacent directions in the real spherical scene. Standard diffusion inference does not know this — the model sees the panorama as a flat rectangle, so its convolutional and attention behavior treats the seam as a hard image border.

The result without mitigation:

- a visible vertical seam at the longitude wrap,
- mismatched objects or lighting across the left/right edge,
- dark bands or vignette-like artifacts at processing boundaries,
- polar void zones and stretched geometry at the top and bottom,
- decoded VAE artifacts caused by non-periodic padding.

## Core Method: Per-Step Circular Latent Rolling

`EquirectPerStepRoll` wraps the ComfyUI model object and intercepts each model call during sampling:

1. Detect the video latent layout.
2. Roll the video latent along the width / longitude axis.
3. Roll `denoise_mask` in sync when present.
4. Run the model on the shifted latent.
5. Optionally feather the known processing boundary in the prediction.
6. Unroll the prediction back to the original longitude.

For LTX-2.3, the node supports both plain 5D video latents and packed audio/video latents. Audio tokens are left untouched.

Modes:

- `random_per_step`: deterministic per-sigma shifts keyed by sigma and seed. One forward pass per step.
- `dual_average`: evaluates two roughly opposite shifted views per step and averages them. Cancels boundary banding at ~2x sampling cost. Note: can introduce repeated geometry (twin subjects 180° apart) due to the compositional bias locking in at two opposite positions.
- `half_turn`: diagnostic mode using a constant half-width shift.
- `feather_columns`: cross-fade the boundary columns of each step's prediction (2-4 is sensible).

## SpheRoPE Spherical RoPE (separate custom node: `equirect_rope`)

The spherical positional encoding described in the SpheRoPE paper (https://arxiv.org/abs/2606.32033) is implemented as a **separate custom node** (`EquirectPeriodicWRoPE` in the `equirect_rope/` directory). It monkeypatches `comfy.ldm.lightricks.model.generate_freqs` — the single entry point where LTX-2.3 computes all rotary position embedding angles — replacing the width-axis phase with a two-path spectral split:

- **Path A** (high-frequency channels ≥ 10 cycles across the width): harmonic snapping, snapping the linear frequency to the nearest integer cycle count so the phase repeats exactly across the wrap.
- **Path B** (low-frequency channels < 10 cycles): spherical Cartesian (X, Y) coordinates on the unit sphere. X = cos(lat)·cos(lon), Y = cos(lat)·sin(lon). At the poles (lat=±π/2), cos(lat)=0 so X=Y=0 for all longitudes, naturally encoding polar convergence.

The two-path split uses SpheRoPE's own frequency vector (`theta^(k/D) · π/2`), not the RoPE frequency ladder — this distinction is critical and was the cause of several failed implementations.

Key parameters: `blend` (0=linear, 1=spherical), `path_b_threshold` (default 10.0, lower = more spherical channels), `r_scale` (default 1.0, spherical magnitude).

The node is idempotent and can be toggled at runtime.

**License:** This nodepack is licensed under the PolyForm Noncommercial License 1.0.0 (https://polyformproject.org/licenses/noncommercial/1.0.0). The spherical positional encoding and wrapped noise implementations are based on ideas from the SpheRoPE paper (https://arxiv.org/abs/2606.32033) by Or Hirshfeld, Lior Talker, and colleagues — see the LICENSE file for details.

## Circular VAE Padding

`EquirectCircularVAE` patches loaded VAE convolution modules so width padding is circular while height and time remain non-periodic. The implementation wraps convolution forwards rather than replacing model classes, preserving ComfyUI's weight-casting and offload paths.

Important caveat: ComfyUI caches VAEs by file. This node modifies the loaded VAE object in place, so every consumer of the same loaded VAE file in a run sees the same patched or restored state. The node resets to stock before applying its selected state, but true stock-vs-circular A/B testing needs separate runs or separate loaded VAE objects.

## Wrapped Noise (SpheRoPE Spherical Phase Remaster)

`EquirectWrappedNoise` wraps a NOISE generator and applies SpheRoPE's spherical phase remaster to the initial noise latent before denoising. The algorithm:

1. Creates a "macro-structure" version by weighting each row toward its row mean, with strength alpha = 1 − cos(latitude) — zero at the equator, ~1 at the poles.
2. Computes 2D FFT of both original and macro noise.
3. Replaces the low-frequency phase (inside a circular mask of configurable radius) with the macro version's phase.
4. Reconstructs using the original magnitude spectrum.

This encodes polar convergence into the noise structure before denoising begins, providing a subtle but consistent improvement to pole quality.

The `radius` parameter defaults to 4 for typical LTX-2.3 latent resolutions (20×40). SpheRoPE's default of 12 targets their 64×128 latent grid.

## Installation

Clone into `ComfyUI/custom_nodes`:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Burgstall-labs/ComfyUI-Seamless-Equirectangular
```

The `equirect_rope/` custom node (SpheRoPE spherical RoPE) is bundled alongside.

Restart ComfyUI. The nodes appear under their respective categories:
- `model_patches/equirect`
- `latent/video/equirect`
- `noise/equirect`
- `360/seamless` (equirect_rope node)

No extra runtime dependency is required beyond the torch runtime already used by ComfyUI.

## Recommended Pipeline

```
UNETLoader → Equirect Spherical RoPE → LTX2_NAG → PerStepRoll → CFGGuider → SamplerCustomAdvanced
```

With:
- EquirectCircularVAE: enabled, decoder=True
- EquirectWrappedNoise: enabled, radius=4
- EquirectPerStepRoll: try `random_per_step` first (avoids the repeated-geometry issue); switch to `dual_average` if seam banding is visible and you don't see twin artifacts
- Equirect Spherical RoPE: blend=1.0, threshold=10.0, r_scale=1.0

Your equirectangular LoRA (trained on zero-padded VAE) should be loaded — the SpheRoPE RoPE patch is not a replacement for the LoRA, which carries the essential equirectangular output knowledge that the frozen base model lacks.

## Tests

Run with the ComfyUI Python environment:

```bash
/home/juho/ComfyUI/venv/bin/python tests/test_seam_nodes.py
```

The tests cover circular Conv2d/Conv3d patch/unpatch behavior, VAE encoder/decoder scoping, deterministic per-sigma shift generation, LTX packed audio/video latent rolling, per-step model wrapper round trips, dual_average behavior, and latent seam blending.

## Related Burgstall Labs Work

Burgstall Labs develops practical open-source tools for 360-degree AI video, VR-outpainting, panoramic generation, and ComfyUI-based production workflows. ComfyUI-Seamless-Equirectangular provides the seam-inference and geometry-enhancement layer that complements projection, outpainting, and LoRA workflows for immersive video.
