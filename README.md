# ComfyUI-Seamless-Equirectangular

**ComfyUI-Seamless-Equirectangular** is a ComfyUI custom node pack by **Burgstall Labs** for seamless equirectangular, 360-degree, VR, and panoramic video generation with LTX-2.3. It provides training-free seam mitigation and geometry enhancement for equirectangular inference.

This is the result of extensive experimentation: training a dedicated equirectangular LoRA, then developing circular VAE padding, per-step latent rolling, and spherical phase remaster noise to handle the seam and pole convergence — then implementing the spherical positional encoding ideas from the SpheRoPE paper (https://arxiv.org/abs/2606.32033) on top of that foundation. The combined pipeline produces seamless 360° video that neither approach achieves alone. Continued testing produced **EquiRoPE** — an original, architecturally distinct latitude-scaled RoPE with pole gating and hemisphere selection — plus a latitudinal geometric CFG, which now form the recommended refinement layer over the LoRA + roll + circular-VAE foundation.

The LoRA is available on Hugging Face: [TheBurgstall/Seamless-Equirectangular-LTX2.3-LoRA](https://huggingface.co/TheBurgstall/Seamless-Equirectangular-LTX2.3-LoRA)

**Keywords:** ComfyUI seamless equirectangular, LTX-2.3 360 video, SpheRoPE, circular VAE padding, per-step latent roll, panoramic video generation, Burgstall Labs.

## Nodes

| Node | Type | What it does |
|---|---|---|
| **EquirectCircularVAE** | `VAE → VAE` | Patches VAE decoder convolutions to use circular padding along the width (longitude) axis. Fixes the decode-side seam caused by zero-padded convolutions at the 360° wrap boundary. |
| **EquirectPerStepRoll** | `MODEL → MODEL` | Wraps the diffusion model so every denoising step runs on a W-rolled latent. Each step sees a different longitudinal shift, so no column is always at the model's processing boundary — forces content coherent across the wrap. Based on the Diffusion360 / PanoDiffusion circular shift method. `start_percent` holds the roll boundary at the true wrap edge during the early layout-forming steps — fixes structural closure failures (mismatched buildings at a frozen column) in fast-motion scenes. |
| **EquirectLatentSeamBlend** | `LATENT → LATENT` | Thin cross-fade of the W-edge latent columns. Cosmetic fallback — smooths the boundary band but does not make content wrap. Off by default. |
| **EquirectWrappedNoise** | `NOISE → NOISE` | Spherical phase remaster on the initial noise latent. Encodes polar convergence into low-frequency phase, reducing stretching at the poles. Original method developed during this project. |
| **EquirectSphericalRoPE** | `MODEL → MODEL` | Spherical-aware width-axis RoPE with two-path spectral split (harmonic snapping + spherical coordinates). Based on the SpheRoPE paper. |
| **EquiRoPE** | `MODEL → MODEL` | Original latitude-scaled width RoPE with spectral split: low-frequency (structure) channels get pole-gated `1−(1−\|cos lat\|)^γ` position scaling, high-frequency channels stay stock for LoRA compatibility. `pole_side` restricts the collapse to one hemisphere (`bottom` = nadir-only, keeps skylines straight). Architecturally distinct from SpheRoPE. |
| **EquiRoPEGeometricCFG** | `MODEL → MODEL` | Latitude-dependent horizontal blur of the CFG delta — smooths polar artifacts without touching the equator. Handles LTX packed audio+video latents; exact no-op at cfg=1, so it's safe to feed every sampler stage. |
| **EquirectSoftPoleConstraint** | `MODEL → MODEL` | Per-step polar row blend. Insurance against rare polar voids; usually unnecessary when noise + roll are active. |

## History

### Pre-SpheRoPE development

I started with training a dedicated equirectangular LoRA — a rank-128 LoRA for 15k steps. In my previous experiments, the model could not produce text-to-video equirectangular output without it; the frozen 22B base model's rectilinear bias is too strong even with inference-time tricks.

With the LoRA producing equirectangular video, two problems remained:

1. **The longitude seam**: column 0 and column W-1 represent adjacent directions in the real spherical scene, but the model treats them as unrelated edges. Standard zero-padded VAE convolutions amplify this with a visible seam at decode time — even WITH the LoRA.

2. **Polar stretching**: the equirectangular projection compresses the poles into single rows, but the model has no inductive bias for this. Sometimes this would result in polar voids, even with the LoRA (the base model has clearly seen equirectangular footage during training, but not enough to overcome the problem consistently).

The first solutions I implemented were:

- **Circular VAE padding** (`EquirectCircularVAE`): wrapping each VAE convolution's forward so width padding is circular while height and time remain non-periodic. This eliminates the decode-side seam.

- **Circular (periodic) RoPE** was also explored as a possible solution — making the width axis of the transformer's position encoding periodic so column 0 and column W-1 become adjacent in rotary phase. This was tested but didn't produce usable results; the cylindrical topology helped conceptually but the implementation wasn't effective without the spherical component I later found in the SpheRoPE paper.

- **Per-step latent rolling** (`EquirectPerStepRoll`): rolling the latent by a per-sigma deterministic shift before each model forward, then unrolling the prediction. This forces every longitude to be generated as "interior" at some step and proved critical for seam continuity. **Note:** the `dual_average` mode (evaluating two opposite shifts per step) can introduce repeated geometry — twin subjects or mirrored buildings appearing roughly 180° apart. If you see this, try `random_per_step` mode instead, or disable the roll node entirely.

The LoRA, circular VAE, and per-step roll together produced seamless equirectangular video that was already working in production. In a batch of 20 evaluation prompts, this setup produced good results but would occasionally (roughly 2 out of 20) lose the equirectangular projection shape and produce polar voids. This instability is what SpheRoPE's method for spherical positional encoding helps resolve — it provides a persistent geometric signal that keeps the model's output structurally consistent across diverse prompts.

### Post-SpheRoPE integration

After the SpheRoPE paper (https://arxiv.org/abs/2606.32033) was released, I implemented its spherical positional encoding ideas — the two-path spectral split (Path A for harmonic snapping at the seam, Path B for spherical Cartesian coordinates at the poles). This complemented the wrapped noise and circular VAE methods I had already developed.

In my testing, a partial implementation of SpheRoPE's methods on their own was not sufficient to produce fully seamless equirectangular output. However, combining their spherical positional encoding ideas with my existing equirectangular LoRA and pre-SpheRoPE findings produced excellent results:

- My **LoRA** provides the essential foundation — the model cannot produce text-to-video equirectangular output without it. Not even with my partial implementation of the SpheRoPE ideas. (Note: This is not to say the full implementation cannot do this, I have not had the opportunity to test the full original implementation yet.)
- My **per-step roll** handles the seam boundary more effectively than Path A alone, by ensuring no column always sits at the processing boundary.
- SpheRoPE methodology for **Path B (spherical coordinates)** straightens building geometry and reduces polar stretching, improving overall structural quality.
- My **circular VAE** ensures the decode doesn't reintroduce seam artifacts.
- My **wrapped noise** gives a subtle but consistent improvement to polar coherence (original method, not from SpheRoPE).

The combination of my LoRA-based pipeline with SpheRoPE's positional encoding ideas produces seamless equirectangular 360° video that neither approach achieves alone (in my testing, admittedly missing the tests with the original full SpheRoPE implementation).

I was unable to get SpheRoPE's **semantic distortion CFG** (the three-way guidance term) working reliably due to conflicts with TiledDiffusion's monkeypatching of ComfyUI's `calc_cond_batch`. The rest of their described methodologies integrate cleanly.

### EquiRoPE and the division-of-labor findings (July 2026)

A controlled test ladder (same seeds, one variable per run) sharpened the picture of what each layer actually contributes:

- **LoRA + per-step roll + circular VAE alone produces seamless output** — the wrap-edge difference is statistically indistinguishable from any interior column pair. Position-encoding patches are refinements, not the backbone.
- Without the LoRA, no inference-time patch produces equirectangular output at all: RoPE modifications are geometric *consistency* constraints, not a content prior.
- EquiRoPE with the classic `|cos(latitude)|` profile cleaned up nadir melt but bent long horizontals (rooflines) at mid-latitudes — the cos profile compresses positions ~30% already at 45°. The `pole_gamma` gate (default 3.0) keeps mid-latitudes at ~0.95–1.0 while still collapsing the pole rows; `pole_side=bottom` removes the cost entirely for outdoor scenes, where the zenith is just sky.
- In fast-panning scenes, per-step rolling during the early layout-forming steps let the panorama fail to close structurally — a hard cut between two different buildings, frozen at a latent column. `start_percent` (0.2–0.3) holds the boundary at the true wrap edge until the layout is settled, then rolls to heal seam tone.

**Division of labor: LoRA = equirectangular appearance · per-step roll + circular VAE = seam · wrapped noise = closure and pole coherence · EquiRoPE + geometric CFG (bottom) = nadir.**

## Why Equirectangular Inference Needs These Tools

An equirectangular panorama is periodic along longitude: column 0 and the final column represent adjacent directions in the real spherical scene. Standard diffusion inference does not know this — the model sees the panorama as a flat rectangle, so its convolutional and attention behavior treats the seam as a hard image border.

The result without mitigation:

- a visible vertical seam at the longitude wrap,
- mismatched objects or lighting across the left/right edge,
- dark bands or vignette-like artifacts at processing boundaries,
- polar void zones and stretched geometry at the top and bottom,
- decoded VAE artifacts caused by non-periodic padding.

## Per-Step Circular Latent Rolling

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
- `start_percent`: hold rolling for the first fraction of each sampler run so scene layout closes with the boundary at the true wrap edge (where wrapped noise correlates content); rolling then heals seam tone/texture. 0 = roll every step; 0.2–0.3 recommended for scenes with fast lateral motion.

## SpheRoPE Spherical RoPE

The spherical positional encoding described in the SpheRoPE paper (https://arxiv.org/abs/2606.32033) is implemented as the **EquirectSphericalRoPE** node. It monkeypatches `comfy.ldm.lightricks.model.generate_freqs` — the single entry point where LTX-2.3 computes all rotary position embedding angles — replacing the width-axis phase with a two-path spectral split:

- **Path A** (high-frequency channels ≥ 10 cycles across the width): harmonic snapping, snapping the linear frequency to the nearest integer cycle count so the phase repeats exactly across the wrap.
- **Path B** (low-frequency channels < 10 cycles): spherical Cartesian (X, Y) coordinates on the unit sphere. X = cos(lat)·cos(lon), Y = cos(lat)·sin(lon). At the poles (lat=±π/2), cos(lat)=0 so X=Y=0 for all longitudes, naturally encoding polar convergence.

The two-path split uses SpheRoPE's own frequency vector (`theta^(k/D) · π/2`), not the RoPE frequency ladder — this distinction is critical and was the cause of several failed implementations.

Key parameters: `blend` (0=linear, 1=spherical), `path_b_threshold` (default 10.0, lower = more spherical channels), `r_scale` (default 1.0, spherical magnitude).

The node is idempotent and can be toggled at runtime.

**Status:** validated on slow scenes. Under fast camera motion it currently produces severe artifacts (scattered debris, duplicated content) — use EquiRoPE there.

**License:** This nodepack is licensed under the PolyForm Noncommercial License 1.0.0 (https://polyformproject.org/licenses/noncommercial/1.0.0). Noncommercial use (research, academic, personal, hobbyist) is free. Commercial use requires a separate license — contact **howdy@theaiwrangler.com**. The spherical positional encoding implementation is based on ideas from the SpheRoPE paper (https://arxiv.org/abs/2606.32033) by Or Hirshfeld, Lior Talker, and colleagues — see the LICENSE file for details.

## EquiRoPE (Latitude-Scaled RoPE) and Geometric CFG

`EquiRoPE` scales each token's width position on low-frequency channels by a pole-gated latitude profile, collapsing polar tokens toward co-location so attention treats them as adjacent. High-frequency channels keep stock positions — this spectral split is critical for LoRA compatibility. Parameters: `blend` (0=stock, 1=full), `epsilon` (minimum scale at the exact pole), `threshold` (spectral split, cycles), `pole_gamma` (1.0 = classic `|cos|`, higher = tighter pole hugging; default 3.0), `pole_side` (`both`/`bottom`/`top`).

`EquiRoPEGeometricCFG` blurs the CFG delta horizontally with strength rising toward the poles — smoothing polar starburst/pinch without touching equatorial detail. It supports plain 4D/5D latents, LTX flat-packed audio+video latents (audio passes through untouched), and token-packed formats. At cfg=1 it reproduces stock behavior exactly, so it can safely feed a distilled/NAG stage. It is a per-model patch: it only affects samplers wired downstream of its output — put it in the model path of your cfg>1 stage.

**Note on global patches:** EquiRoPE and EquirectSphericalRoPE both monkeypatch the same LTX RoPE entry point — they are mutually exclusive (last enabled wins), and the patch survives node deletion/bypass. Only running the node with `enabled=False` or restarting ComfyUI restores stock RoPE. Both nodes log a warning on every sampling run while active so silent carry-over can't skew A/B comparisons.

## Circular VAE Padding

`EquirectCircularVAE` patches loaded VAE convolution modules so width padding is circular while height and time remain non-periodic. The implementation wraps convolution forwards rather than replacing model classes, preserving ComfyUI's weight-casting and offload paths.

Important caveat: ComfyUI caches VAEs by file. This node modifies the loaded VAE object in place, so every consumer of the same loaded VAE file in a run sees the same patched or restored state. The node resets to stock before applying its selected state, but true stock-vs-circular A/B testing needs separate runs or separate loaded VAE objects.

## Wrapped Noise (Spherical Phase Remaster)

`EquirectWrappedNoise` wraps a NOISE generator and applies a spherical phase remaster (original method, developed during this project) to the initial noise latent before denoising. The algorithm:

1. Creates a "macro-structure" version by weighting each row toward its row mean, with strength alpha = 1 − cos(latitude) — zero at the equator, ~1 at the poles.
2. Computes 2D FFT of both original and macro noise.
3. Replaces the low-frequency phase (inside a circular mask of configurable radius) with the macro version's phase.
4. Reconstructs using the original magnitude spectrum.

This encodes polar convergence into the noise structure before denoising begins, providing a subtle but consistent improvement to pole quality.

The `radius` parameter defaults to 4 for typical LTX-2.3 latent resolutions (20×40). A higher value like 12 matches the 64×128 latent grid used by the SpheRoPE paper's position encoding experiments but is too broad for LTX's smaller latents.

## Installation

Clone into `ComfyUI/custom_nodes`:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Burgstall-labs/ComfyUI-Seamless-Equirectangular
```

Restart ComfyUI. The nodes appear under their respective categories:
- `model_patches/equirect`
- `latent/video/equirect`
- `noise/equirect`

No extra runtime dependency is required beyond the torch runtime already used by ComfyUI.

## Recommended Pipeline

```
UNETLoader → LoraLoaderModelOnly → EquiRoPE → EquiRoPEGeometricCFG → LTX2_NAG → EquirectPerStepRoll → CFGGuider (cfg>1) → SamplerCustomAdvanced
```

With:
- EquirectCircularVAE: enabled, decoder=True
- EquirectWrappedNoise: enabled, radius=4 (first/base sampling stage only; re-noise in a refine stage is seam-neutral)
- EquirectPerStepRoll: `random_per_step`, `start_percent=0.25`, feed both stages. Switch to `dual_average` if seam banding is visible and you don't see twin artifacts.
- EquiRoPE: blend=1.0, threshold=10.0, pole_gamma=3.0, pole_side=bottom (outdoor content; use `both` and higher gamma for scenes with structured ceilings)
- EquiRoPEGeometricCFG: radius=3, strength=3.0, pole_side=bottom
- EquirectSphericalRoPE: disabled (see status note above)

Your equirectangular LoRA should be loaded — none of these nodes replace the LoRA, which carries the essential equirectangular output knowledge that the frozen base model lacks.

## Known Limitations

In videos with a moving camera, expect some **morphing** of scene content as it translates across longitudes, and occasional **polar voiding** at the nadir/zenith. For the latter, `EquirectSoftPoleConstraint` is worth testing as a mitigation — it blends the polar rows toward coherence each step and is cheap insurance, though it showed no measurable effect when the poles were already clean.

## Tests

Run with the ComfyUI Python environment (so `comfy` is importable):

```bash
PYTHONPATH=/path/to/ComfyUI python tests/test_seam_nodes.py
```

The tests cover circular Conv2d/Conv3d patch/unpatch behavior, VAE encoder/decoder scoping, deterministic per-sigma shift generation, LTX packed audio/video latent rolling, per-step model wrapper round trips, dual_average behavior, and latent seam blending.

## Related Burgstall Labs Work

Burgstall Labs develops practical open-source tools for 360-degree AI video, VR-outpainting, panoramic generation, and ComfyUI-based production workflows. ComfyUI-Seamless-Equirectangular provides the seam-inference and geometry-enhancement layer that complements projection, outpainting, and LoRA workflows for immersive video.
