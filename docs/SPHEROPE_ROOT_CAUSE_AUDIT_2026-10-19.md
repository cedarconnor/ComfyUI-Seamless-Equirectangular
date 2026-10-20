# SpheRoPE: reset the diagnosis

The 37-case panorama comparison measured output differences, but did not locate
the first failing component. A successful execution, a small boundary metric,
and a faithful reproduction of the reference method are three different claims.
We have not yet established the last claim for LTX.

## Reference contract

The [paper](https://arxiv.org/html/2606.32033v1) combines spherical width RoPE,
three-way semantic/geometry guidance, and circular VAE decoding. Appendix B
specifies harmonic tolerance 0.06 and geometry strength 6. Its geometry prompt
locks the camera, while its Stress-20 benchmark includes moving-camera prompts.
That tension prevents dismissing every motion failure as outside its evidence.

The released [LTX source](https://github.com/orhir/SpheRoPE/blob/a40a86b49c25dbd4d0191591f28a0d5baf0c4ca4/panorama_diffusers/pipelines/pipeline_ltx23_erp.py)
uses a cycle-count threshold instead of the paper's tolerance rule. Its pipeline
defaults to threshold 10, though its lower-level builder defaults to 1.
The CLI defaults to geometry strength 6; its help recommends 3 for LTX.
These are upstream inconsistencies to track explicitly, not evidence that
changing our port to the paper formula will solve the artifacts.

## What our experiments actually tested

| Experiment | Important departure from a reference reproduction |
|---|---|
| Recent 32-case distilled grid | Single stage, CFG 1, no three-way geometry prediction; optional panorama LoRA and denoiser rolling |
| Four LTX 2.3 dev cases | Ordinary CFG, single stage; still no full reference guidance |
| Earlier standalone native LTX | Existing FP8 checkpoint with measured recasting error; 1024×512 output rather than CLI's 2048×1024; shortened clips |
| Earlier standalone FLUX.1 | BF16 native code, but 1024×512 and 28 steps; not the paper's stated image evaluation settings |

The current model library contains LTX 2.3 dev and distilled **FP8** checkpoints.
The reference CLI requests the dev BF16 checkpoint. No additional weights were
downloaded for this audit. Converting FP8 to BF16 would not restore lost precision.

Reduced resolution also changes the positional intervention itself. With 682
width-frequency channels and threshold 10, spherical-path channel counts are
325 at 512-pixel width, 273 at 1024, and 222 at 2048. Consequently a half-size
screen is not just a blurrier version of the full-size experiment.

## New test: positional preprocessing through query/key rotation

[Probe](../scripts/probe_ltx_rope_application.py) executes the original native
builder and patch function definitions, ComfyUI's actual model-owned patch
resolver, and both implementations' query/key rotary functions. No weights or
sampling are involved. Upstream function bodies are extracted unchanged with
AST to avoid unrelated heavyweight pipeline imports.

At LTX 2.3's 4096 inner dimension / 32 heads, two temporal slices, midpoint
coordinates, threshold 10 and radius 1:

| Spatial token grid | FP32 cosine/sine and rotated Q/K maximum error | BF16 maximum error |
|---|---:|---:|
| 8×16 | 0 | 0 |
| 16×32 | 0 | 0 |
| 32×64 | 0 | 0 |

This rules out a port-specific axis-order, phase-format, or rotation-layout
mismatch for these tested inputs. It does **not** establish equality of loaded
weights, prompt embeddings, attention outputs, guidance or complete denoising.
The probe uses CPU tensors, not the server's GPU attention kernel.

Evidence: `E:/ComfyUI-test-results/erp-boundary-20261019/rope_application.json`.

## Boundary experiment

[VAE experiment builder/analyzer](../scripts/diagnose_erp_vae.py) prepares a
continuous analytic sphere and the user's mountain panorama at 1024×512.
The analytic control has known longitude closure and pole convergence.
Each is encoded and decoded with LTX 2.3 and 2.5, without any diffusion,
prompt, LoRA, spherical patch or rolling:

1. Ordinary encode and decode.
2. The same encoded latent, circularly padded only for decoding.
3. Wrap-extended source, ordinary encode/decode, then crop back to the source.

Case 2 isolates decoder boundary handling. Case 3 tests the combined boundary
effect of encoding and decoding; it does not isolate encoder changes alone.
The tests use eight latent columns / 256 pixels of context on each side.
This is a diagnostic setting, not a new recommended artist preset.

Results and execution records are stored under
`E:/ComfyUI-test-results/erp-boundary-20261019`.

### Measured results

All three MCP workflows succeeded, producing 18 lossless images. The follow-up
compares the existing LTX 2.5 dual-rotation decoder at blend widths 0, 1 and 4.
The zero-blend outputs are pixel-identical to the initial ordinary decodes for
both inputs, confirming repeatability of those decodes.

Mean first/last-column RGB difference, on a 0–255 scale:

| Input / model | Source | Ordinary VAE | Circular decode | Wrap-extended encode/decode | Dual rotation, blend 1 |
|---|---:|---:|---:|---:|---:|
| Analytic sphere / LTX 2.3 | 0.539 | 0.661 | 0.336 | 0.709 | — |
| Analytic sphere / LTX 2.5 | 0.539 | 5.212 | 1.001 | 1.229 | 0.270 |
| Mountain photograph / LTX 2.3 | 6.768 | 10.369 | 6.404 | 7.672 | — |
| Mountain photograph / LTX 2.5 | 6.768 | 17.224 | 17.191 | 18.104 | 11.540 |

First and last columns are adjacent samples, not identical coordinates, so a
valid continuous source has a nonzero difference. Reducing this number below
the source is not proof of better reconstruction. For example, LTX 2.3's
circular decode has a lower raw jump than the wrap-extended round trip, but
a higher source-relative boundary residual (0.667 versus 0.465) on the sphere.

Rectilinear join inspection shows a new vertical boundary on LTX 2.5's analytic
control. The dual-rotation decoder removes most of it. The photograph's LTX 2.5
round trip has visibly altered, noisy mountain/ground detail; the dual decoder
retains much of that reconstruction error. LTX 2.3 preserves this detail better
in this experiment. This does not measure video temporal reconstruction.

**Causal result:** the ordinary LTX 2.5 encode/decode path can introduce a seam
before transformer denoising, and replacing only its decode boundary treatment
substantially reduces it on a known-continuous control. This component failure
cannot be caused by the panorama LoRA, motion prompt, or spherical RoPE, since
none is present. It does not explain all remaining artifacts in generated clips
that already use the dual decoder.

The local LTX 2.5 decoder is a neighborhood-attention **diffusion decoder**,
whereas the reference paper's LTX 2.3 path uses a convolutional VAE.
`CausalDiffusionVAE.decode` initializes decoder noise with seed 0 on each call.
Thus decodes repeat, but changing tensor width or rotating only the latent does
not preserve the noise field in corresponding spherical coordinates. That is
a specific hypothesis for additional 2.5 reconstruction/compositing errors,
not a tested root cause. Both noise alignment and neighborhood boundary handling
need isolation before changing the implementation.

Local evidence:

- `vae_metrics.json`, `vae_dual_metrics.json`: measurements.
- `vae_join_comparison.jpg`, `vae_dual_comparison.jpg`: perspective inspections.
- `audit.json`: successful job IDs, source/output hashes and repeatability.
- `<prompt_id>_history.json`: actual server execution graphs and results.

## Next acceptance gates

1. **VAE boundary:** locate errors introduced before denoising using the analytic
   sphere and source-relative boundary residual, with rectilinear join inspection.
2. **Denoiser boundary:** save the initial conditioning latent, stage-one output,
   upscaled latent and stage-two output. Decode each with a common decoder.
   Identify the first new discontinuity rather than judging only the final MP4.
3. **Reference generation:** reproduce released LTX settings at the intended
   resolution with no panorama LoRA or rolling. Record precision deviations.
   Compare a fixed-camera reference before adding subject/camera motion.
4. **Port localization:** if native succeeds and Comfy fails, compare prompt
   embeddings and single-step predictions on identical inputs and sigmas before
   another full video.
5. **Artist use cases:** add first-frame conditioning, motion, gray-guide control,
   and other backbones only after the baseline passes.

There is no new production fix in this audit. The aim is to identify a causal
failure before selecting another parameter or modifying the algorithm.
