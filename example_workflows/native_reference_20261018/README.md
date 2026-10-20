# Native-reference follow-up experiments

> Historical comparison graphs: settings are preserved to reproduce their recorded results. Use the [current canvas workflow index](../README.md) for October 20 wrap defaults.

These ten API graphs transfer ideas from the standalone SpheRoPE comparison: six FLUX.2 Klein image experiments and four LTX 2.3 video experiments. Their original versions all completed through MCP; current versions use integrated `Equirect` node names. For the tested fixed-camera ERP first-frame case, start with [ltx23_firstframe_stockrope.json](ltx23_firstframe_stockrope.json). LTX 2.5 users should use the separate [artist workflows](../ltx25/README.md).

| Graph | Purpose |
|---|---|
| [Klein padding](klein_decode_padding.json) | Distilled Klein 9B, seed 80084: decode one latent at 0, 8 and 18 circular columns |
| [Klein base padding](klein_base_decode_padding.json) | Base 9B FP8, seed 80084: same decoder comparison |
| [Klein base delayed guidance](klein_base_warmup2.json) | Two positive-only steps, then 18 steps of three-way ERP guidance |
| [Split control](klein_base_split_control.json) | Same two-sampler structure, with three-way guidance in both stages |
| [Second-seed baseline](klein_base_seed80085.json) | Seed 80085, guidance from the first step |
| [Second-seed delayed guidance](klein_base_warmup2_seed80085.json) | Seed 80085, two-step warmup |

All use 1024x512 and Euler/simple sampling. Distilled Klein uses four steps at CFG 1; base uses 20 steps with CFG 4 and geometry gamma 3. Models and prompts are embedded as node settings. These graphs use the three-way guider, spherical FLUX.2 patch, circular decoder and diagnostics included in Seamless-Equirectangular. No companion pack is required. They produce images, not videos.

Eighteen-column padding gave no meaningful improvement over eight columns. Delayed guidance improved the zenith in one seed but not the other, and did not correct the nadir. Splitting the sampler introduces small numerical differences even with unchanged guidance; the control records that limit. Read the [report and comparison views](../../docs/NATIVE_SPHEROPE_COMPARISON_2026-10-18.md) before adopting these settings.

## LTX 2.3 two-stage transfer

**Preferred first-frame variant for the tested sample:** [ltx23_firstframe_stockrope.json](ltx23_firstframe_stockrope.json). It disables spherical RoPE in both stages and retains circular processing. The spherical and gamma-zero graphs below are comparison controls.

These are ComfyUI API-format JSON graphs, not frontend canvas exports. Submit them through ComfyUI's local API or the MCP `run_workflow` tool. Set `LoadImage.image` to your own seamless 2:1 ERP frame already in ComfyUI's input folder. The recorded test used `Living_Room_01_Couch_color_firstframe.png`; that input image is not bundled here.

Required local files (select your installed filenames if different):

| ComfyUI model category | Filename |
|---|---|
| checkpoints | `ltx-2.3-22b-dev-fp8.safetensors` |
| text_encoders | `gemma_3_12B_it.safetensors` |
| loras | `ltx-2.3-22b-distilled-lora-384.safetensors` |
| latent_upscale_models | `ltx-2.3-spatial-upscaler-x2-1.1.safetensors` |

Update Seamless-Equirectangular and restart ComfyUI before use. The current circular-upscaler fix requires ComfyUI's managed `LTXVLatentUpsampler.execute` API (tested with ComfyUI 0.39.0). These graphs save lossless PNG sequences under `output/native-ltx-comparison-20261018/`; the report's MP4s are 24 fps review encodes.

- [Text-to-video](ltx23_two_stage.json): 17 frames, 512x256 dev sampling, then 1024x512 refinement.
- [First-frame](ltx23_firstframe.json): 33 frames, reference image re-encoded and pinned at both stage resolutions. Put `Living_Room_01_Couch_color_firstframe.png` in ComfyUI's input folder, or select your own seamless ERP image in `LoadImage`.

All four LTX graphs use the installed LTX 2.3 dev FP8 checkpoint, full Gemma 3 encoder, x2 1.1
upscaler, 30 Euler dev steps at CFG 3 (gamma 3 except in the gamma-zero control), and three positive-only refinement
steps with the distilled LoRA at 0.8. Wrapped noise is disabled. Circular
upscale and decode use four columns. No panorama LoRA is required.

**Status:** both graphs completed through MCP after the circular
upscaler fix and server restart: 17 and 33 lossless 1024x512 frames respectively.
The integrated suite passes 116 tests (4 skipped); the upscaler also passed a real-model CPU check. Mean boundary ratios
were 0.684 and 0.528, but floor-pole distortion remains. Restart ComfyUI after
updating this pack to load the fix.
All seven formerly external node classes now have local `Equirect` equivalents, and these saved graphs use them. These are experimental LTX 2.3 video-only graphs,
not replacements for the LTX 2.5 Ripple or moving-camera presets.

### First-frame controls

- [Stock-RoPE first-frame variant](ltx23_firstframe_stockrope.json): disables both spherical patch nodes while retaining circular upscale/decode and geometry gamma 3. In the 33-frame living-room test it preserved the floor better, without the pronounced nadir pinch; the join stayed continuous in sampled views. Prefer this experimental variant for that fixed-camera first-frame case. One seed/image is not general qualification.
- [Gamma-zero control](ltx23_firstframe_gamma0.json): retains spherical RoPE and disables geometry guidance. It completed but did not remove the floor artifact.

Both controls completed through MCP. See the [report](../../docs/NATIVE_SPHEROPE_COMPARISON_2026-10-18.md) for videos, measurements and limitations. All four LTX graphs use existing local models; no panorama LoRA is used.

Historical execution records retain the original `SpheRoPE` class names. Current graphs change only those node names; sampling settings remain unchanged. The preferred first-frame and Klein padding graphs also completed on an isolated server with only this pack enabled. All ten graphs resolve their node classes there. See the [integration evidence](../../docs/NATIVE_SPHEROPE_COMPARISON_2026-10-18.md#integrated-seamless-equirectangular-delivery).
