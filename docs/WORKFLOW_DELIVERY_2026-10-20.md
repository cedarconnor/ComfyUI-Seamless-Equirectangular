# October 20 workflow delivery

The [current workflow index](../example_workflows/README.md) replaces mixed historical defaults with the inspected longitude-wrap profiles. Model-specific encoding and decoding are retained; no universal pole-quality claim is made.

## Changes

- Both LTX 2.5 first-frame canvas filenames now reproduce the final 121-frame fixed-camera API profile: native 2.5 components, spherical RoPE blend 1 / threshold 10 / radius 1, dual-rotation decode blend 1, no panorama LoRA or rolling.
- Both Ripple canvases use those encoding/decode settings while retaining their native Ripple adapter at 1.35 and aligned look-frame/gray-guide conditioning. They remain 65-frame templates; their updated guide behavior is not visually qualified.
- H3 uses its tested native INT8 encoder, periodic longitude blend 1, 56-frame profile, and dual decode blend 4. The conventional post-decode upscaler is removed, so the export receives the inspected decoder branch.
- New LTX 2.3 I2V and three Klein canvas files expose the tested API profiles without stock-decoder comparisons or zero-strength LoRA/disabled-roll dependencies.
- The main LTX 2.3 T2V filename now opens the tested single-stage control. Its old two-stage canvas is preserved under `example_workflows/historical/`. The new control passes longitude wrap on one seed but retains severe geometry defects.
- Outpainting switches to spherical RoPE and explicit circular decode pad 4, with decoder target width driven by the projected guide width. Its required outpainting adapter and camera projection are preserved. It remains dependency-limited, as detailed below.
- Dated API comparisons remain unchanged as experimental evidence. Their READMEs now point to current canvas defaults. The repository includes the final MP4, sampled wrap/pole views, investigation scripts and the FLUX scalar-patch-size rolling fix with regression coverage.

## Fresh verification

On October 20, ComfyUI MCP converted and validated all eleven current canvas files against the live server's node schemas and installed model dropdowns. **Ten pass, with zero warnings.** The outpainting template has eight pre-existing dependency errors and two unreachable-node warnings. [Full schema results](validation/2026-10-20/workflow_schema_validation.json).

The LTX 2.5 first-frame canvas was submitted through MCP as job `379cdcae-a571-401e-a110-14a5135ef11b`. It completed, exported a 1024×512 MP4 with 121 frames at 24 fps, and saved 121 PNGs. Every PNG is pixel-identical to the final accepted API run, job `119d216b-e38b-4970-a6a7-2fe9c8d1f7cc`. All reference core inputs match. ComfyUI reused the original model/sampling/decoder cache: this verifies canvas conversion and export equivalence, not independent resampling. [Machine-readable evidence](validation/2026-10-20/canvas_runtime_equivalence.json).

The repository regression suite reports **118 passed, 4 skipped**. The installed ComfyUI Python runtime was used: `python -m pytest -q`. Skipped tests remain unqualified; successful schema validation alone is not a render-quality test.

## Dependency-limited outpainting

The live installation lacks these exact files/nodes from the original template:

- `ltx-2.3-22b-dev.safetensors` for the text encoder loader.
- `ltx-2.3-22b-distilled-1.1_transformer_only_mxfp8_block32.safetensors`.
- `LTX23_audio_vae_bf16.safetensors` and `LTX23_video_vae_bf16.safetensors`.
- `ltx-2/360vroutpaint_v2_step_07000.safetensors`.
- `EstimateCameraGeoCalib`, `RectilinearToEquirect`, and `EquirectSourceComposite`.

These are not replaced with arbitrary checkpoints or bypassed: the projection and adapter define the outpainting task. The template is explicitly marked as unqualified in its canvas note and index. Its new wrap settings do not establish a working outpainting result on this installation.

## Scope of the visual evidence

The [render validation](COMFY_WRAP_VALIDATION_2026-10-20.md) covers Klein base/distilled stills, short LTX 2.3/2.5 and H3 first-frame clips, diagnostic T2V, and the final LTX 2.5 panorama. Longitude continuity passes for the documented samples. Pole stretching, radial textures and subject/motion defects remain. Moving-camera behavior, updated Ripple guide behavior, outpainting, FLUX reference editing, full FLUX.2-dev, long clips and 4K are not newly qualified by this delivery.
