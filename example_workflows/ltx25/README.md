# Artist handoff: LTX 2.5 equirectangular video

Start with the October 20 presets below. Drag a JSON onto the ComfyUI canvas, read its START HERE note, load your inputs, edit ARTIST PROMPT and queue. Models and sample media are not bundled; replace the living-room filenames with your own uploaded 2:1 assets.

| Workflow | Inputs | Conditioning |
|---|---|---|
| [Ripple without panorama LoRA](Ripple_NoPanoramaLoRA_LTX25_360.json) | Gray ERP playblast plus a textured look frame aligned with its first frame | Native LTX 2.5 Ripple adapter at 1.35 |
| [First-frame I2V without panorama LoRA](FirstFrame_NoPanoramaLoRA_LTX25_360.json) | One already-textured, seamless panorama | Native first-frame conditioning; no LoRA |

Ripple appearance transfer requires both inputs. The graph prepends the look frame to the guide sequence as required by this tested setup; it does not promise frame-exact source timing. Keep the guide at least as long as the requested output at the selected FPS.

First-frame I2V defaults to the tested fixed-camera jumping-boy prompt. You can experiment with camera motion and newly revealed geometry from the prompt. It is not a measured camera-path controller. Start with a short, slow move and a level horizon. A gray first frame needs a separate look-development step to establish the intended textured appearance.

## Shared settings

- Tested defaults: 1024x512, 121 frames for first-frame I2V or 65 for Ripple, at 24 fps, fixed seed 42, eight Euler/simple steps, CFG 1, denoise 1.
- Native `EquirectLTX25VAEDecode`, `blend_columns=1` latent columns. This uses two VAE decodes and composites the half-rotated result around the longitude boundary.
- Spherical RoPE is enabled at blend 1 / threshold 10 / radius 1. Panorama LoRA, EquiRoPE and rolling are omitted. Wrapped noise is omitted. Ripple still requires its native adapter.
- Width must be twice height. Use dimensions divisible by 32 and frame counts 8n+1, such as 33, 65, 97 or 121. For Ripple, shared FPS controls guide resampling and export. In the first-frame graph, update both the FPS constant and VideoCombine frame rate if changing 24 fps.
- Inputs are resized without cropping or letterboxing. Stretching a perspective image does not create a panorama.
- Keep eight steps and CFG 1 for this distilled baseline. Negative prompts have no CFG effect at CFG 1; more steps are not a validated quality improvement.
- Outputs are MP4 (H.264, CRF 12) and lossless PNG sequences. Audio is not exported, although sampling uses an internal AV latent. See the save nodes for output prefixes.
- Headsets may require separate spherical metadata injection; these workflows do not inject it. Spatial wraparound does not imply a temporal loop.

## Required installation

Use core ComfyUI with LTX 2.5 and ComfyNumberConvert support, plus:

- ComfyUI-Seamless-Equirectangular, including `EquirectLTX25VAEDecode`.
- comfyui-kjnodes for constants and resizing.
- comfyui-videohelpersuite for video input/export.
- ComfyUI-LTXVideo for `LTXAddVideoICLoRAGuideAdvanced` in Ripple.

These LTX graphs do not require ComfyUI-SpheRoPE or LatLong. Restart ComfyUI once when upgrading from this pack's older global RoPE implementation; the new model-owned patches prevent cached branches from changing other workflows.

| Directory | Dropdown filename | Used by |
|---|---|---|
| models/diffusion_models | ltx-2.5-22b-distilled-transformer-bf16.safetensors | Both |
| models/text_encoders | gemma4-12b-with-proj-ltx-2.5-bf16.safetensors | Both |
| models/vae | ltx-2.5-video-vae-bf16.safetensors | Both |
| models/vae | ltx-2.5-audio-vae-bf16.safetensors | Both |
| models/loras | LTX25_Ripple_v11.safetensors | Ripple |

Extra-model paths are fine if the same filenames appear in ComfyUI. No panorama LoRA download is required for these two presets.

## Validation and limits

The final first-frame API profile rendered 121 frames through ComfyUI. Every encoded MP4 frame was inspected at the longitude join; wrap passed. Subject scale/appearance changes, duplication and pole stretching remain. The canvas version contains the same sampling and decoder branch. See the [current validation report](../../docs/COMFY_WRAP_VALIDATION_2026-10-20.md) and [delivery checks](../../docs/WORKFLOW_DELIVERY_2026-10-20.md).

The updated Ripple graph passes live schema validation but has not been visually requalified with spherical encoding. Its older geometry-off / blend-4 graph rendered successfully on October 18; that result does not prove the new settings preserve guide motion or appearance. Inspect the entire join and guide adherence on your own input. Neither profile establishes a precise camera path, long-clip stability or 4K quality.

## Legacy filenames

[FirstFrame_CameraMove_LTX25_360.json](FirstFrame_CameraMove_LTX25_360.json) now contains the same tested fixed-camera baseline as the main first-frame file. Its filename is retained for compatibility, not as a moving-camera qualification. [Ripple_Playblast_LTX25_360.json](Ripple_Playblast_LTX25_360.json) has the same updated wrap settings as the main Ripple template. Neither uses the 2.3-trained panorama LoRA.
