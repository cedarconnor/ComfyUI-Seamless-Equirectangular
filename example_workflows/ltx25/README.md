# Artist handoff: LTX 2.5 equirectangular video

Two ComfyUI UI-format workflows: drag either JSON onto the canvas. Read its START HERE note, load your inputs, edit ARTIST PROMPT, and queue. Sample media and models are not bundled. Replace the example image/video selections with your own uploaded 2:1 assets before queuing.

## 1. Ripple_Playblast_LTX25_360.json

Requires a gray equirectangular playblast **and a textured look frame aligned with its first frame**. This is Ripple appearance transfer, not gray-video-plus-text alone. The graph prepends the look frame to the source guide sequence as required by this tested Ripple setup; it does not promise frame-exact source timing. Keep the guide video at least as long as the requested output at the selected FPS.

Ripple strength 1.35; no panorama LoRA; full guide and attention strength; EquiRoPE enabled at blend 1, threshold 10, pole_gamma 3, bottom pole; native LTX 2.5 dual-rotation decoder with blend_columns 1. Per-step roll and wrapped noise are omitted. EquiRoPE on/off differences were small on this shot; the enabled setting is a geometric aid, not a proven universal quality gain.

## 2. FirstFrame_CameraMove_LTX25_360.json

Requires one **already-textured, seamless 2:1 panorama**. No conditioning video, Ripple, Union or 3DREAL is used. Replace the example living-room description and camera sentence in ARTIST PROMPT. Start with one short, slow dolly and a level horizon. Camera motion and newly revealed geometry are generated; this is not a measured camera-path controller. If the starting frame is gray, it needs a separate look-development step to get the intended textured starting appearance.

Uses Seamless-Equirectangular-LTX2.3-LoRA.safetensors at 0.20 with the Equirectangular trigger, EquiRoPE (both poles), first-frame strength 1, and native LTX 2.5 dual-rotation decoding. The companion LoRA was trained for 2.3; this handoff validates its execution and this particular result on 2.5, not general native 2.5 compatibility. It is not the unverified LTX-2_360Panaram download.

## Shared controls and outputs

- Tested defaults: 1024 x 512, 65 frames at 24fps (2.708 seconds), seed 80084, Euler/simple, eight steps, CFG 1, denoise 1.
- Width must be twice height. Use dimensions divisible by 32 and frame counts 8n+1, such as 33, 65 or 97. The shared controls feed generation and resizing; FPS also feeds video export and guide resampling.
- Inputs are resized without cropping or letterboxing. Supply true 2:1 panoramas; stretching a perspective image does not make a panorama.
- Keep eight-step / CFG 1 settings for this distilled model. More steps are not a validated quality improvement. The negative prompt has no CFG effect at CFG 1.
- MP4 (H.264, CRF 12) and lossless PNG sequences save to ComfyUI/output/artist-handoff/<workflow name>. Audio is not exported, although the model uses an internal AV latent.
- The MP4 is a 2:1 video. Automatic headset recognition may require separate spherical metadata injection; the workflow does not inject it.
- 2048 x 1024 and 4096 x 2048 can be entered for larger jobs, but these exact handoff graphs have not been validated at those resolutions. Start with the tested preview. A first-frame source at 4K does not make this 1024-wide output a 4K validation.

## Required installation

Current core ComfyUI with LTX 2.5 and ComfyNumberConvert support, plus:

- ComfyUI-Seamless-Equirectangular, including EquirectLTX25VAEDecode.
- comfyui-kjnodes (constants and ImageResizeKJv2).
- comfyui-videohelpersuite (video input/export).
- ComfyUI-LTXVideo (LTXAddVideoICLoRAGuideAdvanced), for Ripple only.

No ComfyUI-SpheRoPE or LatLong dependency is required. The graphs use the Seamless-Equirectangular pack requested for this handoff.

Models, using these exact dropdown filenames:

| Directory | Filename | Used by |
|---|---|---|
| models/diffusion_models | ltx-2.5-22b-distilled-transformer-bf16.safetensors | Both |
| models/text_encoders | gemma4-12b-with-proj-ltx-2.5-bf16.safetensors | Both |
| models/vae | ltx-2.5-video-vae-bf16.safetensors | Both |
| models/vae | ltx-2.5-audio-vae-bf16.safetensors | Both |
| models/loras | LTX25_Ripple_v11.safetensors | Ripple |
| models/loras | Seamless-Equirectangular-LTX2.3-LoRA.safetensors | First frame |

Extra-model paths are fine if the same filenames appear in ComfyUI. The [companion LoRA model card](https://huggingface.co/TheBurgstall/Seamless-Equirectangular-LTX2.3-LoRA) identifies its 2.3 training and Equirectangular trigger; its commercial-use terms require a separate license for the creator's contribution.

## Validation boundary

The [main README](../../README.md#ltx-25-support-and-artist-workflows) summarizes the dated validation and remaining defects. Both graphs were loaded in the installed ComfyUI frontend, exported back to API format with all execution inputs matching, and rendered successfully at their delivered defaults. This confirms working workflows; it does not establish a global optimum or guarantee an artifact-free seam on new shots. The installed frontend displayed pre-existing version warnings; it was not upgraded during this handoff.

Before delivery, inspect a longitude-rolled view, the poles and the whole clip in a 360 viewer. Decoder seam handling does not eliminate changing geometry, texture swimming, floor creases or temporal ghosts. These workflows produce spatial 360 wraparound video, not automatically looping video in time.

EquiRoPE changes global LTX RoPE state in this installed pack. Deleting or bypassing its node does not turn it off. For comparisons against unrelated workflows, execute EquiRoPE with enabled=False or restart ComfyUI. For exact reproduction of a changed pole configuration, use a fresh ComfyUI session rather than relying on cached patch nodes.
