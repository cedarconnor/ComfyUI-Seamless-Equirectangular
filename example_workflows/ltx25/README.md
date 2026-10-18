# Artist handoff: LTX 2.5 equirectangular video

Start with the October 18 presets below. Drag a JSON onto the ComfyUI canvas, read its START HERE note, load your inputs, edit ARTIST PROMPT and queue. Models and sample media are not bundled; replace the living-room filenames with your own uploaded 2:1 assets.

| Workflow | Inputs | Conditioning |
|---|---|---|
| [Ripple without panorama LoRA](Ripple_NoPanoramaLoRA_LTX25_360.json) | Gray ERP playblast plus a textured look frame aligned with its first frame | Native LTX 2.5 Ripple adapter at 1.35 |
| [First-frame I2V without panorama LoRA](FirstFrame_NoPanoramaLoRA_LTX25_360.json) | One already-textured, seamless panorama | Native first-frame conditioning; no LoRA |

Ripple appearance transfer requires both inputs. The graph prepends the look frame to the guide sequence as required by this tested setup; it does not promise frame-exact source timing. Keep the guide at least as long as the requested output at the selected FPS.

First-frame I2V generates camera motion and newly revealed geometry from the prompt. It is not a measured camera-path controller. Start with a short, slow move and a level horizon. A gray first frame needs a separate look-development step to establish the intended textured appearance.

## Shared settings

- Tested defaults: 1024x512, 65 frames at 24 fps, fixed seed 80084, eight Euler/simple steps, CFG 1, denoise 1.
- Native `EquirectLTX25VAEDecode`, `blend_columns=4` latent columns. This uses two VAE decodes and composites the half-rotated result around the longitude boundary.
- Panorama LoRA, EquiRoPE, spherical RoPE and per-step rolling are off. Wrapped noise is omitted. Ripple still requires its native adapter.
- Width must be twice height. Use dimensions divisible by 32 and frame counts 8n+1, such as 33, 65 or 97. Shared FPS controls guide resampling and video export.
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

Both UI files rendered through MCP UI conversion (34/27 executable nodes). All 65 PNG frames from each exactly matched the corresponding tested API output. This is live graph validation through MCP, not a separate frontend interaction test. [API equivalents](../seam_diagnosis_20261018/README.md) and [comparison evidence](../../docs/validation/2026-10-18/ui_runtime_comparison.json) are included.

Separate 1536x768, 33-frame runs with seed 80085 completed. These results do not qualify 4K. A 4K input frame does not make the default 1024-wide output a 4K test.

Inspect the longitude join, poles and whole clip in a 360 viewer. Decoder blending reduces the measured pixel discontinuity, but floor texture swimming, creases, ceiling deformation and geometry drift remain. Early rolling and guide-release variants introduced new folded geometry in this shot and were rejected as defaults. See the [October 18 investigation](../../docs/SEAM_DIAGNOSIS_2026-10-18.md) for comparisons and limitations.

## Historical September presets

[Ripple_Playblast_LTX25_360.json](Ripple_Playblast_LTX25_360.json) and [FirstFrame_CameraMove_LTX25_360.json](FirstFrame_CameraMove_LTX25_360.json) retain their earlier settings for comparison: EquiRoPE enabled and one-column native decode. The first-frame preset additionally uses `Seamless-Equirectangular-LTX2.3-LoRA.safetensors` at 0.20. This is a 2.3-trained adapter, not a native 2.5 adapter; consult its [model card and license](https://huggingface.co/TheBurgstall/Seamless-Equirectangular-LTX2.3-LoRA) before use. The October presets above supersede these as the tested starting points for existing ERP inputs.
