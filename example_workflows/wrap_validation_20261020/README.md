# ComfyUI longitude-wrap profiles

These are **API-format graphs**, executed through ComfyUI MCP on October 20.
They are not canvas-layout JSON files. Submit with MCP `run_workflow` or
ComfyUI's `/prompt` API. See the [validation report](../../docs/COMFY_WRAP_VALIDATION_2026-10-20.md)
for acceptance, checkpoint versions, evidence and limitations.

| Executed graph | Recommended output branch | Qualification |
|---|---|---|
| [Klein base, spherical scale 2](klein_base_spherical_seed42_API.json) | `circular`, pad 9 | Text-to-image, 2048×1024; wrap pass on seeds 42/43 |
| [Klein base, smaller radius / scale 4](klein_base_radius_half_seed42_API.json) | `circular`, pad 9 | Both seeds pass wrap; reduced distinct pole bullseye/spokes in this scene |
| [Klein distilled, spherical scale 2](klein_distilled_spherical_seed42_API.json) | `circular`, pad 9 | Text-to-image, 2048×1024; wrap pass on seeds 42/43 |
| [LTX 2.3 first-frame](ltx23_i2v_spherical_seed42_API.json) | `seam`, circular pad 4 | 1024×512, 65 frames; wrap pass on seeds 42/43 |
| [LTX 2.5 first-frame](ltx25_i2v_spherical_seed42_API.json) | `narrow`, dual rotation, blend 1 | 1024×512, 65 frames; wrap pass on seeds 42/43 |
| [MiniMax H3 first-frame](h3_i2v_periodic_seed42_API.json) | `dual`, dual rotation, blend 4 | 1024×512, 56 frames; wrap pass on seeds 42/43 |
| [Final LTX 2.5 panorama video, 121 frames](ltx25_pano13_final_121f_API.json) | Matching MP4 and PNG outputs, dual rotation blend 1 | 1024×512, 24 fps, 5.04 seconds; seed 42 wrap pass after inspecting all 121 exported MP4 frames |

Graphs retain the same-latent stock-decoder and alternate-padding controls
used in testing. Those branches cost additional decode time and save extra
files; their presence does not make them recommended outputs. Graphs preserve
the executed model, sampling and decoder settings. The LTX 2.5 graph's video
export is routed to the inspected `narrow` decoder instead of the original
blend-4 preview; both decoder branches were rendered in the test.
Output prefixes are relative to ComfyUI's output folder.

For first-frame workflows, place the supplied `pano_2k_00013_.png` in
ComfyUI's input directory, or change the `LoadImage` filename. The image is
not bundled. Model filenames match the local test installation; change loader
paths if the same checkpoints are stored elsewhere. Klein needs Qwen3 8B
and the FLUX.2 VAE; LTX 2.3 and 2.5 need their respective native components.
The LTX graphs also use KJNodes for constants and full-panorama resizing.
H3 uses the FL2VA INT8 checkpoint, its Qwen3-VL 32B encoder, and native video VAE.
Video preview branches use VideoHelperSuite.

The profiles use **no panorama LoRA and no rolling**. Keep the panorama at
2:1 and resize the full image without cropping. A continuous longitude join
does not establish correct poles, subject identity, prompt adherence, or
stable camera motion. These short fixed-camera tests do not qualify
outpainting, gray-guide video, 4K or long clips.

The final 121-frame graph removes the disabled LoRA/rolling nodes and diagnostic
decoder branches, retaining only the selected decoder and matching video/PNG
outputs. It uses the supplied `pano_2k_00013_.png`. Its longitude join passes,
but subject scale/appearance changes and pole stretching remain.

Full FLUX.2-dev is not installed and is not qualified by the Klein tests.
The FLUX spherical patch deliberately defers on reference-image token
layouts; these text-to-image results do not establish Klein image-edit support.

## Diagnostic text-to-video controls

These use no starting image. Each passed the longitude-wrap inspection on
seed 42 only, while retaining substantial pole and geometry defects. They
are included for reproducibility, not recommended as artist-quality presets:

- [LTX 2.3 control](controls/ltx23_t2v_spherical_seed42_API.json), select `seam`.
- [LTX 2.5 control](controls/ltx25_t2v_spherical_seed42_API.json), select `narrow`.
  The historical MP4 preview uses blend 4; the inspected blend-1 frames are
  saved by the `narrow` PNG branch.
- [H3 control](controls/h3_t2v_periodic_seed42_API.json), select `dual`.
