# Current ComfyUI workflows

Drag these **canvas JSON files** into ComfyUI. They use the October 20 longitude-wrap profiles: model-specific positional encoding, wrap-aware decoding, fixed seed 42, no panorama LoRA, no rolling and no wrapped noise. Export nodes receive the selected decoder output directly. Model files and input media are not bundled.

| Workflow | Default | Validation scope |
|---|---|---|
| [LTX 2.5 first-frame](ltx25/FirstFrame_NoPanoramaLoRA_LTX25_360.json) | 1024×512, 121 frames, 24 fps; spherical RoPE; dual decode blend 1 | Tested panorama video; every exported frame inspected for wrap |
| [LTX 2.3 first-frame](ltx23/FirstFrame_LTX23_360.json) | 1024×512, 65 frames; spherical RoPE; circular decode pad 4 | API profile passed wrap on seeds 42/43 |
| [MiniMax H3 first-frame](MiniMax-H3-SEAMLESS-FL2VA.json) | 1024×512, 56 frames; periodic longitude RoPE blend 1; dual decode blend 4 | API profile passed wrap on seeds 42/43; no post-decode upscaler |
| [Klein base](flux2/Klein_Base_360.json) | 2048×1024, 50 Euler steps, CFG 4, ERP gamma 6, spherical scale 2, decode pad 9 | Text-to-image wrap passed seeds 42/43 |
| [Klein base, smaller radius](flux2/Klein_Base_SmallerRadius_360.json) | Same, spherical scale 4 | Reduced distinct pole bullseye/spokes on both tested seeds; radial distortion remains |
| [Klein distilled](flux2/Klein_Distilled_360.json) | 2048×1024, 4 Euler steps, CFG 1, spherical scale 2, decode pad 9 | Text-to-image wrap passed seeds 42/43 |
| [LTX 2.3 text-only control](Burgstall-SEAMLESS-equirectangular-T2V.json) | 1024×512, 65 frames; single-stage spherical profile, circular decode pad 4 | One seed passed wrap; severe pole/global geometry defects remain |
| [LTX 2.5 Ripple guide template](ltx25/Ripple_NoPanoramaLoRA_LTX25_360.json) | 1024×512, 65 frames; spherical RoPE, dual decode blend 1; native Ripple adapter 1.35 | Live schema validation only for the updated spherical profile; guide appearance/motion requires visual requalification |
| [LTX 2.3 outpainting template](Burgstall-SEAMLESS-VR-Outpaint.json) | Spherical RoPE, circular decode pad 4; preserves projection and outpainting adapter | Dependency-limited; not executable in the validation installation and not visually qualified |

`FirstFrame_CameraMove_LTX25_360.json` now contains the same fixed-camera baseline as the main first-frame preset. Its old filename remains for compatibility; edit the prompt to experiment with movement and inspect the result. `Ripple_Playblast_LTX25_360.json` uses the same updated wrap settings as the Ripple template. Neither legacy filename enables a cross-version panorama LoRA.

## Inputs and installation

For I2V, upload `pano_2k_00013_.png`, or select your own already-seamless 2:1 panorama. LTX resizes the entire panorama without cropping. Keep output width twice height. Update the LTX 2.3 decoder's `target_width` when changing width. These fixed-camera tests do not qualify moving cameras, long clips or 4K.

Use this pack plus core ComfyUI with the relevant model support, KJNodes for LTX constants/resizing, and VideoHelperSuite for video export. LTX 2.3 text/audio loading and Ripple guide nodes also require ComfyUI-LTXVideo. Native model filenames are saved in the loaders; use the matching checkpoint family. See [LTX 2.5 setup](ltx25/README.md) and [executed API profiles](wrap_validation_20261020/README.md). No ComfyUI-SpheRoPE dependency is needed.

Ripple requires both a gray ERP guide video and an aligned textured look frame. Its native Ripple LoRA is retained because it supplies guide conditioning. The outpainting graph similarly retains its required `360vroutpaint_v2_step_07000.safetensors` adapter; removing it would change the task.

The outpainting template also needs its original LTX 2.3 component files and `EstimateCameraGeoCalib`, `RectilinearToEquirect`, and `EquirectSourceComposite`. Live validation reports eight missing-node/model errors on this installation. Do not interpret its migrated settings as a successful outpainting test. Its decoder target width follows the projected guide width.

## What a pass means

Longitude continuity is the acceptance criterion. Pole pinching, radial textures, subject duplication and motion/identity drift remain possible. H3's encoding is periodic in longitude, not the full latitude-dependent spherical method. Klein results do not qualify FLUX.2-dev or reference-image editing. Negative prompts have no CFG effect in the CFG-1 distilled video profiles.

The [validation report](../docs/COMFY_WRAP_VALIDATION_2026-10-20.md) records the renders and their limits. [Workflow delivery checks](../docs/WORKFLOW_DELIVERY_2026-10-20.md) distinguish runtime evidence from schema validation.

## Historical comparisons

The dated [native-reference](native_reference_20261018/README.md), [seam-diagnosis](seam_diagnosis_20261018/README.md) and [wrap-validation](wrap_validation_20261020/README.md) API graphs preserve the experiments as executed, including intentionally disabled patches and alternate decoder branches. They are not all recommended presets. The former large two-stage T2V canvas is retained in [historical/](historical/README.md); the main T2V filename now opens the simpler tested single-stage control.
