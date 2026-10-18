# Seam investigation — October 18, 2026

Status: 34 MCP jobs completed successfully, including LTX 2.5, MiniMax H3, Klein 9B distilled and Klein base 9B FP8. Three LTX guide comparisons were confounded by a reproduced cache/global-state bug and are excluded as stock controls. All three implementation fixes now have live render proof. The model-scoped LTX patch also passes 41 tests (4 skipped). Longer validation is recorded below; remaining pole/texture defects prevent a claim of universally artifact-free output.

## Reproduced problems

1. `periodic_rope.build_freqs_periodic` inferred spatial height from total token count divided by width. Video frames and appended guides became additional latitude rows. The same spatial location received different width phases in different frames. A new regression test failed on 37.5% of the width-angle elements before the fix. The fix obtains row/column coordinates from the position grid and gathers the spatial angle table by those coordinates, preserving temporal and height phases. It also handles shuffled token order.
2. The MiniMax H3 RoPE wrapper passed the removed `frame_count` argument to the installed `PackedLayout` constructor. The existing lifecycle test reproduced the crash. The wrapper now matches the installed core constructor.
3. The handoff uses EquiRoPE, which scales low-frequency width positions toward the poles but preserves high-frequency linear positions. It is distinct from periodic spherical width encoding and cannot establish longitude closure by itself.
4. The native LTX 2.5 dual-rotation decoder suppresses the pixel discontinuity, but a narrow blend cannot correct incompatible generated structure. Circular padding alone did not eliminate the reproduced wider band either.
5. Both legacy LTX RoPE nodes retained global process state. ComfyUI cached a disabled node, another graph enabled the global patch, and returning to the cached disabled graph silently left spherical RoPE active. This affected `ripple_guide0` and `ripple_guide4`; `ripple_guide4_spherical` was also part of that confounded comparison. The nodes now return model clones with object patches and captured parameters. New regressions failed before the change and now pass, including independent cached branches, split/interleaved layouts, audio passthrough and actual ModelPatcher patch/unpatch restoration. Legacy standalone install helpers remain available, but the nodes no longer call them. Restart clears the old implementation from the running process.

## Verification completed

Environment: ComfyUI 0.39.0, PyTorch 2.12.0+cu130, embedded Python 3.12.10, NVIDIA RTX A6000. MCP `server_info`, `system_stats`, `validate_workflow`, `run_workflow`, and `job` used against local port 8188. Only one ComfyUI server process was found. MCP memory release recovered about 46 GiB free VRAM.

- Initial suite: 35 passed, 4 skipped, 1 failed (`test_minimax_h3_rope_math_and_lifecycle`).
- New spherical-coordinate regression: failed before implementation.
- After initial fixes: 37 passed, 4 skipped. After the model-scoped patch: 41 passed, 4 skipped; `git diff --check` clean.
- Spatial angle math compared directly against `_build_ltx23_width_angles` extracted from the original repository: maximum absolute difference 0 for H/W/channel counts 16/32/22, 32/64/22, 16/32/42.

Actual MCP runs use the supplied handoff's API graphs, seed 80084, 1024x512, 17 frames, eight Euler/simple steps, CFG 1. The Ripple run has no panorama LoRA; first-frame baseline retains the handoff's 0.2 panorama LoRA. Ordinary and dual-rotation decodes share the same sampled latent within each baseline. All images are raw saved PNGs, before video compression.

| Run | Prompt ID | Result |
|---|---|---|
| Ripple baseline | `eb50e567-7593-4cdb-a8fc-76b3dfb437ea` | Completed, stock + dual decode + saved latent |
| First-frame baseline | `03ed06a7-aa39-43f5-ab1f-c641e693ec22` | Completed, stock + dual decode + saved latent |
| Ripple decoder comparison | `98ad4387-9fbf-435f-bb6a-81eeee9b56ea` | Completed, circular padding 4/8 and dual rotation 4 |
| Input VAE roundtrip | `647c85b0-bdab-4b03-b8e3-2560fa8e76b3` | Completed, stock/circular encode crossed with three decoders |

Mean boundary-jump / median interior horizontal-difference ratio (lower is not automatically better):

| Case | Mean ratio | Maximum over frames |
|---|---:|---:|
| Ripple stock decode | 14.757 | 26.137 |
| Ripple dual decode, 1 column | 1.194 | 5.193 |
| Ripple circular decode, 4 columns | 2.581 | 8.570 |
| Ripple circular decode, 8 columns | 2.131 | 6.047 |
| Ripple dual decode, 4 columns | 1.197 | 5.198 |
| First-frame stock decode | 8.350 | 9.304 |
| First-frame dual decode, 1 column | 0.407 | 0.466 |

These measurements establish a VAE boundary contribution, not universal closure. Full seam-facing, zenith and nadir views accompany the measurements. The supplied look frame itself contains some irregular floor texture when projected onto a sphere; do not attribute every inherited defect to generation.

## Additional completed comparisons

The user restarted ComfyUI at 10:02:55 on October 18 (PID 36132). The next runs loaded the temporal-coordinate and H3 constructor fixes. All runs listed below completed through MCP; raw output paths and all prompt IDs are archived in `output/seam_diagnosis_20261018/all_jobs.json` and corresponding history files.

LTX screening: seed 80084, 1024x512, 17 frames, eight Euler/simple steps, CFG 1. Ripple adapter 1.35; no panorama LoRA in any of these rows. Dual decode uses four latent blend columns. The first six comparisons below preceded the cached-toggle contamination and have explicit matching toggle executions in the logs.

| Case | Ordinary decode mean edge ratio | Dual-rotation mean edge ratio |
|---|---:|---:|
| Ripple, geometry patches off | 14.113 | 1.106 |
| Ripple, corrected spherical width | 25.243 | 1.716 |
| Ripple, spherical width plus dual-average denoising | 23.928 | 2.079 |
| First frame, geometry patches off | 7.753 | 0.404 |
| First frame, corrected spherical width | 10.583 | 0.431 |
| First frame, circular first-frame encode plus spherical width | 20.217 | 0.939 |
| H3 first frame, periodic width off | 2.368 | 2.110 |
| H3 first frame, periodic width on | 4.569 | 1.341 |
| H3 gray-video Canny control, periodic width off | 3.177 | 1.886 |
| H3 gray-video Canny control, periodic width on | 5.073 | 2.061 |

H3 uses its native 17k+5 frame grid: 22 frames, res_multistep/simple, 20 steps, seed 80084. Canny control uses the installed Fun ControlNet Union adapter, strength 1, full schedule; no per-step roll because its separate control stream has not been qualified for roll alignment. The control run follows changing camera orientation, but floorboard mismatch remains. This is a tested edge-conditioned path, not proof of reliable gray-material transfer.

FLUX screening uses the alternate ComfyUI-SpheRoPE pack, 1024x512, seed 80084, a misty forest prompt and circular decode (8 latent columns) in every case:

| Model and treatment | Mean edge ratio |
|---|---:|
| Klein 9B distilled, stock, 4 steps / CFG 1 | 1.149 |
| Klein 9B distilled, spherical width, 4 steps / CFG 1 | 0.838 |
| Klein base 9B FP8, stock, 20 steps / CFG 4 | 1.411 |
| Klein base 9B FP8, spherical width plus geometric guidance 3 | 0.896 |

Klein outputs are images. The base comparison changes both RoPE and geometric guidance, so it does not isolate their individual effects. The seam-facing views improve, but zenith/nadir views still show radial convergence and stretched vegetation. No claim of distortion-free poles follows from these scores. FLUX.2-dev and its Mistral encoder are absent; they were not downloaded or tested.

## Usable starting points and remaining validation

For this living-room sample, use native LTX 2.5 dual-rotation decode with `blend_columns=4`, geometry patches off, eight Euler/simple steps and CFG 1. This reduces the measured pixel boundary jump about 13x for Ripple and 19x for first-frame I2V relative to ordinary decoding of the same latents. It does not remove all floor texture mismatch. Both paths work without the panorama LoRA; Ripple still needs its native 2.5 adapter and an aligned textured look frame.

`output/seam_diagnosis_20261018/Screened_Workflows.zip` contains two editable UI starting points and six API graphs. The UI copies inherit the handoff layout, pass local link checks, and completed live renders through MCP UI conversion (34/27 executable nodes). All 65 PNG frames from each exactly match the tested API reference. This verifies the graphs through MCP, not a separate frontend interaction session. API graphs were checked against live schemas. The first-frame copies remove the zero-strength panorama loader completely. Other model filenames match the tested installed weights.

Native ERP MP4s and half-rotated join previews are saved beside the evidence for Ripple, LTX first-frame, H3 first-frame and H3 Canny. These short previews preserve the actual 17/22-frame durations at 24 fps. PNGs remain the metric source; MP4 compression is not used to judge pixel closure.

The prepared 65-frame comparisons completed after the second restart. Both native LTX paths also completed at 1536x768, 33 frames, seed 80085. No panorama LoRA was used. The current evidence does not establish a universal solution for seamless moving-camera ERP video.

## Post-restart verification and longer clips

The second user restart created PID 52980 at 11:36:01. The live system endpoint and MCP render calls succeeded, even though comfy-cli `server_info` incorrectly reported its unmanaged background server as stopped.

**Cache isolation is verified on the live model.** Run stock first-frame I2V, then spherical I2V, then reuse the cached disabled model nodes and force fresh sampling through an identity `SplitSigmas(step=0)` path. History confirms model nodes 1501/1950/1951 were cached and sampler 2999 was not. All 65 resampled PNG frames match the initial stock run exactly: maximum pixel difference 0. The spherical run differs from stock by a mean 9.69/255, confirming that the enabled branch actually changed behavior. Evidence: `cache_isolation_runtime.json`, the three API graphs and their history files. Changing a sampler node ID alone is insufficient to invalidate ComfyUI's input-signature cache; the initial queued probe was replaced before execution.

| Longer LTX case | Frames / resolution | Ordinary decode edge ratio | Dual-rotation edge ratio |
|---|---|---:|---:|
| Ripple, patches off, seed 80084 | 65 / 1024x512 | 15.161 | 1.341 |
| Ripple, spherical width, seed 80084 | 65 / 1024x512 | 13.506 | 1.138 |
| First frame, patches off, seed 80084 | 65 / 1024x512 | 8.644 | 0.595 |
| First frame, spherical width, seed 80084 | 65 / 1024x512 | 11.861 | 0.767 |
| Ripple, patches off, seed 80085 | 33 / 1536x768 | 14.387 | 2.143 |
| First frame, patches off, seed 80085 | 33 / 1536x768 | 10.728 | 0.555 |
| Ripple, dual-average roll during first half | 65 / 1024x512 | 16.177 | 1.428 |
| Ripple, early roll plus 64px guide-release band | 65 / 1024x512 | 15.794 | 1.580 |

The higher-resolution runs change both resolution and seed; they establish execution and another visual sample, not an isolated resolution effect. The identity cache probe is not a second independent seed.

**Rejected:** both early-roll variants invent a folded white floor/wall patch by frame 64. They do not belong in the default workflow despite reducing some score peaks. Spherical width does not consistently improve the living-room result: Ripple's average score improves slightly at 65 frames, while first-frame scores worsen and floor deformation remains.

**Metric limitation verified:** the worst-scoring stock Ripple frame (index 10) has a bright window edge crossing the longitude boundary. A native-resolution half-rolled strip shows the window continuing correctly; that high edge gradient is not automatically a seam defect. See `worst_edge_strip.png` and `worst_edge_review.json`. Inspect geometry and texture continuity, not just first/last-column similarity.

The two 65-frame native LTX results were projected into seam-facing, zenith and nadir videos. Watch ran locally with transcription disabled; all five selected views for each clip were inspected, alongside first/middle/last and worst-edge native PNG views. This samples temporal behavior; it is not a per-frame human review. Floor texture creases/swimming and pole geometry changes remain visible. There was no cloud upload.

The practical default remains: native 2.5 dual-rotation decode, four latent blend columns, geometry patches and per-step roll off, eight Euler/simple steps, CFG 1. Ripple requires its 1.35 native adapter plus aligned gray video and textured look frame. First-frame I2V requires only the already-textured panorama. Negative prompts do not contribute CFG guidance at CFG 1.

Longer H3 first-frame verification also completed: 56 frames at 1024x512, periodic width on, no rolling, 20 res_multistep/simple steps. Mean edge ratio is 8.336 with ordinary decoding and 1.398 with dual rotation. Motion is slight in this prompt; the gray-video control test separately exercises camera motion.

The H3 gray-video Canny control extension completed at 56 frames: ordinary decode mean edge ratio 4.601, dual-rotation 1.774. The changing camera orientation follows the guide qualitatively; floor texture and ceiling deformation remain. Both longer H3 API graphs are included in the package. All 34 recorded jobs have successful completion histories; three earlier guide comparisons remain excluded as confounded controls. The queue is empty.

## Evidence and reference

- Included in Git: [compact validation records](validation/2026-10-18/README.md), [six selected API workflows](../example_workflows/seam_diagnosis_20261018/README.md) and [two LTX UI presets](../example_workflows/ltx25/README.md). Raw media, full histories and the ZIP remain in the local diagnostic archive; `output/` references below and elsewhere in this report describe that archive, not files bundled in the repository.
- `output/seam_diagnosis_20261018/jobs.json`, `*_history.json`, `environment.json`, `pre_restart_logs.json`.
- `output/seam_diagnosis_20261018/metrics.json`, `review.py`, `*_sphere.jpg`, and paired review sheets.
- Raw PNGs and latents: ComfyUI `output/seam-diagnosis-20261018`.
- [Original implementation](https://github.com/orhir/SpheRoPE), [ComfyUI alternate implementation](https://github.com/cedarconnor/ComfyUI-SpheRoPE), [paper](https://arxiv.org/html/2606.32033v1).

The paper's video extension assumes a static camera and minimal atmospheric motion; complex camera trajectories remain a stated limitation. Its supported reference video backbone is LTX 2.3, not a demonstrated guarantee for LTX 2.5 or MiniMax H3.

Upgrade note: restart ComfyUI after installing these fixes to clear the old process-wide RoPE patch. The test server was restarted before the longer-clip and cache-isolation verification.
