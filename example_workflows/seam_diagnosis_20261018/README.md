# October 18 API workflows

These API-format graphs are for MCP or ComfyUI API submission. For editable canvas workflows, use the [LTX 2.5 UI presets](../ltx25/README.md). Models and sample media are not included. Replace the living-room input filenames with your own 2:1 assets; preserve alignment between a gray video and its textured look frame.

| Graph | Tested settings | Dependencies beyond core and this pack |
|---|---|---|
| [Ripple](ripple_stock_nolora_API.json) | LTX 2.5, 65 frames, eight Euler/simple steps, CFG 1, Ripple 1.35 | ComfyUI-LTXVideo, KJNodes, VideoHelperSuite |
| [First-frame I2V](firstframe_stock_nolora_API.json) | LTX 2.5, 65 frames, eight Euler/simple steps, CFG 1 | KJNodes, VideoHelperSuite |
| [H3 first frame](h3_rope_API.json) | 56 frames, 20 res_multistep/simple steps, periodic width enabled | VideoHelperSuite; see model filenames in graph |
| [H3 gray-video Canny](h3_canny_rope_API.json) | 56 frames, same H3 sampling, edge control | H3 Fun ControlNet Union weights, VideoHelperSuite |
| [Klein 9B distilled](flux_klein_rope_API.json) | Image, four steps, CFG 1, spherical width | [ComfyUI-SpheRoPE](https://github.com/cedarconnor/ComfyUI-SpheRoPE) |
| [Klein base 9B FP8](flux_base_erp_g3_API.json) | Image, 20 steps, CFG 4, spherical width and geometric guidance 3 | ComfyUI-SpheRoPE |

All use 1024x512 and fixed seed 80084. LTX uses no panorama LoRA, geometry patches off and native dual-rotation decode with four latent blend columns. Ripple still needs its native adapter plus both guide inputs. H3 uses dual-rotation decode with four columns and rolling off; alignment of its separate ControlNet stream under rolling is unqualified. Canny control is an edge-guided experiment, not a qualified material-transfer replacement for Ripple.

Klein graphs use circular VAE decoding with eight latent columns and produce images, not videos. They use nodes from the separate SpheRoPE pack. Full FLUX.2-dev and its Mistral encoder were not installed or tested. The base comparison changes RoPE and guidance together, so it does not isolate either effect.

All six graphs passed live schema checks, and their selected settings completed render tests. Both LTX UI counterparts also rendered and matched all 65 API reference PNG frames exactly. See the [investigation](../../docs/SEAM_DIAGNOSIS_2026-10-18.md) and [recorded validation](../../docs/validation/2026-10-18/README.md).

These are reproducible starting points, not a guarantee of seamless production output. Floor texture mismatch, temporal geometry changes and radial pole distortion remain. A small boundary-jump score does not certify coherent geometry.
