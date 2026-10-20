# Sequential SpheRoPE validation

> Historical strict full-sphere gate. The user subsequently accepted pole distortion and prioritized longitude continuity. The [October 20 wrap report](COMFY_WRAP_VALIDATION_2026-10-20.md) supersedes the status below for subsequent video work.

User instruction: validate a working text-generated still first, then image
outpainting, then text-to-video, then image-to-video. Do not advance past an
unvalidated stage. Use `pano_2k_00013_.png` (the jumping boy on an alpine trail)
as the content reference and later conditioning source.

**Status, October 20:** eleven generation profiles completed (six native FLUX.1
and five ComfyUI Klein), plus decoder-only controls. None passes the complete
still-image gate. No outpainting or video stage has started. Following the
user's morning follow-up, MCP successfully launched ComfyUI. The rolling
comparison completed with the saved node fix, but failed visual acceptance.
Browse the outputs at `E:/ComfyUI-test-results/first-principles-20261019/review.html`.

## Acceptance rules fixed before inspecting new outputs

- Correct 2:1 ERP dimensions and full surrounding scene.
- No visible hard longitude join in a rectilinear view centered on the wrap.
- Zenith and nadir must remain plausible in rectilinear views: no conspicuous
  radial smear, pinhole, or unrelated duplicated scene.
- Content must match the request, including one subject rather than repeated
  people around the panorama.
- Boundary-jump measurements support visual inspection; they cannot override
  an obvious artifact. A small score obtained through blur does not pass.
- Before calling a still workflow repeatable, validate at least two seeds.
- Video must additionally preserve the join and scene through its duration.
  Inspect moving views and temporal samples; one successful first frame is not
  a video pass.
- Failures remain recorded. Do not silently replace unsuccessful seeds.

## Gates

1. **Text-to-still:** original standalone SpheRoPE FLUX.1-dev with installed
   BF16 weights. Start at 2048×1024, 50 steps, seed 42, geometry gamma 6,
   spherical RoPE and circular decoding. No panorama LoRA or rolling.
   This matches the paper's stated image resolution/step count while retaining
   the released implementation's guidance warmup.
2. **Image outpainting:** only after gate 1 passes. Preserve a specified region
   of the boy image and reconstruct surrounding ERP content. Record projection
   and mask handling explicitly; ordinary rectangular widening is not enough.
3. **Text-to-video:** only after gate 2 passes. Start with the original LTX 2.3
   method and fixed camera, then test motion separately.
4. **Image-to-video:** only after gate 3 passes. Use the boy image, first with
   a fixed camera and subject motion, then camera motion.

## First-principles code checks

The released FLUX.1 builder currently uses `R = W/(2*pi)`, while the paper's
implementation appendix states `R = W/2`. It also gives image tokens time
position 1; stock FLUX image IDs in this pipeline use 0. These must be measured
against the actual stock embedding and recorded as reference deviations.
First render the released implementation unchanged. If it fails, investigate
one discrepancy at a time; do not combine speculative repairs.

The upstream default branch was checked live on October 19 and still points to
`a40a86b49c25dbd4d0191591f28a0d5baf0c4ca4`, matching the local reference.
The paper's scheduling discussion also specifies geometry guidance throughout
sampling, while the released image CLI defaults to a two-step warmup.
Diagnostic switches preserve the original settings by default and record every
override. CPU probes verify that radius and time interventions affect only their
intended axes.

Source: [paper](https://arxiv.org/html/2606.32033v1),
[released helper](https://github.com/orhir/SpheRoPE/blob/a40a86b49c25dbd4d0191591f28a0d5baf0c4ca4/panorama_diffusers/erp_utils.py).

Equation 2 also shifts spherical X/Y by +R; the FLUX.1 helper omits this
translation, whereas the FLUX.2 helper includes it. A common translation
cancels from image-to-image relative angles in a channel, but not from
image-to-text angles when text retains its original coordinates. Therefore it
cannot simply be dismissed as an equivalent coordinate convention. This is a
measured implementation discrepancy, not yet a proven cause of the artifacts.

## Gate 1 observations

All runs below use the same alpine/jumping-boy prompt, seed 42, BF16 FLUX.1,
2048×1024, 50 steps, semantic CFG 3.5 and circular decoding.
Geometry gamma is 6 except for the final geometry-off control. The two-step
guidance warmup is removed in the last two rows. No LoRA,
rolling or source-image conditioning.

| Run folder | Change from previous row | Edge MAE / ratio | Visual decision |
|---|---|---|---|
| `01_text_still/seed42` | Released reference | 8.923 / 1.708 | Fail: two boys at the wrap, radial stone pattern at nadir |
| `01_text_still/seed42_paper_radius` | Radius becomes W/2 | 8.645 / 1.446 | Fail: one primary boy, but radial sky streaks and swirling vegetation |
| `01_text_still/seed42_paper_radius_stock_time` | Restore time coordinate 0 | 7.115 / 1.246 | Fail: two boys, repeated bright sunset, radial sky and nadir artifacts |
| `01_text_still/seed42_paper_coordinates` | Add +R to spherical X/Y | 6.471 / 1.101 | Fail: two boys and pronounced radial sky/ground artifacts |
| `01_text_still/seed42_paper_coordinates_all_steps` | Start three-way guidance at step 0 | 6.373 / 1.257 | Fail: duplicate boys, repeated sunset and pole spokes persist |
| `01_text_still/seed42_paper_coordinates_no_geometry` | Disable geometry term, gamma 0 | 7.769 / 1.227 | Fail: duplicate boys, striped sky, repeated scenery and severe pole spokes remain |

The radius intervention changes the failure substantially; it does not yet
validate a solution. Each lossless output has a full-image and six-view review.
Restoring the stock time coordinate, adding the paper's positive offset and
activating guidance from step zero did not solve the visual problem. The released warmup gates
the entire three-way branch, including real semantic CFG; this schedule test
does not isolate the geometry term from the semantic term during those first
two steps.
An independent CPU probe checks the paper's +R offset
at the pole and verifies unchanged height and high-frequency width channels.

Supporting controls: the base prompt uses 74 CLIP tokens (limit 77), so it is
not truncated. All installed transformer, VAE and text-encoder tensor headers
are BF16. Transformer shard sizes match the publisher's public manifest;
masked upstream hashes prevent a cryptographic identity claim. An analytic
XYZ sphere verifies the six review viewing directions with maximum center RGB
error below 0.002/255. These checks qualify the diagnostics, not the generated
panoramas.

The installed Diffusers embedding and rotary application were also exercised
through the original wrapper at the actual 64×128 token grid. With the
diagnostic paper coordinates, text embeddings and image time/height rotations
match stock exactly in FP32 and BF16. The released helper changes image time
rotation as expected. This CPU check excludes that interface mismatch; it
does not prove GPU attention or full-generation parity.

## Check of the published visual evidence

The paper's own
[FLUX.1 epsilon-0.06 image](https://arxiv.org/html/2606.32033v1/figures/ablation_eps/flux1/eps_06.jpg)
was reprojected with the same six-view reviewer. Its exact zenith has a severe
radial starburst. It fails our full-sphere visual gate despite plausible
horizontal views. This is one published example, not a conclusion about every
paper result or proof that all local defects are upstream.

The corresponding
[FLUX.2 image](https://arxiv.org/html/2606.32033v1/figures/ablation_eps/flux2/eps_06.jpg)
has more coherent overhead branches, with residual local deformation. It shows
a different scene, so these two examples are not a controlled backbone
comparison. Original JPEGs, source URLs, hashes and six-view reprojections are
retained in `published_flux1_review/` and `published_flux2_review/`.

This changes the reference assumption: a flat image displayed in the paper
does not by itself validate the exact poles. The local still gate remains open.
The installed Klein base model was tested as a **still** experiment, using
the FLUX.2 method; it is FP8 Klein 9B rather than the paper's FLUX.2-dev BF16.
Graph validation is a wiring check and cannot advance the stage.

Klein job `73ebf082-5833-446f-9676-8d3f0bf63f38` ran on isolated port 8189. It used
2048×1024, 50 Euler steps with `Flux2Scheduler`, seed 42, CFG 4, geometry gamma
6 and spherical scale 2. The negative prompt is
empty; the geometry branch appends the original image geometry clause without
video-motion instructions. There is no LoRA, rolling or image conditioning.
A stock decode and 9-/18-column circular decodes share the same final latent.
This is a different backbone and runtime, not a single-variable comparison
against the native FLUX.1 images.

It completed successfully but **fails the visual gate**. There is one boy and
the circular decode's horizontal wrap looks continuous; a conspicuous
concentric grass/ground pattern remains at the nadir in all three decodes.

| Same Klein latent | Boundary MAE / ratio | Pole decision |
|---|---:|---|
| Stock decode | 16.161 / 2.132 | Fail: concentric nadir |
| Circular pad 9 (native-equivalent extent) | 7.314 / 0.966 | Fail: concentric nadir |
| Circular pad 18 | 7.323 / 0.969 | Fail: concentric nadir |

Job `d23e6513-08b3-4e3c-9e4b-4be742a75995` tested the same Klein seed with
geometry gamma changed from 6 to 0. It also fails: the concentric nadir changes
into a severe radial grass/rock pinch. Boundary MAE is 8.800/255, ratio 1.031.
The subject remains singular. Removing guidance changes the defect rather
than removing it.

Job `dca1f71d-820c-470e-8858-4f58cd2ce8ba` bypassed spherical RoPE while
retaining gamma 6 and the other settings from the first Klein run. It fails:
a conspicuous vertical brightness and landscape discontinuity crosses the
wrap view, and the poles remain distorted. Boundary MAE is 8.370/255, ratio
1.224; those modest numbers understate the visible seam. In this matched
comparison, spherical encoding materially improves horizontal closure without
solving the poles.

The final component control, job `9fbed860-fe36-49e6-b757-94220b8b2bac`,
also disables geometry guidance (gamma 0), completing the 2×2 comparison of
spherical encoding and geometry guidance. It completed in 364.35 seconds but
fails visual inspection: a hard landscape/brightness join, a sky seam into
the zenith, and a radial grass/rock pinch at the nadir.

| Spherical encoding | Geometry gamma | Edge MAE / ratio | Visual gate |
|---|---:|---:|---|
| On | 6 | 7.314 / 0.966 | Fail: concentric nadir; wrap looks continuous |
| On | 0 | 8.800 / 1.031 | Fail: radial nadir pinch; wrap looks continuous |
| Off | 6 | 8.370 / 1.224 | Fail: hard wrap discontinuity and distorted poles |
| Off | 0 | 12.917 / 1.692 | Fail: hard wrap discontinuity and distorted poles |

This matrix uses one seed and the same prompt, backbone, CFG, sampler,
resolution and circular decoder. It demonstrates a useful horizontal effect
of spherical encoding in this configuration, not robustness across seeds.
Since no configuration passes the first seed, none advances to the repeat-seed
acceptance check. All four retain one boy; the native FLUX.1 subject-duplication
failure is therefore not universal across the tested backbones.

The original MCP job on port 8188 was removed while pending behind a separate
H3 render; that H3 render remains active. The MCP wrapper forces port 8188,
so the isolated job uses its underlying comfy-cli with an explicit port.

The reference FLUX.2 pipeline pads 18 columns **after** unpatchifying the
latent (8 pixels per column). ComfyUI exposes the packed 128-channel latent
(16 pixels per column), so 9 columns reproduce that 144-pixel context.
A CPU tensor-layout probe verifies that padding 9 packed columns and then
unpatchifying is exactly equal to unpatchifying and padding 18 columns.
The 18-column ComfyUI decoder is retained as a larger-context control;
this discrepancy alone is not evidence of a visual root cause.

## What the encoding actually guarantees

Section 3.2 explicitly acknowledges that the high-frequency cyclic subspace
does **not** converge at the poles. Only the spherical low-frequency subspace
does. This is a stated method tradeoff, not a newly discovered port bug.

`scripts/probe_flux_polar_contract.py` measures the released tables directly.
At the 64×128 token grid used for a 2048×1024 still, 4/28 FLUX.1 width rotation
pairs and 6/16 FLUX.2 pairs retain longitude dependence at either pole. Their
maximum unit-circle chord distance is 2, while the remaining pairs converge
within floating-point precision. The local FLUX.2 table matches the native
table exactly at all three tested grids (32×64, 64×128, 128×256).

This establishes a limitation of the embedding contract. It does not establish
which channels caused the visible spokes in a complete model. The paper's
Figure 6 discussion also reports better zenith convergence without semantic
distortion guidance in its example. The next native control therefore keeps
the paper-coordinate/step-zero profile and disables only geometry guidance
(`cfg_only`, gamma 0), preserving real semantic CFG at 3.5. It completed but
retains duplicate subjects and severe polar distortion. Thus the extra
geometry term is not the sole cause in this paper-coordinate configuration.
This control is still within gate 1; it is not a new production fix.

The earlier integration test used width 16, which selects no harmonic
channels. It has been renamed to state that scope; a new 64×128 test checks
the actual hybrid behavior. The focused integration file passes 9 tests.
The FLUX.2 angle-table docstring now describes convergence of the spherical
subspace, rather than implying convergence of the complete width embedding.
No inference behavior was changed by these corrections.

## FLUX.1 decoder-only control

`scripts/probe_flux_vae_roundtrip.py` encodes a continuous analytic sphere and
the supplied panorama using deterministic posterior mode, then decodes each
identical latent normally and with 18-column circular padding. This is CPU
FP32 at 1024×512, not exact parity with the BF16 2K generation pipeline.

| Input | Source edge MAE | Stock decode | Circular decode |
|---|---:|---:|---:|
| Analytic sphere | 0.539 | 2.048 | 1.681 |
| Supplied panorama | 6.768 | 11.812 | 7.889 |

Pole-view inspection shows no new severe sky starburst in either control.
The supplied panorama's existing ground convergence remains in both.
This narrows the explanation for the generated sky spokes, but does not rule
out decoder effects on generated latents. Circular decoding also does not
restore exact pixel continuity: the photo's boundary residual against its
source changes only from 9.896 to 9.697/255, despite the lower edge jump.
The lower jump alone is therefore insufficient evidence of a repaired image.

## FLUX.2 decoder-only control

An additional isolated ComfyUI graph tests the actual FLUX.2 VAE in BF16 on
CUDA at 2048×1024. No model sampler or text encoder is involved. Job
`f5c54236-a28b-467a-a7fe-6e26b0c78224` completed successfully.

| Input | Source edge MAE | Stock decode | Circular decode, pad 9 |
|---|---:|---:|---:|
| Analytic sphere | 0.274 | 4.383 | 0.245 |
| Supplied panorama | 4.709 | 6.741 | 5.188 |

The inspected poles retain the input geometry without the new concentric
grass pattern of the generated Klein image. This does not prove that a decoder
cannot amplify defects in generated latents. As in the FLUX.1 control, a lower
edge jump does not imply exact reconstruction: the photo's boundary residual
against its input increases from 4.808 to 6.392/255 with circular decoding.
The complete controls, lossless images and review sheets are retained under
`flux2_vae_roundtrip/`.

## Current causal conclusions

- **Observed:** a VAE-only roundtrip can introduce left/right boundary error.
  That part of the failure does not require a LoRA, denoiser or video model.
- **Observed:** the released FLUX.1 helper differs from the paper's radius,
  coordinate offset and stated time-axis preservation. Aligning those
  coordinates did not pass the visual gate; the radius change also introduced
  pronounced sky artifacts in this seed. Paper alignment is not automatically
  a quality improvement for the installed backbone.
- **Excluded as sole explanation in the tested configuration:** the extra
  geometry-guidance term. Turning it off retains duplicated subjects and pole
  artifacts.
- **Observed in a matched Klein comparison:** removing spherical encoding
  reintroduces a conspicuous horizontal wrap discontinuity. The working
  horizontal effect must be distinguished from the unresolved pole defect.
- **Method limitation, measured in the actual tables:** full polar convergence
  is not enforced by the hybrid encoding. This agrees with the paper's stated
  high-frequency tradeoff and with a distorted published pole example.
- **Still unresolved:** a repeatable, visually acceptable text-to-still profile
  for the supplied scene. Unit-test success and boundary statistics cannot
  close that gap. Outpainting and both video stages remain unstarted.

## Rolling compatibility defect found during the audit

`EquirectPerStepRoll.patch()` called `len(patch_size)` whenever a model exposed
that attribute. FLUX's `patch_size` is an integer, so enabling rolling could
raise `TypeError: object of type 'int' has no len()` before sampling.

A regression test reproduced that failure for scalar patch metadata. The
minimal fix restricts the video `(T,H,W)` handling to tuple/list values; scalar
image metadata no longer enters that branch. The test also exercises a real
roll/unroll of a 4D image latent with a hand-checked processing-boundary effect.
The full suite passes **118 tests, 4 skipped**. The skips concern the absent
development-only `EquirectSemanticDistortionCFG` node, not the integrated
three-way ERP guider used in these renders.

This setup defect cannot explain any of the preceding images: all had rolling
disabled. Runtime verification requires a process that has loaded the fix.
Automatic approval review rejected both stopping the owned idle test server
for a restart and the alternative of freeing its models and launching a
separate test server, each with “blocked by policy.” Neither action executed.
The running test server therefore still contains the old node implementation.

The concrete follow-up graph is saved at
`01_text_still/klein_base_seed42_stock_rope_roll/workflow_api.json` beneath the
evidence directory. It adds random per-step rolling (seed 0, all steps, no
feathering) to the stock-RoPE, gamma-6 Klein control. It was initially blocked
awaiting a fresh process; the successful morning launch and completed render
below supersede that blocker.

### Morning follow-up: MCP launch succeeded

After the user's October 20 follow-up, both ports 8188 and 8189 were down.
`server_info` confirmed ComfyUI was stopped. `launch_comfyui` successfully
started the normal local server on port 8188, and the live queue was empty.
Validation passed and job `6e65950a-b28b-49f0-b029-1981778dac95` started the
prepared rolling graph. Its log reports `EquirectPerStepRoll active: W=128`,
confirming that the scalar-patch setup crash no longer prevents sampling.
The job completed and all three decodes plus the final latent were retained.
The primary output **fails** visual acceptance: its wrap appears continuous,
but a second person appears opposite the main boy and a conspicuous grass
bullseye remains at the nadir. Boundary MAE is 9.052/255, ratio 1.266.
This verifies runtime compatibility, not seamless-image quality.

This normal server loads all installed custom nodes, whereas the previous
Klein controls used an isolated whitelist. A passing candidate therefore
requires confirmation in this runtime; do not attribute every image difference
to rolling without checking that runtime difference.

## Evidence location

`E:/ComfyUI-test-results/first-principles-20261019/`

Every run retains the prompt, seed, model/reference versions, configuration,
lossless output, metrics, perspective inspection and an explicit gate decision.
The source checkout remains unchanged unless a separately tested modification
is needed. No later-stage generation is queued while an earlier gate is open.
