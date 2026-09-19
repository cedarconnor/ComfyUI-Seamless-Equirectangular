# Ripple + Seamless LTX 2.5 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and evaluate a controlled LTX 2.5 matrix combining Ripple with the spherical seam stack and optional low-strength LTX 2.3 seamless LoRA.

**Architecture:** Generate four ComfyUI workflows from the verified native LTX 2.5 Ripple graph. Each workflow shares the same Ripple guide, seed, sampling, spherical model patches, and dual-rotation decoder; only seamless-LoRA strength changes.

**Tech Stack:** Python, ComfyUI workflow JSON, LTX 2.5, ComfyUI-LTXVideo, ComfyUI-Seamless-Equirectangular, OpenCV, Pillow.

## Global Constraints

- Use native LTX 2.5 models and Ripple LoRA strength 1.35.
- Use the existing gray living-room video and colored first frame.
- Render 1024×512, 33 frames, at the same seed and sampler settings.
- Disable circular VAE, latent seam blend, and pixel seam blend.
- Use the LTX 2.5 dual-rotation decoder with `blend_columns=1`.
- Select the best output even when seamless-LoRA strength 0.0 wins.

---

### Task 1: Generate the controlled workflow matrix

**Files:**
- Create: `user/default/workflows/build_ripple_seamless_ltx25_matrix.py`
- Create: `user/default/workflows/ripple-seamless-ltx25/R0_ripple_control.json`
- Create: `user/default/workflows/ripple-seamless-ltx25/R2_seamless_020.json`
- Create: `user/default/workflows/ripple-seamless-ltx25/R4_seamless_040.json`
- Create: `user/default/workflows/ripple-seamless-ltx25/R6_seamless_060.json`

**Interfaces:**
- Consumes: `LTX_Ripple_FFAF_Edit_v11-LTX25.json`
- Produces: Four validation-ready workflow JSON files differing only in seamless-LoRA strength and output prefixes.

- [ ] **Step 1: Write a failing matrix assertion script**

Create graph assertions that require:

- Native LTX 2.5 transformer, text encoder, and VAEs.
- Ripple LoRA strength 1.35.
- A second model-only LoRA loader using
  `Seamless-Equirectangular-LTX2.3-LoRA.safetensors`.
- Wrapped noise, EquiRoPE, geometric CFG, and random per-step roll at start 0.
- `EquirectLTX25VAEDecode` with `blend_columns=1`.
- Raw and exact 180-degree-rolled PNG and MP4 outputs.
- No image or latent seam-blend nodes.

- [ ] **Step 2: Run the assertions and confirm they fail**

Run the assertion script against the not-yet-created matrix directory.

Expected: failure reporting that the four workflows do not exist.

- [ ] **Step 3: Implement the workflow builder**

Clone the working LTX 2.5 Ripple graph, chain Ripple then seamless LoRA, insert
the spherical model/noise patches, replace stock `VAEDecode` with
`EquirectLTX25VAEDecode`, and create raw/rolled output branches.

- [ ] **Step 4: Run assertions**

Expected: all four workflows satisfy the graph invariants.

### Task 2: Validate and render the matrix

**Files:**
- Consume: `user/default/workflows/ripple-seamless-ltx25/*.json`
- Produce: `output/ripple-seamless-ltx25/<variant>/`

**Interfaces:**
- Consumes: Four UI workflows.
- Produces: Four 33-frame raw/rolled PNG sequences and MP4 pairs.

- [ ] **Step 1: Validate all workflows**

Run ComfyUI validation for each workflow. Require zero errors.

- [ ] **Step 2: Render all workflows**

Queue the four workflows and wait for completion. Record any model-patch
warnings or guide-layout errors from the ComfyUI log.

- [ ] **Step 3: Verify output counts**

Require 33 raw PNGs, 33 rolled PNGs, one raw MP4, and one rolled MP4 per
variant.

### Task 3: Analyze seam and Ripple fidelity

**Files:**
- Create: `user/default/workflows/analyze_ripple_seamless_ltx25.py`
- Produce: `output/ripple-seamless-ltx25/metrics.json`
- Produce: `output/ripple-seamless-ltx25/comparison.png`

**Interfaces:**
- Consumes: Lossless output frames.
- Produces: Per-variant wrap metrics and a flat/rolled visual comparison.

- [ ] **Step 1: Measure every frame**

For each raw PNG compute:

```python
joins = np.abs(image[:, 1:] - image[:, :-1]).mean(axis=(0, 2))
wrap = float(np.abs(image[:, 0] - image[:, -1]).mean())
interior = joins[16:-16]
ratio = wrap / np.median(interior)
percentile = (interior <= wrap).mean() * 100.0
```

- [ ] **Step 2: Create a comparison image**

Place the middle raw frame and exact 180-degree-rolled frame side-by-side for
all four variants.

- [ ] **Step 3: Select the winner**

Reject candidates with ratio above 1.0, a visible center cut or stripe,
duplicated/mirrored room geometry, missing couch, or failed color propagation.
Choose the strongest remaining visual result regardless of seamless-LoRA
strength.

- [ ] **Step 4: Run one delayed-roll follow-up only if needed**

If the winner closes the seam but weakens Ripple fidelity, duplicate only that
workflow with `start_percent=0.25`, render, and compare it against the winner.
