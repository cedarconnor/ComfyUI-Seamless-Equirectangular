# LTX 2.5 Seamless Decode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add and verify a dual-rotation seamless decoder for LTX 2.5's neighborhood-attention VAE.

**Architecture:** A dedicated node performs stock and half-rotated decodes, unrolls the second result, and composites only the original boundary band. A pure helper owns the tensor math so it can be tested independently.

**Tech Stack:** Python, PyTorch, ComfyUI node API, pytest.

## Global Constraints

- Accept only `CausalDiffusionVAE`.
- Accept only 5D video latents.
- Leave audio and diffusion sampling unchanged.
- Preserve stock decode behavior when the blend width is zero.

---

### Task 1: Dual-rotation decode helper and node

**Files:**
- Create: `seamless_equirect/ltx25.py`
- Create: `tests/test_ltx25.py`
- Modify: `seamless_equirect/nodes.py`

**Interfaces:**
- Produces: `dual_rotation_composite(stock_images, shifted_images, shift_lat, latent_width, blend_columns) -> torch.Tensor`
- Produces: `EquirectLTX25VAEDecode.decode(samples, vae, blend_columns=2) -> tuple[torch.Tensor]`

- [ ] **Step 1: Write failing tests**

Cover non-LTX rejection, non-5D rejection, zero-width stock decoding, shifted
decode alignment, edge replacement, unchanged interior, and node registration.

- [ ] **Step 2: Verify tests fail**

Run: `python -m pytest tests/test_ltx25.py -q`

Expected: collection failure because `seamless_equirect.ltx25` does not exist.

- [ ] **Step 3: Implement the minimal node**

Create the helper and node, then register `EquirectLTX25VAEDecode` through
`seamless_equirect.nodes`.

- [ ] **Step 4: Verify unit and regression tests**

Run:

```powershell
python -m pytest tests/test_ltx25.py -q
python -m pytest tests -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```powershell
git add seamless_equirect/ltx25.py seamless_equirect/nodes.py tests/test_ltx25.py docs/superpowers
git commit -m "feat: add seamless LTX 2.5 VAE decode"
```

### Task 2: Workflow integration and render validation

**Files:**
- Modify: `user/default/workflows/build_seamless_ltx25_tests.py`
- Create: `user/default/workflows/seamless-ltx25/E25_dual_decode.json`

**Interfaces:**
- Consumes: `EquirectLTX25VAEDecode`
- Produces: raw and exact 180-degree-rolled PNG sequences and MP4 outputs.

- [ ] **Step 1: Replace `VAEDecode` in the full random-roll graph**

Connect the separated video latent and native LTX 2.5 VAE to the new node with
`blend_columns=2`; retain the raw and rolled QA branches.

- [ ] **Step 2: Validate and render**

Validate the workflow, render 33 frames at 1024×512, and save lossless frames.

- [ ] **Step 3: Measure continuity**

Run `user/default/workflows/analyze_seamless_ltx25.py` and require the median
wrap jump to be no greater than an ordinary interior join.

- [ ] **Step 4: Inspect the rolled seam**

Confirm that the original wrap, moved to image center, has continuous geometry
without a blurred vertical stripe.
