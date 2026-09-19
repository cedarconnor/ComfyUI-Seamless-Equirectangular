# Ripple + Seamless LTX 2.5 Integration Design

## Goal

Determine whether LTX Ripple and the LTX 2.3 seamless equirectangular LoRA can
produce a better seamless 360-degree edit together on LTX 2.5 than Ripple with
the model-independent seam stack alone.

The best visually correct result wins. The seamless LoRA may remain disabled if
its cross-version application reduces fidelity or wrap quality.

## Version Decision

Use LTX 2.5. Ripple is an LTX 2.5 FFAF IC-LoRA; its model repository declares
`Lightricks/LTX-2.5` as the base model and provides only
`LTX25_Ripple_v11.safetensors`.

Do not test Ripple on LTX 2.3. A supported LTX 2.3 path would require retraining
Ripple against LTX 2.3.

## Conditioning and Seam Architecture

`LTXAddVideoICLoRAGuide` VAE-encodes Ripple's image/video guide and appends the
guide frames to the video latent before AV packing. `EquirectPerStepRoll`
therefore rolls the generated video frames and guide frames together.

The LTX 2.5 graph will use:

1. Native LTX 2.5 transformer, text encoder, video VAE, and audio VAE.
2. Ripple LoRA at strength 1.35.
3. Optional LTX 2.3 seamless LoRA at a low test strength.
4. Ripple IC-LoRA guide built from the gray source video and colored first
   frame.
5. Wrapped noise, EquiRoPE/geometric CFG, and random per-step latent roll.
6. `EquirectLTX25VAEDecode` with one latent-column boundary replacement.

Do not use `EquirectCircularVAE` on LTX 2.5 because its neighborhood-attention
decoder has no compatible decoder convolutions to patch. Do not use pixel or
latent seam blending.

## Controlled Test Matrix

All tests use the same source assets, prompt, seed, sampler, 1024×512 output,
and 33 frames.

1. Control: Ripple 1.35; seamless LoRA disabled.
2. Ripple 1.35 + seamless LoRA 0.2.
3. Ripple 1.35 + seamless LoRA 0.4.
4. Ripple 1.35 + seamless LoRA 0.6.

Use random per-step roll beginning at 0.0 for the initial matrix. If the best
candidate closes the seam but weakens source motion or edit propagation, repeat
only that candidate with roll start 0.25.

## Outputs and Evaluation

Save for every candidate:

- Raw lossless PNG sequence and MP4.
- Exact 180-degree-rolled PNG sequence and MP4.
- Median wrap jump, median ordinary interior join, their ratio, and the wrap
  percentile among ordinary joins.

A candidate passes when:

- Median wrap-to-interior ratio is at most 1.0.
- The rolled-center view has no visible cut or blur stripe.
- The room and couch are not mirrored or duplicated.
- The colored first-frame edit propagates through the gray source video while
  retaining its motion and layout.

Select the best output, even if the selected workflow has the cross-version
seamless LoRA disabled.
