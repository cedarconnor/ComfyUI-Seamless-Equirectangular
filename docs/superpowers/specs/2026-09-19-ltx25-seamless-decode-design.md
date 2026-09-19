# LTX 2.5 Seamless Decode Design

## Goal

Remove the longitude boundary introduced by LTX 2.5's neighborhood-attention
video VAE decoder without pixel-blending unrelated panorama content.

## Evidence

`EquirectCircularVAE` patches zero decoder convolutions on
`CausalDiffusionVAE`. LTX 2.5 uses `NADiffusionDecoder`, whose spatial
neighborhood attention is unaffected by convolution padding patches. The
diffusion latent roll is active and recognizes the packed AV width, but the
decoded image still has a boundary artifact.

## Design

Add `EquirectLTX25VAEDecode`, accepting a separated 5D video latent, an LTX 2.5
VAE, and a latent-column blend width.

1. Decode the original latent.
2. Circularly roll the latent by half its width and decode it again.
3. Roll the second image back by the proportional pixel distance.
4. Use the shifted decode only around the original left/right boundary, with a
   cosine transition into the stock decode.

The shifted decoder's own boundary lands near image center, outside the region
used from that decode. This preserves the stock decode over most of the image
while sourcing the longitude boundary from an artifact-free interior region.

## Constraints

- Reject VAEs other than `CausalDiffusionVAE`.
- Require a 5D `[B,C,T,H,W]` video latent.
- Do not alter diffusion sampling, conditioning, or audio.
- Do not use final-image seam blending between unrelated edge content.
- Retain the stock decode when `blend_columns=0`.

## Verification

- Unit-test validation, zero-width passthrough, roll/unroll alignment, and
  boundary replacement with a deterministic fake VAE.
- Run the existing node-pack tests.
- Render native LTX 2.5 at 1024×512 for 33 frames.
- Compare the lossless left/right boundary with all ordinary vertical joins and
  inspect an exact 180-degree rolled video.
