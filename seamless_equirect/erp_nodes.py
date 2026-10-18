"""Integrated ERP guidance, circular processing and diagnostics.

Adapted from cedarconnor/ComfyUI-SpheRoPE c8148da; see THIRD_PARTY_NOTICES.md.
"""
from comfy_extras.nodes_lt_upsampler import LTXVLatentUpsampler
from .circular_video import (
    circular_pad_video_width,
    circular_upsample_width,
    crop_decoded_frame_width,
    assert_width_dims_match,
)
from .erp_guider import Guider_ERP3Way
from .flux2_rope import install_flux_spherical_rope
from .erp_metrics import (
    latent_seam_metrics, pole_consistency, metrics_to_json, metrics_to_csv,
    seam_mae, seam_correlation, per_frame_seam_mae, temporal_seam_variation,
    seam_difference_heatmap, pole_variance_heatmap,
)


SPATIAL_COMPRESSION = 32


DEFAULT_PAD_W = 4


class EquirectCircularUpscale:
    CATEGORY = "equirect/circular"
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "upscale"
    DESCRIPTION = (
        "Wrap-aware x2 latent upscale for the LTX two-stage pipeline. Circularly pads "
        "the latent width before the spatial upscaler and crops it back after, so the "
        "0/360 seam stays continuous through stage 2. Preserves any conditioning "
        "noise_mask (needed for image-to-video). Use between the stage-1 sampler and "
        "the stage-2 sampler."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "samples": ("LATENT", {"tooltip": "Stage-1 LTX latent to upscale."}),
            "upscale_model": ("LATENT_UPSCALE_MODEL", {"tooltip": "The LTX x2 spatial upscaler (LatentUpscaleModelLoader)."}),
            "vae": ("VAE", {"tooltip": "The LTX VAE (used for its per-channel un/normalize statistics)."}),
            "pad_w": ("INT", {"default": DEFAULT_PAD_W, "min": 0, "max": 64,
                              "tooltip": "Latent columns of circular padding on each side. 4 is plenty; "
                                         "higher is safer but slower. 0 disables the seam wrap."}),
        }}

    def upscale(self, samples, upscale_model, vae, pad_w):
        latents = samples["samples"]

        def _upsample(x):
            # Core owns ModelPatcher loading, dtype/device and VAE statistics.
            # Passing the padded tensor also gives its memory estimate the real size.
            return LTXVLatentUpsampler.execute({"samples": x}, upscale_model, vae)[0]["samples"]

        upsampled = circular_upsample_width(latents, _upsample, pad_w)
        out = dict(samples)
        out["samples"] = upsampled
        # Preserve any conditioning noise_mask: it is per-frame (B,1,T,1,1),
        # spatially agnostic, and spatial upscale leaves T unchanged.
        return (out,)


class EquirectCircularVAEDecodeLTX:
    CATEGORY = "equirect/circular"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    DESCRIPTION = (
        "Wrap-aware VAE decode for LTX ERP video. Circularly pads the latent width, "
        "decodes, and crops the padding back off in pixel space so the decoded frames "
        "close cleanly at the 0/360 seam. Use as the final decode instead of VAEDecode."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "samples": ("LATENT", {"tooltip": "The final LTX video latent to decode."}),
            "vae": ("VAE", {"tooltip": "The LTX VAE."}),
            "pad_w": ("INT", {"default": DEFAULT_PAD_W, "min": 0, "max": 64,
                              "tooltip": "Latent columns of circular padding (x32 px for LTX). 0 disables the wrap."}),
            "target_width": ("INT", {"default": 2048, "min": 64, "max": 8192,
                                     "tooltip": "Expected decoded pixel width; the node errors if the crop "
                                                "did not restore it (a guard against a wrong pad/scale)."}),
        }}

    def decode(self, samples, vae, pad_w, target_width):
        latent = samples["samples"]
        padded = circular_pad_video_width(latent, pad_w)
        frames = vae.decode(padded)                          # (F,H,W,C) or (B,F,H,W,C)
        cropped = crop_decoded_frame_width(frames, pad_w * SPATIAL_COMPRESSION)
        if cropped.dim() == 5:                               # (B,F,H,W,C) -> combine batches
            cropped = cropped.reshape(-1, *cropped.shape[-3:])
        assert_width_dims_match(got=cropped.shape[-2], want=target_width)
        return (cropped,)


class EquirectCircularVAEDecodeFlux:
    """Seam-safe VAE decode for a single ERP image latent (Flux.2 and other image
    models). Wrap-pads the latent width, decodes, and crops the padding back off in
    pixel space. The pixel crop is derived from the actual decode ratio (decoded width
    / padded latent width), so it is VAE-agnostic (Flux 8x, LTX 32x, etc.)."""

    CATEGORY = "equirect/circular"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    DESCRIPTION = (
        "Wrap-aware VAE decode for a single ERP image latent (FLUX.2 and other image "
        "models). Circularly pads the latent width, decodes, and crops the padding back "
        "off in pixel space so the panorama closes at the 0/360 seam. The pixel crop is "
        "derived from the actual decode ratio, so it is VAE-agnostic (Flux 8x etc.). "
        "Use as the final decode instead of VAEDecode."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "samples": ("LATENT", {"tooltip": "The final FLUX.2 image latent to decode."}),
            "vae": ("VAE", {"tooltip": "The image VAE (e.g. flux2-vae)."}),
            "pad_w": ("INT", {"default": 8, "min": 0, "max": 64,
                              "tooltip": "Latent columns of circular padding on each side. 8 is a good "
                                         "default for Flux; 0 disables the seam wrap."}),
        }}

    def decode(self, samples, vae, pad_w):
        latent = samples["samples"]
        padded = circular_pad_video_width(latent, pad_w)     # pad width (dim -1)
        images = vae.decode(padded)                          # (B, H, W, C)
        if pad_w > 0:
            ratio = images.shape[-2] // padded.shape[-1]     # px per latent col
            images = crop_decoded_frame_width(images, pad_w * ratio)
        return (images,)


class EquirectERPGuider:
    CATEGORY = "equirect/guidance"
    RETURN_TYPES = ("GUIDER",)
    FUNCTION = "get_guider"
    DESCRIPTION = (
        "3-way semantic-distortion ERP CFG guider (LTX + Flux). Combines "
        "x0_cond + (cfg-1)(cond-uncond) + erp_gamma(geo-cond): a geometry-anchored "
        "branch nudges the sample toward valid spherical ERP structure. Feed to "
        "SamplerCustomAdvanced. The tested LTX dev stage uses cfg=3 and erp_gamma=3. "
        "NOTE: when erp_gamma > 0 the 3-way combine runs its own CFG, so pre-CFG / "
        "cfg_function hooks (PAG, perp-neg, RescaleCFG) are bypassed; set erp_gamma = 0 "
        "for plain 2-way CFG if you need those nodes to take effect."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL", {"tooltip": "The (spherical-RoPE-patched) diffusion model."}),
            "positive": ("CONDITIONING", {"tooltip": "Your prompt conditioning."}),
            "negative": ("CONDITIONING", {"tooltip": "Negative conditioning (ERP Motion Policy supplies a good default)."}),
            "geometry": ("CONDITIONING", {"tooltip": "Prompt + ERP geometry clause (from ERP Motion Policy). The geometry-anchored branch."}),
            "cfg": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01,
                              "tooltip": "Prompt CFG. The tested LTX dev stage uses 3; geometry guidance "
                                         "is an independent term and remains active at CFG 1."}),
            "erp_gamma": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 20.0, "step": 0.1, "round": 0.01,
                                    "tooltip": "Geometry-branch strength. 0 = plain 2-way CFG (stock). "
                                               "Higher pulls harder toward ERP geometry but softens detail; "
                                               "~1-3 is a good range. Reference LTX default 3."}),
        }}

    def get_guider(self, model, positive, negative, geometry, cfg, erp_gamma):
        guider = Guider_ERP3Way(model)
        guider.set_conds(positive, negative, geometry)
        guider.set_params(cfg, erp_gamma)
        return (guider,)


class EquirectSphericalRoPEFlux:
    CATEGORY = "equirect/rope"
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    DESCRIPTION = (
        "Patch a FLUX.2 MODEL's pe_embedder width block ([32:48] of the [T,H,W,L] "
        "RoPE) with sphere-aware embeddings; time/height/level stay identical. FLUX.2 "
        "ONLY (dev or klein) - errors on non-Flux.2 models. It changes longitude "
        "encoding; low pole variance does not guarantee correct geometry. "
        "Insert right after your UNETLoader; feed the patched "
        "model to the guider and the scheduler. Text-to-panorama only (no reference "
        "images); a normal Flux workflow is unaffected when unused."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL", {"tooltip": "A FLUX.2 diffusion model (dev or klein), loaded via UNETLoader."}),
            "scale": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 8.0, "step": 0.1,
                                "tooltip": "Spherical RoPE radius denominator (SFC scale). 2.0 = reference. "
                                           "Controls how strongly the width axis is warped toward the sphere."}),
        }}

    def apply(self, model, scale):
        patched = install_flux_spherical_rope(model, scale=scale)
        return (patched,)


class EquirectLatentDiagnostics:
    """Pre-decode latent seam metrics: the reference pipeline's three
    d(col*) numbers plus pole consistency, read off the (B,C,F,H,W) LATENT
    (width already the last axis, so it is already canonical (...,H,W))."""
    CATEGORY = "equirect/diagnostics"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("metrics_json", "metrics_csv")
    FUNCTION = "analyze"
    DESCRIPTION = (
        "Pre-decode latent seam + pole metrics (the reference's d(col*) numbers). "
        "Wire the sampler's LATENT here and view via PreviewAny. NOTE: these measure "
        "seam continuity and pole uniformity ONLY - a degenerate banded image scores "
        "'best', so never use them alone as a quality signal; judge the decoded image by eye."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "samples": ("LATENT",),
            "pole_rows": ("INT", {"default": 1, "min": 1, "max": 16}),
        }}

    def analyze(self, samples, pole_rows):
        latent = samples["samples"]
        metrics = {}
        metrics.update(latent_seam_metrics(latent))
        metrics.update(pole_consistency(latent, rows=pole_rows))
        return (metrics_to_json(metrics), metrics_to_csv(metrics))


class EquirectFrameDiagnostics:
    """Post-decode diagnostics on decoded ERP frames (IMAGE, (F,H,W,C) in [0,1]):
    seam MAE + circular correlation, pole consistency, temporal seam std, plus
    seam and pole heatmap images rendered from the first frame."""
    CATEGORY = "equirect/diagnostics"
    RETURN_TYPES = ("STRING", "STRING", "IMAGE", "IMAGE")
    RETURN_NAMES = ("metrics_json", "metrics_csv", "seam_heatmap", "pole_heatmap")
    FUNCTION = "analyze"
    DESCRIPTION = (
        "Post-decode ERP metrics on decoded frames: seam MAE + circular correlation, "
        "pole variance, temporal seam std, plus seam/pole heatmap previews. Comparative "
        "only (A/B two runs). Same caveat as Latent Diagnostics: these reward a banded "
        "image, so pair them with a visual coherence check."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "seam_width_k": ("INT", {"default": 1, "min": 1, "max": 64}),
            "pole_rows": ("INT", {"default": 1, "min": 1, "max": 16}),
        }}

    def analyze(self, images, seam_width_k, pole_rows):
        imgs = images
        if imgs.dim() == 5:                       # (B,F,H,W,C) -> (B*F,H,W,C)
            imgs = imgs.reshape(-1, *imgs.shape[-3:])
        # (F,H,W,C) -> canonical (F,C,H,W): move channels to -3, width lands last.
        stack = imgs.movedim(-1, -3)
        per_frame = per_frame_seam_mae(stack, k=seam_width_k)
        metrics = {
            "seam_mae": seam_mae(stack, k=seam_width_k),
            "seam_correlation": seam_correlation(stack, k=seam_width_k),
            "temporal_seam_std": temporal_seam_variation(per_frame),
        }
        metrics.update(pole_consistency(stack, rows=pole_rows))
        first = imgs[0]                           # (H,W,C)
        seam_hm = seam_difference_heatmap(first)
        pole_hm = pole_variance_heatmap(first)
        return (metrics_to_json(metrics), metrics_to_csv(metrics), seam_hm, pole_hm)

NODE_CLASS_MAPPINGS = {
    "EquirectCircularUpscale": EquirectCircularUpscale,
    "EquirectCircularVAEDecodeFlux": EquirectCircularVAEDecodeFlux,
    "EquirectCircularVAEDecodeLTX": EquirectCircularVAEDecodeLTX,
    "EquirectERPGuider": EquirectERPGuider,
    "EquirectSphericalRoPEFlux": EquirectSphericalRoPEFlux,
    "EquirectLatentDiagnostics": EquirectLatentDiagnostics,
    "EquirectFrameDiagnostics": EquirectFrameDiagnostics,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EquirectCircularUpscale": "Equirect Circular Upscale (LTX)",
    "EquirectCircularVAEDecodeFlux": "Equirect Circular VAE Decode (Flux.2)",
    "EquirectCircularVAEDecodeLTX": "Equirect Circular VAE Decode (LTX 2.3)",
    "EquirectERPGuider": "Equirect ERP Guider",
    "EquirectSphericalRoPEFlux": "Equirect Spherical RoPE (Flux.2)",
    "EquirectLatentDiagnostics": "Equirect Latent Diagnostics",
    "EquirectFrameDiagnostics": "Equirect Frame Diagnostics",
}
