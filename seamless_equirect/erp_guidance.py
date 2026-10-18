"""Pure 3-way semantic-distortion ERP guidance combine (SpheRoPE reference formula).

    x0_final = x0_cond + (cfg - 1) * (x0_cond - x0_uncond) + gamma * (x0_geo - x0_cond)

Operates in comfy's denoised (x0) space, where calc_cond_batch predictions live and
predict_noise must return. Torch-only, no ComfyUI deps.
"""


def combine_erp_3way(x0_cond, x0_uncond, x0_geo, cfg: float, gamma: float):
    """Combine the positive/negative/geometry x0 predictions. At gamma == 0 this reduces to
    standard 2-way CFG (`x0_uncond + (x0_cond - x0_uncond) * cfg`)."""
    cfg_delta = (cfg - 1.0) * (x0_cond - x0_uncond)
    geo_delta = gamma * (x0_geo - x0_cond)
    return x0_cond + cfg_delta + geo_delta
