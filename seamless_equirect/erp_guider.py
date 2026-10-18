"""ERP 3-way guider: a comfy CFGGuider subclass applying SpheRoPE semantic-distortion CFG.

Mirrors comfy's Guider_DualCFG. predict_noise runs calc_cond_batch on [negative, geometry,
positive] and combines the x0 predictions via combine_erp_3way. gamma <= 0 (or missing
geometry) delegates to comfy's standard 2-way sampling_function (stock behavior).
"""
import comfy.samplers

from .erp_guidance import combine_erp_3way


class Guider_ERP3Way(comfy.samplers.CFGGuider):
    def set_conds(self, positive, negative, geometry):
        self.inner_set_conds({
            "positive": positive,
            "negative": negative,
            "geometry": geometry,
        })

    def set_params(self, cfg, gamma):
        self.cfg = cfg
        self.gamma = gamma

    def predict_noise(self, x, timestep, model_options={}, seed=None):
        positive = self.conds.get("positive", None)
        negative = self.conds.get("negative", None)
        geometry = self.conds.get("geometry", None)

        if getattr(self, "gamma", 0.0) <= 0 or geometry is None:
            # Stock 2-way CFG (pos/neg). Reproduces comfy default behavior.
            return comfy.samplers.sampling_function(
                self.inner_model, x, timestep, negative, positive, self.cfg,
                model_options=model_options, seed=seed,
            )

        out = comfy.samplers.calc_cond_batch(
            self.inner_model, [negative, geometry, positive], x, timestep, model_options,
        )
        x0_uncond, x0_geo, x0_cond = out[0], out[1], out[2]
        denoised = combine_erp_3way(x0_cond, x0_uncond, x0_geo, self.cfg, self.gamma)

        # Fire post-CFG hooks so CFG++ samplers (e.g. euler_cfg_pp, the LTX default)
        # can capture uncond_denoised; comfy's cfg_function does the same after CFG.
        # The stock cfg_pp hook returns args["denoised"] unchanged, so our ERP combine
        # is preserved. x0_uncond is the negative-cond prediction (the cfg_pp uncond).
        for fn in model_options.get("sampler_post_cfg_function", []):
            denoised = fn({
                "denoised": denoised, "cond": positive, "uncond": negative,
                "cond_scale": self.cfg, "model": self.inner_model,
                "uncond_denoised": x0_uncond, "cond_denoised": x0_cond,
                "sigma": timestep, "model_options": model_options, "input": x,
            })
        return denoised
