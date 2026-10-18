import pytest
import torch

pytest.importorskip("comfy.samplers")

import comfy.samplers  # noqa: E402
from erp_test_helpers import FakeModelPatcher, FakeLTXDiffusionModel  # noqa: E402
from seamless_equirect.erp_guider import Guider_ERP3Way  # noqa: E402
from seamless_equirect.erp_guidance import combine_erp_3way  # noqa: E402

COND = [[torch.zeros(1, 8, 16), {}]]   # minimal comfy conditioning


def _guider(cfg=3.0, gamma=3.0):
    mp = FakeModelPatcher(FakeLTXDiffusionModel())
    g = Guider_ERP3Way(mp)
    g.set_conds(positive=COND, negative=COND, geometry=COND)
    g.set_params(cfg, gamma)
    return g


def test_set_params_and_conds_stored():
    g = _guider(cfg=2.5, gamma=1.5)
    assert g.cfg == 2.5 and g.gamma == 1.5
    assert set(["positive", "negative", "geometry"]).issubset(g.original_conds.keys())


def test_predict_noise_three_way(monkeypatch):
    captured = {}

    def fake_ccb(model, conds, x, timestep, model_options):
        captured["conds"] = conds
        # neg -> 10, geo -> 20, pos -> 30
        return [torch.full_like(x, 10.0), torch.full_like(x, 20.0), torch.full_like(x, 30.0)]

    monkeypatch.setattr(comfy.samplers, "calc_cond_batch", fake_ccb)
    g = _guider(cfg=3.0, gamma=3.0)
    g.inner_model = object()
    g.conds = {"negative": "N", "geometry": "G", "positive": "P"}
    x = torch.zeros(1, 4)
    ts = torch.tensor([1.0])
    out = g.predict_noise(x, ts, {}, None)
    assert captured["conds"] == ["N", "G", "P"]          # order neg, geo, pos
    expect = combine_erp_3way(torch.full_like(x, 30.0), torch.full_like(x, 10.0),
                              torch.full_like(x, 20.0), cfg=3.0, gamma=3.0)
    assert torch.allclose(out, expect)


def test_predict_noise_fires_post_cfg_hook_for_cfgpp(monkeypatch):
    """CFG++ samplers (euler_cfg_pp, the LTX default) read uncond_denoised from a
    sampler_post_cfg_function. The ERP guider must fire it, passing x0_uncond as
    uncond_denoised, and must preserve its combine when the hook is identity."""
    def fake_ccb(model, conds, x, timestep, model_options):
        return [torch.full_like(x, 10.0), torch.full_like(x, 20.0), torch.full_like(x, 30.0)]

    monkeypatch.setattr(comfy.samplers, "calc_cond_batch", fake_ccb)
    seen = {}

    def post_cfg(args):
        seen["uncond_denoised"] = args["uncond_denoised"]
        seen["cond_denoised"] = args["cond_denoised"]
        return args["denoised"]                      # identity, like comfy's cfg_pp hook

    g = _guider(cfg=3.0, gamma=3.0)
    g.inner_model = object()
    g.conds = {"negative": "N", "geometry": "G", "positive": "P"}
    x = torch.zeros(1, 4)
    out = g.predict_noise(x, torch.tensor([1.0]),
                          {"sampler_post_cfg_function": [post_cfg]}, None)
    assert torch.allclose(seen["uncond_denoised"], torch.full_like(x, 10.0))  # x0_uncond
    assert torch.allclose(seen["cond_denoised"], torch.full_like(x, 30.0))    # x0_cond
    expect = combine_erp_3way(torch.full_like(x, 30.0), torch.full_like(x, 10.0),
                              torch.full_like(x, 20.0), cfg=3.0, gamma=3.0)
    assert torch.allclose(out, expect)               # identity hook preserves combine


def test_predict_noise_gamma0_falls_back_to_two_way(monkeypatch):
    captured = {}

    def fake_sf(model, x, timestep, uncond, cond, cond_scale, model_options={}, seed=None):
        captured["args"] = (uncond, cond, cond_scale)
        return torch.full_like(x, 7.0)

    monkeypatch.setattr(comfy.samplers, "sampling_function", fake_sf)
    g = _guider(cfg=3.0, gamma=0.0)
    g.inner_model = object()
    g.conds = {"negative": "N", "geometry": "G", "positive": "P"}
    x = torch.zeros(1, 4)
    ts = torch.tensor([1.0])
    out = g.predict_noise(x, ts, {}, None)
    assert torch.allclose(out, torch.full_like(x, 7.0))
    assert captured["args"] == ("N", "P", 3.0)           # (uncond, cond, cfg)
