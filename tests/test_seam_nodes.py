"""Unit tests for the 360-seam nodes (circular VAE conv patch + per-step roll wrapper).

Run with any python that has torch (e.g. the ComfyUI venv):
    /home/juho/ComfyUI/venv/bin/python tests/test_seam_nodes.py
No comfy imports needed — seam_nodes.py is deliberately comfy-free.
"""

import importlib
import math
import pathlib
import sys
import types

import torch

PACK_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK_DIR))

cc = importlib.import_module("seamless_equirect.circular_conv")
ll = importlib.import_module("seamless_equirect.latent_layouts")
sn = importlib.import_module("seamless_equirect.nodes")

torch.manual_seed(0)


# ---------------------------------------------------------------------------
# circular_conv
# ---------------------------------------------------------------------------
def test_circular_conv_patch_unpatch():
    for mode in ("zeros", "reflect", "replicate"):
        conv = torch.nn.Conv3d(4, 4, 3, padding=1, padding_mode=mode)
        x = torch.randn(1, 4, 3, 8, 12)
        stock = conv(x)

        assert cc._patch_conv_leaf(conv)
        patched = conv(x)
        assert patched.shape == stock.shape, f"size not preserved for {mode}"
        assert not cc._patch_conv_leaf(conv), "patch not idempotent"

        # circular-W shift equivariance: conv(roll(x)) == roll(conv(x))
        for shift in (1, 5):
            a = conv(torch.roll(x, shifts=shift, dims=-1))
            b = torch.roll(conv(x), shifts=shift, dims=-1)
            assert torch.allclose(a, b, atol=1e-5), f"not W-shift-equivariant ({mode}, shift={shift})"

        # interior (away from W edges by receptive field) unchanged vs stock
        assert torch.allclose(patched[..., 2:-2], stock[..., 2:-2], atol=1e-5), f"interior changed ({mode})"

        assert cc._unpatch_conv_leaf(conv)
        restored = conv(x)
        assert torch.equal(restored, stock), f"unpatch not bit-exact ({mode})"
    print("ok: circular conv patch/unpatch (zeros/reflect/replicate)")


def test_patch_vae_walk_scoping():
    class FakeVAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Sequential(torch.nn.Conv3d(2, 2, 3, padding=1))
            self.decoder = torch.nn.Sequential(torch.nn.Conv3d(2, 2, 3, padding=1, padding_mode="reflect"))

    vae = FakeVAE()
    n = cc.patch_vae_circular(vae, encoder=False, decoder=True)
    assert n == 1, f"expected only decoder conv patched, got {n}"
    assert not getattr(vae.encoder[0], cc._SENTINEL, False)
    assert getattr(vae.decoder[0], cc._SENTINEL, False)
    assert cc.unpatch_vae_circular(vae) == 1
    print("ok: patch_vae_circular encoder/decoder scoping + unpatch count")


def test_ltx_empty_chunk_shape_preserved():
    class CausalConv3d(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.out_channels = 4
            self.conv = torch.nn.Conv3d(4, 4, 3, padding=(0, 1, 1), padding_mode="reflect")

        def _empty_output(self, x):
            h = (x.shape[3] + 2 * self.conv.padding[1] - self.conv.kernel_size[1]) // self.conv.stride[1] + 1
            w = (x.shape[4] + 2 * self.conv.padding[2] - self.conv.kernel_size[2]) // self.conv.stride[2] + 1
            return x.new_empty((x.shape[0], self.out_channels, 0, h, w))

    class FakeVAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder = torch.nn.ModuleList([CausalConv3d(), CausalConv3d()])

    vae = FakeVAE()
    x = torch.empty(1, 4, 0, 120, 240)
    assert cc.patch_vae_circular(vae, encoder=False, decoder=True) == 2
    for conv in vae.decoder:
        x = conv._empty_output(x)
    assert x.shape[-2:] == (120, 240), f"LTX empty chunk shrank to {x.shape[-2:]}"
    assert cc.unpatch_vae_circular(vae) == 2
    assert all("_empty_output" not in conv.__dict__ for conv in vae.decoder)
    print("ok: LTX empty chunk shape preserved through circular residual convs")


# ---------------------------------------------------------------------------
# per-step roll helpers
# ---------------------------------------------------------------------------
def test_shift_for_sigma():
    w = 60
    s1 = sn.shift_for_sigma(0.7321, w, seed=42, mode="random_per_step")
    s2 = sn.shift_for_sigma(0.7321, w, seed=42, mode="random_per_step")
    assert s1 == s2, "shift not deterministic for same sigma"
    assert 0 <= s1 < w
    assert sn.shift_for_sigma(0.7321, w, 42, "half_turn") == 30
    assert sn.shift_for_sigma(0.5, 1, 42, "random_per_step") == 0
    # different sigmas should (almost always) give different shifts
    shifts = {sn.shift_for_sigma(s / 37.0 + 0.01, w, 42, "random_per_step") for s in range(40)}
    assert len(shifts) > 10, f"shift distribution suspiciously degenerate: {shifts}"
    print("ok: shift_for_sigma determinism/range/distribution")


def test_roll_model_io_5d_and_packed():
    vshape = [1, 8, 3, 5, 7]  # B,C,T,H,W
    vid = torch.randn(*vshape)
    assert torch.equal(sn.roll_model_io(sn.roll_model_io(vid, None, 3), None, -3), vid)
    assert ll.find_layout(vid, None).name == "plain_video_5d"

    audio = torch.randn(1, 4, 11)
    packed = torch.cat([vid.reshape(1, 1, -1), audio.reshape(1, 1, -1)], dim=-1)
    rolled = sn.roll_model_io(packed, vshape, 2)
    assert ll.find_layout(packed, vshape).name == "ltx_packed_av"
    n = math.prod(vshape[1:])
    # audio slice untouched
    assert torch.equal(rolled[:, :, n:], packed[:, :, n:]), "audio tokens were modified"
    # video slice is the rolled video
    expect = torch.roll(vid, shifts=2, dims=-1).reshape(1, 1, -1)
    assert torch.equal(rolled[:, :, :n], expect)
    # round trip
    assert torch.equal(sn.roll_model_io(rolled, vshape, -2), packed)
    # batched (cond+uncond) packed input with batch-1 video_shape still works
    packed2 = torch.cat([packed, packed], dim=0)
    rolled2 = sn.roll_model_io(packed2, vshape, 2)
    assert torch.equal(rolled2[0:1], rolled) and torch.equal(rolled2[1:2], rolled)
    # unknown layout -> None
    assert sn.roll_model_io(torch.randn(1, 1, 50), None, 2) is None
    print("ok: roll_model_io 5D + packed-AV + batched + unknown-layout")


def test_per_step_roll_wrapper_end_to_end():
    class StubModel:
        def clone(self):
            return self

        def set_model_unet_function_wrapper(self, w):
            self.wrapper = w

    stub = StubModel()
    sn.EquirectPerStepRoll().patch(stub, enabled=True, mode="random_per_step", seed=7)

    vshape = torch.Size([1, 8, 3, 5, 7])
    vid = torch.randn(1, 8, 3, 5, 7)
    audio = torch.randn(1, 1, 13)
    packed = torch.cat([vid.reshape(1, 1, -1), audio], dim=-1)
    # pick a sigma whose deterministic shift is nonzero so the roll is observable
    sigma_val = next(s / 100.0 for s in range(1, 100) if sn.shift_for_sigma(s / 100.0, 7, 7, "random_per_step") != 0)
    sigma = torch.tensor([sigma_val])
    seen = {}

    def fake_apply_model(x, t, **c):
        seen["x"] = x.clone()
        return x  # identity model

    mask = torch.ones(1, 1, 3, 5, 7)
    c = {"latent_shapes": [vshape, torch.Size([1, 1, 13])], "denoise_mask": mask}
    out = stub.wrapper(fake_apply_model, {"input": packed, "timestep": sigma, "c": c, "cond_or_uncond": [0]})

    # identity model + roll/unroll must cancel exactly
    assert torch.equal(out, packed), "roll/unroll round trip broke the identity model"
    # the model must have actually seen a rolled input (shift != 0 by construction)
    shift = sn.shift_for_sigma(sigma_val, 7, 7, "random_per_step")
    assert not torch.equal(seen["x"], packed), "model input was not rolled"
    assert torch.equal(seen["x"], sn.roll_model_io(packed, list(vshape), shift))
    # original conditioning dict must not be mutated
    assert c["denoise_mask"] is mask
    # disabled node returns the original model object
    out_model, = sn.EquirectPerStepRoll().patch(stub, enabled=False, mode="random_per_step", seed=7)
    assert out_model is stub
    print("ok: EquirectPerStepRoll wrapper end-to-end (roll seen by model, exact round trip)")


def test_per_step_roll_supports_flux_integer_patch_size():
    """FLUX has scalar patch_size; treating it as a video tuple crashed setup."""
    class FluxPatcher:
        def __init__(self, patch_size):
            self.diffusion_model = types.SimpleNamespace(patch_size=patch_size)

        def clone(self):
            return FluxPatcher(self.diffusion_model.patch_size)

        def get_model_object(self, name):
            assert name == "diffusion_model"
            return self.diffusion_model

        def set_model_unet_function_wrapper(self, wrapper):
            self.wrapper = wrapper

    def boundary_model(x, timestep, **conditioning):
        output = x.clone()
        output[..., 0] = 99
        return output

    for patch_size in (1, 2):  # FLUX.2 and FLUX.1 scalar forms
        original = FluxPatcher(patch_size)
        patched, = sn.EquirectPerStepRoll().patch(
            original, enabled=True, mode="half_turn", seed=0)
        x = torch.arange(8, dtype=torch.float32).reshape(1, 1, 1, 8)
        output = patched.wrapper(
            boundary_model, {"input": x, "timestep": torch.tensor([.5]), "c": {}})
        # Half-turn processing moves the model's boundary effect to column 4.
        assert output.flatten().tolist() == [0, 1, 2, 3, 99, 5, 6, 7]
        assert not hasattr(original, "wrapper")


def test_per_step_roll_stops_after_end_percent():
    assert "end_percent" in sn.EquirectPerStepRoll.INPUT_TYPES()["optional"]

    class StubModel:
        def clone(self):
            return self

        def set_model_unet_function_wrapper(self, wrapper):
            self.wrapper = wrapper

    stub = StubModel()
    sn.EquirectPerStepRoll().patch(
        stub,
        enabled=True,
        mode="random_per_step",
        seed=7,
        start_percent=0.0,
        end_percent=0.5,
    )

    vshape = torch.Size([1, 8, 3, 5, 7])
    video = torch.randn(*vshape)
    audio = torch.randn(1, 1, 13)
    packed = torch.cat([video.reshape(1, 1, -1), audio], dim=-1)
    schedule = torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0])
    conditioning = {
        "latent_shapes": [vshape, torch.Size([1, 1, 13])],
        "transformer_options": {"sample_sigmas": schedule},
    }

    seen = []

    def identity(x, timestep, **kwargs):
        seen.append(x.clone())
        return x

    stub.wrapper(
        identity,
        {
            "input": packed,
            "timestep": torch.tensor([0.75]),
            "c": conditioning,
            "cond_or_uncond": [0],
        },
    )
    assert not torch.equal(seen[-1], packed), "early step was not rolled"

    stub.wrapper(
        identity,
        {
            "input": packed,
            "timestep": torch.tensor([0.25]),
            "c": conditioning,
            "cond_or_uncond": [0],
        },
    )
    assert torch.equal(seen[-1], packed), "late step rolled past end_percent"
    print("ok: per-step roll end_percent limits rolling to early denoise steps")


def test_dual_average_and_feather():
    class StubModel:
        def clone(self):
            return self

        def set_model_unet_function_wrapper(self, w):
            self.wrapper = w

    vshape = torch.Size([1, 8, 3, 5, 12])
    vid = torch.randn(1, 8, 3, 5, 12)
    audio = torch.randn(1, 1, 13)
    packed = torch.cat([vid.reshape(1, 1, -1), audio], dim=-1)
    c = {"latent_shapes": [vshape, torch.Size([1, 1, 13])]}
    sigma = torch.tensor([0.5])

    # dual_average with identity model: mean of two exact round trips == input
    stub = StubModel()
    calls = []
    sn.EquirectPerStepRoll().patch(stub, enabled=True, mode="dual_average", seed=3)
    out = stub.wrapper(lambda x, t, **kw: (calls.append(x.clone()), x)[1],
                       {"input": packed, "timestep": sigma, "c": c, "cond_or_uncond": [0]})
    assert len(calls) == 2, f"dual_average should evaluate twice, got {len(calls)}"
    assert not torch.equal(calls[0], calls[1]), "the two evaluations must use different shifts"
    assert torch.allclose(out, packed, atol=1e-6), "dual_average identity round trip broke"

    # jittered pairing: separation stays in [3W/8, 5W/8], and is NOT always exactly W/2
    # (a fixed half-turn phase-lock amplifies into twin subjects / mirrored structures)
    w = 60
    seps = set()
    for k in range(1, 80):
        s1, s2 = sn.shifts_for_sigma(k / 80.0, w, seed=9, mode="dual_average")
        d = abs(s2 - s1)
        d = min(d, w - d)
        assert 3 * w // 8 <= d <= 5 * w // 8, f"dual separation {d} outside [3W/8, 5W/8]"
        seps.add(d)
    assert len(seps) > 1, "dual_average second shift is phase-locked to exactly W/2"

    # feather: identity model -> only the boundary columns of each shift change, audio untouched
    stub2 = StubModel()
    sn.EquirectPerStepRoll().patch(stub2, enabled=True, mode="half_turn", seed=0, feather_columns=2)
    out2 = stub2.wrapper(lambda x, t, **kw: x,
                         {"input": packed, "timestep": sigma, "c": c, "cond_or_uncond": [0]})
    n = math.prod(vshape[1:])
    assert torch.equal(out2[:, :, n:], packed[:, :, n:]), "feather touched audio tokens"
    ov = out2[:, :, :n].reshape(1, 8, 3, 5, 12)
    # half_turn shift = 6; boundary in rolled coords at wrap -> unrolled, junction at cols 5|6
    diff_cols = [wcol for wcol in range(12) if not torch.allclose(ov[..., wcol], vid[..., wcol], atol=1e-6)]
    assert diff_cols and set(diff_cols) <= {4, 5, 6, 7}, f"feather modified wrong columns: {diff_cols}"
    print("ok: dual_average (2 calls, exact identity) + feather (boundary-local, audio intact)")


def test_seam_blend_w():
    x = torch.randn(1, 8, 3, 5, 12)
    out = cc.seam_blend_w(x, 2)
    assert out.shape == x.shape
    assert torch.equal(out[..., 3:-3], x[..., 3:-3]), "interior changed"
    assert torch.allclose(out[..., 0], out[..., -1], atol=1e-6), "edges not continuous after blend"
    assert torch.equal(cc.seam_blend_w(x, 0), x)
    print("ok: seam_blend_w edge continuity + interior preservation")


# ---------------------------------------------------------------------------
# Semantic Distortion CFG (dev-branch node — tests skip when absent on main)
# ---------------------------------------------------------------------------
def test_semantic_cfg_node_disabled():
    """enabled=False must return the original model object."""
    if not hasattr(sn, "EquirectSemanticDistortionCFG"):
        import pytest
        pytest.skip("EquirectSemanticDistortionCFG not present on this branch (dev-only node)")
    class StubModel:
        def clone(self):
            return self

    stub = StubModel()
    out_model, = sn.EquirectSemanticDistortionCFG().patch(
        stub, geo_conditioning=[[torch.zeros(2, 4), {}]],
        w_sem=7.0, gamma=3.0, enabled=False,
    )
    assert out_model is stub
    print("ok: SemanticDistortionCFG disabled returns original model")


def test_semantic_cfg_three_way_formula():
    """The three-way formula is algebraically correct."""
    if not hasattr(sn, "EquirectSemanticDistortionCFG"):
        import pytest
        pytest.skip("EquirectSemanticDistortionCFG not present on this branch (dev-only node)")
    cfg_fn = sn.EquirectSemanticDistortionCFG._make_cfg()

    # Toy tensor: shape (B=1, C=4, T=3, H=2, W=6) for visual clarity
    x = torch.arange(1.0, 1 + 4 * 3 * 2 * 6).reshape(1, 4, 3, 2, 6)
    eps_cond = torch.randn_like(x)
    eps_uncond = torch.randn_like(x)
    eps_geo = torch.randn_like(x)

    w_sem = 7.0
    gamma = 3.0

    args = {
        "cond": eps_cond,
        "uncond": eps_uncond,
        "input": x,
        "model_options": {
            "equirect_w_sem": w_sem,
            "equirect_gamma": gamma,
            "equirect_geo_pred": x - eps_geo,  # geo_pred = x - eps_geo
        },
    }

    result = cfg_fn(args)
    expected = eps_uncond + w_sem * (eps_cond - eps_uncond) + gamma * (eps_geo - eps_cond)
    assert torch.allclose(result, expected, rtol=1e-4, atol=1e-4), "three-way formula mismatch"
    print("ok: SemanticDistortionCFG three-way formula")


def test_semantic_cfg_gamma_zero_fallback():
    """gamma=0.0 must produce standard CFG behavior without crashing."""
    if not hasattr(sn, "EquirectSemanticDistortionCFG"):
        import pytest
        pytest.skip("EquirectSemanticDistortionCFG not present on this branch (dev-only node)")
    cfg_fn = sn.EquirectSemanticDistortionCFG._make_cfg()

    x = torch.randn(1, 4, 3, 5, 7)
    eps_cond = torch.randn_like(x)
    eps_uncond = torch.randn_like(x)

    args = {
        "cond": eps_cond,
        "uncond": eps_uncond,
        "input": x,
        "model_options": {
            "equirect_w_sem": 7.0,
            "equirect_gamma": 0.0,
            "equirect_geo_pred": None,  # also tests the None-geo_pred path
        },
    }

    result = cfg_fn(args)
    expected = eps_uncond + 7.0 * (eps_cond - eps_uncond)
    assert torch.allclose(result, expected, atol=1e-6), "gamma=0 fallback mismatch"
    print("ok: SemanticDistortionCFG gamma=0 fallback to standard CFG")


def test_semantic_cfg_model_options_set_correctly():
    """The node must set disable_cfg1_optimization and stashed params."""
    if not hasattr(sn, "EquirectSemanticDistortionCFG"):
        import pytest
        pytest.skip("EquirectSemanticDistortionCFG not present on this branch (dev-only node)")
    class StubModel:
        def __init__(self):
            self.model_options = {}

        def clone(self):
            return self

        def set_model_sampler_calc_cond_batch_function(self, fn):
            self._calc_batch_fn = fn

        def set_model_sampler_cfg_function(self, fn):
            self._cfg_fn = fn

    geo_cond = [[torch.tensor([[1.0, 2.0, 3.0]]), {"pooled_output": torch.zeros(1, 4)}]]
    stub = StubModel()
    sn.EquirectSemanticDistortionCFG().patch(
        stub, geo_conditioning=geo_cond, w_sem=5.5, gamma=2.0, enabled=True,
    )

    assert stub.model_options.get("disable_cfg1_optimization") is True
    assert stub.model_options.get("equirect_geo_cond") is geo_cond
    assert stub.model_options.get("equirect_w_sem") == 5.5
    assert stub.model_options.get("equirect_gamma") == 2.0
    assert hasattr(stub, "_cfg_fn")
    assert hasattr(stub, "_calc_batch_fn")
    print("ok: SemanticDistortionCFG model_options set correctly")


if __name__ == "__main__":
    test_circular_conv_patch_unpatch()
    test_patch_vae_walk_scoping()
    test_ltx_empty_chunk_shape_preserved()
    test_shift_for_sigma()
    test_roll_model_io_5d_and_packed()
    test_per_step_roll_wrapper_end_to_end()
    test_per_step_roll_stops_after_end_percent()
    test_dual_average_and_feather()
    test_seam_blend_w()
    if hasattr(sn, "EquirectSemanticDistortionCFG"):
        test_semantic_cfg_node_disabled()
        test_semantic_cfg_three_way_formula()
        test_semantic_cfg_gamma_zero_fallback()
        test_semantic_cfg_model_options_set_correctly()
    else:
        print("skip: EquirectSemanticDistortionCFG not present on this branch (dev-only node)")
    print("ALL TESTS PASSED")
