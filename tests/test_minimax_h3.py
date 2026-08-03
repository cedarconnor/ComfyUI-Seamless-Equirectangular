"""Unit tests for MiniMax H3 seamless equirectangular helpers and nodes.

Run with:
    python -m pytest -q
or:
    python tests/test_minimax_h3.py
"""

import importlib
import logging
import math
import pathlib
import sys
import torch

# Ensure ComfyUI and repository roots are in sys.path
repo_dir = pathlib.Path(__file__).resolve().parents[1]
comfy_dir = repo_dir.parents[1]
for d in (str(comfy_dir), str(repo_dir)):
    if d not in sys.path:
        sys.path.insert(0, d)

mh3 = importlib.import_module("seamless_equirect.minimax_h3")
sn = importlib.import_module("seamless_equirect.nodes")
ll = importlib.import_module("seamless_equirect.latent_layouts")

try:
    import comfy.patcher_extension
    from comfy.nested_tensor import NestedTensor
except ImportError:
    comfy = None
    class NestedTensor:
        def __init__(self, tensors):
            self.tensors = list(tensors)
            self.is_nested = True
        def unbind(self):
            return tuple(self.tensors)

torch.manual_seed(42)


def test_quantize_roll_shift():
    w = 128
    multiple = 2
    # Even shifts only
    for s in range(w):
        q = mh3.quantize_roll_shift(s, w, multiple)
        assert q % multiple == 0, f"quantized shift {q} for input {s} is not even"
        assert 0 <= q < w

    # multiple=1 matches exact shift
    for s in range(w):
        assert mh3.quantize_roll_shift(s, w, 1) == s

    # dual_average shifts stay distinct
    for sigma_val in [0.1, 0.3, 0.5, 0.7, 0.9]:
        shifts = mh3.shifts_for_sigma(sigma_val, w, seed=123, mode="dual_average", multiple=2)
        assert len(shifts) == 2
        assert shifts[0] != shifts[1]
        assert shifts[0] % 2 == 0 and shifts[1] % 2 == 0
        sep = abs(shifts[1] - shifts[0])
        sep = min(sep, w - sep)
        assert sep > 0

    print("ok: test_quantize_roll_shift")


def test_roll_minimax_h3_fl2va_conditioning():
    vshape = [1, 24, 3, 16, 32]
    guide1 = torch.randn(*vshape)
    guide2 = torch.randn(*vshape)
    kf_latents = [guide1, guide2]
    c = {
        "minimax_payload": {
            "keyframes": [{"resolved_frame_index": 0}, {"resolved_frame_index": 21}],
            "cond_video_latents": kf_latents,
            "layout": "fake_layout",
        }
    }

    # Shift 0 returns original c
    assert mh3.roll_minimax_h3_fl2va_conditioning(c, 0, vshape) is c

    # Non-zero shift rolls guides
    shift = 4
    c_rolled = mh3.roll_minimax_h3_fl2va_conditioning(c, shift, vshape)
    assert c_rolled is not c
    assert c_rolled["minimax_payload"] is not c["minimax_payload"]
    rolled_guides = c_rolled["minimax_payload"]["cond_video_latents"]
    assert len(rolled_guides) == 2
    assert torch.equal(rolled_guides[0], torch.roll(guide1, shifts=shift, dims=-1))
    assert torch.equal(rolled_guides[1], torch.roll(guide2, shifts=shift, dims=-1))

    # Originals unchanged
    assert c["minimax_payload"]["cond_video_latents"][0] is guide1

    # Reference mode (refs non-empty) is NOT rolled
    c_ref = {
        "minimax_payload": {
            "keyframes": [{"resolved_frame_index": 0}],
            "cond_video_latents": [guide1],
            "refs": [{"kind": "image"}],
        }
    }
    assert mh3.roll_minimax_h3_fl2va_conditioning(c_ref, shift, vshape) is c_ref

    # Mismatched guide size raises ValueError
    bad_guide = torch.randn(1, 24, 3, 16, 30)
    c_bad = {
        "minimax_payload": {
            "keyframes": [{"resolved_frame_index": 0}],
            "cond_video_latents": [bad_guide],
        }
    }
    try:
        mh3.roll_minimax_h3_fl2va_conditioning(c_bad, shift, vshape)
        assert False, "Should have raised ValueError on spatial mismatch"
    except ValueError as e:
        assert "spatial dimensions" in str(e)

    print("ok: test_roll_minimax_h3_fl2va_conditioning")


def test_wrapped_noise_h3_nested_tensor():
    class DummyNoise:
        def __init__(self, nested):
            self.nested = nested
        def generate_noise(self, latent):
            return self.nested

    vid = torch.randn(1, 24, 3, 16, 32)
    audio = torch.randn(1, 32, 2, 128)
    nested_in = NestedTensor([vid, audio])

    dummy = DummyNoise(nested_in)
    wrapped, = sn.EquirectWrappedNoise().wrap(dummy, enabled=True, radius=6)
    out_nested = wrapped.generate_noise(None)

    assert getattr(out_nested, "is_nested", False)
    parts = list(out_nested.unbind())
    out_vid, out_audio = parts[0], parts[1]

    # Audio is untouched
    assert out_audio is audio
    assert torch.equal(out_audio, audio)

    # Video is remastered
    assert out_vid.shape == vid.shape
    assert out_vid.dtype == vid.dtype
    assert out_vid.device == vid.device
    assert not torch.equal(out_vid, vid)

    # Input not mutated
    assert nested_in.unbind()[0] is vid

    print("ok: test_wrapped_noise_h3_nested_tensor")


def test_h3_latent_seam_blend():
    vid = torch.randn(1, 24, 3, 16, 32)
    audio = torch.randn(1, 32, 2, 128)
    nested_in = NestedTensor([vid, audio])
    latent_dict = {"samples": nested_in}

    out_dict, = sn.EquirectLatentSeamBlend().blend(latent_dict, blend_columns=2)
    out_samples = out_dict["samples"]
    parts = list(out_samples.unbind())

    # Audio untouched and exact object
    assert parts[1] is audio

    # Video blended at W edges
    out_vid = parts[0]
    assert out_vid.shape == vid.shape
    assert torch.allclose(out_vid[..., 0], out_vid[..., -1], atol=1e-5)
    # Interior untouched
    assert torch.equal(out_vid[..., 3:-3], vid[..., 3:-3])

    # blend_columns=0 returns exact
    out_zero, = sn.EquirectLatentSeamBlend().blend(latent_dict, blend_columns=0)
    out_z_vid = list(out_zero["samples"].unbind())[0]
    assert torch.equal(out_z_vid, vid)

    print("ok: test_h3_latent_seam_blend")


def test_minimax_h3_rope_math_and_lifecycle():
    try:
        from comfy.ldm.minimax.model import MiniMaxH3Model, PackedLayout
    except ImportError:
        print("skip: comfy.ldm.minimax.model not available")
        return

    # Disabled or blend=0 returns original model
    class DummyModel:
        pass

    dummy = DummyModel()
    res, = mh3.EquirectMiniMaxH3RoPE().patch(dummy, enabled=False, blend=0.5)
    assert res is dummy

    # Non-H3 model raises ValueError
    try:
        mh3.EquirectMiniMaxH3RoPE().patch(dummy, enabled=True, blend=0.5)
        assert False, "Should raise ValueError for non-H3 model"
    except ValueError as e:
        assert "MiniMax H3" in str(e)

    class FakeRoPE:
        def __init__(self):
            self.inv_freq = torch.tensor([1.0 / (10000.0 ** (i / 16.0)) for i in range(16)], dtype=torch.float64)

    class FakeAttn(torch.nn.Module):
        def forward(self, x, rope_freqs=None, transformer_options={}):
            self.seen_rope_freqs = rope_freqs
            self.seen_tx_opts = transformer_options
            return x

    class FakeBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = FakeAttn()

    class FakeH3Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.is_minimax_h3 = True
            self.rope = FakeRoPE()
            self.patch_size = (1, 2, 2)
            self.blocks = torch.nn.ModuleList([FakeBlock()])

        def rope_freqs(self, position_ids, device):
            pos = position_ids.to(torch.float32).to(device)
            inv = self.rope.inv_freq.to(device)
            per_axis = pos.unsqueeze(-1) * inv.view(1, 1, -1)
            t_f, h_f, w_f = per_axis.unbind(dim=1)
            half = torch.cat((t_f, h_f, w_f), dim=-1)
            return torch.cat((half, half), dim=-1)

    fake_h3 = FakeH3Model()

    class FakePatcher:
        def __init__(self, m):
            self.model = m
            self.wrappers = {}
            self.object_patches = {}

        def clone(self):
            cp = FakePatcher(self.model)
            cp.wrappers = dict(self.wrappers)
            cp.object_patches = dict(self.object_patches)
            return cp

        def get_model_object(self, name):
            if name == "diffusion_model":
                return self.model
            if name.startswith("diffusion_model.blocks.") and name.endswith(".attn.forward"):
                idx = int(name.split(".")[2])
                return self.model.blocks[idx].attn.forward
            return getattr(self.model, name, None)

        def remove_wrappers_with_key(self, wrapper_type, key):
            w = self.wrappers.get(wrapper_type, {})
            w.pop(key, None)

        def add_wrapper_with_key(self, wrapper_type, key, w):
            self.wrappers.setdefault(wrapper_type, {})[key] = w

        def add_object_patch(self, name, obj):
            self.object_patches[name] = obj

    patcher = FakePatcher(fake_h3)
    patched_patcher, = mh3.EquirectMiniMaxH3RoPE().patch(patcher, enabled=True, blend=1.0)

    # Check wrappers and object patches added
    wrapper_type = getattr(comfy.patcher_extension.WrappersMP, "DIFFUSION_MODEL", "diffusion_model")
    assert wrapper_type in patched_patcher.wrappers
    assert "equirect_minimax_h3_rope" in patched_patcher.wrappers[wrapper_type]
    assert "diffusion_model.blocks.0.attn.forward" in patched_patcher.object_patches

    # Idempotency check: patching again replaces wrapper/patch rather than accumulating
    patched_again, = mh3.EquirectMiniMaxH3RoPE().patch(patched_patcher, enabled=True, blend=1.0)
    assert len(patched_again.wrappers[wrapper_type]) == 1

    # Execute wrapper with list x = [video_x, audio_x]
    vid_x = torch.randn(1, 24, 3, 16, 32)
    aud_x = torch.randn(1, 32, 2, 64)
    x_input = [vid_x, aud_x]
    timestep = torch.tensor([500.0])
    context = [torch.randn(10, 768)]
    payload = {}

    wrapper_fn = patched_again.wrappers[wrapper_type]["equirect_minimax_h3_rope"]
    executed_kwargs = {}

    def dummy_executor(x, ts, ctx, transformer_options={}, minimax_payload=None, **kw):
        executed_kwargs["transformer_options"] = transformer_options
        return x

    res_x = wrapper_fn(dummy_executor, x_input, timestep, context, transformer_options={}, minimax_payload=payload)
    assert res_x is x_input
    assert "equirect_minimax_h3_rope_table" in executed_kwargs["transformer_options"]

    # Test patched attention forward
    attn_fn = patched_again.object_patches["diffusion_model.blocks.0.attn.forward"]
    fake_x = torch.randn(1, 100, 768)
    res_attn = attn_fn(fake_x, rope_freqs=None, transformer_options=executed_kwargs["transformer_options"])
    assert fake_h3.blocks[0].attn.seen_rope_freqs is executed_kwargs["transformer_options"]["equirect_minimax_h3_rope_table"]

    print("ok: test_minimax_h3_rope_math_and_lifecycle")


def test_h3_vae_dual_decode():
    class FakeH3VAEModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.is_minimax_h3_vae = True
            self.patch_size = (1, 2, 2)
            self.calls = 0
            self.last_inputs = []

        def decode(self, x):
            self.calls += 1
            self.last_inputs.append(x.clone())
            B, C, T, H, W = x.shape
            img = torch.ones(B * T, H * 16, W * 16, 3, dtype=torch.float32)
            img[:, :, 0, :] = 0.0
            img[:, :, -1, :] = 0.0
            return img

    class FakeVAEWrapper:
        def __init__(self, m):
            self.first_stage_model = m
        def decode(self, x):
            return self.first_stage_model.decode(x)

    fake_model = FakeH3VAEModel()
    fake_vae = FakeVAEWrapper(fake_model)

    video_lat = torch.randn(1, 24, 3, 16, 32)
    samples = {"samples": video_lat}

    out0, = mh3.EquirectMiniMaxH3VAEDecode().decode(samples, fake_vae, blend_columns=0)
    assert fake_model.calls == 1
    assert out0.shape == (3, 256, 512, 3)
    assert out0[:, :, 0, 0].max().item() == 0.0

    fake_model.calls = 0
    fake_model.last_inputs.clear()
    out_dual, = mh3.EquirectMiniMaxH3VAEDecode().decode(samples, fake_vae, blend_columns=2)
    assert fake_model.calls == 2
    assert out_dual.shape == (3, 256, 512, 3)

    assert torch.equal(fake_model.last_inputs[1], torch.roll(video_lat, shifts=16, dims=-1))

    assert out_dual[:, :, 0, :].mean().item() > 0.5
    assert out_dual[:, :, -1, :].mean().item() > 0.5

    assert torch.equal(out_dual[:, :, 35:-35, :], out0[:, :, 35:-35, :])

    fake_model.is_minimax_h3_vae = False
    fake_model.__class__.__name__ = "OtherVAE"
    try:
        mh3.EquirectMiniMaxH3VAEDecode().decode(samples, fake_vae, blend_columns=2)
        assert False, "Should raise ValueError for non-H3 VAE"
    except ValueError as e:
        assert "MiniMaxH3VideoVAE" in str(e)

    print("ok: test_h3_vae_dual_decode")


def test_image_seam_blend():
    img = torch.randn(2, 64, 128, 3)
    out0, = sn.EquirectImageSeamBlend().blend(img, blend_pixels=0)
    assert out0 is img

    out, = sn.EquirectImageSeamBlend().blend(img, blend_pixels=24)
    assert out.shape == img.shape
    assert torch.allclose(out[:, :, 0, :], out[:, :, -1, :], atol=1e-5)
    assert torch.equal(out[:, :, 30:-30, :], img[:, :, 30:-30, :])

    print("ok: test_image_seam_blend")


def test_circular_vae_h3_warning(caplog):
    class FakeH3VideoVAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.is_minimax_h3_vae = True
            self.decoder = torch.nn.Sequential()

    class FakeVAEWrapper:
        def __init__(self, m):
            self.first_stage_model = m

    fake_vae = FakeVAEWrapper(FakeH3VideoVAE())
    if caplog is not None:
        with caplog.at_level(logging.WARNING):
            res, = sn.EquirectCircularVAE().patch(fake_vae, patch_encoder=False, patch_decoder=True, enabled=True)
        assert any("tiled ViT decoder" in record.message for record in caplog.records)
    else:
        res, = sn.EquirectCircularVAE().patch(fake_vae, patch_encoder=False, patch_decoder=True, enabled=True)
    assert res is fake_vae

    print("ok: test_circular_vae_h3_warning")


def test_node_registration():
    for node_id in (
        "EquirectPerStepRoll",
        "EquirectMiniMaxH3RoPE",
        "EquirectWrappedNoise",
        "EquirectLatentSeamBlend",
        "EquirectMiniMaxH3VAEDecode",
        "EquirectImageSeamBlend",
        "EquirectCircularVAE",
    ):
        assert node_id in sn.NODE_CLASS_MAPPINGS, f"Missing registration for {node_id}"
        assert node_id in sn.NODE_DISPLAY_NAME_MAPPINGS, f"Missing display name for {node_id}"

    print("ok: test_node_registration")


if __name__ == "__main__":
    test_quantize_roll_shift()
    test_roll_minimax_h3_fl2va_conditioning()
    test_wrapped_noise_h3_nested_tensor()
    test_h3_latent_seam_blend()
    test_minimax_h3_rope_math_and_lifecycle()
    test_h3_vae_dual_decode()
    test_image_seam_blend()
    test_circular_vae_h3_warning(caplog=None)
    test_node_registration()
    print("ALL MINIMAX H3 TESTS PASSED")
