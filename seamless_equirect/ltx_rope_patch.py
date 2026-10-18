"""Model-owned LTX video RoPE patch; audio retains the original precompute."""
from comfy.ldm.lightricks.model import LTXRopeType, freqs_cis_matrix


def patch_ltx_rope(model, build_freqs):
    patched = model.clone()
    dm = patched.get_model_object("diffusion_model")
    key = "diffusion_model._precompute_freqs_cis"
    original = patched.get_model_object(key)

    def precompute(indices_grid, dim, out_dtype, theta=10000.0,
                   max_pos=(20, 2048, 2048), use_middle_indices_grid=False,
                   num_attention_heads=32):
        if indices_grid.shape[1] != 3:
            return original(indices_grid, dim, out_dtype, theta, max_pos,
                            use_middle_indices_grid, num_attention_heads)
        indices = dm.freq_grid_generator(theta, 3, dim, indices_grid.device)
        freqs = build_freqs(indices, indices_grid, max_pos, use_middle_indices_grid)
        split = dm.split_positional_embedding == LTXRopeType.SPLIT
        pad = dim // 2 - freqs.shape[-1] if split else dim % 6
        return freqs_cis_matrix(freqs, pad, split, num_attention_heads, out_dtype)

    patched.add_object_patch(key, precompute)
    return patched
