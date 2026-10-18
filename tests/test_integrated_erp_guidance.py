import torch

from seamless_equirect.erp_guidance import combine_erp_3way


def test_gamma0_cfg1_returns_cond():
    c = torch.randn(2, 4)
    u = torch.randn(2, 4)
    g = torch.randn(2, 4)
    out = combine_erp_3way(c, u, g, cfg=1.0, gamma=0.0)
    assert torch.allclose(out, c)


def test_gamma0_matches_two_way_cfg():
    c = torch.randn(2, 4)
    u = torch.randn(2, 4)
    g = torch.randn(2, 4)
    out = combine_erp_3way(c, u, g, cfg=3.0, gamma=0.0)
    two_way = u + (c - u) * 3.0            # comfy cfg_function form
    assert torch.allclose(out, two_way, atol=1e-6)


def test_full_formula_known_values():
    c = torch.full((1, 3), 2.0)
    u = torch.full((1, 3), 1.0)
    g = torch.full((1, 3), 5.0)
    # 2 + (3-1)*(2-1) + 3*(5-2) = 2 + 2 + 9 = 13
    out = combine_erp_3way(c, u, g, cfg=3.0, gamma=3.0)
    assert torch.allclose(out, torch.full((1, 3), 13.0))


def test_geometry_term_scales_with_gamma():
    c = torch.zeros(1, 4)
    u = torch.zeros(1, 4)
    g = torch.ones(1, 4)
    out0 = combine_erp_3way(c, u, g, cfg=1.0, gamma=0.0)
    out2 = combine_erp_3way(c, u, g, cfg=1.0, gamma=2.0)
    assert torch.allclose(out0, torch.zeros(1, 4))
    assert torch.allclose(out2, torch.full((1, 4), 2.0))


def test_shape_and_dtype_preserved():
    c = torch.randn(1, 2, 8, dtype=torch.float32)
    out = combine_erp_3way(c, c.clone(), c.clone(), cfg=3.0, gamma=3.0)
    assert out.shape == c.shape and out.dtype == c.dtype
