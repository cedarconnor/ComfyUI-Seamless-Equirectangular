import torch

from seamless_equirect.flux2_angles import build_flux2_width_angles


def test_shape_and_dtype():
    ang = build_flux2_width_angles(H=4, W=8)
    assert ang.shape == (4 * 8, 16)          # w_dim=32 -> 16 half-angle pairs
    assert ang.dtype == torch.float64


def test_deterministic():
    a = build_flux2_width_angles(H=6, W=12)
    b = build_flux2_width_angles(H=6, W=12)
    assert torch.equal(a, b)


def test_pole_rows_converge_on_path_b():
    # At the poles cos(latitude)=0 -> spherical X,Y are column-independent, so the
    # last (Path B) channel is constant across all columns of a pole row.
    H, W = 5, 16
    ang = build_flux2_width_angles(H=H, W=W)
    top_row = ang[0:W, -1]                    # row 0 = top pole, last channel (Path B)
    bot_row = ang[(H - 1) * W:H * W, -1]       # row H-1 = bottom pole
    assert torch.allclose(top_row, top_row[0].expand_as(top_row))
    assert torch.allclose(bot_row, bot_row[0].expand_as(bot_row))
    # An interior (non-pole) row is NOT column-constant on that channel.
    mid = ang[2 * W:3 * W, -1]
    assert not torch.allclose(mid, mid[0].expand_as(mid))


def test_path_a_is_linear_ramp_in_column():
    # Channel 0 is the lowest frequency -> Path A (harmonic). Its angle must be a pure
    # linear ramp in column within a row (constant per-column step), i.e. horizontal
    # periodicity is preserved.
    H, W = 3, 32
    ang = build_flux2_width_angles(H=H, W=W)
    row = ang[0:W, 0]
    diffs = row[1:] - row[:-1]
    assert torch.allclose(diffs, diffs[0].expand_as(diffs), atol=1e-9)
    assert float(diffs[0]) > 0.0             # strictly increasing (nonzero harmonic)


def test_scale_changes_path_b():
    # Path B radius R_width = W/scale, so changing scale must change spherical angles.
    a = build_flux2_width_angles(H=4, W=16, scale=2.0)
    b = build_flux2_width_angles(H=4, W=16, scale=4.0)
    assert not torch.allclose(a[:, -1], b[:, -1])
