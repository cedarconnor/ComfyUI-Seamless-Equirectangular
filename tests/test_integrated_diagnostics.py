import math
import torch

from seamless_equirect.erp_metrics import seam_mae, seam_correlation


def test_seam_mae_zero_when_edges_identical():
    x = torch.arange(4 * 5, dtype=torch.float32).reshape(4, 5)
    x[:, -1] = x[:, 0]                       # right edge == left edge
    assert seam_mae(x, k=1) == 0.0


def test_seam_mae_matches_manual_meandiff():
    x = torch.zeros(3, 4)
    x[:, 0] = 1.0                            # left col = 1, right col = 0
    assert math.isclose(seam_mae(x, k=1), 1.0, rel_tol=1e-6)


def test_seam_mae_batched_leading_dims():
    x = torch.zeros(2, 3, 4, 5)             # (B, C, H, W)
    x[..., 0] = 2.0
    x[..., -1] = 0.0
    assert math.isclose(seam_mae(x, k=1), 2.0, rel_tol=1e-6)


def test_seam_correlation_high_for_wrapping_gradient():
    # A column that continues smoothly across the wrap: col0 ~ colW-1 shifted.
    row = torch.linspace(0.0, 1.0, 6)
    frame = row.unsqueeze(0).repeat(4, 1)   # (H=4, W=6), each row identical ramp
    # left strip (col0) and right strip (col-1) both vary across H identically here -> corr ~1
    frame = frame + torch.linspace(0, 3, 4).unsqueeze(1)  # add per-row offset
    c = seam_correlation(frame, k=1)
    assert c > 0.99


def test_seam_correlation_constant_returns_one():
    x = torch.full((3, 4), 5.0)
    assert seam_correlation(x, k=1) == 1.0


from seamless_equirect.erp_metrics import latent_seam_metrics, pole_consistency


def test_latent_seam_metrics_keys_and_values():
    x = torch.zeros(2, 4)                    # (H=2, W=4)
    x[:, 0] = 0.0
    x[:, 1] = 1.0
    x[:, 2] = 2.0
    x[:, 3] = 9.0
    m = latent_seam_metrics(x)
    assert set(m) == {"d_col0_col1", "d_col0_colWm1", "d_col1_col2"}
    assert math.isclose(m["d_col0_col1"], 1.0, rel_tol=1e-6)   # |0-1|
    assert math.isclose(m["d_col0_colWm1"], 9.0, rel_tol=1e-6) # |0-9|
    assert math.isclose(m["d_col1_col2"], 1.0, rel_tol=1e-6)   # |1-2|


def test_pole_consistency_zero_when_longitude_constant():
    # Every column identical at top and bottom row -> zero longitude variance.
    x = torch.zeros(3, 5)
    x[0, :] = 7.0                            # top row constant across longitude
    x[-1, :] = 4.0                           # bottom row constant across longitude
    m = pole_consistency(x, rows=1)
    assert math.isclose(m["pole_top_var"], 0.0, abs_tol=1e-6)
    assert math.isclose(m["pole_bottom_var"], 0.0, abs_tol=1e-6)


def test_pole_consistency_positive_when_longitude_varies():
    x = torch.zeros(3, 4)
    x[0, :] = torch.tensor([0.0, 1.0, 2.0, 3.0])   # top row varies across longitude
    m = pole_consistency(x, rows=1)
    assert m["pole_top_var"] > 0.0


from seamless_equirect.erp_metrics import per_frame_seam_mae, temporal_seam_variation


def test_per_frame_seam_mae_shape_and_values():
    v = torch.zeros(3, 2, 4)                 # (F=3, H=2, W=4)
    v[0, :, 0] = 1.0                         # frame0 seam diff = 1
    v[1, :, 0] = 3.0                         # frame1 seam diff = 3
    # frame2 all zero -> 0
    s = per_frame_seam_mae(v, k=1)
    assert s.shape == (3,)
    assert torch.allclose(s, torch.tensor([1.0, 3.0, 0.0]))


def test_temporal_seam_variation_is_population_std():
    s = torch.tensor([1.0, 3.0, 0.0])
    expect = s.std(unbiased=False).item()
    assert math.isclose(temporal_seam_variation(s), expect, rel_tol=1e-6)


def test_temporal_seam_variation_zero_for_constant():
    assert temporal_seam_variation(torch.tensor([2.0, 2.0, 2.0])) == 0.0


import json
from seamless_equirect.erp_metrics import (
    seam_difference_heatmap, pole_variance_heatmap, metrics_to_json, metrics_to_csv,
)


def test_seam_heatmap_shape_and_range():
    frame = torch.rand(8, 10, 3)
    hm = seam_difference_heatmap(frame)
    assert hm.shape == (1, 8, 10, 3)
    assert float(hm.min()) >= 0.0 and float(hm.max()) <= 1.0


def test_seam_heatmap_flat_frame_is_zero():
    frame = torch.full((4, 6, 3), 0.5)
    hm = seam_difference_heatmap(frame)
    assert torch.allclose(hm, torch.zeros_like(hm))


def test_pole_heatmap_shape_and_range():
    frame = torch.rand(8, 10, 3)
    hm = pole_variance_heatmap(frame)
    assert hm.shape == (1, 8, 10, 3)
    assert float(hm.min()) >= 0.0 and float(hm.max()) <= 1.0


def test_metrics_to_json_roundtrips_floats():
    s = metrics_to_json({"seam_mae": 0.123456789, "corr": 1.0})
    d = json.loads(s)
    assert math.isclose(d["seam_mae"], 0.123457, rel_tol=1e-6)
    assert d["corr"] == 1.0


def test_metrics_to_csv_header_and_sorted_rows():
    s = metrics_to_csv({"b": 2.0, "a": 1.0})
    lines = s.strip().splitlines()
    assert lines[0] == "metric,value"
    assert lines[1].startswith("a,")
    assert lines[2].startswith("b,")
