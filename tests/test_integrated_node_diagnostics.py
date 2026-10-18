import json
import pytest
import torch

pytest.importorskip("comfy.samplers")   # keep node tests uniform with the pack

from seamless_equirect.erp_nodes import EquirectLatentDiagnostics  # noqa: E402
from seamless_equirect.erp_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS  # noqa: E402


def test_latent_node_contract():
    it = EquirectLatentDiagnostics.INPUT_TYPES()["required"]
    assert "samples" in it and "pole_rows" in it
    assert EquirectLatentDiagnostics.RETURN_TYPES == ("STRING", "STRING")
    assert EquirectLatentDiagnostics.RETURN_NAMES == ("metrics_json", "metrics_csv")


def test_latent_node_reports_seam_numbers():
    node = EquirectLatentDiagnostics()
    lat = torch.zeros(1, 4, 2, 3, 5)         # (B,C,F,H,W), W=5
    lat[..., 0] = 0.0
    lat[..., -1] = 8.0                        # big col0 vs colW-1 gap
    (js, csv) = node.analyze({"samples": lat}, pole_rows=1)
    d = json.loads(js)
    assert d["d_col0_colWm1"] == 8.0
    assert "pole_top_var" in d and "pole_bottom_var" in d
    assert csv.startswith("metric,value")


def test_latent_node_registered():
    assert "EquirectLatentDiagnostics" in NODE_CLASS_MAPPINGS
    assert (NODE_DISPLAY_NAME_MAPPINGS["EquirectLatentDiagnostics"]
            == "Equirect Latent Diagnostics")


from seamless_equirect.erp_nodes import EquirectFrameDiagnostics  # noqa: E402


def test_frame_node_contract():
    it = EquirectFrameDiagnostics.INPUT_TYPES()["required"]
    for k in ("images", "seam_width_k", "pole_rows"):
        assert k in it
    assert EquirectFrameDiagnostics.RETURN_TYPES == ("STRING", "STRING", "IMAGE", "IMAGE")
    assert EquirectFrameDiagnostics.RETURN_NAMES == (
        "metrics_json", "metrics_csv", "seam_heatmap", "pole_heatmap")


def test_frame_node_outputs_metrics_and_heatmaps():
    node = EquirectFrameDiagnostics()
    imgs = torch.rand(3, 8, 10, 3)           # (F,H,W,C)
    js, csv, seam_hm, pole_hm = node.analyze(imgs, seam_width_k=1, pole_rows=1)
    d = json.loads(js)
    for key in ("seam_mae", "seam_correlation", "temporal_seam_std",
                "pole_top_var", "pole_bottom_var"):
        assert key in d
    assert seam_hm.shape == (1, 8, 10, 3)
    assert pole_hm.shape == (1, 8, 10, 3)
    assert csv.startswith("metric,value")


def test_frame_node_handles_batched_5d():
    node = EquirectFrameDiagnostics()
    imgs = torch.rand(2, 3, 8, 10, 3)        # (B,F,H,W,C)
    js, csv, seam_hm, pole_hm = node.analyze(imgs, seam_width_k=1, pole_rows=1)
    assert seam_hm.shape == (1, 8, 10, 3)    # heatmap of first frame


def test_frame_node_registered():
    assert "EquirectFrameDiagnostics" in NODE_CLASS_MAPPINGS
    assert (NODE_DISPLAY_NAME_MAPPINGS["EquirectFrameDiagnostics"]
            == "Equirect Frame Diagnostics")
