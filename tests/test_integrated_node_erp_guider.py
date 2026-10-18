import pytest
import torch

pytest.importorskip("comfy.samplers")

from erp_test_helpers import FakeModelPatcher, FakeLTXDiffusionModel  # noqa: E402
from seamless_equirect.erp_nodes import EquirectERPGuider  # noqa: E402
from seamless_equirect.erp_guider import Guider_ERP3Way  # noqa: E402
from seamless_equirect.erp_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS  # noqa: E402

COND = [[torch.zeros(1, 8, 16), {}]]


def test_input_types_and_contract():
    it = EquirectERPGuider.INPUT_TYPES()["required"]
    for k in ("model", "positive", "negative", "geometry", "cfg", "erp_gamma"):
        assert k in it
    assert EquirectERPGuider.RETURN_TYPES == ("GUIDER",)


def test_get_guider_returns_configured_guider():
    node = EquirectERPGuider()
    mp = FakeModelPatcher(FakeLTXDiffusionModel())
    (guider,) = node.get_guider(mp, COND, COND, COND, cfg=2.0, erp_gamma=4.0)
    assert isinstance(guider, Guider_ERP3Way)
    assert guider.cfg == 2.0 and guider.gamma == 4.0
    assert "geometry" in guider.original_conds


def test_node_registered():
    assert "EquirectERPGuider" in NODE_CLASS_MAPPINGS
    assert NODE_DISPLAY_NAME_MAPPINGS["EquirectERPGuider"] == "Equirect ERP Guider"
