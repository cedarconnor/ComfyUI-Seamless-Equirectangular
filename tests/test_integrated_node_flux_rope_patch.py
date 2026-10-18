import pytest

pytest.importorskip("comfy.ldm.flux.layers")

from erp_test_helpers import FakeModelPatcher, FakeFlux2DM  # noqa: E402
from seamless_equirect.erp_nodes import EquirectSphericalRoPEFlux  # noqa: E402
from seamless_equirect.flux2_rope import FLUX_PE_KEY  # noqa: E402
from seamless_equirect.erp_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS  # noqa: E402


def test_input_types_and_contract():
    it = EquirectSphericalRoPEFlux.INPUT_TYPES()
    assert "model" in it["required"]
    assert "scale" in it["required"]
    assert EquirectSphericalRoPEFlux.RETURN_TYPES == ("MODEL",)
    assert EquirectSphericalRoPEFlux.CATEGORY == "equirect/rope"


def test_apply_returns_patched_clone():
    node = EquirectSphericalRoPEFlux()
    p = FakeModelPatcher(FakeFlux2DM())
    (out,) = node.apply(p, scale=2.0)
    assert out is not p
    assert FLUX_PE_KEY in out.object_patches


def test_apply_rejects_non_flux2():
    from comfy.ldm.flux.layers import EmbedND

    class _Bad:
        def __init__(self):
            self.pe_embedder = EmbedND(dim=128, theta=10000, axes_dim=[16, 56, 56])
    node = EquirectSphericalRoPEFlux()
    with pytest.raises(Exception):
        node.apply(FakeModelPatcher(_Bad()), scale=2.0)


def test_node_registered():
    assert "EquirectSphericalRoPEFlux" in NODE_CLASS_MAPPINGS
    assert (NODE_DISPLAY_NAME_MAPPINGS["EquirectSphericalRoPEFlux"]
            == "Equirect Spherical RoPE (Flux.2)")
