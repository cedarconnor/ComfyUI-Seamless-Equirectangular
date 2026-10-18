import pytest

pytest.importorskip("comfy.ldm.flux.layers")

import torch  # noqa: E402
from comfy.ldm.flux.layers import EmbedND  # noqa: E402

from erp_test_helpers import FakeModelPatcher, FakeFlux2DM, make_flux2_ids  # noqa: E402
from seamless_equirect.flux2_rope import (  # noqa: E402
    FLUX_PE_KEY, _SphericalFluxEmbedND, install_flux_spherical_rope,
)

AXES = [32, 32, 32, 32]
W_START, W_END = 32, 48                       # width block = pairs [32:48]


def _stock_and_wrapped(H, W, n_txt=7):
    emb_mod = EmbedND(dim=sum(AXES), theta=2000, axes_dim=AXES)
    ids = make_flux2_ids(H, W, n_txt=n_txt)
    stock = emb_mod(ids)
    wrapped = _SphericalFluxEmbedND(emb_mod, scale=2.0)(ids)
    return stock, wrapped, H * W, n_txt


def test_non_width_blocks_bit_identical():
    stock, wrapped, _, _ = _stock_and_wrapped(4, 16)
    assert torch.equal(wrapped[..., 0:W_START, :, :], stock[..., 0:W_START, :, :])   # T,H
    assert torch.equal(wrapped[..., W_END:64, :, :], stock[..., W_END:64, :, :])     # L


def test_width_block_changes_on_image_rows():
    stock, wrapped, n_img, _ = _stock_and_wrapped(4, 16)
    w_stock = stock[..., -n_img:, W_START:W_END, :, :]
    w_wrap = wrapped[..., -n_img:, W_START:W_END, :, :]
    assert not torch.allclose(w_wrap, w_stock)


def test_text_rows_untouched():
    stock, wrapped, n_img, n_txt = _stock_and_wrapped(4, 16, n_txt=9)
    assert torch.equal(wrapped[..., :n_txt, :, :, :], stock[..., :n_txt, :, :, :])


def test_pole_row_width_converges_through_wrapper():
    _, wrapped, _, n_txt = _stock_and_wrapped(5, 16)
    W = 16
    # First image row (row 0 = top pole): width block equal across all columns.
    row0 = wrapped[0, 0, n_txt:n_txt + W, W_START:W_END, :, :]
    assert torch.allclose(row0, row0[0:1].expand_as(row0), atol=1e-5)


def test_guard_defers_on_reference_tokens():
    emb_mod = EmbedND(dim=sum(AXES), theta=2000, axes_dim=AXES)
    ids = make_flux2_ids(4, 16)
    ids[:, -1, 0] = 10.0                       # a T>0 (reference) token present
    stock = emb_mod(ids)
    wrapped = _SphericalFluxEmbedND(emb_mod, scale=2.0)(ids)
    assert torch.equal(wrapped, stock)


def test_guard_defers_on_non_flux2_axes():
    emb3 = EmbedND(dim=128, theta=10000, axes_dim=[16, 56, 56])   # Flux.1 layout
    ids = torch.zeros((1, 20, 3), dtype=torch.float32)
    ids[:, :, 2] = torch.arange(20, dtype=torch.float32)
    stock = emb3(ids)
    wrapped = _SphericalFluxEmbedND(emb3, scale=2.0)(ids)
    assert torch.equal(wrapped, stock)


def test_install_clones_and_patches():
    p = FakeModelPatcher(FakeFlux2DM())
    out = install_flux_spherical_rope(p, scale=2.0)
    assert out is not p
    assert FLUX_PE_KEY in out.object_patches


def test_install_rejects_non_flux2():
    class _Bad:
        def __init__(self):
            self.pe_embedder = EmbedND(dim=128, theta=10000, axes_dim=[16, 56, 56])
    p = FakeModelPatcher(_Bad())
    with pytest.raises(ValueError):
        install_flux_spherical_rope(p, scale=2.0)
