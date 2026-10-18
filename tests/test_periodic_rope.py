import math

import torch

from seamless_equirect import periodic_rope as rope


def test_spherical_width_is_identical_across_video_frames_and_token_order():
    # Latitude depends on spatial row, never frame number or token order.
    t, h, w = torch.meshgrid(torch.arange(3), torch.arange(4), torch.arange(8), indexing="ij")
    grid = torch.stack((t * 8, h * 32, w * 32)).reshape(1, 3, -1).float()
    indices = torch.logspace(0, 4, 8) * (math.pi / 2)
    rope.set_params(enabled=True, alpha=1, w_scale=32)
    result = rope.build_freqs_periodic(indices, grid, [20, 2048, 2048], False)
    width = result[..., 2::3].reshape(3, 32, 8)
    torch.testing.assert_close(width[0], width[1])
    torch.testing.assert_close(width[0], width[2])
    permutation = torch.randperm(grid.shape[-1], generator=torch.Generator().manual_seed(42))
    shuffled = rope.build_freqs_periodic(indices, grid[..., permutation], [20, 2048, 2048], False)
    torch.testing.assert_close(shuffled, result[:, permutation])
    stock = indices * (grid.transpose(1, 2).unsqueeze(-1) / torch.tensor([20, 2048, 2048]).view(1, 1, 3, 1) * 2 - 1)
    stock = stock.transpose(-1, -2).flatten(2)
    torch.testing.assert_close(result[..., 0::3], stock[..., 0::3])
    torch.testing.assert_close(result[..., 1::3], stock[..., 1::3])
