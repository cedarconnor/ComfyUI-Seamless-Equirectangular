"""Pure-torch ERP diagnostics: seam / pole / temporal metrics and heatmaps.

Canonical layout for every function here is (..., H, W) with the WIDTH axis LAST.
Node adapters (erp_nodes.py) convert real ComfyUI LATENT
(B,C,F,H,W) and IMAGE (F,H,W,C) tensors into this layout before calling in.
No ComfyUI imports — this module is standalone and unit-testable.
"""
import json

import torch

_EPS = 1e-8


def seam_mae(spatial: torch.Tensor, k: int = 1) -> float:
    """Mean abs diff between the leftmost k and rightmost k width columns.

    seam_mae = mean(abs(x[..., :k] - x[..., -k:]))  over a canonical (..., H, W).
    """
    left = spatial[..., :k]
    right = spatial[..., -k:]
    return (left - right).abs().mean().item()


def seam_correlation(spatial: torch.Tensor, k: int = 1) -> float:
    """Pearson correlation of the flattened left-k strip vs right-k strip.

    High correlation with low seam_mae indicates a clean 360 wrap; low
    correlation flags a genuine boundary break rather than merely different
    content. Constant-and-equal strips return 1.0 (degenerate, no variance).
    """
    left = spatial[..., :k].reshape(-1).to(torch.float64)
    right = spatial[..., -k:].reshape(-1).to(torch.float64)
    lc = left - left.mean()
    rc = right - right.mean()
    denom = torch.sqrt((lc * lc).sum() * (rc * rc).sum())
    if denom < _EPS:
        # No variance in one/both strips: equal -> perfectly consistent.
        return 1.0 if torch.allclose(left, right) else 0.0
    return float((lc * rc).sum() / denom)


def latent_seam_metrics(spatial: torch.Tensor) -> dict:
    """Reproduce the reference pipeline's three post-stage-1 latent-seam numbers
    on a canonical (..., H, W) tensor (requires W >= 3):

      d_col0_col1   = mean(abs(col0 - col1))     -- adjacent internal gradient
      d_col0_colWm1 = mean(abs(col0 - col(W-1))) -- across the physical seam
      d_col1_col2   = mean(abs(col1 - col2))     -- adjacent internal gradient
    """
    if spatial.shape[-1] < 3:
        raise ValueError(f"latent_seam_metrics needs W>=3, got W={spatial.shape[-1]}")
    col0, col1, col2, colWm1 = (
        spatial[..., 0], spatial[..., 1], spatial[..., 2], spatial[..., -1],
    )
    return {
        "d_col0_col1": (col0 - col1).abs().mean().item(),
        "d_col0_colWm1": (col0 - colWm1).abs().mean().item(),
        "d_col1_col2": (col1 - col2).abs().mean().item(),
    }


def pole_consistency(spatial: torch.Tensor, rows: int = 1) -> dict:
    """Mean variance across longitude (width) at the top/bottom `rows` token rows.
    Reveals pathological longitude-dependent behavior at the ERP poles; not a
    pass/fail score. Uses population variance (unbiased=False) for determinism.
    """
    top = spatial[..., :rows, :]
    bottom = spatial[..., -rows:, :]
    return {
        "pole_top_var": top.var(dim=-1, unbiased=False).mean().item(),
        "pole_bottom_var": bottom.var(dim=-1, unbiased=False).mean().item(),
    }


def per_frame_seam_mae(video: torch.Tensor, k: int = 1) -> torch.Tensor:
    """seam_mae for each frame along dim 0 of a canonical (F, ..., H, W) tensor.
    Returns a 1-D tensor of length F."""
    diff = (video[..., :k] - video[..., -k:]).abs()   # (F, ..., H, k)
    return diff.flatten(1).mean(dim=1)


def temporal_seam_variation(per_frame: torch.Tensor) -> float:
    """Population std of a 1-D per-frame seam_mae sequence (std over frames)."""
    per_frame = torch.as_tensor(per_frame, dtype=torch.float32)
    if per_frame.numel() <= 1:
        return 0.0
    return per_frame.std(unbiased=False).item()


def _normalize01(x: torch.Tensor) -> torch.Tensor:
    """Min-max to [0,1]; a flat input maps to all-zeros."""
    lo = x.min()
    span = x.max() - lo
    if span < _EPS:
        return torch.zeros_like(x)
    return (x - lo) / span


def _gray_to_image(gray: torch.Tensor) -> torch.Tensor:
    """(H, W) in [0,1] -> comfy IMAGE (1, H, W, 3)."""
    return gray.clamp(0.0, 1.0).unsqueeze(-1).repeat(1, 1, 3).unsqueeze(0)


def seam_difference_heatmap(frame: torch.Tensor) -> torch.Tensor:
    """Circular column-gradient magnitude of a single (H, W, C) frame -> IMAGE.
    roll-by-1 on width makes column 0's neighbor column W-1, so a seam break
    shows as a bright column at width index 0."""
    f = frame.to(torch.float32)
    grad = (f - torch.roll(f, shifts=1, dims=1)).abs().mean(dim=-1)  # (H, W)
    return _gray_to_image(_normalize01(grad))


def pole_variance_heatmap(frame: torch.Tensor) -> torch.Tensor:
    """Per-row across-longitude variance of a single (H, W, C) frame -> IMAGE."""
    f = frame.to(torch.float32).mean(dim=-1)                        # (H, W)
    row_var = f.var(dim=-1, unbiased=False, keepdim=True)           # (H, 1)
    gray = _normalize01(row_var).expand(-1, f.shape[-1])            # (H, W)
    return _gray_to_image(gray)


def metrics_to_json(metrics: dict) -> str:
    """Pretty JSON with floats rounded to 6 dp (stable, comparable across runs)."""
    rounded = {k: (round(float(v), 6) if isinstance(v, (int, float)) else v)
               for k, v in metrics.items()}
    return json.dumps(rounded, indent=2)


def metrics_to_csv(metrics: dict) -> str:
    """CSV: 'metric,value' header then one sorted key,value line per entry."""
    lines = ["metric,value"]
    for k in sorted(metrics):
        lines.append(f"{k},{round(float(metrics[k]), 6)}")
    return "\n".join(lines) + "\n"
