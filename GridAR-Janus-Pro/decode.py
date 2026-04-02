from typing import Optional

import torch


# Config scale follows the paper's w, but decoding keeps the Janus-Pro unconditioned-anchor form.
def cfg_decode(logit_cond: torch.Tensor, logit_uncond: torch.Tensor, scale: float, **_: object) -> torch.Tensor:
    direction_original = logit_cond - logit_uncond
    internal_scale = scale + 1.0
    return logit_uncond + internal_scale * direction_original

# Keep the same convention for 3-way decode:
# logits = uncond + (w + 1) * direction_original + w * orthogonal_refined
def cfg_decode_3_way(
    logit_cond_modified: torch.Tensor,
    logit_cond: torch.Tensor,
    logit_uncond: torch.Tensor,
    scale: float,
    scale_refined: Optional[float] = None,
    **_: object,
) -> torch.Tensor:
    direction_original = logit_cond - logit_uncond
    direction_refined = logit_cond_modified - logit_uncond

    eps = 1e-12
    projection = (direction_refined * direction_original).sum(dim=-1, keepdim=True)
    norm_sq = (direction_original * direction_original).sum(dim=-1, keepdim=True).clamp_min(eps)
    orthogonal_refined = direction_refined - (projection / norm_sq) * direction_original

    refined_scale = scale if scale_refined is None else scale_refined
    internal_original_scale = scale + 1.0
    return logit_uncond + internal_original_scale * direction_original + refined_scale * orthogonal_refined
