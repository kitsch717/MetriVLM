"""Selected MetriVLM core components, with tensor-only interfaces.

This reference contains no pretrained weights, Qwen integration, task heads,
training recipe or dataset. It is not an end-to-end measurement system.
All modules start untrained; callers must set their own training/freeze policy.
RGB tokens for injection are visual-encoder tokens; tokens for measurement
readout are the final Qwen hidden states at the aligned image-token positions.
"""

import math

import torch
from torch import nn
import torch.nn.functional as F


def prepare_metric_depth(depth_m, focal_px, grid, max_depth_m=80.0):
    """Validity-aware pooling of predicted depth (metres), not ground truth.

    Args:
        depth_m: [B, 1, H, W], at the original RGB image resolution.
        focal_px: [B], predicted focal length in original-image pixels.
        grid: (height, width) of the intermediate depth feature grid.
    Returns:
        depth, coverage: [B, 1, *grid]; focal: [B, 2], (f/W, f/H).
        Coverage is the fraction of valid source pixels in each pooled cell.
    """
    if depth_m.ndim != 4 or depth_m.shape[1] != 1:
        raise ValueError("depth_m must have shape [B, 1, H, W]")
    b, _, h, w = depth_m.shape
    focal_px = torch.as_tensor(focal_px, device=depth_m.device).float()
    if focal_px.shape != (b,) or not torch.isfinite(focal_px).all() or (focal_px <= 0).any():
        raise ValueError("One finite, positive predicted focal length per image is required")
    z = depth_m.float()
    valid = torch.isfinite(z) & (z > 0.05) & (z < max_depth_m)
    if not valid.flatten(1).any(1).all():
        raise ValueError("Each image must contain valid predicted depth")
    coverage = F.interpolate(valid.float(), grid, mode="area")
    weighted = F.interpolate(torch.where(valid, z, 0.0), grid, mode="area")
    depth = weighted / coverage.clamp_min(1e-6)
    focal = torch.stack((focal_px / w, focal_px / h), dim=1)
    return depth, coverage, focal


class MetricDepthTokenAdapter(nn.Module):
    """Depth encoding, support-aware pooling and RGB-conditioned modulation.

    The gate operates on 64-channel depth features BEFORE linear projection.
    Its zero initialization makes the gate equal to one. The projection also
    starts at zero, preserving the RGB visual tokens before optimization.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(6, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.GELU(),
        )
        self.projection = nn.Linear(64, hidden_size, bias=False)
        self.rgb_norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.rgb_gate = nn.Linear(hidden_size, 64)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.rgb_gate.weight)
        nn.init.zeros_(self.rgb_gate.bias)

    def encode_tokens(self, depth, coverage, focal, token_grid):
        """Inputs are prepared metric depth, support fractions and (f/W, f/H)."""
        if depth.ndim != 4 or depth.shape[1] != 1 or depth.shape != coverage.shape:
            raise ValueError("depth and coverage must have shape [B, 1, H, W]")
        if focal.shape != (len(depth), 2):
            raise ValueError("focal must have shape [B, 2]")
        if not all(torch.isfinite(t).all() for t in (depth, coverage, focal)):
            raise ValueError("Adapter inputs must be finite; preprocess invalid depth first")
        if (coverage < 0).any() or (coverage > 1).any() or (focal <= 0).any():
            raise ValueError("Require coverage in [0, 1] and positive focal lengths")
        if not (coverage > 1e-6).flatten(1).any(1).all():
            raise ValueError("Each image must contain supported depth")
        depth, coverage, focal = depth.float(), coverage.float(), focal.float()
        reference = (depth * coverage).sum((2, 3), keepdim=True)
        reference = reference / coverage.sum((2, 3), keepdim=True)
        raw = torch.cat((
            depth / 10.0, depth.clamp_min(0.05).log() / 4.0,
            depth / reference.clamp_min(0.05), coverage,
            focal.log()[:, :, None, None].expand(-1, -1, *depth.shape[-2:]),
        ), dim=1)
        features = self.encoder(raw)
        support = F.interpolate(coverage, token_grid, mode="area")
        pooled = F.interpolate(features * coverage, token_grid, mode="area")
        tokens = (pooled / support.clamp_min(1e-6)).flatten(2).transpose(1, 2)
        return tokens, support.flatten(2).transpose(1, 2) > 1e-6

    def forward(self, depth, coverage, focal, token_grid, rgb_tokens, *, modulate=True):
        """Return aligned depth tokens [B, N, d], N = product(token_grid).

        rgb_tokens must be [B, N, d], in the same spatial order. Add the returned
        tokens to the corresponding RGB visual tokens before Qwen processing;
        retain the SAME depth tokens for the depth measurement-value branch.
        """
        tokens, valid = self.encode_tokens(depth, coverage, focal, token_grid)
        expected = (len(depth), tokens.shape[1], self.projection.out_features)
        if rgb_tokens.shape != expected:
            raise ValueError(f"Aligned RGB tokens must have shape {expected}")
        if modulate:
            gate = 1.0 + self.rgb_gate(self.rgb_norm(rgb_tokens.float())).tanh()
            tokens = tokens * gate
        return self.projection(tokens) * valid


class RGBMeasurementReadout(nn.Module):
    """Measurement-specific RGB contexts from final image-position hidden states.

    RGB values are the supplied hidden states directly, without a value
    projection. This branch has its own keys and learned measurement queries.
    """

    def __init__(self, hidden_size, num_queries=8, key_size=64):
        super().__init__()
        self.normalization = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.keys = nn.Linear(hidden_size, key_size, bias=False)
        self.queries = nn.Parameter(torch.zeros(num_queries, key_size))

    def forward(self, rgb_hidden):
        """rgb_hidden: [B, N, d], containing image-token positions ONLY."""
        if rgb_hidden.ndim != 3 or rgb_hidden.shape[1] == 0:
            raise ValueError("Supply nonempty, aligned image-token hidden states [B, N, d]")
        values = rgb_hidden.float()
        keys = self.keys(self.normalization(values))
        scores = torch.einsum("bnk,qk->bqn", keys, self.queries)
        attention = (scores / math.sqrt(keys.shape[-1])).softmax(-1)
        return torch.einsum("bqn,bnd->bqd", attention, values), attention


class RGBGuidedDepthMeasurementBottleneck(nn.Module):
    """RGB selects positions; projected depth tokens supply metric features.

    These keys and queries are independent of RGBMeasurementReadout. Depth
    values are normalized, then projected; they are not raw depths in metres.
    """

    def __init__(self, hidden_size, num_queries=8, key_size=64):
        super().__init__()
        self.rgb_normalization = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.depth_normalization = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.keys = nn.Linear(hidden_size, key_size, bias=False)
        self.queries = nn.Parameter(torch.empty(num_queries, key_size))
        self.residual_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.normal_(self.queries, std=0.01)
        nn.init.normal_(self.residual_projection.weight, std=1e-4)

    def forward(self, rgb_hidden, depth_tokens):
        """Both inputs: [B, N, d]; outputs: contexts [B, Q, d], attention [B, Q, N]."""
        if rgb_hidden.ndim != 3 or rgb_hidden.shape != depth_tokens.shape or not rgb_hidden.shape[1]:
            raise ValueError("RGB hidden states and depth tokens must align as [B, N, d]")
        keys = self.keys(self.rgb_normalization(rgb_hidden.float()))
        scores = torch.einsum("bnk,qk->bqn", keys, self.queries)
        attention = (scores / math.sqrt(keys.shape[-1])).softmax(-1)
        values = self.residual_projection(self.depth_normalization(depth_tokens.float()))
        return torch.einsum("bqn,bnd->bqd", attention, values), attention


class AdditiveMeasurementContextFusion(nn.Module):
    """Independent RGB/depth readout followed by elementwise context addition.

    Returns FEATURES, not physical measurements or compliance judgments.
    The depth locator receives detached RGB hidden states, matching the
    original additive path; its keys and queries can still receive gradients.
    """

    def __init__(self, hidden_size, num_queries=8, key_size=64):
        super().__init__()
        self.rgb = RGBMeasurementReadout(hidden_size, num_queries, key_size)
        self.depth = RGBGuidedDepthMeasurementBottleneck(hidden_size, num_queries, key_size)

    def forward(self, rgb_hidden, depth_tokens):
        rgb_context, rgb_attention = self.rgb(rgb_hidden)
        depth_context, depth_attention = self.depth(rgb_hidden.detach(), depth_tokens)
        return {
            "measurement_context": rgb_context + depth_context,
            "rgb_context": rgb_context,
            "depth_context": depth_context,
            "rgb_attention": rgb_attention,
            "depth_attention": depth_attention,
        }
