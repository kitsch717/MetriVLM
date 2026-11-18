"""CPU smoke check with RANDOM tensors; not an image inference/evaluation demo.

Run: python demo_components.py
No data, pretrained model, network access or GPU is needed.
"""

import torch

from metrivlm import (
    AdditiveMeasurementContextFusion,
    MetricDepthTokenAdapter,
    prepare_metric_depth,
)


def main():
    torch.manual_seed(7)
    torch.set_num_threads(1)
    batch, hidden_size, queries = 2, 32, 8
    token_grid = (3, 4)
    count = token_grid[0] * token_grid[1]
    raw_depth = 0.1 + 10 * torch.rand(batch, 1, 24, 32)
    raw_depth[:, :, :4, :4] = float("nan")
    depth, coverage, focal = prepare_metric_depth(
        raw_depth, torch.tensor([35.0, 40.0]), (12, 16)
    )
    visual_tokens = torch.randn(batch, count, hidden_size)
    adapter = MetricDepthTokenAdapter(hidden_size)
    initial = adapter(depth, coverage, focal, token_grid, visual_tokens)
    assert torch.count_nonzero(initial) == 0  # Identity RGB injection at initialization.

    # Stand in for learned projection weights, solely to exercise nonzero paths.
    # These random values are NOT released weights or a performance result.
    with torch.no_grad():
        torch.nn.init.normal_(adapter.projection.weight, std=0.02)
    depth_tokens = adapter(depth, coverage, focal, token_grid, visual_tokens)
    unmodulated = adapter(depth, coverage, focal, token_grid, visual_tokens, modulate=False)
    torch.testing.assert_close(depth_tokens, unmodulated)  # Zero gate => multiplier 1.
    injected_visual_tokens = visual_tokens + depth_tokens
    assert injected_visual_tokens.shape == (batch, count, hidden_size)

    # A full model would run injected tokens through Qwen, then extract its
    # final image-position hidden states. Random stand-ins keep this demo small.
    final_image_hidden = torch.randn(batch, count, hidden_size, requires_grad=True)
    reader = AdditiveMeasurementContextFusion(hidden_size, queries, key_size=16)
    output = reader(final_image_hidden, depth_tokens)
    assert output["measurement_context"].shape == (batch, queries, hidden_size)
    for name in ("rgb_attention", "depth_attention"):
        torch.testing.assert_close(output[name].sum(-1), torch.ones(batch, queries))
    torch.testing.assert_close(
        output["measurement_context"], output["rgb_context"] + output["depth_context"]
    )
    # Zero RGB queries implement uniform image-token pooling at initialization.
    torch.testing.assert_close(
        output["rgb_context"], final_image_hidden.mean(1, keepdim=True).expand(-1, queries, -1)
    )
    assert reader.rgb.keys.weight.data_ptr() != reader.depth.keys.weight.data_ptr()
    assert reader.rgb.queries.data_ptr() != reader.depth.queries.data_ptr()
    zero_depth = reader(final_image_hidden, torch.zeros_like(depth_tokens))["depth_context"]
    assert torch.count_nonzero(zero_depth) == 0
    assert output["depth_context"].abs().max() > 0

    output["measurement_context"].square().mean().backward()
    for parameter in (adapter.rgb_gate.weight, adapter.projection.weight,
                      reader.rgb.queries, reader.depth.queries, reader.depth.keys.weight):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    try:
        prepare_metric_depth(torch.zeros_like(raw_depth), torch.ones(batch), (12, 16))
    except ValueError:
        pass
    else:
        raise AssertionError("All-invalid depth must be rejected")
    print("Passed: depth preprocessing, identity gate, aligned injection, independent")
    print("RGB/depth attention, additive contexts, gradient flow and invalid-depth handling.")
    print("Context shape:", tuple(output["measurement_context"].shape))
    print("Random tensors only; no physical measurement accuracy is evaluated.")


if __name__ == "__main__":
    main()
