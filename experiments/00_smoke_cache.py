"""CPU/MPS-safe functional smoke test; not a performance benchmark."""

from __future__ import annotations

import torch

from factory_diffusion.cache import AdaptiveCacheConfig, AdaptiveResidualCache


def main() -> None:
    cache = AdaptiveResidualCache(
        AdaptiveCacheConfig(threshold=1.0, warmup_steps=2, force_compute_last=1)
    )
    cache.reset(total_steps=6)
    calls = 0

    for step in range(6):
        model_input = torch.full((1, 8, 6), float(step))

        def compute() -> torch.Tensor:
            nonlocal calls
            calls += 1
            return model_input + 2.0

        result = cache.run(step, model_input, compute)
        expected = model_input + 2.0
        torch.testing.assert_close(result.output, expected)
        print(
            f"step={step} recomputed={result.recomputed} reason={result.reason} "
            f"predicted_error={result.predicted_error:.6f}"
        )

    print(f"model calls: {calls}/6")


if __name__ == "__main__":
    main()
