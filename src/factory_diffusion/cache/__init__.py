"""Runtime-adaptive caching primitives."""

from factory_diffusion.cache.adaptive import (
    AdaptiveCacheConfig,
    AdaptiveResidualCache,
    CacheStep,
)

__all__ = ["AdaptiveCacheConfig", "AdaptiveResidualCache", "CacheStep"]
