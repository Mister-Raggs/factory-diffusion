"""Runtime-adaptive caching primitives."""

from factory_diffusion.cache.adaptive import (
    AdaptiveCacheConfig,
    AdaptiveResidualCache,
    CacheStep,
)
from factory_diffusion.cache.fixed import FixedResidualCache, guarded_uniform_schedule

__all__ = [
    "AdaptiveCacheConfig",
    "AdaptiveResidualCache",
    "CacheStep",
    "FixedResidualCache",
    "guarded_uniform_schedule",
]
