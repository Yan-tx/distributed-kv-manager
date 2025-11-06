"""Prefetch & IO Aggregation subsystem.

Exports the core classes for consumption by caching/storage layers.
"""
from .core import (
    EntryState,
    PrefetchEntry,
    PrefetchBuffer,
    BudgetEstimator,
    RateLimiter,
    IOAggregator,
    PlanBuilder,
)

__all__ = [
    "EntryState",
    "PrefetchEntry",
    "PrefetchBuffer",
    "BudgetEstimator",
    "RateLimiter",
    "IOAggregator",
    "PlanBuilder",
]
