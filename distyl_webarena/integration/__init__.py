"""
Integration module for Distyl-WebArena

Contains WebArena system integration and parallel execution support.
"""

from .webarena_adapter import WebArenaAdapter
from .parallel_integration import ParallelIntegration

__all__ = [
    "WebArenaAdapter", 
    "ParallelIntegration"
]