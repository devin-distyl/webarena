"""
Controller module for Distyl-WebArena

Contains the main controller that orchestrates all components and provides
WebArena-compatible interface for seamless integration.
"""

from .controller import DistylWebArenaController
from .interface_adapter import WebArenaInterfaceAdapter

__all__ = [
    "DistylWebArenaController",
    "WebArenaInterfaceAdapter"
]