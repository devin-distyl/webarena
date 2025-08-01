"""
Utils module for Distyl-WebArena

Contains web-specific utilities and enhanced logging.
"""

from .web_utils import WebUtils
from .logging import DistylLogger

__all__ = [
    "WebUtils",
    "DistylLogger"
]