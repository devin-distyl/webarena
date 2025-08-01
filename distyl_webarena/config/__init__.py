"""
Configuration module for Distyl-WebArena

Contains default configurations and site-specific settings.
"""

from .default_config import DefaultConfig
from .site_configs import SiteConfigs

__all__ = [
    "DefaultConfig",
    "SiteConfigs"
]