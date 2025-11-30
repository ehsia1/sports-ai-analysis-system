"""Configuration module.

Usage:
    from sports_betting.config import get_settings

    settings = get_settings()
    print(settings.odds_api_key)
    print(settings.min_edge_pct)
"""

from .settings import Settings, get_settings

__all__ = ["Settings", "get_settings"]
