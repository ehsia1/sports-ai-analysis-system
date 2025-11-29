"""Minimal configuration module."""

import os
from pathlib import Path
from dataclasses import dataclass


@dataclass
class Settings:
    """Application settings."""
    database_url: str = ""

    def __post_init__(self):
        if not self.database_url:
            # Default to SQLite in data directory
            data_dir = Path(__file__).parent.parent.parent.parent.parent / "data"
            data_dir.mkdir(exist_ok=True)
            self.database_url = f"sqlite:///{data_dir / 'sports_betting.db'}"


_settings = None


def get_settings() -> Settings:
    """Get application settings."""
    global _settings
    if _settings is None:
        _settings = Settings(
            database_url=os.getenv("DATABASE_URL", "")
        )
    return _settings
