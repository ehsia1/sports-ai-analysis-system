"""Centralized configuration using pydantic-settings.

All configuration values are loaded from environment variables with sensible defaults.
Environment variables can be set in .env file or directly in the environment.

Usage:
    from sports_betting.config import get_settings
    settings = get_settings()
    print(settings.odds_api_key)
"""

from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent.parent


def _get_current_nfl_season() -> int:
    """Determine the current NFL season based on date.

    NFL season runs September to February, so Jan/Feb belongs to previous year's season.
    """
    now = datetime.now()
    if now.month < 3:  # Jan/Feb = previous year's season
        return now.year - 1
    return now.year


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # Ignore extra env vars
    )

    # ==========================================================================
    # Database
    # ==========================================================================
    database_url: str = Field(
        default="sqlite:///data/sports_betting.db",
        description="SQLAlchemy database URL"
    )

    # ==========================================================================
    # API Keys
    # ==========================================================================
    odds_api_key: Optional[str] = Field(
        default=None,
        description="The Odds API key for fetching betting lines"
    )
    weather_api_key: Optional[str] = Field(
        default=None,
        description="Weather API key (optional)"
    )

    # ==========================================================================
    # API Limits
    # ==========================================================================
    odds_api_monthly_limit: int = Field(
        default=500,
        description="Monthly API credit limit for The Odds API"
    )
    odds_api_daily_budget: int = Field(
        default=30,
        description="Conservative daily API credit budget"
    )

    # ==========================================================================
    # NFL Season
    # ==========================================================================
    current_season: int = Field(
        default_factory=_get_current_nfl_season,
        description="Current NFL season year"
    )

    # ==========================================================================
    # Betting Thresholds
    # ==========================================================================
    min_edge_pct: float = Field(
        default=3.0,
        description="Minimum edge percentage to consider a bet (e.g., 3.0 = 3%)"
    )
    min_confidence: float = Field(
        default=0.65,
        description="Minimum model confidence to place a bet (0-1)"
    )
    max_edge_pct: float = Field(
        default=50.0,
        description="Maximum edge percentage - higher is suspicious"
    )

    # ==========================================================================
    # Bankroll Management
    # ==========================================================================
    default_bankroll: float = Field(
        default=10000.0,
        description="Default bankroll for paper trading"
    )
    max_bet_fraction: float = Field(
        default=0.05,
        description="Maximum bet size as fraction of bankroll"
    )
    default_stake: float = Field(
        default=100.0,
        description="Default stake per bet in paper trading"
    )

    # ==========================================================================
    # Paths
    # ==========================================================================
    data_dir: Path = Field(
        default_factory=lambda: _get_project_root() / "data",
        description="Directory for data files"
    )
    models_dir: Path = Field(
        default_factory=lambda: _get_project_root() / "models",
        description="Directory for ML model files"
    )
    logs_dir: Path = Field(
        default_factory=lambda: _get_project_root() / "logs",
        description="Directory for log files"
    )
    cache_dir: Path = Field(
        default_factory=lambda: Path.home() / ".sports_betting" / "cache",
        description="Directory for cache files"
    )

    # ==========================================================================
    # Logging
    # ==========================================================================
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR)"
    )
    log_to_file: bool = Field(
        default=True,
        description="Whether to write logs to file"
    )

    # ==========================================================================
    # Notifications
    # ==========================================================================
    discord_webhook_url: Optional[str] = Field(
        default=None,
        description="Discord webhook URL for notifications"
    )
    discord_notify_min_edge: float = Field(
        default=5.0,
        description="Minimum edge % to trigger Discord notification"
    )

    # ==========================================================================
    # Scheduling
    # ==========================================================================
    odds_fetch_hour: int = Field(
        default=10,
        description="Hour (ET) to fetch daily odds"
    )
    predictions_day: str = Field(
        default="Thursday",
        description="Day of week to generate predictions"
    )
    scoring_day: str = Field(
        default="Tuesday",
        description="Day of week to score previous week's results"
    )

    # ==========================================================================
    # Validators
    # ==========================================================================
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        return v_upper

    @field_validator("min_confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        if not 0 <= v <= 1:
            raise ValueError("min_confidence must be between 0 and 1")
        return v

    @field_validator("max_bet_fraction")
    @classmethod
    def validate_bet_fraction(cls, v: float) -> float:
        if not 0 < v <= 1:
            raise ValueError("max_bet_fraction must be between 0 and 1")
        return v

    # ==========================================================================
    # Computed Properties
    # ==========================================================================
    @property
    def min_edge(self) -> float:
        """Min edge as decimal (e.g., 0.03 for 3%)."""
        return self.min_edge_pct / 100

    @property
    def max_edge(self) -> float:
        """Max edge as decimal."""
        return self.max_edge_pct / 100

    @property
    def db_path(self) -> Path:
        """Extract SQLite database path from URL."""
        if self.database_url.startswith("sqlite:///"):
            return Path(self.database_url.replace("sqlite:///", ""))
        return self.data_dir / "sports_betting.db"

    def ensure_directories(self) -> None:
        """Create all required directories if they don't exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings.

    Settings are loaded once and cached for performance.
    Call get_settings.cache_clear() to reload.
    """
    settings = Settings()
    settings.ensure_directories()
    return settings
