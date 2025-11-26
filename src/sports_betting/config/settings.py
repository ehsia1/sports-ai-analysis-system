"""Application settings and configuration."""

import os
from pathlib import Path
from typing import Optional

try:
    from pydantic_settings import BaseSettings
    from pydantic import Field
except ImportError:
    from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Configuration
    odds_api_key: str = Field(..., env="ODDS_API_KEY")
    weather_api_key: Optional[str] = Field(None, env="WEATHER_API_KEY")

    # Database
    database_url: str = Field("sqlite:///data/sports_betting.db", env="DATABASE_URL")

    # File paths
    model_cache_dir: Path = Field(Path("data/models"), env="MODEL_CACHE_DIR")
    feature_cache_dir: Path = Field(Path("data/features"), env="FEATURE_CACHE_DIR")
    output_dir: Path = Field(Path("outputs"), env="OUTPUT_DIR")

    # Logging
    log_level: str = Field("INFO", env="LOG_LEVEL")
    log_file: Optional[Path] = Field(Path("logs/sports_betting.log"), env="LOG_FILE")

    # Analysis Configuration
    default_bankroll: float = Field(10000.0, env="DEFAULT_BANKROLL")
    max_bet_size: float = Field(0.05, env="MAX_BET_SIZE")  # 5% of bankroll
    min_edge_threshold: float = Field(0.02, env="MIN_EDGE_THRESHOLD")  # 2% edge
    confidence_threshold: float = Field(0.7, env="CONFIDENCE_THRESHOLD")

    # Model Configuration
    model_retrain_weeks: int = Field(4, env="MODEL_RETRAIN_WEEKS")
    feature_lookbook_weeks: int = Field(10, env="FEATURE_LOOKBACK_WEEKS")
    ensemble_models: list[str] = Field(
        ["xgboost", "neural_net", "bayesian"], env="ENSEMBLE_MODELS"
    )

    # Smart API Management
    odds_api_monthly_limit: int = Field(500, env="ODDS_API_MONTHLY_LIMIT")
    cache_ttl_props: int = Field(24 * 3600, env="CACHE_TTL_PROPS")  # 24 hours
    cache_ttl_odds: int = Field(12 * 3600, env="CACHE_TTL_ODDS")    # 12 hours
    cache_ttl_scores: int = Field(6 * 3600, env="CACHE_TTL_SCORES")  # 6 hours
    
    # Priority thresholds
    high_priority_threshold: float = Field(7.0, env="HIGH_PRIORITY_THRESHOLD")
    medium_priority_threshold: float = Field(4.0, env="MEDIUM_PRIORITY_THRESHOLD")
    
    # Update schedule (weekday: hour in UTC)
    weekly_update_day: int = Field(2, env="WEEKLY_UPDATE_DAY")  # Wednesday = 2
    weekly_update_hour: int = Field(16, env="WEEKLY_UPDATE_HOUR")  # 4 PM UTC

    # Data Sources
    odds_api_base_url: str = Field(
        "https://api.the-odds-api.com/v4", env="ODDS_API_BASE_URL"
    )
    weather_api_base_url: str = Field(
        "https://api.openweathermap.org/data/2.5", env="WEATHER_API_BASE_URL"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create directories if they don't exist
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)
        self.feature_cache_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings