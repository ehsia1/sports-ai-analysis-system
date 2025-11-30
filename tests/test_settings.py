"""Tests for the settings/configuration module."""

import pytest
import os
from unittest.mock import patch


class TestSettings:
    """Test the Settings class and configuration loading."""

    def test_settings_loads_defaults(self):
        """Settings should load with sensible defaults."""
        # Clear cache to ensure fresh settings
        from src.sports_betting.config.settings import get_settings
        get_settings.cache_clear()

        settings = get_settings()

        assert settings.min_edge_pct == 3.0
        assert settings.min_confidence == 0.65
        assert settings.log_level == "INFO"
        assert settings.default_stake == 100.0

    def test_min_edge_property(self):
        """min_edge property should convert percentage to decimal."""
        from src.sports_betting.config.settings import get_settings
        get_settings.cache_clear()

        settings = get_settings()

        # 3.0% should become 0.03
        assert settings.min_edge == pytest.approx(0.03)

    def test_max_edge_property(self):
        """max_edge property should convert percentage to decimal."""
        from src.sports_betting.config.settings import get_settings
        get_settings.cache_clear()

        settings = get_settings()

        # 50.0% should become 0.50
        assert settings.max_edge == pytest.approx(0.50)

    def test_log_level_validation(self):
        """Invalid log level should raise validation error."""
        from pydantic import ValidationError
        from src.sports_betting.config.settings import Settings

        with pytest.raises(ValidationError):
            Settings(log_level="INVALID")

    def test_confidence_validation(self):
        """Confidence must be between 0 and 1."""
        from pydantic import ValidationError
        from src.sports_betting.config.settings import Settings

        with pytest.raises(ValidationError):
            Settings(min_confidence=1.5)

        with pytest.raises(ValidationError):
            Settings(min_confidence=-0.1)

    def test_bet_fraction_validation(self):
        """Bet fraction must be between 0 and 1."""
        from pydantic import ValidationError
        from src.sports_betting.config.settings import Settings

        with pytest.raises(ValidationError):
            Settings(max_bet_fraction=1.5)

        with pytest.raises(ValidationError):
            Settings(max_bet_fraction=0)

    def test_current_season_auto_detection(self):
        """current_season should auto-detect based on date."""
        from src.sports_betting.config.settings import _get_current_nfl_season
        from datetime import datetime

        # This should return a reasonable year
        season = _get_current_nfl_season()
        current_year = datetime.now().year

        # Season should be either current year or previous year
        assert season in [current_year, current_year - 1]

    def test_settings_caching(self):
        """get_settings should return cached instance."""
        from src.sports_betting.config.settings import get_settings
        get_settings.cache_clear()

        settings1 = get_settings()
        settings2 = get_settings()

        assert settings1 is settings2

    def test_ensure_directories_creates_folders(self, tmp_path):
        """ensure_directories should create required folders."""
        from src.sports_betting.config.settings import Settings

        settings = Settings(
            data_dir=tmp_path / "data",
            models_dir=tmp_path / "models",
            logs_dir=tmp_path / "logs",
            cache_dir=tmp_path / "cache"
        )

        settings.ensure_directories()

        assert (tmp_path / "data").exists()
        assert (tmp_path / "models").exists()
        assert (tmp_path / "logs").exists()
        assert (tmp_path / "cache").exists()
