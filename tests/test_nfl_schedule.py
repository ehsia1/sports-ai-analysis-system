"""Tests for NFL schedule utilities."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from src.sports_betting.utils.nfl_schedule import (
    GamePhase,
    NFLSchedule,
    WeekInfo,
    get_current_week,
    get_week_info,
)


class TestGamePhase:
    """Tests for GamePhase enum."""

    def test_phase_values(self):
        """Test that all expected phases exist."""
        assert GamePhase.PRE_GAME.value == "pre_game"
        assert GamePhase.IN_PROGRESS.value == "in_progress"
        assert GamePhase.POST_GAME.value == "post_game"
        assert GamePhase.OFFSEASON.value == "offseason"


class TestWeekInfo:
    """Tests for WeekInfo dataclass."""

    def test_all_games_completed_true(self):
        """Test all_games_completed when all games are done."""
        info = WeekInfo(
            season=2024,
            week=10,
            phase=GamePhase.POST_GAME,
            first_game=datetime(2024, 11, 7),
            last_game=datetime(2024, 11, 11),
            games_completed=16,
            total_games=16,
        )
        assert info.all_games_completed is True
        assert info.has_upcoming_games is False

    def test_all_games_completed_false(self):
        """Test all_games_completed when games remain."""
        info = WeekInfo(
            season=2024,
            week=10,
            phase=GamePhase.IN_PROGRESS,
            first_game=datetime(2024, 11, 7),
            last_game=datetime(2024, 11, 11),
            games_completed=10,
            total_games=16,
        )
        assert info.all_games_completed is False
        assert info.has_upcoming_games is True

    def test_empty_week(self):
        """Test properties for a week with no games."""
        info = WeekInfo(
            season=2024,
            week=1,
            phase=GamePhase.OFFSEASON,
            first_game=None,
            last_game=None,
            games_completed=0,
            total_games=0,
        )
        assert info.all_games_completed is False
        assert info.has_upcoming_games is False


class TestNFLScheduleDateCalculation:
    """Tests for date-based week calculation (fallback logic)."""

    def test_september_week_1(self):
        """Test early September returns week 1."""
        schedule = NFLSchedule()
        season, week = schedule._calculate_week_from_date(datetime(2024, 9, 6))
        assert season == 2024
        assert week == 1

    def test_mid_october(self):
        """Test mid-October returns appropriate week."""
        schedule = NFLSchedule()
        # Oct 15 is roughly 5-6 weeks after Sept 5
        season, week = schedule._calculate_week_from_date(datetime(2024, 10, 15))
        assert season == 2024
        assert 5 <= week <= 7

    def test_late_november(self):
        """Test late November returns appropriate week."""
        schedule = NFLSchedule()
        # Nov 29 is roughly 12 weeks after Sept 5
        season, week = schedule._calculate_week_from_date(datetime(2024, 11, 29))
        assert season == 2024
        assert 11 <= week <= 14

    def test_january_is_previous_season(self):
        """Test January belongs to previous season."""
        schedule = NFLSchedule()
        season, week = schedule._calculate_week_from_date(datetime(2025, 1, 15))
        assert season == 2024  # 2024 season
        assert week == 18  # Playoffs

    def test_february_is_previous_season(self):
        """Test February belongs to previous season."""
        schedule = NFLSchedule()
        season, week = schedule._calculate_week_from_date(datetime(2025, 2, 5))
        assert season == 2024
        assert week == 18

    def test_offseason_defaults_week_1(self):
        """Test offseason months default to week 1."""
        schedule = NFLSchedule()
        season, week = schedule._calculate_week_from_date(datetime(2025, 5, 15))
        assert season == 2025
        assert week == 1

    def test_week_never_exceeds_18(self):
        """Test week is capped at 18."""
        schedule = NFLSchedule()
        # Late December should not exceed week 18
        season, week = schedule._calculate_week_from_date(datetime(2024, 12, 31))
        assert week <= 18


class TestNFLScheduleDeterminePhase:
    """Tests for phase determination logic."""

    def test_pre_game_phase(self):
        """Test phase is PRE_GAME before first game."""
        schedule = NFLSchedule()
        now = datetime(2024, 11, 5, 10, 0)  # Tuesday morning
        first_game = datetime(2024, 11, 7, 20, 15)  # Thursday night
        last_game = datetime(2024, 11, 11, 20, 15)  # Monday night

        phase = schedule._determine_phase(now, first_game, last_game, 0, 16)
        assert phase == GamePhase.PRE_GAME

    def test_in_progress_phase(self):
        """Test phase is IN_PROGRESS during games."""
        schedule = NFLSchedule()
        now = datetime(2024, 11, 10, 15, 0)  # Sunday afternoon
        first_game = datetime(2024, 11, 7, 20, 15)
        last_game = datetime(2024, 11, 11, 20, 15)

        phase = schedule._determine_phase(now, first_game, last_game, 8, 16)
        assert phase == GamePhase.IN_PROGRESS

    def test_post_game_phase(self):
        """Test phase is POST_GAME when all games complete."""
        schedule = NFLSchedule()
        now = datetime(2024, 11, 12, 10, 0)  # Tuesday after MNF
        first_game = datetime(2024, 11, 7, 20, 15)
        last_game = datetime(2024, 11, 11, 20, 15)

        phase = schedule._determine_phase(now, first_game, last_game, 16, 16)
        assert phase == GamePhase.POST_GAME


class TestGetCurrentWeekIntegration:
    """Integration tests for get_current_week function."""

    def test_returns_tuple(self):
        """Test that get_current_week returns a tuple of two ints."""
        result = get_current_week()
        assert isinstance(result, tuple)
        assert len(result) == 2
        season, week = result
        assert isinstance(season, int)
        assert isinstance(week, int)

    def test_valid_season_range(self):
        """Test that season is reasonable."""
        season, week = get_current_week()
        current_year = datetime.now().year
        # Season should be within 1 year of current
        assert current_year - 1 <= season <= current_year + 1

    def test_valid_week_range(self):
        """Test that week is within valid range."""
        season, week = get_current_week()
        assert 1 <= week <= 18

    def test_with_specific_date(self):
        """Test get_current_week with a specific date."""
        # Test with a date in September 2024
        season, week = get_current_week(datetime(2024, 9, 10))
        assert season == 2024
        assert week >= 1


class TestGetWeekInfoIntegration:
    """Integration tests for get_week_info function."""

    def test_returns_week_info(self):
        """Test that get_week_info returns WeekInfo."""
        info = get_week_info(2024, 10)
        assert isinstance(info, WeekInfo)
        assert info.season == 2024
        assert info.week == 10

    def test_phase_is_valid(self):
        """Test that phase is a valid GamePhase."""
        info = get_week_info(2024, 10)
        assert isinstance(info.phase, GamePhase)
