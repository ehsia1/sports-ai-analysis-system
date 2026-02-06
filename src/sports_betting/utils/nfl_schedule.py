"""NFL schedule utilities for determining current week."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from functools import lru_cache
from typing import Optional, Tuple

from sqlalchemy import func

from ..database.models import Game
from ..database.session import get_session
from .logging import get_logger

logger = get_logger(__name__)


class GamePhase(Enum):
    """Phase of the NFL week."""

    PRE_GAME = "pre_game"  # Before games start (Tue-Thu morning)
    IN_PROGRESS = "in_progress"  # Games happening (Thu evening - Mon night)
    POST_GAME = "post_game"  # After games complete (Tue morning)
    OFFSEASON = "offseason"  # No games scheduled


@dataclass
class WeekInfo:
    """Detailed information about an NFL week."""

    season: int
    week: int
    phase: GamePhase
    first_game: Optional[datetime]
    last_game: Optional[datetime]
    games_completed: int
    total_games: int

    @property
    def all_games_completed(self) -> bool:
        """Check if all games in the week have been completed."""
        return self.total_games > 0 and self.games_completed == self.total_games

    @property
    def has_upcoming_games(self) -> bool:
        """Check if there are games yet to be played."""
        return self.games_completed < self.total_games


class NFLSchedule:
    """NFL schedule utilities using database-first approach."""

    def __init__(self):
        self._cache_timestamp: Optional[datetime] = None
        self._cache_duration = timedelta(minutes=5)

    def get_current_week(self, as_of_date: Optional[datetime] = None) -> Tuple[int, int]:
        """
        Get current NFL season and week.

        Uses database-first approach: queries the games table to find
        the week with upcoming or recent games. Falls back to date-based
        calculation if no database data available.

        Args:
            as_of_date: Date to calculate for (defaults to now)

        Returns:
            Tuple of (season, week)
        """
        now = as_of_date or datetime.now()

        # Try database first
        db_result = self._get_week_from_db(now)
        if db_result:
            season, week = db_result
            logger.debug(f"Week from database: {season} Week {week}")
            return season, week

        # Fallback to date-based calculation
        season, week = self._calculate_week_from_date(now)
        logger.debug(f"Week from date calculation: {season} Week {week}")
        return season, week

    def _get_week_from_db(self, now: datetime) -> Optional[Tuple[int, int]]:
        """
        Query database for current week based on game schedule.

        Logic:
        1. Find games happening today or in the next 7 days
        2. If found, return that week
        3. If not, find the most recent completed week
        4. Return None if no schedule data available
        """
        try:
            with get_session() as session:
                # Find the current season
                season = self._get_current_season_from_db(session, now)
                if not season:
                    return None

                # Find week with games closest to now (upcoming or recent)
                # Look for games within -2 to +7 days of now
                window_start = now - timedelta(days=2)
                window_end = now + timedelta(days=7)

                game = (
                    session.query(Game)
                    .filter(
                        Game.season == season,
                        Game.season_type.in_(["REG", "POST"]),
                        Game.game_date >= window_start,
                        Game.game_date <= window_end,
                    )
                    .order_by(Game.game_date)
                    .first()
                )

                if game:
                    return season, game.week

                # No games in window - check if we're before or after season
                first_game = (
                    session.query(Game)
                    .filter(Game.season == season, Game.season_type.in_(["REG", "POST"]))
                    .order_by(Game.game_date)
                    .first()
                )

                if first_game and now < first_game.game_date:
                    # Before season starts
                    return season, 1

                # Find the last completed week (including playoffs)
                last_completed = (
                    session.query(Game.week)
                    .filter(
                        Game.season == season,
                        Game.season_type.in_(["REG", "POST"]),
                        Game.is_completed == True,
                    )
                    .order_by(Game.week.desc())
                    .first()
                )

                if last_completed:
                    # Check if all games in that week are done
                    week = last_completed[0]
                    incomplete_in_week = (
                        session.query(Game)
                        .filter(
                            Game.season == season,
                            Game.week == week,
                            Game.is_completed == False,
                        )
                        .count()
                    )

                    if incomplete_in_week == 0:
                        # All games done - we're in post-game or next week
                        # Cap at week 22 (Super Bowl) for playoffs
                        next_week = min(week + 1, 22)
                        return season, next_week

                    return season, week

                return None

        except Exception as e:
            logger.warning(f"Error querying database for week: {e}")
            return None

    def _get_current_season_from_db(self, session, now: datetime) -> Optional[int]:
        """Get the current NFL season from the database."""
        # Determine likely season based on date
        if now.month >= 9:
            likely_season = now.year
        elif now.month <= 2:
            likely_season = now.year - 1
        else:
            likely_season = now.year

        # Check if this season exists in DB
        has_games = (
            session.query(Game)
            .filter(Game.season == likely_season)
            .first()
        )

        if has_games:
            return likely_season

        # Try adjacent seasons
        for season in [likely_season - 1, likely_season + 1]:
            has_games = session.query(Game).filter(Game.season == season).first()
            if has_games:
                return season

        return None

    def _calculate_week_from_date(self, now: datetime) -> Tuple[int, int]:
        """
        Calculate NFL week from date (fallback when no DB data).

        NFL Week 1 typically starts the Thursday after Labor Day
        (first Monday in September).
        """
        # Determine season
        if now.month >= 9:
            season = now.year
        elif now.month <= 2:
            season = now.year - 1  # Jan/Feb = previous year's season
        else:
            season = now.year  # Offseason - use current year

        # Calculate week
        if now.month >= 9:
            # NFL regular season: September - December
            # Week 1 starts around Sept 5-10
            week_start = datetime(now.year, 9, 5)
            days_since_start = (now - week_start).days
            week = min(18, max(1, (days_since_start // 7) + 1))
        elif now.month <= 2:
            # January/February: Playoffs and Super Bowl
            week = 18  # Final regular season week
        else:
            # Offseason (March - August)
            week = 1  # Default for offseason

        return season, week

    def get_week_info(
        self, season: int, week: int, as_of_date: Optional[datetime] = None
    ) -> WeekInfo:
        """
        Get detailed information about a specific NFL week.

        Args:
            season: NFL season year
            week: Week number (1-18)
            as_of_date: Reference date for phase calculation

        Returns:
            WeekInfo with game dates and completion status
        """
        now = as_of_date or datetime.now()

        try:
            with get_session() as session:
                games = (
                    session.query(Game)
                    .filter(
                        Game.season == season,
                        Game.week == week,
                        Game.season_type == "REG",
                    )
                    .order_by(Game.game_date)
                    .all()
                )

                if not games:
                    return WeekInfo(
                        season=season,
                        week=week,
                        phase=GamePhase.OFFSEASON,
                        first_game=None,
                        last_game=None,
                        games_completed=0,
                        total_games=0,
                    )

                first_game = games[0].game_date
                last_game = games[-1].game_date
                games_completed = sum(1 for g in games if g.is_completed)
                total_games = len(games)

                # Determine phase
                phase = self._determine_phase(
                    now, first_game, last_game, games_completed, total_games
                )

                return WeekInfo(
                    season=season,
                    week=week,
                    phase=phase,
                    first_game=first_game,
                    last_game=last_game,
                    games_completed=games_completed,
                    total_games=total_games,
                )

        except Exception as e:
            logger.warning(f"Error getting week info: {e}")
            return WeekInfo(
                season=season,
                week=week,
                phase=GamePhase.OFFSEASON,
                first_game=None,
                last_game=None,
                games_completed=0,
                total_games=0,
            )

    def _determine_phase(
        self,
        now: datetime,
        first_game: datetime,
        last_game: datetime,
        games_completed: int,
        total_games: int,
    ) -> GamePhase:
        """Determine the current phase of an NFL week."""
        # Add buffer for game completion (Monday night + few hours)
        last_game_end = last_game + timedelta(hours=4)

        if now < first_game:
            return GamePhase.PRE_GAME
        elif now <= last_game_end and games_completed < total_games:
            return GamePhase.IN_PROGRESS
        elif games_completed == total_games:
            return GamePhase.POST_GAME
        else:
            # Some games played but not all - still in progress
            return GamePhase.IN_PROGRESS

    def get_week_date_range(
        self, season: int, week: int
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Get the date range for a specific week.

        Args:
            season: NFL season year
            week: Week number

        Returns:
            Tuple of (start_date, end_date) as ISO format strings,
            or (None, None) if no data available
        """
        info = self.get_week_info(season, week)

        if info.first_game and info.last_game:
            return (
                info.first_game.strftime("%Y-%m-%d"),
                info.last_game.strftime("%Y-%m-%d"),
            )

        return None, None


# Module-level convenience functions
_schedule = None


def _get_schedule() -> NFLSchedule:
    """Get or create the singleton NFLSchedule instance."""
    global _schedule
    if _schedule is None:
        _schedule = NFLSchedule()
    return _schedule


def get_current_week(as_of_date: Optional[datetime] = None) -> Tuple[int, int]:
    """
    Get current NFL season and week.

    Args:
        as_of_date: Date to calculate for (defaults to now)

    Returns:
        Tuple of (season, week)

    Example:
        >>> season, week = get_current_week()
        >>> print(f"Currently in {season} Week {week}")
    """
    return _get_schedule().get_current_week(as_of_date)


def get_week_info(
    season: int, week: int, as_of_date: Optional[datetime] = None
) -> WeekInfo:
    """
    Get detailed information about a specific NFL week.

    Args:
        season: NFL season year
        week: Week number (1-18)
        as_of_date: Reference date for phase calculation

    Returns:
        WeekInfo dataclass with phase, game dates, completion status
    """
    return _get_schedule().get_week_info(season, week, as_of_date)


def get_week_date_range(season: int, week: int) -> Tuple[Optional[str], Optional[str]]:
    """
    Get the date range for a specific week.

    Args:
        season: NFL season year
        week: Week number

    Returns:
        Tuple of (start_date, end_date) as ISO format strings
    """
    return _get_schedule().get_week_date_range(season, week)
