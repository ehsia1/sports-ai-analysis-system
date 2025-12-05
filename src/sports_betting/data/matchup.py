"""Matchup context service for game-level betting factors."""

import logging
from dataclasses import dataclass
from typing import Dict, Optional

from ..database import get_session, Game, Team

logger = logging.getLogger(__name__)


@dataclass
class MatchupContext:
    """Game matchup context for betting decisions."""

    game_id: int
    home_team: str
    away_team: str
    spread: Optional[float]  # Home team spread (negative = home favorite)
    total: Optional[float]  # Over/under total points
    home_implied_total: Optional[float]
    away_implied_total: Optional[float]
    home_moneyline: Optional[int]
    away_moneyline: Optional[int]

    @property
    def home_is_favorite(self) -> Optional[bool]:
        """Check if home team is favored."""
        if self.spread is not None:
            return self.spread < 0
        return None

    @property
    def spread_magnitude(self) -> Optional[float]:
        """Get absolute value of spread (how lopsided)."""
        if self.spread is not None:
            return abs(self.spread)
        return None

    @property
    def is_high_total(self) -> bool:
        """Game total is above average (47+)."""
        return self.total is not None and self.total >= 47

    @property
    def is_low_total(self) -> bool:
        """Game total is below average (< 43)."""
        return self.total is not None and self.total < 43

    def get_team_implied_total(self, team_abbr: str) -> Optional[float]:
        """Get implied total for a specific team."""
        if team_abbr == self.home_team:
            return self.home_implied_total
        elif team_abbr == self.away_team:
            return self.away_implied_total
        return None

    def is_team_favorite(self, team_abbr: str) -> Optional[bool]:
        """Check if a specific team is the favorite."""
        if self.spread is None:
            return None
        if team_abbr == self.home_team:
            return self.spread < 0
        elif team_abbr == self.away_team:
            return self.spread > 0
        return None

    def get_confidence_multiplier(self, team_abbr: str, market: str) -> float:
        """
        Get confidence multiplier based on matchup context.

        Higher implied totals boost passing/receiving confidence.
        Lower implied totals boost rushing confidence.

        Args:
            team_abbr: Team abbreviation
            market: Market type (player_pass_yds, player_reception_yds, etc.)

        Returns:
            Confidence multiplier (0.8 - 1.2)
        """
        implied_total = self.get_team_implied_total(team_abbr)
        if implied_total is None:
            return 1.0

        # Base multiplier
        multiplier = 1.0

        if market in ('player_pass_yds', 'player_reception_yds', 'player_receptions'):
            # Passing/receiving benefits from higher team totals
            if implied_total >= 27:  # High scoring offense
                multiplier = 1.05
            elif implied_total >= 24:
                multiplier = 1.02
            elif implied_total < 20:  # Low scoring
                multiplier = 0.95
            elif implied_total < 17:
                multiplier = 0.90

        elif market == 'player_rush_yds':
            # Rushing can benefit from favorable game script (favorites in control)
            is_fav = self.is_team_favorite(team_abbr)
            spread_mag = self.spread_magnitude

            if is_fav and spread_mag and spread_mag >= 7:
                # Big favorites may run more to kill clock
                multiplier = 1.05
            elif is_fav and spread_mag and spread_mag >= 3:
                multiplier = 1.02
            # Underdogs trailing may abandon run game
            elif is_fav is False and spread_mag and spread_mag >= 7:
                multiplier = 0.95

        return multiplier

    @property
    def summary(self) -> str:
        """Get human-readable matchup summary."""
        parts = []

        if self.spread is not None:
            if self.spread < 0:
                parts.append(f"{self.home_team} -{abs(self.spread):.1f}")
            else:
                parts.append(f"{self.away_team} -{self.spread:.1f}")

        if self.total is not None:
            parts.append(f"O/U {self.total:.1f}")

        return " | ".join(parts) if parts else "No odds"

    @property
    def detailed_summary(self) -> str:
        """Get detailed matchup summary for display."""
        lines = []

        if self.spread is not None:
            fav = self.home_team if self.spread < 0 else self.away_team
            dog = self.away_team if self.spread < 0 else self.home_team
            lines.append(f"Spread: {fav} {-abs(self.spread):.1f} (vs {dog})")

        if self.total is not None:
            lines.append(f"Total: {self.total:.1f}")
            if self.home_implied_total and self.away_implied_total:
                lines.append(f"Implied: {self.away_team} {self.away_implied_total:.1f} - {self.home_team} {self.home_implied_total:.1f}")

        return "\n".join(lines) if lines else "No matchup data"


class MatchupService:
    """Service for getting game matchup context."""

    def __init__(self):
        self._cache: Dict[int, MatchupContext] = {}

    def get_matchup_context(self, game_id: int) -> Optional[MatchupContext]:
        """
        Get matchup context for a game.

        Args:
            game_id: Database game ID

        Returns:
            MatchupContext with spread, total, and implied totals
        """
        if game_id in self._cache:
            return self._cache[game_id]

        with get_session() as session:
            game = session.query(Game).get(game_id)
            if not game:
                return None

            context = MatchupContext(
                game_id=game_id,
                home_team=game.home_team.abbreviation,
                away_team=game.away_team.abbreviation,
                spread=game.spread,
                total=game.total,
                home_implied_total=game.home_implied_total,
                away_implied_total=game.away_implied_total,
                home_moneyline=game.home_moneyline,
                away_moneyline=game.away_moneyline,
            )

            self._cache[game_id] = context
            return context

    def get_week_matchups(self, season: int, week: int) -> Dict[int, MatchupContext]:
        """Get matchup context for all games in a week."""
        matchups = {}

        with get_session() as session:
            games = session.query(Game).filter_by(
                season=season,
                week=week,
                season_type='REG'
            ).all()

            for game in games:
                context = MatchupContext(
                    game_id=game.id,
                    home_team=game.home_team.abbreviation,
                    away_team=game.away_team.abbreviation,
                    spread=game.spread,
                    total=game.total,
                    home_implied_total=game.home_implied_total,
                    away_implied_total=game.away_implied_total,
                    home_moneyline=game.home_moneyline,
                    away_moneyline=game.away_moneyline,
                )
                matchups[game.id] = context
                self._cache[game.id] = context

        return matchups

    def format_week_matchups(self, season: int, week: int) -> str:
        """Get formatted matchup report for a week."""
        matchups = self.get_week_matchups(season, week)

        if not matchups:
            return f"No matchup data for Week {week}"

        lines = [f"MATCHUP CONTEXT - Week {week}"]
        lines.append("=" * 60)

        for context in sorted(matchups.values(), key=lambda x: x.game_id):
            game_str = f"{context.away_team} @ {context.home_team}"
            lines.append(f"\n{game_str}")
            lines.append("-" * 30)

            if context.spread is not None:
                fav = context.home_team if context.spread < 0 else context.away_team
                lines.append(f"  Favorite: {fav} by {abs(context.spread):.1f}")

            if context.total is not None:
                lines.append(f"  Total: {context.total:.1f}")

            if context.home_implied_total and context.away_implied_total:
                lines.append(f"  Implied: {context.away_team} {context.away_implied_total:.1f} vs {context.home_team} {context.home_implied_total:.1f}")

        return "\n".join(lines)


# Singleton instance
_matchup_service: Optional[MatchupService] = None


def get_matchup_service() -> MatchupService:
    """Get the singleton matchup service instance."""
    global _matchup_service
    if _matchup_service is None:
        _matchup_service = MatchupService()
    return _matchup_service
