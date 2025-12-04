"""Injury data service for checking player injury status."""

import logging
import requests
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

from ..database import get_session, InjuryReport, Player

logger = logging.getLogger(__name__)

# ESPN team ID mapping
ESPN_TEAM_IDS = {
    'ARI': 22, 'ATL': 1, 'BAL': 33, 'BUF': 2, 'CAR': 29, 'CHI': 3,
    'CIN': 4, 'CLE': 5, 'DAL': 6, 'DEN': 7, 'DET': 8, 'GB': 9,
    'HOU': 34, 'IND': 11, 'JAX': 30, 'KC': 12, 'LAC': 24, 'LAR': 14,
    'LV': 13, 'MIA': 15, 'MIN': 16, 'NE': 17, 'NO': 18, 'NYG': 19,
    'NYJ': 20, 'PHI': 21, 'PIT': 23, 'SEA': 26, 'SF': 25, 'TB': 27,
    'TEN': 10, 'WAS': 28,
}


@dataclass
class InjuryStatus:
    """Player injury status for betting decisions."""

    player_id: int
    player_name: str
    status: str  # Healthy, Questionable, Doubtful, Out, IR, PUP, Suspended
    primary_injury: Optional[str]
    practice_friday: Optional[str]  # Full, Limited, DNP
    games_missed: int

    @property
    def is_out(self) -> bool:
        """Player is definitely not playing."""
        status_lower = self.status.lower() if self.status else ''
        return status_lower in ('out', 'ir', 'pup', 'suspended', 'injured reserve')

    @property
    def is_questionable(self) -> bool:
        """Player game status is uncertain."""
        status_lower = self.status.lower() if self.status else ''
        return status_lower in ('questionable', 'doubtful')

    @property
    def is_limited(self) -> bool:
        """Player had limited practice participation."""
        return self.practice_friday in ('Limited', 'DNP')

    @property
    def confidence_multiplier(self) -> float:
        """
        Get confidence multiplier based on injury status.

        Returns value 0-1 to multiply model confidence by.
        """
        if self.is_out:
            return 0.0  # Don't bet on players who are out

        status_lower = self.status.lower() if self.status else ''

        if status_lower == 'doubtful':
            return 0.3  # Very unlikely to play, high uncertainty

        if status_lower == 'questionable':
            # Check practice status for better signal
            practice = self.practice_friday.upper() if self.practice_friday else ''
            if practice == 'DNP':
                return 0.5  # Questionable + DNP = concerning
            elif practice == 'LIMITED':
                return 0.7  # Questionable + Limited = moderate concern
            else:
                return 0.85  # Questionable + Full practice = likely playing

        # Healthy but check for recent missed games
        if self.games_missed > 0:
            return 0.9  # Just returning from injury

        return 1.0  # Healthy, no concerns

    @property
    def warning_message(self) -> Optional[str]:
        """Get warning message for display."""
        status_lower = self.status.lower() if self.status else ''

        if self.is_out:
            return f"OUT ({self.primary_injury or 'injury'})"

        if status_lower == 'doubtful':
            return f"DOUBTFUL ({self.primary_injury or 'injury'})"

        if status_lower == 'questionable':
            practice = f", {self.practice_friday}" if self.practice_friday else ""
            return f"QUESTIONABLE ({self.primary_injury or 'injury'}{practice})"

        if self.games_missed > 0:
            return f"Returning from injury ({self.games_missed} games missed)"

        return None


class InjuryService:
    """Service for checking player injury status."""

    # Injury statuses that mean player is OUT (case-insensitive matching)
    OUT_STATUSES = {
        'Out', 'out', 'IR', 'ir', 'PUP', 'pup', 'Suspended', 'suspended',
        'Reserve/COVID-19', 'Injured Reserve', 'injured reserve'
    }

    # Injury statuses that mean uncertain
    UNCERTAIN_STATUSES = {'Questionable', 'questionable', 'Doubtful', 'doubtful', 'Probable', 'probable'}

    def __init__(self):
        self._cache: Dict[int, InjuryStatus] = {}
        self._cache_week: Optional[int] = None
        self._cache_season: Optional[int] = None
        self._espn_cache: Dict[str, Dict] = {}  # player_name -> injury data

    def fetch_espn_injuries(self, team_abbr: str) -> Dict[str, Dict]:
        """
        Fetch current injuries from ESPN's free API.

        Args:
            team_abbr: Team abbreviation (e.g., 'KC', 'PHI')

        Returns:
            Dict mapping player names to injury info
        """
        team_id = ESPN_TEAM_IDS.get(team_abbr)
        if not team_id:
            logger.warning(f"Unknown team abbreviation: {team_abbr}")
            return {}

        url = f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/{team_id}/injuries"

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

            injuries = {}
            for item in data.get('items', []):
                athlete = item.get('athlete', {})
                player_name = athlete.get('displayName', '')
                status = item.get('status', 'Unknown')
                injury_type = item.get('type', {}).get('text', '')
                details = item.get('details', {})

                if player_name:
                    injuries[player_name] = {
                        'status': status,
                        'injury_type': injury_type,
                        'long_comment': details.get('longComment', ''),
                        'short_comment': details.get('shortComment', ''),
                    }

            return injuries

        except requests.RequestException as e:
            logger.warning(f"Failed to fetch ESPN injuries for {team_abbr}: {e}")
            return {}

    def fetch_all_espn_injuries(self) -> Dict[str, Dict]:
        """Fetch injuries for all teams from ESPN's main injuries endpoint."""
        all_injuries = {}

        url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries"

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()

            # Main endpoint returns injuries grouped by team
            for team_data in data.get('injuries', []):
                team_name = team_data.get('displayName', 'Unknown')

                for injury in team_data.get('injuries', []):
                    athlete = injury.get('athlete', {})
                    player_name = athlete.get('displayName', '')
                    status = injury.get('status', 'Unknown')
                    injury_type = injury.get('type', {})
                    if isinstance(injury_type, dict):
                        injury_type = injury_type.get('description', '')

                    if player_name:
                        all_injuries[player_name] = {
                            'status': status,
                            'injury_type': injury_type,
                            'team': team_name,
                            'long_comment': injury.get('longComment', ''),
                            'short_comment': injury.get('shortComment', ''),
                        }

            logger.info(f"Fetched {len(all_injuries)} injured players from ESPN")

        except requests.RequestException as e:
            logger.warning(f"Failed to fetch ESPN injuries: {e}")

        self._espn_cache = all_injuries
        return all_injuries

    def get_injury_status(
        self,
        player_id: int,
        season: int,
        week: int
    ) -> InjuryStatus:
        """
        Get injury status for a player for a specific week.

        Args:
            player_id: Database player ID
            season: Season year
            week: Week number

        Returns:
            InjuryStatus dataclass with injury information
        """
        # Check cache
        if (self._cache_week == week and
            self._cache_season == season and
            player_id in self._cache):
            return self._cache[player_id]

        # Clear cache if week/season changed
        if self._cache_week != week or self._cache_season != season:
            self._cache.clear()
            self._cache_week = week
            self._cache_season = season

        with get_session() as session:
            player = session.query(Player).get(player_id)
            if not player:
                # Return healthy default if player not found
                return InjuryStatus(
                    player_id=player_id,
                    player_name="Unknown",
                    status="Healthy",
                    primary_injury=None,
                    practice_friday=None,
                    games_missed=0
                )

            # Get most recent injury report for this week
            injury_report = session.query(InjuryReport).filter(
                InjuryReport.player_id == player_id,
                InjuryReport.season == season,
                InjuryReport.week == week
            ).order_by(InjuryReport.report_date.desc()).first()

            if injury_report:
                status = InjuryStatus(
                    player_id=player_id,
                    player_name=player.name,
                    status=injury_report.injury_status,
                    primary_injury=injury_report.primary_injury,
                    practice_friday=injury_report.practice_friday,
                    games_missed=injury_report.games_missed
                )
            else:
                # No DB injury report - check ESPN cache
                espn_injury = self._espn_cache.get(player.name)
                if espn_injury:
                    status = InjuryStatus(
                        player_id=player_id,
                        player_name=player.name,
                        status=espn_injury.get('status', 'Unknown'),
                        primary_injury=espn_injury.get('injury_type'),
                        practice_friday=None,
                        games_missed=0
                    )
                else:
                    # No injury report - check player's current_status field
                    status = InjuryStatus(
                        player_id=player_id,
                        player_name=player.name,
                        status=player.current_status or "Healthy",
                        primary_injury=None,
                        practice_friday=None,
                        games_missed=0
                    )

            self._cache[player_id] = status
            return status

    def get_week_injuries(
        self,
        season: int,
        week: int
    ) -> Dict[int, InjuryStatus]:
        """
        Get all injury statuses for a week.

        Returns dict of player_id -> InjuryStatus
        """
        injuries = {}

        with get_session() as session:
            # Get all injury reports for this week
            reports = session.query(InjuryReport).filter(
                InjuryReport.season == season,
                InjuryReport.week == week
            ).all()

            # Group by player, take most recent
            player_reports: Dict[int, InjuryReport] = {}
            for report in reports:
                if (report.player_id not in player_reports or
                    report.report_date > player_reports[report.player_id].report_date):
                    player_reports[report.player_id] = report

            # Convert to InjuryStatus
            for player_id, report in player_reports.items():
                player = session.query(Player).get(player_id)
                if player:
                    injuries[player_id] = InjuryStatus(
                        player_id=player_id,
                        player_name=player.name,
                        status=report.injury_status,
                        primary_injury=report.primary_injury,
                        practice_friday=report.practice_friday,
                        games_missed=report.games_missed
                    )

        return injuries

    def get_injured_players_summary(
        self,
        season: int,
        week: int,
        use_espn: bool = True
    ) -> str:
        """Get formatted summary of injured players for the week."""
        injuries = self.get_week_injuries(season, week)

        # If no DB injuries and ESPN enabled, fetch from ESPN
        if not injuries and use_espn:
            espn_injuries = self.fetch_all_espn_injuries()
            if espn_injuries:
                # Convert ESPN injuries to display format
                lines = [f"INJURY REPORT - Week {week} (from ESPN)"]
                lines.append("=" * 50)

                out_players = []
                doubtful_players = []
                questionable_players = []

                for name, info in espn_injuries.items():
                    status = info.get('status', 'Unknown')
                    status_lower = status.lower() if status else ''
                    injury_type = info.get('injury_type', 'injury')

                    if status_lower in ('out', 'ir', 'pup', 'suspended', 'injured reserve'):
                        out_players.append((name, injury_type))
                    elif status_lower == 'doubtful':
                        doubtful_players.append((name, injury_type))
                    elif status_lower == 'questionable':
                        questionable_players.append((name, injury_type))

                if out_players:
                    lines.append(f"\n🚫 OUT ({len(out_players)}):")
                    for name, injury in sorted(out_players):
                        lines.append(f"  {name}: {injury}")

                if doubtful_players:
                    lines.append(f"\n⚠️ DOUBTFUL ({len(doubtful_players)}):")
                    for name, injury in sorted(doubtful_players):
                        lines.append(f"  {name}: {injury}")

                if questionable_players:
                    lines.append(f"\n❓ QUESTIONABLE ({len(questionable_players)}):")
                    for name, injury in sorted(questionable_players):
                        lines.append(f"  {name}: {injury}")

                lines.append("")
                lines.append(f"Total: {len(espn_injuries)} players with injury designations")

                return "\n".join(lines)

        if not injuries:
            return "No injury reports found for this week."

        # Group by status
        out_players = []
        doubtful_players = []
        questionable_players = []

        for status in injuries.values():
            if status.is_out:
                out_players.append(status)
            elif status.status == 'Doubtful':
                doubtful_players.append(status)
            elif status.status == 'Questionable':
                questionable_players.append(status)

        lines = [f"INJURY REPORT - Week {week}"]
        lines.append("=" * 50)

        if out_players:
            lines.append(f"\n🚫 OUT ({len(out_players)}):")
            for p in sorted(out_players, key=lambda x: x.player_name):
                lines.append(f"  {p.player_name}: {p.primary_injury or 'injury'}")

        if doubtful_players:
            lines.append(f"\n⚠️ DOUBTFUL ({len(doubtful_players)}):")
            for p in sorted(doubtful_players, key=lambda x: x.player_name):
                practice = f" ({p.practice_friday})" if p.practice_friday else ""
                lines.append(f"  {p.player_name}: {p.primary_injury or 'injury'}{practice}")

        if questionable_players:
            lines.append(f"\n❓ QUESTIONABLE ({len(questionable_players)}):")
            for p in sorted(questionable_players, key=lambda x: x.player_name):
                practice = f" ({p.practice_friday})" if p.practice_friday else ""
                lines.append(f"  {p.player_name}: {p.primary_injury or 'injury'}{practice}")

        return "\n".join(lines)


# Singleton instance
_injury_service: Optional[InjuryService] = None


def get_injury_service() -> InjuryService:
    """Get the singleton injury service instance."""
    global _injury_service
    if _injury_service is None:
        _injury_service = InjuryService()
    return _injury_service
