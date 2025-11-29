"""
Situational context for prediction adjustments.

Tracks factors that affect predictions but aren't captured in basic stats:
- Coaching changes (new OC = more uncertainty)
- QB changes (affects target distribution)
- Teammate injuries (opportunity shifts)
- Weather conditions
- Home/away/divisional matchups
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class SituationalContext:
    """
    Track situational factors that affect receiving predictions.

    These factors create adjustments to predictions beyond what
    the statistical model captures.
    """

    # Adjustment factors (multipliers on prediction)
    ADJUSTMENTS = {
        # QB changes
        'new_qb_starter': 0.85,      # New QB = fewer connections
        'backup_qb': 0.80,           # Backup QB even worse
        'elite_qb_upgrade': 1.15,    # Traded to team with better QB

        # Coaching changes
        'new_oc': 0.90,              # New offensive coordinator = less predictable
        'new_hc': 0.92,              # New head coach

        # Team situation
        'divisional_game': 1.05,     # More familiarity, higher variance
        'primetime_game': 1.03,      # Slight boost for primetime
        'bad_weather': 0.90,         # Rain/snow/wind hurts passing

        # Player situation
        'returning_injury': 0.85,    # First game back from injury
        'wr1_injured': 1.20,         # Target share boost if WR1 out
        'te1_injured': 1.10,         # Boost if primary TE out
    }

    def __init__(self):
        """Initialize context tracker."""
        # Load or initialize situational data
        self._qb_changes = {}       # team -> new_qb info
        self._coaching_changes = {} # team -> coaching info
        self._injuries = {}         # player -> injury status
        self._weather = {}          # game -> weather info

    def set_qb_change(
        self,
        team: str,
        new_qb: str,
        reason: str = 'injury',  # 'injury', 'trade', 'benched'
        games_started: int = 0
    ):
        """Record a QB change for a team."""
        self._qb_changes[team] = {
            'new_qb': new_qb,
            'reason': reason,
            'games_started': games_started,
            'recorded_at': datetime.now(),
        }
        logger.info(f"QB change recorded: {team} -> {new_qb} ({reason})")

    def set_coaching_change(
        self,
        team: str,
        position: str,  # 'HC', 'OC', 'WR_coach'
        new_coach: str
    ):
        """Record a coaching change."""
        if team not in self._coaching_changes:
            self._coaching_changes[team] = {}

        self._coaching_changes[team][position] = {
            'new_coach': new_coach,
            'recorded_at': datetime.now(),
        }
        logger.info(f"Coaching change: {team} {position} -> {new_coach}")

    def set_player_injury(
        self,
        player_name: str,
        status: str,  # 'out', 'doubtful', 'questionable', 'returning'
        position: str = 'WR'
    ):
        """Record player injury status."""
        self._injuries[player_name] = {
            'status': status,
            'position': position,
            'recorded_at': datetime.now(),
        }

    def set_weather(
        self,
        game_key: str,  # e.g., "BUF@NE_2025_13"
        conditions: str,  # 'clear', 'rain', 'snow', 'wind'
        temp_f: int = 70,
        wind_mph: int = 10
    ):
        """Record weather for a game."""
        self._weather[game_key] = {
            'conditions': conditions,
            'temp_f': temp_f,
            'wind_mph': wind_mph,
        }

    def get_adjustment_factor(
        self,
        player_name: str,
        team: str,
        opponent: str,
        game_key: Optional[str] = None
    ) -> tuple[float, List[str]]:
        """
        Calculate total adjustment factor for a player.

        Returns:
            Tuple of (adjustment_factor, list_of_reasons)
        """
        factor = 1.0
        reasons = []

        # Check QB situation
        if team in self._qb_changes:
            qb_info = self._qb_changes[team]
            if qb_info['games_started'] < 3:
                factor *= self.ADJUSTMENTS['new_qb_starter']
                reasons.append(f"New QB ({qb_info['new_qb']})")

        # Check coaching changes
        if team in self._coaching_changes:
            if 'OC' in self._coaching_changes[team]:
                factor *= self.ADJUSTMENTS['new_oc']
                reasons.append("New OC")
            if 'HC' in self._coaching_changes[team]:
                factor *= self.ADJUSTMENTS['new_hc']
                reasons.append("New HC")

        # Check player injury status
        if player_name in self._injuries:
            status = self._injuries[player_name]['status']
            if status == 'returning':
                factor *= self.ADJUSTMENTS['returning_injury']
                reasons.append("Returning from injury")

        # Check teammate injuries (look for WR1/TE1 out)
        for name, injury in self._injuries.items():
            if injury['status'] == 'out' and name != player_name:
                # This would need roster data to know team
                # For now, skip this check
                pass

        # Check weather
        if game_key and game_key in self._weather:
            weather = self._weather[game_key]
            if weather['conditions'] in ['rain', 'snow'] or weather['wind_mph'] > 20:
                factor *= self.ADJUSTMENTS['bad_weather']
                reasons.append(f"Weather: {weather['conditions']}, {weather['wind_mph']}mph wind")

        return factor, reasons

    def apply_context_to_predictions(
        self,
        predictions_df: pd.DataFrame,
        game_key: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Apply situational adjustments to a predictions dataframe.

        Args:
            predictions_df: DataFrame with predictions
            game_key: Optional game key for weather lookup

        Returns:
            DataFrame with adjusted predictions and adjustment details
        """
        if len(predictions_df) == 0:
            return predictions_df

        df = predictions_df.copy()

        # Apply adjustments per player
        adjustments = []
        reasons_list = []

        for _, row in df.iterrows():
            player = row.get('player_name', '')
            team = row.get('recent_team', '')
            opponent = row.get('opponent_team', '')

            factor, reasons = self.get_adjustment_factor(
                player_name=player,
                team=team,
                opponent=opponent,
                game_key=game_key
            )

            adjustments.append(factor)
            reasons_list.append('; '.join(reasons) if reasons else 'None')

        df['situation_adjustment'] = adjustments
        df['situation_reasons'] = reasons_list

        # Apply adjustment to prediction
        pred_col = 'predicted_yards' if 'predicted_yards' in df.columns else 'predicted_receiving_yards'
        if pred_col in df.columns:
            df['adjusted_prediction'] = df[pred_col] * df['situation_adjustment']

        return df

    def get_context_summary(self) -> Dict:
        """Get summary of current situational context."""
        return {
            'qb_changes': len(self._qb_changes),
            'coaching_changes': len(self._coaching_changes),
            'injuries_tracked': len(self._injuries),
            'weather_recorded': len(self._weather),
            'details': {
                'qb_changes': dict(self._qb_changes),
                'coaching_changes': dict(self._coaching_changes),
                'injuries': dict(self._injuries),
            }
        }


# Singleton instance for easy access
_context = SituationalContext()


def get_context() -> SituationalContext:
    """Get the global situational context instance."""
    return _context


def set_qb_change(team: str, new_qb: str, reason: str = 'injury', games_started: int = 0):
    """Convenience function to set QB change."""
    _context.set_qb_change(team, new_qb, reason, games_started)


def set_injury(player_name: str, status: str, position: str = 'WR'):
    """Convenience function to set player injury."""
    _context.set_player_injury(player_name, status, position)
