"""Historical performance tracking service.

Aggregates weekly betting results into WeeklySummary records for
long-term performance analysis and trend tracking.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from ..database import get_session, Game, PaperTrade, Player, WeeklySummary

logger = logging.getLogger(__name__)


@dataclass
class WeeklyStats:
    """Weekly statistics computed from paper trades."""

    season: int
    week: int
    total_bets: int = 0
    wins: int = 0
    losses: int = 0
    pushes: int = 0
    total_staked: float = 0.0
    total_profit: float = 0.0
    over_wins: int = 0
    over_losses: int = 0
    under_wins: int = 0
    under_losses: int = 0
    market_breakdown: Dict = None
    position_breakdown: Dict = None
    avg_edge_pct: float = 0.0
    avg_confidence: float = 0.0
    best_bet: Tuple[str, float] = None  # (player, profit)
    worst_bet: Tuple[str, float] = None  # (player, loss)

    @property
    def win_rate(self) -> float:
        """Calculate win rate percentage."""
        total = self.wins + self.losses
        return (self.wins / total * 100) if total > 0 else 0.0

    @property
    def roi(self) -> float:
        """Calculate ROI percentage."""
        return (self.total_profit / self.total_staked * 100) if self.total_staked > 0 else 0.0


class HistoricalTracker:
    """Service for tracking and analyzing historical betting performance."""

    def __init__(self):
        self._cache: Dict[Tuple[int, int], WeeklySummary] = {}

    def compute_weekly_stats(self, season: int, week: int) -> Optional[WeeklyStats]:
        """
        Compute statistics from paper trades for a specific week.

        Args:
            season: Season year
            week: Week number

        Returns:
            WeeklyStats with computed metrics, or None if no trades found
        """
        with get_session() as session:
            # Get all evaluated trades for this week
            trades = (
                session.query(PaperTrade)
                .join(Game)
                .filter(
                    Game.season == season,
                    Game.week == week,
                    PaperTrade.evaluated_at.isnot(None),
                )
                .all()
            )

            if not trades:
                logger.info(f"No evaluated trades found for {season} Week {week}")
                return None

            stats = WeeklyStats(season=season, week=week)
            stats.market_breakdown = {}
            stats.position_breakdown = {}

            edge_sum = 0.0
            conf_sum = 0.0
            best_profit = float('-inf')
            worst_loss = float('inf')

            for trade in trades:
                stats.total_bets += 1
                stats.total_staked += trade.stake or 0

                # Get player info for position breakdown
                player = session.query(Player).get(trade.player_id)
                position = player.position if player else 'Unknown'
                market = trade.market or 'unknown'

                # Initialize breakdown dicts if needed
                if market not in stats.market_breakdown:
                    stats.market_breakdown[market] = {
                        'wins': 0, 'losses': 0, 'profit': 0.0, 'staked': 0.0
                    }
                if position not in stats.position_breakdown:
                    stats.position_breakdown[position] = {
                        'wins': 0, 'losses': 0, 'profit': 0.0, 'staked': 0.0
                    }

                # Track edge and confidence
                if trade.edge_percentage:
                    edge_sum += trade.edge_percentage
                if trade.model_confidence:
                    conf_sum += trade.model_confidence

                # Count wins/losses
                if trade.won is True:
                    stats.wins += 1
                    stats.market_breakdown[market]['wins'] += 1
                    stats.position_breakdown[position]['wins'] += 1
                    if trade.bet_side == 'over':
                        stats.over_wins += 1
                    else:
                        stats.under_wins += 1
                elif trade.won is False:
                    stats.losses += 1
                    stats.market_breakdown[market]['losses'] += 1
                    stats.position_breakdown[position]['losses'] += 1
                    if trade.bet_side == 'over':
                        stats.over_losses += 1
                    else:
                        stats.under_losses += 1
                else:
                    stats.pushes += 1

                # Track profit/loss
                profit = trade.profit_loss or 0
                stats.total_profit += profit
                stats.market_breakdown[market]['profit'] += profit
                stats.market_breakdown[market]['staked'] += trade.stake or 0
                stats.position_breakdown[position]['profit'] += profit
                stats.position_breakdown[position]['staked'] += trade.stake or 0

                # Track best/worst bets
                player_name = player.name if player else 'Unknown'
                if profit > best_profit:
                    best_profit = profit
                    stats.best_bet = (player_name, profit)
                if profit < worst_loss:
                    worst_loss = profit
                    stats.worst_bet = (player_name, profit)

            # Calculate averages
            if stats.total_bets > 0:
                stats.avg_edge_pct = edge_sum / stats.total_bets
                stats.avg_confidence = conf_sum / stats.total_bets

            # Calculate ROI for breakdowns
            for market_data in stats.market_breakdown.values():
                if market_data['staked'] > 0:
                    market_data['roi'] = market_data['profit'] / market_data['staked'] * 100
                else:
                    market_data['roi'] = 0.0

            for pos_data in stats.position_breakdown.values():
                if pos_data['staked'] > 0:
                    pos_data['roi'] = pos_data['profit'] / pos_data['staked'] * 100
                else:
                    pos_data['roi'] = 0.0

            logger.info(
                f"Computed stats for {season} Week {week}: "
                f"{stats.wins}-{stats.losses} ({stats.win_rate:.1f}%), "
                f"ROI: {stats.roi:.1f}%"
            )

            return stats

    def save_weekly_summary(self, stats: WeeklyStats) -> Dict:
        """
        Save computed weekly stats to the database.

        Args:
            stats: Computed WeeklyStats

        Returns:
            Dict with summary data (to avoid SQLAlchemy detached instance issues)
        """
        with get_session() as session:
            # Check for existing summary
            existing = session.query(WeeklySummary).filter_by(
                season=stats.season,
                week=stats.week
            ).first()

            # Compute season cumulative totals
            season_stats = self._get_season_cumulative(session, stats.season, stats.week)

            if existing:
                # Update existing record
                summary = existing
                logger.info(f"Updating existing summary for {stats.season} Week {stats.week}")
            else:
                # Create new record
                summary = WeeklySummary(season=stats.season, week=stats.week)
                session.add(summary)
                logger.info(f"Creating new summary for {stats.season} Week {stats.week}")

            # Set all fields
            summary.total_bets = stats.total_bets
            summary.wins = stats.wins
            summary.losses = stats.losses
            summary.pushes = stats.pushes
            summary.total_staked = stats.total_staked
            summary.total_profit = stats.total_profit
            summary.roi_pct = stats.roi
            summary.win_rate_pct = stats.win_rate
            summary.over_wins = stats.over_wins
            summary.over_losses = stats.over_losses
            summary.under_wins = stats.under_wins
            summary.under_losses = stats.under_losses
            summary.market_breakdown = stats.market_breakdown
            summary.position_breakdown = stats.position_breakdown
            summary.avg_edge_pct = stats.avg_edge_pct
            summary.avg_confidence = stats.avg_confidence

            # Best/worst bets
            if stats.best_bet:
                summary.best_bet_player = stats.best_bet[0]
                summary.best_bet_profit = stats.best_bet[1]
            if stats.worst_bet:
                summary.worst_bet_player = stats.worst_bet[0]
                summary.worst_bet_loss = stats.worst_bet[1]

            # Season cumulative
            summary.season_total_bets = season_stats['total_bets'] + stats.total_bets
            summary.season_total_profit = season_stats['total_profit'] + stats.total_profit
            if season_stats['total_staked'] + stats.total_staked > 0:
                summary.season_roi_pct = (
                    (season_stats['total_profit'] + stats.total_profit) /
                    (season_stats['total_staked'] + stats.total_staked) * 100
                )

            summary.updated_at = datetime.utcnow()

            session.commit()

            # Capture data before session closes (to avoid detached instance issues)
            result_data = {
                'id': summary.id,
                'season': summary.season,
                'week': summary.week,
                'wins': summary.wins,
                'losses': summary.losses,
                'total_profit': summary.total_profit,
                'win_rate_pct': summary.win_rate_pct,
                'roi_pct': summary.roi_pct,
            }

            return result_data

    def _get_season_cumulative(
        self, session, season: int, up_to_week: int
    ) -> Dict[str, float]:
        """Get cumulative stats for all weeks before the specified week."""
        summaries = session.query(WeeklySummary).filter(
            WeeklySummary.season == season,
            WeeklySummary.week < up_to_week
        ).all()

        return {
            'total_bets': sum(s.total_bets or 0 for s in summaries),
            'total_profit': sum(s.total_profit or 0 for s in summaries),
            'total_staked': sum(s.total_staked or 0 for s in summaries),
        }

    def generate_and_save_summary(self, season: int, week: int) -> Optional[Dict]:
        """
        Compute stats and save summary in one operation.

        Args:
            season: Season year
            week: Week number

        Returns:
            Dict with summary data if trades exist, None otherwise
        """
        stats = self.compute_weekly_stats(season, week)
        if stats is None:
            return None

        return self.save_weekly_summary(stats)

    def _summary_to_dict(self, summary: WeeklySummary) -> Dict:
        """Convert WeeklySummary ORM object to dict (avoids detached instance issues)."""
        return {
            'id': summary.id,
            'season': summary.season,
            'week': summary.week,
            'total_bets': summary.total_bets,
            'wins': summary.wins,
            'losses': summary.losses,
            'pushes': summary.pushes,
            'total_staked': summary.total_staked,
            'total_profit': summary.total_profit,
            'roi_pct': summary.roi_pct,
            'win_rate_pct': summary.win_rate_pct,
            'over_wins': summary.over_wins,
            'over_losses': summary.over_losses,
            'under_wins': summary.under_wins,
            'under_losses': summary.under_losses,
            'market_breakdown': summary.market_breakdown,
            'position_breakdown': summary.position_breakdown,
            'avg_edge_pct': summary.avg_edge_pct,
            'avg_confidence': summary.avg_confidence,
            'best_bet_player': summary.best_bet_player,
            'best_bet_profit': summary.best_bet_profit,
            'worst_bet_player': summary.worst_bet_player,
            'worst_bet_loss': summary.worst_bet_loss,
            'season_total_bets': summary.season_total_bets,
            'season_total_profit': summary.season_total_profit,
            'season_roi_pct': summary.season_roi_pct,
        }

    def get_summary(self, season: int, week: int) -> Optional[Dict]:
        """Get stored weekly summary as dict."""
        with get_session() as session:
            summary = session.query(WeeklySummary).filter_by(
                season=season, week=week
            ).first()
            return self._summary_to_dict(summary) if summary else None

    def get_season_summaries(self, season: int) -> List[Dict]:
        """Get all weekly summaries for a season as list of dicts."""
        with get_session() as session:
            summaries = session.query(WeeklySummary).filter_by(
                season=season
            ).order_by(WeeklySummary.week).all()
            return [self._summary_to_dict(s) for s in summaries]

    def get_all_summaries(self) -> List[Dict]:
        """Get all weekly summaries across all seasons as list of dicts."""
        with get_session() as session:
            summaries = session.query(WeeklySummary).order_by(
                WeeklySummary.season.desc(),
                WeeklySummary.week.desc()
            ).all()
            return [self._summary_to_dict(s) for s in summaries]

    def format_history_report(
        self,
        summaries: List[Dict],
        include_breakdowns: bool = False
    ) -> str:
        """
        Format a list of weekly summaries into a readable report.

        Args:
            summaries: List of summary dicts
            include_breakdowns: Include market/position breakdowns

        Returns:
            Formatted report string
        """
        if not summaries:
            return "No historical data found."

        lines = []
        lines.append("=" * 70)
        lines.append("HISTORICAL PERFORMANCE")
        lines.append("=" * 70)
        lines.append("")

        # Summary table header
        lines.append(f"{'Week':<8} {'Record':<12} {'Win%':>8} {'Profit':>12} {'ROI':>8} {'Season P/L':>12}")
        lines.append("-" * 70)

        for s in summaries:
            record = f"{s['wins']}-{s['losses']}"
            if s.get('pushes'):
                record += f"-{s['pushes']}"

            lines.append(
                f"Wk {s['week']:<4} {record:<12} {s['win_rate_pct']:>7.1f}% "
                f"${s['total_profit']:>+10.2f} {s['roi_pct']:>+7.1f}% "
                f"${s.get('season_total_profit') or 0:>+10.2f}"
            )

        lines.append("-" * 70)

        # Calculate totals
        total_bets = sum(s['total_bets'] for s in summaries)
        total_wins = sum(s['wins'] for s in summaries)
        total_losses = sum(s['losses'] for s in summaries)
        total_profit = sum(s['total_profit'] for s in summaries)
        total_staked = sum(s['total_staked'] for s in summaries)
        overall_roi = (total_profit / total_staked * 100) if total_staked > 0 else 0
        overall_win_rate = (total_wins / (total_wins + total_losses) * 100) if (total_wins + total_losses) > 0 else 0

        lines.append(
            f"{'TOTAL':<8} {f'{total_wins}-{total_losses}':<12} {overall_win_rate:>7.1f}% "
            f"${total_profit:>+10.2f} {overall_roi:>+7.1f}%"
        )
        lines.append("")

        # Over/Under breakdown
        over_wins = sum(s['over_wins'] for s in summaries)
        over_losses = sum(s['over_losses'] for s in summaries)
        under_wins = sum(s['under_wins'] for s in summaries)
        under_losses = sum(s['under_losses'] for s in summaries)

        lines.append("BY DIRECTION:")
        if over_wins + over_losses > 0:
            over_pct = over_wins / (over_wins + over_losses) * 100
            lines.append(f"  OVER:  {over_wins}-{over_losses} ({over_pct:.1f}%)")
        if under_wins + under_losses > 0:
            under_pct = under_wins / (under_wins + under_losses) * 100
            lines.append(f"  UNDER: {under_wins}-{under_losses} ({under_pct:.1f}%)")

        lines.append("")

        # Market breakdown (aggregated across all weeks)
        if include_breakdowns:
            market_totals = {}
            for s in summaries:
                if s.get('market_breakdown'):
                    for market, data in s['market_breakdown'].items():
                        if market not in market_totals:
                            market_totals[market] = {'wins': 0, 'losses': 0, 'profit': 0, 'staked': 0}
                        market_totals[market]['wins'] += data.get('wins', 0)
                        market_totals[market]['losses'] += data.get('losses', 0)
                        market_totals[market]['profit'] += data.get('profit', 0)
                        market_totals[market]['staked'] += data.get('staked', 0)

            if market_totals:
                lines.append("BY MARKET:")
                for market, data in sorted(market_totals.items()):
                    total = data['wins'] + data['losses']
                    win_pct = data['wins'] / total * 100 if total > 0 else 0
                    roi = data['profit'] / data['staked'] * 100 if data['staked'] > 0 else 0
                    market_name = market.replace('player_', '').replace('_', ' ').title()
                    lines.append(
                        f"  {market_name:<20} {data['wins']}-{data['losses']} "
                        f"({win_pct:.1f}%) ${data['profit']:+.2f} ({roi:+.1f}% ROI)"
                    )
                lines.append("")

        return "\n".join(lines)

    def get_performance_trends(self, season: int) -> Dict:
        """
        Analyze performance trends over a season.

        Returns dict with trend analysis.
        """
        summaries = self.get_season_summaries(season)
        if not summaries:
            return {}

        # Calculate rolling averages and trends
        profits = [s['total_profit'] for s in summaries]
        win_rates = [s['win_rate_pct'] for s in summaries]

        # Simple trend: compare first half to second half
        mid = len(summaries) // 2
        if mid > 0:
            first_half_profit = sum(profits[:mid])
            second_half_profit = sum(profits[mid:])
            first_half_wr = sum(win_rates[:mid]) / mid
            second_half_wr = sum(win_rates[mid:]) / (len(summaries) - mid)
        else:
            first_half_profit = second_half_profit = 0
            first_half_wr = second_half_wr = 0

        return {
            'season': season,
            'weeks_tracked': len(summaries),
            'total_profit': sum(profits),
            'avg_weekly_profit': sum(profits) / len(profits) if profits else 0,
            'best_week': max(summaries, key=lambda s: s['total_profit']),
            'worst_week': min(summaries, key=lambda s: s['total_profit']),
            'profit_trend': 'improving' if second_half_profit > first_half_profit else 'declining',
            'win_rate_trend': 'improving' if second_half_wr > first_half_wr else 'declining',
        }


# Singleton instance
_tracker: Optional[HistoricalTracker] = None


def get_historical_tracker() -> HistoricalTracker:
    """Get the singleton historical tracker instance."""
    global _tracker
    if _tracker is None:
        _tracker = HistoricalTracker()
    return _tracker
