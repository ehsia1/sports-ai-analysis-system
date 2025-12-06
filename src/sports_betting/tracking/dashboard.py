"""Result tracking dashboard for analyzing betting performance."""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from ..database import get_session
from ..database.models import PaperTrade, Player, Game

logger = logging.getLogger(__name__)


@dataclass
class BreakdownStats:
    """Statistics for a single breakdown category."""

    category: str
    total_bets: int = 0
    wins: int = 0
    losses: int = 0
    pushes: int = 0
    total_staked: float = 0.0
    total_profit: float = 0.0

    @property
    def win_rate(self) -> float:
        """Win rate as percentage."""
        decided = self.wins + self.losses
        return (self.wins / decided * 100) if decided > 0 else 0.0

    @property
    def roi(self) -> float:
        """Return on investment as percentage."""
        return (self.total_profit / self.total_staked * 100) if self.total_staked > 0 else 0.0


@dataclass
class DashboardData:
    """Aggregated dashboard data."""

    season: int
    week: Optional[int] = None  # None = all weeks

    # Overall stats
    total_bets: int = 0
    wins: int = 0
    losses: int = 0
    pushes: int = 0
    total_staked: float = 0.0
    total_profit: float = 0.0

    # Breakdowns
    by_market: Dict[str, BreakdownStats] = field(default_factory=dict)
    by_direction: Dict[str, BreakdownStats] = field(default_factory=dict)
    by_position: Dict[str, BreakdownStats] = field(default_factory=dict)
    by_tier: Dict[str, BreakdownStats] = field(default_factory=dict)
    by_edge_bucket: Dict[str, BreakdownStats] = field(default_factory=dict)
    by_confidence_bucket: Dict[str, BreakdownStats] = field(default_factory=dict)
    by_week: Dict[int, BreakdownStats] = field(default_factory=dict)

    @property
    def win_rate(self) -> float:
        """Overall win rate."""
        decided = self.wins + self.losses
        return (self.wins / decided * 100) if decided > 0 else 0.0

    @property
    def roi(self) -> float:
        """Overall ROI."""
        return (self.total_profit / self.total_staked * 100) if self.total_staked > 0 else 0.0


class ResultDashboard:
    """Dashboard for analyzing betting results across multiple dimensions."""

    # Elite player lists (mirrors edge_calculator.py)
    ELITE_PLAYERS = {
        'WR': {
            "Ja'Marr Chase", "CeeDee Lamb", "Justin Jefferson", "Tyreek Hill",
            "A.J. Brown", "Amon-Ra St. Brown", "Davante Adams", "Nico Collins",
            "Malik Nabers", "Puka Nacua", "Garrett Wilson", "Chris Olave",
        },
        'RB': {
            "Derrick Henry", "Saquon Barkley", "Josh Jacobs", "Breece Hall",
            "Bijan Robinson", "De'Von Achane", "Jahmyr Gibbs", "Jonathan Taylor",
        },
        'TE': {
            "Travis Kelce", "George Kittle", "Mark Andrews", "T.J. Hockenson",
            "Sam LaPorta", "Trey McBride", "Brock Bowers",
        },
        'QB': {
            "Patrick Mahomes", "Josh Allen", "Lamar Jackson", "Jalen Hurts",
            "Joe Burrow", "Jared Goff", "Jayden Daniels",
        },
    }

    # Market display names
    MARKET_NAMES = {
        'player_reception_yds': 'Receiving Yards',
        'player_rush_yds': 'Rushing Yards',
        'player_pass_yds': 'Passing Yards',
        'player_receptions': 'Receptions',
    }

    def __init__(self):
        self._data: Optional[DashboardData] = None

    def get_player_tier(self, player_name: str, position: str) -> str:
        """Determine player tier based on name and position."""
        elite_set = self.ELITE_PLAYERS.get(position, set())
        if player_name in elite_set:
            return 'Elite'
        # Could expand with starter/backup logic based on snap %
        return 'Other'

    def get_edge_bucket(self, edge_pct: float) -> str:
        """Categorize edge percentage into buckets."""
        edge = abs(edge_pct)
        if edge >= 20:
            return '20%+'
        elif edge >= 15:
            return '15-20%'
        elif edge >= 10:
            return '10-15%'
        elif edge >= 5:
            return '5-10%'
        else:
            return '<5%'

    def get_confidence_bucket(self, confidence: float) -> str:
        """Categorize model confidence into buckets."""
        if confidence >= 0.8:
            return 'High (80%+)'
        elif confidence >= 0.6:
            return 'Medium (60-80%)'
        else:
            return 'Low (<60%)'

    def _add_to_breakdown(
        self,
        breakdown: Dict[str, BreakdownStats],
        key: str,
        trade: PaperTrade
    ):
        """Add a trade's stats to a breakdown dictionary."""
        if key not in breakdown:
            breakdown[key] = BreakdownStats(category=key)

        stats = breakdown[key]
        stats.total_bets += 1
        stats.total_staked += trade.stake

        if trade.won is True:
            stats.wins += 1
        elif trade.won is False:
            stats.losses += 1
        else:
            stats.pushes += 1

        if trade.profit_loss is not None:
            stats.total_profit += trade.profit_loss

    def load_data(
        self,
        season: int,
        week: Optional[int] = None,
        evaluated_only: bool = True
    ) -> DashboardData:
        """
        Load and aggregate dashboard data from the database.

        Args:
            season: Season year
            week: Specific week (None = all weeks)
            evaluated_only: Only include trades with results

        Returns:
            DashboardData with all breakdowns populated
        """
        data = DashboardData(season=season, week=week)

        with get_session() as session:
            # Build query
            query = session.query(PaperTrade).join(Player).join(Game)
            query = query.filter(Game.season == season)

            if week is not None:
                query = query.filter(Game.week == week)

            if evaluated_only:
                query = query.filter(PaperTrade.won.isnot(None))

            trades = query.all()

            for trade in trades:
                player = trade.player
                game = trade.game

                # Overall stats
                data.total_bets += 1
                data.total_staked += trade.stake

                if trade.won is True:
                    data.wins += 1
                elif trade.won is False:
                    data.losses += 1
                else:
                    data.pushes += 1

                if trade.profit_loss is not None:
                    data.total_profit += trade.profit_loss

                # By market
                market_name = self.MARKET_NAMES.get(trade.market, trade.market)
                self._add_to_breakdown(data.by_market, market_name, trade)

                # By direction (normalize case)
                direction = trade.bet_side.upper()
                self._add_to_breakdown(data.by_direction, direction, trade)

                # By position
                position = player.position
                self._add_to_breakdown(data.by_position, position, trade)

                # By tier
                tier = self.get_player_tier(player.name, player.position)
                self._add_to_breakdown(data.by_tier, tier, trade)

                # By edge bucket
                edge_bucket = self.get_edge_bucket(trade.edge_percentage)
                self._add_to_breakdown(data.by_edge_bucket, edge_bucket, trade)

                # By confidence bucket
                conf_bucket = self.get_confidence_bucket(trade.model_confidence)
                self._add_to_breakdown(data.by_confidence_bucket, conf_bucket, trade)

                # By week
                week_num = game.week
                self._add_to_breakdown(data.by_week, str(week_num), trade)

        self._data = data
        logger.info(f"Loaded {data.total_bets} trades for dashboard")
        return data

    def format_breakdown_table(
        self,
        breakdown: Dict[str, BreakdownStats],
        title: str,
        sort_by: str = 'profit'
    ) -> str:
        """Format a breakdown as a table string."""
        if not breakdown:
            return f"{title}\n  No data\n"

        lines = [title]
        lines.append("-" * 70)
        lines.append(f"{'Category':<20} {'Record':<12} {'Win%':>8} {'Profit':>12} {'ROI':>8}")
        lines.append("-" * 70)

        # Sort by profit or win rate
        items = list(breakdown.values())
        if sort_by == 'profit':
            items.sort(key=lambda x: x.total_profit, reverse=True)
        elif sort_by == 'win_rate':
            items.sort(key=lambda x: x.win_rate, reverse=True)
        elif sort_by == 'bets':
            items.sort(key=lambda x: x.total_bets, reverse=True)

        for stats in items:
            record = f"{stats.wins}-{stats.losses}"
            if stats.pushes:
                record += f"-{stats.pushes}"

            lines.append(
                f"{stats.category:<20} {record:<12} {stats.win_rate:>7.1f}% "
                f"${stats.total_profit:>+10.2f} {stats.roi:>+7.1f}%"
            )

        lines.append("")
        return "\n".join(lines)

    def format_full_report(self, data: Optional[DashboardData] = None) -> str:
        """Generate a full dashboard report."""
        if data is None:
            data = self._data

        if data is None:
            return "No data loaded. Call load_data() first."

        lines = []

        # Header
        week_str = f"Week {data.week}" if data.week else "All Weeks"
        lines.append("=" * 70)
        lines.append(f"RESULT TRACKING DASHBOARD - {data.season} {week_str}")
        lines.append("=" * 70)
        lines.append("")

        # Overall summary
        lines.append("OVERALL PERFORMANCE")
        lines.append("-" * 70)
        record = f"{data.wins}-{data.losses}"
        if data.pushes:
            record += f"-{data.pushes}"
        lines.append(f"Record: {record} ({data.win_rate:.1f}%)")
        lines.append(f"Total Staked: ${data.total_staked:,.2f}")
        lines.append(f"Total Profit: ${data.total_profit:+,.2f}")
        lines.append(f"ROI: {data.roi:+.1f}%")
        lines.append("")

        # By direction
        lines.append(self.format_breakdown_table(
            data.by_direction, "BY DIRECTION (OVER/UNDER)", sort_by='bets'
        ))

        # By market
        lines.append(self.format_breakdown_table(
            data.by_market, "BY MARKET", sort_by='bets'
        ))

        # By position
        lines.append(self.format_breakdown_table(
            data.by_position, "BY POSITION", sort_by='bets'
        ))

        # By tier
        lines.append(self.format_breakdown_table(
            data.by_tier, "BY PLAYER TIER", sort_by='bets'
        ))

        # By edge bucket
        lines.append(self.format_breakdown_table(
            data.by_edge_bucket, "BY EDGE %", sort_by='bets'
        ))

        # By confidence bucket
        # Note: This is data quality confidence (how much historical data available),
        # NOT prediction accuracy confidence
        lines.append(self.format_breakdown_table(
            data.by_confidence_bucket, "BY MODEL CONFIDENCE (Data Quality)", sort_by='bets'
        ))

        # By week (if showing all weeks)
        if data.week is None and data.by_week:
            lines.append(self.format_breakdown_table(
                data.by_week, "BY WEEK", sort_by='bets'
            ))

        # Insights section
        lines.append("=" * 70)
        lines.append("KEY INSIGHTS")
        lines.append("=" * 70)
        lines.append("")

        insights = self._generate_insights(data)
        for insight in insights:
            lines.append(f"• {insight}")

        lines.append("")
        return "\n".join(lines)

    def _generate_insights(self, data: DashboardData) -> List[str]:
        """Generate actionable insights from the data."""
        insights = []

        # Direction insight
        if data.by_direction:
            over = data.by_direction.get('OVER')
            under = data.by_direction.get('UNDER')

            if over and under:
                if over.win_rate > under.win_rate + 10:
                    insights.append(
                        f"OVERs outperforming UNDERs: {over.win_rate:.0f}% vs {under.win_rate:.0f}% win rate"
                    )
                elif under.win_rate > over.win_rate + 10:
                    insights.append(
                        f"UNDERs outperforming OVERs: {under.win_rate:.0f}% vs {over.win_rate:.0f}% win rate"
                    )

        # Position insight
        if data.by_position:
            best_pos = max(data.by_position.values(), key=lambda x: x.win_rate if x.total_bets >= 3 else 0)
            worst_pos = min(data.by_position.values(), key=lambda x: x.win_rate if x.total_bets >= 3 else 100)

            if best_pos.total_bets >= 3 and best_pos.win_rate > 50:
                insights.append(
                    f"Best position: {best_pos.category} ({best_pos.wins}-{best_pos.losses}, {best_pos.win_rate:.0f}%)"
                )
            if worst_pos.total_bets >= 3 and worst_pos.win_rate < 40:
                insights.append(
                    f"Struggling position: {worst_pos.category} ({worst_pos.wins}-{worst_pos.losses}, {worst_pos.win_rate:.0f}%)"
                )

        # Tier insight
        if data.by_tier:
            elite = data.by_tier.get('Elite')
            other = data.by_tier.get('Other')

            if elite and other and elite.total_bets >= 3 and other.total_bets >= 3:
                if elite.win_rate > other.win_rate + 5:
                    insights.append(
                        f"Elite players more reliable: {elite.win_rate:.0f}% vs {other.win_rate:.0f}%"
                    )
                elif other.win_rate > elite.win_rate + 5:
                    insights.append(
                        f"Non-elite players outperforming: {other.win_rate:.0f}% vs {elite.win_rate:.0f}%"
                    )

        # Edge bucket insight
        if data.by_edge_bucket:
            high_edge_buckets = ['15-20%', '20%+']
            high_edge_stats = BreakdownStats(category='High Edge')

            for bucket in high_edge_buckets:
                if bucket in data.by_edge_bucket:
                    b = data.by_edge_bucket[bucket]
                    high_edge_stats.wins += b.wins
                    high_edge_stats.losses += b.losses
                    high_edge_stats.total_bets += b.total_bets

            if high_edge_stats.total_bets >= 3:
                if high_edge_stats.win_rate > 50:
                    insights.append(
                        f"High edge (15%+) bets hitting: {high_edge_stats.win_rate:.0f}% win rate"
                    )
                elif high_edge_stats.win_rate < 40:
                    insights.append(
                        f"High edge (15%+) bets underperforming: {high_edge_stats.win_rate:.0f}% win rate"
                    )

        # Confidence insight
        # Note: model_confidence is based on DATA QUALITY (historical data availability),
        # NOT prediction accuracy. It indicates how much data the model had to work with.
        if data.by_confidence_bucket:
            high_conf = data.by_confidence_bucket.get('High (80%+)')
            low_conf = data.by_confidence_bucket.get('Low (<60%)')

            if high_conf and high_conf.total_bets >= 3:
                if high_conf.win_rate > 50:
                    insights.append(
                        f"High data-quality bets performing well: {high_conf.win_rate:.0f}% win rate"
                    )
                elif high_conf.win_rate < 40:
                    # Check if low confidence is outperforming - indicates tighter lines on known players
                    low_wr = low_conf.win_rate if low_conf and low_conf.total_bets >= 3 else 0
                    if low_wr > high_conf.win_rate + 10:
                        insights.append(
                            f"⚠️ High data-quality bets ({high_conf.win_rate:.0f}%) beaten by low ({low_wr:.0f}%) - "
                            f"well-known players may have tighter lines"
                        )
                    else:
                        insights.append(
                            f"⚠️ High data-quality bets underperforming: {high_conf.win_rate:.0f}%"
                        )

        if not insights:
            insights.append("Insufficient data for meaningful insights yet.")

        # Add backtest context when paper trade results differ significantly from historical
        if data.by_direction:
            over = data.by_direction.get('OVER')
            under = data.by_direction.get('UNDER')
            # Historical backtest shows UNDERs hit 73%, OVERs hit 54%
            # If current results show opposite, flag it
            if under and under.total_bets >= 5 and under.win_rate < 40:
                insights.append(
                    "📊 Note: Historical backtest (2600 bets) shows UNDERs at 73% win rate. "
                    "Current underperformance may be sample variance."
                )
            elif over and over.total_bets >= 5 and over.win_rate > 60:
                insights.append(
                    "📊 Note: Historical backtest (2600 bets) shows OVERs at 54% win rate. "
                    "Current overperformance may regress to mean."
                )

        return insights

    def get_best_filters(self, data: Optional[DashboardData] = None) -> Dict[str, str]:
        """Identify the best performing filters based on historical data."""
        if data is None:
            data = self._data

        if data is None:
            return {}

        recommendations = {}

        # Best direction
        if data.by_direction:
            best_dir = max(
                data.by_direction.values(),
                key=lambda x: x.win_rate if x.total_bets >= 5 else 0
            )
            if best_dir.total_bets >= 5 and best_dir.win_rate > 50:
                recommendations['direction'] = f"Focus on {best_dir.category} bets ({best_dir.win_rate:.0f}% win rate)"

        # Best position
        if data.by_position:
            best_pos = max(
                data.by_position.values(),
                key=lambda x: x.win_rate if x.total_bets >= 5 else 0
            )
            if best_pos.total_bets >= 5 and best_pos.win_rate > 50:
                recommendations['position'] = f"Best position: {best_pos.category} ({best_pos.win_rate:.0f}%)"

        return recommendations


# Singleton instance
_dashboard: Optional[ResultDashboard] = None


def get_dashboard() -> ResultDashboard:
    """Get or create the dashboard singleton."""
    global _dashboard
    if _dashboard is None:
        _dashboard = ResultDashboard()
    return _dashboard
