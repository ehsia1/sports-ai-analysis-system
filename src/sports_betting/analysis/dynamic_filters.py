"""Dynamic filter service that auto-adjusts based on historical win rates."""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from ..database import get_session
from ..database.models import PaperTrade, Player, Game

logger = logging.getLogger(__name__)


@dataclass
class FilterRule:
    """A single filter rule based on historical performance."""

    dimension: str  # e.g., 'position_direction', 'market', 'tier_direction'
    category: str   # e.g., 'TE_OVER', 'player_reception_yds', 'elite_UNDER'
    action: str     # 'skip' or 'prioritize'
    win_rate: float
    sample_size: int
    total_profit: float
    reason: str

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'FilterRule':
        return cls(**data)


@dataclass
class DynamicFilterConfig:
    """Configuration for dynamic filter generation."""

    # Minimum sample size before applying a filter
    min_sample_size: int = 15

    # Win rate thresholds
    skip_below_win_rate: float = 0.35  # Skip categories with win rate below this
    prioritize_above_win_rate: float = 0.55  # Prioritize categories above this

    # Profit thresholds (per $100 unit)
    skip_below_profit: float = -300.0  # Skip if lost more than $300

    # Which dimensions to analyze
    dimensions: List[str] = field(default_factory=lambda: [
        'direction',           # OVER vs UNDER
        'position',            # WR, RB, TE, QB
        'market',              # player_reception_yds, etc.
        'position_direction',  # WR_OVER, RB_UNDER, etc.
        'tier_direction',      # elite_OVER, other_UNDER, etc.
        'market_direction',    # player_reception_yds_OVER, etc.
    ])


@dataclass
class CategoryStats:
    """Statistics for a single category."""

    wins: int = 0
    losses: int = 0
    pending: int = 0
    total_profit: float = 0.0

    @property
    def total(self) -> int:
        return self.wins + self.losses

    @property
    def win_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.wins / self.total

    @property
    def avg_profit_per_bet(self) -> float:
        if self.total == 0:
            return 0.0
        return self.total_profit / self.total


class DynamicFilterService:
    """Service for generating and managing dynamic bet filters."""

    # Elite player lists (same as dashboard for consistency)
    ELITE_PLAYERS = {
        'WR': {
            "Ja'Marr Chase", "CeeDee Lamb", "Justin Jefferson", "Tyreek Hill",
            "A.J. Brown", "Amon-Ra St. Brown", "Davante Adams", "Stefon Diggs",
            "DK Metcalf", "Chris Olave", "Garrett Wilson", "Nico Collins",
            "Puka Nacua", "Mike Evans", "Deebo Samuel", "Jaylen Waddle",
        },
        'RB': {
            "Derrick Henry", "Saquon Barkley", "Breece Hall", "Bijan Robinson",
            "Jonathan Taylor", "Josh Jacobs", "De'Von Achane", "Jahmyr Gibbs",
            "Kyren Williams", "Joe Mixon", "Alvin Kamara", "James Cook",
        },
        'TE': {
            "Travis Kelce", "George Kittle", "Mark Andrews", "Sam LaPorta",
            "Trey McBride", "T.J. Hockenson", "David Njoku", "Dalton Kincaid",
        },
        'QB': {
            "Patrick Mahomes", "Josh Allen", "Lamar Jackson", "Jalen Hurts",
            "Joe Burrow", "Dak Prescott", "Tua Tagovailoa", "C.J. Stroud",
        },
    }

    def __init__(self, config: Optional[DynamicFilterConfig] = None):
        self.config = config or DynamicFilterConfig()
        self._rules: List[FilterRule] = []
        self._last_update: Optional[datetime] = None
        self._cache_path = Path.home() / ".sports_betting" / "dynamic_filters.json"

    def _is_elite(self, player_name: str, position: str) -> bool:
        """Check if player is in elite tier."""
        return player_name in self.ELITE_PLAYERS.get(position, set())

    def _get_tier(self, player_name: str, position: str) -> str:
        """Get player tier (elite or other)."""
        return 'elite' if self._is_elite(player_name, position) else 'other'

    def analyze_historical_performance(
        self,
        season: int = 2024,
        min_week: Optional[int] = None,
        max_week: Optional[int] = None,
    ) -> Dict[str, Dict[str, CategoryStats]]:
        """
        Analyze historical paper trades to get win rates by category.

        Returns:
            Dict mapping dimension -> category -> CategoryStats
        """
        stats: Dict[str, Dict[str, CategoryStats]] = {
            dim: {} for dim in self.config.dimensions
        }

        with get_session() as session:
            # Join with Player and Game to get season/week and player info
            query = session.query(PaperTrade).join(Player).join(Game)
            query = query.filter(
                Game.season == season,
                PaperTrade.won.isnot(None),  # Only evaluated trades
            )

            if min_week:
                query = query.filter(Game.week >= min_week)
            if max_week:
                query = query.filter(Game.week <= max_week)

            trades = query.all()

            for trade in trades:
                # Access related objects
                player = trade.player
                game = trade.game

                direction = trade.bet_side.upper()  # OVER or UNDER
                position = player.position or 'UNK'
                market = trade.market
                tier = self._get_tier(player.name, position)

                # Calculate profit/loss for this trade
                if trade.won is True:
                    profit = trade.profit_loss if trade.profit_loss is not None else 0
                    is_win = True
                elif trade.won is False:
                    profit = trade.profit_loss if trade.profit_loss is not None else -trade.stake
                    is_win = False
                else:
                    continue  # Skip push/pending

                # Build category keys for each dimension
                categories = {
                    'direction': direction,
                    'position': position,
                    'market': market,
                    'position_direction': f"{position}_{direction}",
                    'tier_direction': f"{tier}_{direction}",
                    'market_direction': f"{market}_{direction}",
                }

                # Update stats for each dimension
                for dim, category in categories.items():
                    if dim not in stats:
                        continue
                    if category not in stats[dim]:
                        stats[dim][category] = CategoryStats()

                    cat_stats = stats[dim][category]
                    if is_win:
                        cat_stats.wins += 1
                    else:
                        cat_stats.losses += 1
                    cat_stats.total_profit += profit

        return stats

    def generate_rules(
        self,
        season: int = 2024,
        min_week: Optional[int] = None,
        max_week: Optional[int] = None,
    ) -> List[FilterRule]:
        """
        Generate filter rules based on historical performance.

        Returns:
            List of FilterRule objects
        """
        stats = self.analyze_historical_performance(season, min_week, max_week)
        rules = []

        for dim, categories in stats.items():
            for category, cat_stats in categories.items():
                # Skip if sample size too small
                if cat_stats.total < self.config.min_sample_size:
                    continue

                # Check for skip conditions
                if (cat_stats.win_rate < self.config.skip_below_win_rate or
                    cat_stats.total_profit < self.config.skip_below_profit):
                    rules.append(FilterRule(
                        dimension=dim,
                        category=category,
                        action='skip',
                        win_rate=cat_stats.win_rate,
                        sample_size=cat_stats.total,
                        total_profit=cat_stats.total_profit,
                        reason=f"Win rate {cat_stats.win_rate:.0%} ({cat_stats.wins}-{cat_stats.losses}), "
                               f"P&L: ${cat_stats.total_profit:+,.0f}",
                    ))

                # Check for prioritize conditions
                elif cat_stats.win_rate >= self.config.prioritize_above_win_rate:
                    rules.append(FilterRule(
                        dimension=dim,
                        category=category,
                        action='prioritize',
                        win_rate=cat_stats.win_rate,
                        sample_size=cat_stats.total,
                        total_profit=cat_stats.total_profit,
                        reason=f"Win rate {cat_stats.win_rate:.0%} ({cat_stats.wins}-{cat_stats.losses}), "
                               f"P&L: ${cat_stats.total_profit:+,.0f}",
                    ))

        # Sort by impact (skip rules first, then by sample size)
        rules.sort(key=lambda r: (r.action != 'skip', -r.sample_size))

        self._rules = rules
        self._last_update = datetime.now()

        return rules

    def get_skip_categories(self) -> Dict[str, Set[str]]:
        """
        Get categories that should be skipped, organized by dimension.

        Returns:
            Dict mapping dimension -> set of categories to skip
        """
        skip = {}
        for rule in self._rules:
            if rule.action == 'skip':
                if rule.dimension not in skip:
                    skip[rule.dimension] = set()
                skip[rule.dimension].add(rule.category)
        return skip

    def get_prioritize_categories(self) -> Dict[str, Set[str]]:
        """
        Get categories that should be prioritized, organized by dimension.

        Returns:
            Dict mapping dimension -> set of categories to prioritize
        """
        prioritize = {}
        for rule in self._rules:
            if rule.action == 'prioritize':
                if rule.dimension not in prioritize:
                    prioritize[rule.dimension] = set()
                prioritize[rule.dimension].add(rule.category)
        return prioritize

    def should_skip(
        self,
        direction: str,
        position: str,
        market: str,
        player_name: str,
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if a bet should be skipped based on dynamic filters.

        Args:
            direction: 'over' or 'under'
            position: Player position (WR, RB, TE, QB)
            market: Market type (player_reception_yds, etc.)
            player_name: Player name for tier lookup

        Returns:
            Tuple of (should_skip, reason)
        """
        direction = direction.upper()
        tier = self._get_tier(player_name, position)

        # Build category keys
        categories = {
            'direction': direction,
            'position': position,
            'market': market,
            'position_direction': f"{position}_{direction}",
            'tier_direction': f"{tier}_{direction}",
            'market_direction': f"{market}_{direction}",
        }

        skip_cats = self.get_skip_categories()

        for dim, category in categories.items():
            if dim in skip_cats and category in skip_cats[dim]:
                # Find the rule for the reason
                for rule in self._rules:
                    if rule.dimension == dim and rule.category == category:
                        return True, f"Dynamic filter: {category} ({rule.reason})"

        return False, None

    def get_confidence_boost(
        self,
        direction: str,
        position: str,
        market: str,
        player_name: str,
    ) -> float:
        """
        Get confidence boost for prioritized categories.

        Returns:
            Multiplier (1.0 = no change, >1.0 = boost, <1.0 = penalty)
        """
        direction = direction.upper()
        tier = self._get_tier(player_name, position)

        # Build category keys
        categories = {
            'direction': direction,
            'position': position,
            'market': market,
            'position_direction': f"{position}_{direction}",
            'tier_direction': f"{tier}_{direction}",
            'market_direction': f"{market}_{direction}",
        }

        prioritize_cats = self.get_prioritize_categories()
        boost = 1.0

        # Small boost for each matching prioritized category
        for dim, category in categories.items():
            if dim in prioritize_cats and category in prioritize_cats[dim]:
                boost *= 1.02  # 2% boost per matching category

        return min(boost, 1.10)  # Cap at 10% total boost

    def save_rules(self) -> None:
        """Save current rules to cache file."""
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'last_update': self._last_update.isoformat() if self._last_update else None,
            'config': asdict(self.config),
            'rules': [r.to_dict() for r in self._rules],
        }

        with open(self._cache_path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved {len(self._rules)} dynamic filter rules to {self._cache_path}")

    def load_rules(self) -> bool:
        """
        Load rules from cache file.

        Returns:
            True if successfully loaded, False otherwise
        """
        if not self._cache_path.exists():
            return False

        try:
            with open(self._cache_path) as f:
                data = json.load(f)

            self._last_update = (
                datetime.fromisoformat(data['last_update'])
                if data.get('last_update') else None
            )
            self._rules = [FilterRule.from_dict(r) for r in data.get('rules', [])]

            logger.info(f"Loaded {len(self._rules)} dynamic filter rules")
            return True

        except Exception as e:
            logger.warning(f"Failed to load dynamic filters: {e}")
            return False

    def format_report(self) -> str:
        """Format rules into a readable report."""
        if not self._rules:
            return "No dynamic filter rules generated. Run generate_rules() first."

        lines = []
        lines.append("DYNAMIC BET FILTERS")
        lines.append(f"Generated: {self._last_update.strftime('%Y-%m-%d %H:%M') if self._last_update else 'N/A'}")
        lines.append(f"Config: min_sample={self.config.min_sample_size}, "
                    f"skip_below={self.config.skip_below_win_rate:.0%}, "
                    f"prioritize_above={self.config.prioritize_above_win_rate:.0%}")
        lines.append("")

        # Group by action
        skip_rules = [r for r in self._rules if r.action == 'skip']
        prioritize_rules = [r for r in self._rules if r.action == 'prioritize']

        if skip_rules:
            lines.append("🚫 SKIP FILTERS (avoid these bets):")
            lines.append("")
            for rule in skip_rules:
                lines.append(f"  {rule.category:<25} {rule.win_rate:>5.0%} ({rule.sample_size:>3} bets) "
                           f"${rule.total_profit:>+8,.0f}")
            lines.append("")

        if prioritize_rules:
            lines.append("✅ PRIORITIZE (favor these bets):")
            lines.append("")
            for rule in prioritize_rules:
                lines.append(f"  {rule.category:<25} {rule.win_rate:>5.0%} ({rule.sample_size:>3} bets) "
                           f"${rule.total_profit:>+8,.0f}")
            lines.append("")

        # Summary
        lines.append("SUMMARY:")
        lines.append(f"  Skip rules: {len(skip_rules)}")
        lines.append(f"  Prioritize rules: {len(prioritize_rules)}")

        return "\n".join(lines)

    def get_filter_summary_for_edge_calculator(self) -> Dict:
        """
        Get a summary dict suitable for EdgeCalculator.BET_FILTERS format.

        This provides backward compatibility with the static filter system.
        """
        skip_cats = self.get_skip_categories()

        summary = {
            'dynamic_filters_enabled': True,
            'last_update': self._last_update.isoformat() if self._last_update else None,
            'skip_rules_count': len([r for r in self._rules if r.action == 'skip']),
        }

        # Check for specific patterns that map to existing filters

        # skip_te_over equivalent
        if 'position_direction' in skip_cats and 'TE_OVER' in skip_cats['position_direction']:
            summary['skip_te_over'] = True

        # skip_under_all equivalent
        if 'direction' in skip_cats and 'UNDER' in skip_cats['direction']:
            summary['skip_under_all'] = True

        # Check for any elite_UNDER skip (maps to elite WR handling)
        if 'tier_direction' in skip_cats and 'elite_UNDER' in skip_cats['tier_direction']:
            summary['skip_elite_under'] = True

        return summary


# Singleton instance
_filter_service: Optional[DynamicFilterService] = None


def get_filter_service() -> DynamicFilterService:
    """Get the singleton DynamicFilterService instance."""
    global _filter_service
    if _filter_service is None:
        _filter_service = DynamicFilterService()
        _filter_service.load_rules()  # Try to load from cache
    return _filter_service
