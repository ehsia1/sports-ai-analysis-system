"""Parlay generation from top betting edges.

Generates optimal parlay combinations from uncorrelated edges,
calculating combined odds and expected value.
"""

import logging
from itertools import combinations
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from ..database import get_session
from ..database.models import Parlay

logger = logging.getLogger(__name__)


@dataclass
class ParlayLeg:
    """Single leg of a parlay."""
    player: str
    game: str
    market: str
    side: str  # "over" or "under"
    line: float
    prediction: float
    probability: float  # Model probability of hitting
    edge_pct: float
    ev_pct: float
    odds: int  # American odds


@dataclass
class ParlayCombo:
    """A parlay combination."""
    legs: List[ParlayLeg]
    joint_probability: float  # Combined probability (assuming independence)
    fair_odds: int  # What odds should be based on probability
    implied_odds: int  # Combined book odds
    expected_value: float  # EV as decimal (0.15 = 15%)
    parlay_type: str  # "cross_game" or "same_game"

    @property
    def leg_count(self) -> int:
        return len(self.legs)

    @property
    def ev_pct(self) -> float:
        return self.expected_value * 100

    def summary(self) -> str:
        """One-line summary of the parlay."""
        legs_str = " + ".join([
            f"{leg.player} {leg.side.upper()} {leg.line}"
            for leg in self.legs
        ])
        return f"{self.leg_count}-leg: {legs_str} | EV: {self.ev_pct:+.1f}%"


class ParlayGenerator:
    """Generate optimal parlay combinations from edges."""

    # Caps for safety
    MAX_LEGS_CAP = 20  # Allow up to 20 legs
    MAX_CANDIDATES_CAP = 50  # Max candidates to consider
    EXHAUSTIVE_THRESHOLD = 4  # Use exhaustive search up to this many legs
    MAX_COMBINATIONS = 50000  # Switch to greedy if combinations exceed this

    def __init__(
        self,
        min_leg_probability: float = 0.55,  # Min probability for each leg
        min_leg_ev: float = 0.03,  # Min EV for each leg (3%)
        max_legs: int = 5,  # Max legs per parlay
        min_parlay_ev: float = 0.10,  # Min combined EV (10%)
        max_candidates: int = 30,  # Max legs to consider (limits combinatorics)
    ):
        self.min_leg_probability = min_leg_probability
        self.min_leg_ev = min_leg_ev
        self.max_legs = min(max_legs, self.MAX_LEGS_CAP)
        self.min_parlay_ev = min_parlay_ev
        self.max_candidates = min(max_candidates, self.MAX_CANDIDATES_CAP)
        self._cached_legs: Optional[List[ParlayLeg]] = None

        if max_legs > self.MAX_LEGS_CAP:
            logger.warning(f"max_legs capped at {self.MAX_LEGS_CAP} (requested {max_legs})")

    def edges_to_legs(self, edges: List[Dict], force_refresh: bool = False) -> List[ParlayLeg]:
        """Convert edge dicts to ParlayLeg objects.

        Returns top candidates sorted by EV, limited to max_candidates.
        Uses caching to avoid recomputing for each leg size.
        """
        # Use cached legs if available
        if self._cached_legs is not None and not force_refresh:
            return self._cached_legs

        legs = []

        for edge in edges:
            # Determine which side to bet
            over_data = edge.get("over", {})
            under_data = edge.get("under", {})

            if under_data.get("should_bet"):
                side = "under"
                prob = float(under_data.get("model_probability", 0.5))
                edge_pct = float(under_data.get("edge_pct", 0))
                ev_pct = float(under_data.get("ev_pct", 0))
                odds = under_data.get("odds", -110)
            elif over_data.get("should_bet"):
                side = "over"
                prob = float(over_data.get("model_probability", 0.5))
                edge_pct = float(over_data.get("edge_pct", 0))
                ev_pct = float(over_data.get("ev_pct", 0))
                odds = over_data.get("odds", -110)
            else:
                continue

            # Filter by minimum probability and EV
            if prob < self.min_leg_probability:
                continue
            if ev_pct / 100 < self.min_leg_ev:
                continue

            leg = ParlayLeg(
                player=edge.get("player", "Unknown"),
                game=edge.get("game", ""),
                market=edge.get("market", ""),
                side=side,
                line=edge.get("line", 0),
                prediction=edge.get("prediction", 0),
                probability=prob,
                edge_pct=edge_pct,
                ev_pct=ev_pct,
                odds=odds,
            )
            legs.append(leg)

        # Sort by EV and limit to top candidates
        legs.sort(key=lambda x: x.ev_pct, reverse=True)

        # Deduplicate - keep only one line per player/market combo (highest EV)
        seen = set()
        unique_legs = []
        for leg in legs:
            key = (leg.player, leg.market, leg.side)
            if key not in seen:
                seen.add(key)
                unique_legs.append(leg)

        # IMPORTANT: Strictly limit to max_candidates to control combinatorics
        limited_legs = unique_legs[:self.max_candidates]
        logger.info(f"Filtered {len(unique_legs)} unique legs down to {len(limited_legs)} candidates")

        # Cache the result
        self._cached_legs = limited_legs
        return limited_legs

    def are_correlated(self, leg1: ParlayLeg, leg2: ParlayLeg) -> bool:
        """Check if two legs are correlated (should not be combined)."""
        # Same player - always correlated
        if leg1.player == leg2.player:
            return True

        # Same game passing stats are somewhat correlated
        # (if QB throws more, WRs catch more)
        if leg1.game == leg2.game:
            passing_markets = {"player_pass_yds", "player_reception_yds", "player_receptions"}
            if leg1.market in passing_markets and leg2.market in passing_markets:
                # Allow combining if opposite directions (one over, one under)
                if leg1.side != leg2.side:
                    return False
                return True

        return False

    def calculate_parlay_odds(self, legs: List[ParlayLeg]) -> Tuple[float, int, int]:
        """
        Calculate combined probability and odds for a parlay.

        Returns:
            Tuple of (joint_probability, fair_odds, implied_book_odds)
        """
        # Joint probability (assuming independence)
        joint_prob = 1.0
        for leg in legs:
            joint_prob *= leg.probability

        # Fair odds based on true probability
        if joint_prob > 0:
            fair_decimal = 1 / joint_prob
            fair_odds = self._decimal_to_american(fair_decimal)
        else:
            fair_odds = 99999

        # Calculate implied book odds (multiply decimal odds)
        implied_decimal = 1.0
        for leg in legs:
            implied_decimal *= self._american_to_decimal(leg.odds)
        implied_odds = self._decimal_to_american(implied_decimal)

        return joint_prob, fair_odds, implied_odds

    def _american_to_decimal(self, american: int) -> float:
        """Convert American odds to decimal."""
        if american > 0:
            return 1 + (american / 100)
        else:
            return 1 + (100 / abs(american))

    def _decimal_to_american(self, decimal: float) -> int:
        """Convert decimal odds to American."""
        if decimal >= 2.0:
            return int((decimal - 1) * 100)
        else:
            return int(-100 / (decimal - 1))

    def generate_parlays(
        self,
        edges: List[Dict],
        num_legs: int = 2,
        max_parlays: int = 10,
    ) -> List[ParlayCombo]:
        """
        Generate parlay combinations from edges.

        Args:
            edges: List of edge dictionaries
            num_legs: Number of legs per parlay (2, 3, or 4)
            max_parlays: Maximum parlays to return

        Returns:
            List of ParlayCombo sorted by EV
        """
        if num_legs < 2 or num_legs > self.max_legs:
            raise ValueError(f"num_legs must be between 2 and {self.max_legs}")

        # Convert edges to legs
        legs = self.edges_to_legs(edges)
        logger.info(f"Found {len(legs)} legs meeting criteria for parlays")

        if len(legs) < num_legs:
            logger.warning(f"Not enough legs ({len(legs)}) for {num_legs}-leg parlays")
            return []

        # Generate all combinations
        parlays = []
        for combo in combinations(legs, num_legs):
            # Check for correlations
            has_correlation = False
            for i, leg1 in enumerate(combo):
                for leg2 in combo[i+1:]:
                    if self.are_correlated(leg1, leg2):
                        has_correlation = True
                        break
                if has_correlation:
                    break

            if has_correlation:
                continue

            # Calculate combined odds
            joint_prob, fair_odds, implied_odds = self.calculate_parlay_odds(list(combo))

            # Calculate EV
            # EV = (probability * payout) - 1
            implied_decimal = self._american_to_decimal(implied_odds)
            ev = (joint_prob * implied_decimal) - 1

            # Filter by minimum EV
            if ev < self.min_parlay_ev:
                continue

            # Determine parlay type
            games = set(leg.game for leg in combo)
            parlay_type = "same_game" if len(games) == 1 else "cross_game"

            parlay = ParlayCombo(
                legs=list(combo),
                joint_probability=joint_prob,
                fair_odds=fair_odds,
                implied_odds=implied_odds,
                expected_value=ev,
                parlay_type=parlay_type,
            )
            parlays.append(parlay)

        # Sort by EV descending
        parlays.sort(key=lambda p: p.expected_value, reverse=True)

        logger.info(f"Generated {len(parlays)} valid {num_legs}-leg parlays")

        return parlays[:max_parlays]

    def _estimate_combinations(self, n: int, k: int) -> int:
        """Estimate C(n,k) without overflow."""
        if k > n:
            return 0
        if k == 0 or k == n:
            return 1
        k = min(k, n - k)
        result = 1
        for i in range(k):
            result = result * (n - i) // (i + 1)
        return result

    def _extend_parlay(
        self,
        parlay: ParlayCombo,
        legs: List[ParlayLeg],
    ) -> List[ParlayCombo]:
        """Extend a parlay by adding one more uncorrelated leg."""
        extensions = []
        existing_legs = set(id(leg) for leg in parlay.legs)

        for leg in legs:
            if id(leg) in existing_legs:
                continue

            # Check correlation with all existing legs
            is_correlated = False
            for existing_leg in parlay.legs:
                if self.are_correlated(leg, existing_leg):
                    is_correlated = True
                    break

            if is_correlated:
                continue

            # Create extended parlay
            new_legs = list(parlay.legs) + [leg]
            joint_prob, fair_odds, implied_odds = self.calculate_parlay_odds(new_legs)

            implied_decimal = self._american_to_decimal(implied_odds)
            ev = (joint_prob * implied_decimal) - 1

            if ev < self.min_parlay_ev:
                continue

            games = set(l.game for l in new_legs)
            parlay_type = "same_game" if len(games) == 1 else "cross_game"

            extensions.append(ParlayCombo(
                legs=new_legs,
                joint_probability=joint_prob,
                fair_odds=fair_odds,
                implied_odds=implied_odds,
                expected_value=ev,
                parlay_type=parlay_type,
            ))

        return extensions

    def generate_all_parlays(
        self,
        edges: List[Dict],
        max_per_size: int = 5,
    ) -> Dict[int, List[ParlayCombo]]:
        """
        Generate parlays of all sizes using hybrid approach.

        Uses exhaustive search for small parlays (2-4 legs) and
        greedy extension for larger parlays (5+ legs).

        Args:
            edges: List of edge dictionaries
            max_per_size: Max parlays per leg count

        Returns:
            Dict mapping leg count to list of parlays
        """
        result = {}
        legs = self.edges_to_legs(edges)

        if len(legs) < 2:
            return result

        # Use exhaustive search for small parlays
        for num_legs in range(2, min(self.EXHAUSTIVE_THRESHOLD + 1, self.max_legs + 1)):
            estimated = self._estimate_combinations(len(legs), num_legs)
            if estimated <= self.MAX_COMBINATIONS:
                parlays = self.generate_parlays(edges, num_legs, max_per_size)
                if parlays:
                    result[num_legs] = parlays
            else:
                logger.info(f"Skipping exhaustive {num_legs}-leg (would be {estimated:,} combinations)")

        # Use greedy extension for larger parlays
        if self.max_legs > self.EXHAUSTIVE_THRESHOLD:
            # Start from the best 4-leg parlays (or smaller if not available)
            base_size = self.EXHAUSTIVE_THRESHOLD
            while base_size >= 2 and base_size not in result:
                base_size -= 1

            if base_size >= 2:
                # Extend greedily
                current_best = result[base_size][:max(10, max_per_size * 2)]  # Keep more for extension

                for num_legs in range(base_size + 1, self.max_legs + 1):
                    next_parlays = []
                    for parlay in current_best:
                        extensions = self._extend_parlay(parlay, legs)
                        next_parlays.extend(extensions)

                    if not next_parlays:
                        logger.info(f"No valid {num_legs}-leg parlays found (greedy)")
                        break

                    # Sort by EV and keep best
                    next_parlays.sort(key=lambda p: p.expected_value, reverse=True)

                    # Deduplicate (same legs in different order)
                    seen = set()
                    unique_parlays = []
                    for p in next_parlays:
                        key = tuple(sorted(id(leg) for leg in p.legs))
                        if key not in seen:
                            seen.add(key)
                            unique_parlays.append(p)

                    result[num_legs] = unique_parlays[:max_per_size]
                    current_best = unique_parlays[:max(10, max_per_size * 2)]

                    logger.info(f"Generated {len(unique_parlays)} valid {num_legs}-leg parlays (greedy)")

        return result

    def format_parlay_report(
        self,
        parlays_by_size: Dict[int, List[ParlayCombo]],
    ) -> str:
        """Format parlays into a readable report."""
        lines = []
        lines.append("=" * 60)
        lines.append("PARLAY OPPORTUNITIES")
        lines.append("=" * 60)
        lines.append("")

        total_parlays = sum(len(p) for p in parlays_by_size.values())
        if total_parlays == 0:
            lines.append("No parlays meeting criteria found.")
            return "\n".join(lines)

        for num_legs in sorted(parlays_by_size.keys()):
            parlays = parlays_by_size[num_legs]
            lines.append(f"--- {num_legs}-LEG PARLAYS ({len(parlays)} found) ---")
            lines.append("")

            for i, parlay in enumerate(parlays, 1):
                lines.append(f"#{i} | EV: {parlay.ev_pct:+.1f}% | Prob: {parlay.joint_probability:.1%} | Odds: {parlay.implied_odds:+d}")
                for leg in parlay.legs:
                    market_short = leg.market.replace("player_", "").replace("_yds", "").replace("_", " ")
                    lines.append(f"   {leg.side.upper()} {leg.line} {market_short} - {leg.player} ({leg.game})")
                lines.append("")

        return "\n".join(lines)

    def store_parlays(self, parlays: List[ParlayCombo]) -> int:
        """Store parlays in database."""
        stored = 0

        with get_session() as session:
            for parlay in parlays:
                # Create leg data for storage
                legs_data = [
                    {
                        "player": leg.player,
                        "game": leg.game,
                        "market": leg.market,
                        "side": leg.side,
                        "line": leg.line,
                        "odds": leg.odds,
                    }
                    for leg in parlay.legs
                ]

                db_parlay = Parlay(
                    legs=legs_data,
                    parlay_type=parlay.parlay_type,
                    offered_odds=parlay.implied_odds,
                    fair_odds=float(parlay.fair_odds),
                    joint_probability=parlay.joint_probability,
                    expected_value=parlay.expected_value,
                    created_at=datetime.utcnow(),
                )
                session.add(db_parlay)
                stored += 1

            session.commit()

        logger.info(f"Stored {stored} parlays in database")
        return stored

    def send_discord_notifications(
        self,
        parlays_by_size: Dict[int, List[ParlayCombo]],
        max_per_size: int = 3,
    ) -> int:
        """Send top parlays to Discord."""
        import time
        from ..notifications.discord import send_parlay_notification

        sent_count = 0
        all_parlays = []

        # Collect top parlays from each size
        for num_legs in sorted(parlays_by_size.keys()):
            parlays = parlays_by_size[num_legs][:max_per_size]
            all_parlays.extend(parlays)

        logger.info(f"Sending {len(all_parlays)} parlays to Discord")

        for i, parlay in enumerate(all_parlays):
            try:
                success = send_parlay_notification(parlay)
                if success:
                    sent_count += 1
                    # Rate limit - wait between messages
                    if i < len(all_parlays) - 1:
                        time.sleep(1.5)
            except Exception as e:
                logger.error(f"Failed to send parlay notification: {e}")

        logger.info(f"Sent {sent_count}/{len(all_parlays)} parlay notifications")
        return sent_count
