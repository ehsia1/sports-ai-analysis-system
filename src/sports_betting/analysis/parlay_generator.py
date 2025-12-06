"""Parlay generation from top betting edges.

Generates optimal parlay combinations from edges,
calculating combined odds and expected value with
same-game parlay correlation adjustments.
"""

import logging
import math
from itertools import combinations
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from ..database import get_session
from ..database.models import Parlay

logger = logging.getLogger(__name__)


@dataclass
class ParlayLeg:
    """Single leg of a parlay."""
    player: str
    team: str  # Team abbreviation (e.g., 'ATL', 'DET')
    game: str
    market: str
    side: str  # "over" or "under"
    line: float
    prediction: float
    probability: float  # Model probability of hitting
    edge_pct: float
    ev_pct: float
    odds: int  # American odds
    position: str = ""  # Position (QB, RB, WR, TE)


# Correlation coefficients for same-game parlays
# Based on empirical NFL data - positive = outcomes move together
# These adjust joint probability: higher correlation = less independence
SGP_CORRELATIONS = {
    # Same team passing connection (QB + receiver)
    # If QB throws more, WR/TE get more yards - strong positive correlation
    ("QB", "WR", "same_team"): 0.45,
    ("QB", "TE", "same_team"): 0.40,

    # Same team, same position group - competition for touches
    # If RB1 gets more carries, RB2 gets fewer - negative correlation
    ("RB", "RB", "same_team"): -0.25,
    ("WR", "WR", "same_team"): -0.15,
    ("WR", "TE", "same_team"): -0.10,

    # Same team, different roles - weak positive (game script)
    ("RB", "WR", "same_team"): 0.10,
    ("RB", "TE", "same_team"): 0.10,

    # Opposite teams - game script effects
    # If one team dominates, other team plays from behind
    ("QB", "QB", "opposite_team"): 0.15,  # Both QBs can have good games in shootout
    ("RB", "RB", "opposite_team"): -0.10,  # Winning team runs more
    ("WR", "WR", "opposite_team"): 0.10,
    ("QB", "WR", "opposite_team"): 0.05,
    ("QB", "RB", "opposite_team"): -0.05,
}


@dataclass
class ParlayCombo:
    """A parlay combination."""
    legs: List[ParlayLeg]
    joint_probability: float  # Combined probability (adjusted for correlation)
    fair_odds: int  # What odds should be based on probability
    implied_odds: int  # Combined book odds
    expected_value: float  # EV as decimal (0.15 = 15%)
    parlay_type: str  # "cross_game" or "same_game"
    # Correlation tracking
    independent_probability: float = 0.0  # Probability if legs were independent
    correlation_adjustment: float = 0.0  # How much correlation changed probability
    correlation_warnings: List[str] = field(default_factory=list)

    @property
    def leg_count(self) -> int:
        return len(self.legs)

    @property
    def ev_pct(self) -> float:
        return self.expected_value * 100

    @property
    def has_significant_correlation(self) -> bool:
        """Returns True if correlation adjustment is significant (>5%)."""
        return abs(self.correlation_adjustment) > 0.05

    def summary(self) -> str:
        """One-line summary of the parlay."""
        legs_str = " + ".join([
            f"{leg.player} {leg.side.upper()} {leg.line}"
            for leg in self.legs
        ])
        corr_note = ""
        if self.has_significant_correlation:
            direction = "↑" if self.correlation_adjustment > 0 else "↓"
            corr_note = f" | Corr: {direction}{abs(self.correlation_adjustment):.0%}"
        return f"{self.leg_count}-leg: {legs_str} | EV: {self.ev_pct:+.1f}%{corr_note}"


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

    def _infer_position_from_market(self, market: str) -> str:
        """Infer player position from market type."""
        market_lower = market.lower()
        if "pass" in market_lower:
            return "QB"
        elif "rush" in market_lower:
            return "RB"  # Could be QB too, but RB is most common
        elif "reception" in market_lower or "receiving" in market_lower:
            return "WR"  # Could be TE/RB, but WR most common
        return ""

    def get_correlation_coefficient(
        self, leg1: ParlayLeg, leg2: ParlayLeg
    ) -> Tuple[float, str]:
        """
        Calculate correlation coefficient between two legs.

        Returns:
            Tuple of (correlation_coefficient, description)
            - Positive: outcomes tend to move together
            - Negative: outcomes tend to move opposite
            - Zero: independent
        """
        # Different games = no correlation (independent)
        if leg1.game != leg2.game:
            return 0.0, "cross_game"

        # Same game - determine relationship
        pos1 = leg1.position.upper() if leg1.position else "UNK"
        pos2 = leg2.position.upper() if leg2.position else "UNK"
        same_team = leg1.team == leg2.team

        team_type = "same_team" if same_team else "opposite_team"

        # Look up correlation coefficient
        # Try both orderings since dict may only have one
        key1 = (pos1, pos2, team_type)
        key2 = (pos2, pos1, team_type)

        if key1 in SGP_CORRELATIONS:
            corr = SGP_CORRELATIONS[key1]
        elif key2 in SGP_CORRELATIONS:
            corr = SGP_CORRELATIONS[key2]
        else:
            # Default: small positive correlation for same game
            corr = 0.05 if same_team else 0.02

        # Adjust correlation based on bet direction
        # If betting same direction (both OVER or both UNDER), correlation stays same
        # If betting opposite directions, flip the sign
        if leg1.side != leg2.side:
            corr = -corr

        description = f"{pos1}+{pos2} ({team_type.replace('_', ' ')})"
        return corr, description

    def _calculate_correlated_probability(
        self, legs: List[ParlayLeg]
    ) -> Tuple[float, float, List[str]]:
        """
        Calculate joint probability with correlation adjustment.

        Uses a simplified copula-like adjustment:
        - For positive correlations: reduce joint probability (harder to hit both)
        - For negative correlations: increase joint probability (easier to hit both)

        Returns:
            Tuple of (adjusted_probability, independent_probability, warnings)
        """
        # Independent probability (naive multiplication)
        independent_prob = 1.0
        for leg in legs:
            independent_prob *= leg.probability

        # Calculate total correlation effect
        # Sum pairwise correlations
        total_corr = 0.0
        warnings = []

        for i, leg1 in enumerate(legs):
            for leg2 in legs[i + 1:]:
                corr, desc = self.get_correlation_coefficient(leg1, leg2)
                if abs(corr) > 0.1:
                    warnings.append(f"{leg1.player} ↔ {leg2.player}: {corr:+.2f} ({desc})")
                total_corr += corr

        # Apply correlation adjustment
        # Positive correlation = lower probability of both hitting
        # Negative correlation = higher probability of both hitting
        # Scale factor based on average individual probability
        avg_prob = independent_prob ** (1 / len(legs))

        # Adjustment formula: multiply by (1 - corr * scale_factor)
        # Scale factor decreases as leg count increases (correlations compound less)
        scale_factor = 0.15 / math.sqrt(len(legs))
        adjustment_factor = 1.0 - (total_corr * scale_factor)

        # Clamp to reasonable range (0.5 to 1.5 of independent)
        adjustment_factor = max(0.5, min(1.5, adjustment_factor))

        adjusted_prob = independent_prob * adjustment_factor

        # Ensure probability stays in valid range
        adjusted_prob = max(0.001, min(0.999, adjusted_prob))

        return adjusted_prob, independent_prob, warnings

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

            # Extract position from edge data or infer from market
            position = edge.get("position", "")
            if not position:
                position = self._infer_position_from_market(edge.get("market", ""))

            leg = ParlayLeg(
                player=edge.get("player", "Unknown"),
                team=edge.get("team", ""),
                game=edge.get("game", ""),
                market=edge.get("market", ""),
                side=side,
                line=edge.get("line", 0),
                prediction=edge.get("prediction", 0),
                probability=prob,
                edge_pct=edge_pct,
                ev_pct=ev_pct,
                odds=odds,
                position=position,
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
        """
        Check if two legs are correlated (should not be combined).

        Same-Game Parlays (SGPs) are valid betting products offered by sportsbooks.
        We only block truly problematic combinations:
        - Same player, same market, opposite sides (e.g., OVER and UNDER on same line)

        Allowed combinations:
        - Same player, different markets (e.g., mobile QB pass yards + rush yards)
        - Same team, same game (SGP) - e.g., ATL QB + ATL RB
        - Different teams, same game (SGP) - e.g., DET QB + DAL RB
        - Cross-game (standard parlay)
        """
        # Same player, same market = correlated (can't bet both sides)
        if leg1.player == leg2.player and leg1.market == leg2.market:
            return True

        # Everything else is allowed:
        # - Same player, different markets (mobile QB pass + rush, RB rush + receptions)
        # - Same game, same team (SGP)
        # - Same game, different teams (SGP)
        # - Cross-game (standard parlay)
        return False

    def calculate_parlay_odds(
        self, legs: List[ParlayLeg], apply_correlation: bool = True
    ) -> Tuple[float, int, int, float, List[str]]:
        """
        Calculate combined probability and odds for a parlay.

        Args:
            legs: List of parlay legs
            apply_correlation: Whether to apply same-game correlation adjustments

        Returns:
            Tuple of (joint_probability, fair_odds, implied_book_odds,
                      independent_probability, correlation_warnings)
        """
        # Calculate probability with correlation adjustment
        if apply_correlation:
            joint_prob, independent_prob, warnings = self._calculate_correlated_probability(legs)
        else:
            # Independent probability (assuming no correlation)
            joint_prob = 1.0
            for leg in legs:
                joint_prob *= leg.probability
            independent_prob = joint_prob
            warnings = []

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

        return joint_prob, fair_odds, implied_odds, independent_prob, warnings

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

            # Calculate combined odds with correlation adjustment
            (
                joint_prob, fair_odds, implied_odds,
                independent_prob, corr_warnings
            ) = self.calculate_parlay_odds(list(combo))

            # Calculate correlation adjustment ratio
            corr_adjustment = (joint_prob / independent_prob) - 1 if independent_prob > 0 else 0

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
                independent_probability=independent_prob,
                correlation_adjustment=corr_adjustment,
                correlation_warnings=corr_warnings,
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

            # Create extended parlay with correlation adjustment
            new_legs = list(parlay.legs) + [leg]
            (
                joint_prob, fair_odds, implied_odds,
                independent_prob, corr_warnings
            ) = self.calculate_parlay_odds(new_legs)

            corr_adjustment = (joint_prob / independent_prob) - 1 if independent_prob > 0 else 0

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
                independent_probability=independent_prob,
                correlation_adjustment=corr_adjustment,
                correlation_warnings=corr_warnings,
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
                # Show correlation adjustment if significant
                corr_info = ""
                if parlay.has_significant_correlation:
                    direction = "↓" if parlay.correlation_adjustment < 0 else "↑"
                    corr_info = f" | Corr: {direction}{abs(parlay.correlation_adjustment):.0%}"

                lines.append(
                    f"#{i} | EV: {parlay.ev_pct:+.1f}% | "
                    f"Prob: {parlay.joint_probability:.1%} | "
                    f"Odds: {parlay.implied_odds:+d}{corr_info}"
                )

                for leg in parlay.legs:
                    market_short = leg.market.replace("player_", "").replace("_yds", "").replace("_", " ")
                    pos_info = f" [{leg.position}]" if leg.position else ""
                    lines.append(
                        f"   {leg.side.upper()} {leg.line} {market_short} - "
                        f"{leg.player}{pos_info} ({leg.game})"
                    )

                # Show correlation warnings for same-game parlays
                if parlay.correlation_warnings:
                    lines.append("   ⚠️ Correlations:")
                    for warning in parlay.correlation_warnings[:3]:  # Limit to top 3
                        lines.append(f"      {warning}")

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
