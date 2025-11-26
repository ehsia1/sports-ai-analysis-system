"""Odds conversion and manipulation utilities."""

from typing import Tuple


def american_to_decimal(american_odds: int) -> float:
    """Convert American odds to decimal odds."""
    if american_odds > 0:
        return (american_odds / 100) + 1
    else:
        return (100 / abs(american_odds)) + 1


def decimal_to_american(decimal_odds: float) -> int:
    """Convert decimal odds to American odds."""
    if decimal_odds >= 2.0:
        return int((decimal_odds - 1) * 100)
    else:
        return int(-100 / (decimal_odds - 1))


def implied_probability(american_odds: int) -> float:
    """Calculate implied probability from American odds."""
    decimal_odds = american_to_decimal(american_odds)
    return 1 / decimal_odds


def devig_odds(over_odds: int, under_odds: int) -> Tuple[float, float]:
    """
    Remove the vig from odds to get true probabilities.
    
    Returns:
        Tuple of (over_probability, under_probability)
    """
    over_prob_raw = implied_probability(over_odds)
    under_prob_raw = implied_probability(under_odds)
    
    total_prob = over_prob_raw + under_prob_raw
    
    # Remove the vig by normalizing
    over_prob_true = over_prob_raw / total_prob
    under_prob_true = under_prob_raw / total_prob
    
    return over_prob_true, under_prob_true


def calculate_ev(
    probability: float, 
    odds: int, 
    bet_amount: float = 1.0
) -> float:
    """
    Calculate expected value of a bet.
    
    Args:
        probability: True probability of the outcome
        odds: American odds offered
        bet_amount: Amount wagered
    
    Returns:
        Expected value of the bet
    """
    decimal_odds = american_to_decimal(odds)
    
    # Profit if win
    profit_if_win = bet_amount * (decimal_odds - 1)
    
    # Loss if lose (the bet amount)
    loss_if_lose = bet_amount
    
    # Expected value
    ev = (probability * profit_if_win) - ((1 - probability) * loss_if_lose)
    
    return ev


def kelly_criterion(
    probability: float, 
    odds: int, 
    bankroll: float = 1.0
) -> float:
    """
    Calculate optimal bet size using Kelly criterion.
    
    Args:
        probability: True probability of winning
        odds: American odds offered
        bankroll: Total bankroll
    
    Returns:
        Optimal fraction of bankroll to bet
    """
    decimal_odds = american_to_decimal(odds)
    
    # Kelly formula: f = (bp - q) / b
    # where b = odds-1, p = probability of win, q = probability of loss
    b = decimal_odds - 1
    p = probability
    q = 1 - probability
    
    kelly_fraction = (b * p - q) / b
    
    # Don't bet if Kelly is negative (negative expected value)
    return max(0, kelly_fraction)


def fractional_kelly(
    probability: float, 
    odds: int, 
    fraction: float = 0.25,
    bankroll: float = 1.0
) -> float:
    """
    Calculate fractional Kelly bet size to reduce variance.
    
    Args:
        probability: True probability of winning
        odds: American odds offered
        fraction: Fraction of Kelly to use (e.g., 0.25 for quarter Kelly)
        bankroll: Total bankroll
    
    Returns:
        Bet amount using fractional Kelly
    """
    full_kelly = kelly_criterion(probability, odds, bankroll)
    return full_kelly * fraction * bankroll


def closing_line_value(
    bet_odds: int, 
    closing_odds: int
) -> float:
    """
    Calculate closing line value (CLV).
    
    Args:
        bet_odds: Odds when bet was placed
        closing_odds: Final closing odds
    
    Returns:
        CLV as a percentage (positive is good)
    """
    bet_prob = implied_probability(bet_odds)
    closing_prob = implied_probability(closing_odds)
    
    # CLV = (closing probability - bet probability) / bet probability
    clv = (closing_prob - bet_prob) / bet_prob
    
    return clv * 100  # Return as percentage