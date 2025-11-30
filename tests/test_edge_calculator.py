"""Tests for the edge calculator module."""

import pytest


class TestOddsConversion:
    """Test odds conversion functions."""

    def test_american_to_decimal_positive_odds(self):
        """Positive American odds conversion to decimal."""
        from src.sports_betting.data.odds_api import american_to_decimal

        # +150 should be 2.5 (win $150 on $100 bet = $250 total = 2.5x)
        assert american_to_decimal(150) == pytest.approx(2.5)

        # +100 should be 2.0
        assert american_to_decimal(100) == pytest.approx(2.0)

        # +200 should be 3.0
        assert american_to_decimal(200) == pytest.approx(3.0)

    def test_american_to_decimal_negative_odds(self):
        """Negative American odds conversion to decimal."""
        from src.sports_betting.data.odds_api import american_to_decimal

        # -110 should be approximately 1.909
        assert american_to_decimal(-110) == pytest.approx(1.909, rel=0.01)

        # -100 should be 2.0
        assert american_to_decimal(-100) == pytest.approx(2.0)

        # -200 should be 1.5
        assert american_to_decimal(-200) == pytest.approx(1.5)

    def test_american_to_probability_positive_odds(self):
        """Positive American odds to implied probability."""
        from src.sports_betting.data.odds_api import american_to_probability

        # +100 should be 50%
        assert american_to_probability(100) == pytest.approx(0.5)

        # +200 should be 33.33%
        assert american_to_probability(200) == pytest.approx(0.333, rel=0.01)

    def test_american_to_probability_negative_odds(self):
        """Negative American odds to implied probability."""
        from src.sports_betting.data.odds_api import american_to_probability

        # -100 should be 50%
        assert american_to_probability(-100) == pytest.approx(0.5)

        # -200 should be 66.67%
        assert american_to_probability(-200) == pytest.approx(0.667, rel=0.01)


class TestVigRemoval:
    """Test vig removal functions."""

    def test_remove_vig_balanced_odds(self):
        """Balanced -110/-110 should normalize to 50/50."""
        from src.sports_betting.data.odds_api import remove_vig, american_to_probability

        over_prob = american_to_probability(-110)
        under_prob = american_to_probability(-110)

        fair_over, fair_under = remove_vig(over_prob, under_prob)

        assert fair_over == pytest.approx(0.5, rel=0.01)
        assert fair_under == pytest.approx(0.5, rel=0.01)
        assert fair_over + fair_under == pytest.approx(1.0)

    def test_remove_vig_unbalanced_odds(self):
        """Unbalanced odds should remove vig proportionally."""
        from src.sports_betting.data.odds_api import remove_vig, american_to_probability

        # -150/+130 (favorite/underdog)
        over_prob = american_to_probability(-150)  # ~60%
        under_prob = american_to_probability(130)  # ~43%

        fair_over, fair_under = remove_vig(over_prob, under_prob)

        # Fair probabilities should sum to 1.0
        assert fair_over + fair_under == pytest.approx(1.0)

        # Fair over should be less than implied (vig removed)
        assert fair_over < over_prob

    def test_remove_vig_no_vig(self):
        """If no vig present, return original probabilities."""
        from src.sports_betting.data.odds_api import remove_vig

        fair_over, fair_under = remove_vig(0.5, 0.5)

        assert fair_over == pytest.approx(0.5)
        assert fair_under == pytest.approx(0.5)


class TestEdgeCalculator:
    """Test the EdgeCalculator class."""

    def test_edge_calculator_initialization(self):
        """EdgeCalculator should initialize with default thresholds."""
        from src.sports_betting.analysis.edge_calculator import EdgeCalculator

        calc = EdgeCalculator()

        assert calc.min_edge == 0.03
        assert calc.min_confidence == 0.65

    def test_calculate_edge_over_bet(self):
        """Should identify OVER bet when prediction > line."""
        from src.sports_betting.analysis.edge_calculator import EdgeCalculator

        calc = EdgeCalculator()

        result = calc.calculate_edge(
            model_prediction=85.0,
            model_confidence=0.80,
            market_line=70.0,
            over_odds=-110,
            under_odds=-110
        )

        # Should recommend OVER
        assert result['over']['edge'] > 0
        assert result['over']['should_bet'] == True
        assert result['under']['should_bet'] == False

    def test_calculate_edge_under_bet(self):
        """Should identify UNDER bet when prediction < line."""
        from src.sports_betting.analysis.edge_calculator import EdgeCalculator

        calc = EdgeCalculator()

        result = calc.calculate_edge(
            model_prediction=55.0,
            model_confidence=0.80,
            market_line=70.0,
            over_odds=-110,
            under_odds=-110
        )

        # Should recommend UNDER
        assert result['under']['edge'] > 0
        assert result['under']['should_bet'] == True
        assert result['over']['should_bet'] == False

    def test_calculate_edge_no_bet_when_close(self):
        """Should not recommend bet when prediction is close to line."""
        from src.sports_betting.analysis.edge_calculator import EdgeCalculator

        calc = EdgeCalculator()

        result = calc.calculate_edge(
            model_prediction=71.0,  # Very close to line
            model_confidence=0.70,
            market_line=70.0,
            over_odds=-110,
            under_odds=-110
        )

        # Neither side should meet minimum edge threshold
        assert result['over']['should_bet'] == False
        assert result['under']['should_bet'] == False

    def test_calculate_edge_low_confidence_no_bet(self):
        """Should not recommend bet when model confidence is too low."""
        from src.sports_betting.analysis.edge_calculator import EdgeCalculator

        calc = EdgeCalculator()

        result = calc.calculate_edge(
            model_prediction=100.0,  # Big edge but...
            model_confidence=0.50,   # Low confidence
            market_line=70.0,
            over_odds=-110,
            under_odds=-110
        )

        # Should not bet despite edge due to low confidence
        assert result['over']['should_bet'] == False
