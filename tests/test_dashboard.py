"""Tests for the dashboard module."""

import pytest
from unittest.mock import MagicMock, patch

from src.sports_betting.tracking.dashboard import (
    BreakdownStats,
    DashboardData,
    ResultDashboard,
)


class TestBreakdownStats:
    """Tests for the BreakdownStats dataclass."""

    def test_breakdown_stats_defaults(self):
        """BreakdownStats should have sensible defaults."""
        stats = BreakdownStats(category='test')

        assert stats.category == 'test'
        assert stats.total_bets == 0
        assert stats.wins == 0
        assert stats.losses == 0
        assert stats.pushes == 0
        assert stats.total_staked == 0.0
        assert stats.total_profit == 0.0

    def test_breakdown_stats_win_rate_calculation(self):
        """BreakdownStats.win_rate should calculate correctly."""
        stats = BreakdownStats(category='test', wins=6, losses=4)

        assert stats.win_rate == 60.0  # 6/10 * 100

    def test_breakdown_stats_win_rate_all_wins(self):
        """BreakdownStats.win_rate should be 100% with all wins."""
        stats = BreakdownStats(category='test', wins=10, losses=0)

        assert stats.win_rate == 100.0

    def test_breakdown_stats_win_rate_all_losses(self):
        """BreakdownStats.win_rate should be 0% with all losses."""
        stats = BreakdownStats(category='test', wins=0, losses=10)

        assert stats.win_rate == 0.0

    def test_breakdown_stats_win_rate_no_decided(self):
        """BreakdownStats.win_rate should return 0 when no decided bets."""
        stats = BreakdownStats(category='test', wins=0, losses=0, pushes=5)

        assert stats.win_rate == 0.0

    def test_breakdown_stats_roi_calculation(self):
        """BreakdownStats.roi should calculate correctly."""
        stats = BreakdownStats(
            category='test',
            total_staked=1000.0,
            total_profit=150.0,
        )

        assert stats.roi == 15.0  # 150/1000 * 100

    def test_breakdown_stats_roi_negative(self):
        """BreakdownStats.roi should handle negative profit."""
        stats = BreakdownStats(
            category='test',
            total_staked=1000.0,
            total_profit=-200.0,
        )

        assert stats.roi == -20.0

    def test_breakdown_stats_roi_zero_staked(self):
        """BreakdownStats.roi should return 0 when nothing staked."""
        stats = BreakdownStats(category='test', total_staked=0.0)

        assert stats.roi == 0.0


class TestDashboardData:
    """Tests for the DashboardData dataclass."""

    def test_dashboard_data_defaults(self):
        """DashboardData should have sensible defaults."""
        data = DashboardData(season=2024)

        assert data.season == 2024
        assert data.week is None
        assert data.total_bets == 0
        assert data.wins == 0
        assert data.losses == 0
        assert data.by_market == {}
        assert data.by_direction == {}
        assert data.by_position == {}

    def test_dashboard_data_with_week(self):
        """DashboardData should accept specific week."""
        data = DashboardData(season=2024, week=13)

        assert data.season == 2024
        assert data.week == 13

    def test_dashboard_data_win_rate(self):
        """DashboardData.win_rate should calculate correctly."""
        data = DashboardData(season=2024, wins=15, losses=10)

        assert data.win_rate == 60.0  # 15/25 * 100

    def test_dashboard_data_win_rate_no_bets(self):
        """DashboardData.win_rate should return 0 when no bets."""
        data = DashboardData(season=2024)

        assert data.win_rate == 0.0

    def test_dashboard_data_roi(self):
        """DashboardData.roi should calculate correctly."""
        data = DashboardData(
            season=2024,
            total_staked=5000.0,
            total_profit=500.0,
        )

        assert data.roi == 10.0  # 500/5000 * 100

    def test_dashboard_data_roi_negative(self):
        """DashboardData.roi should handle negative profit."""
        data = DashboardData(
            season=2024,
            total_staked=5000.0,
            total_profit=-750.0,
        )

        assert data.roi == -15.0

    def test_dashboard_data_roi_zero_staked(self):
        """DashboardData.roi should return 0 when nothing staked."""
        data = DashboardData(season=2024, total_staked=0.0)

        assert data.roi == 0.0


class TestResultDashboard:
    """Tests for the ResultDashboard class."""

    def test_dashboard_initialization(self):
        """ResultDashboard should initialize with None data."""
        dashboard = ResultDashboard()

        assert dashboard._data is None

    def test_elite_players_structure(self):
        """ResultDashboard should have elite player lists."""
        dashboard = ResultDashboard()

        assert 'WR' in dashboard.ELITE_PLAYERS
        assert 'RB' in dashboard.ELITE_PLAYERS
        assert 'TE' in dashboard.ELITE_PLAYERS
        assert 'QB' in dashboard.ELITE_PLAYERS

    def test_elite_players_contain_known_names(self):
        """Elite player lists should contain known star players."""
        dashboard = ResultDashboard()

        assert "Ja'Marr Chase" in dashboard.ELITE_PLAYERS['WR']
        assert 'CeeDee Lamb' in dashboard.ELITE_PLAYERS['WR']
        assert 'Derrick Henry' in dashboard.ELITE_PLAYERS['RB']
        assert 'Travis Kelce' in dashboard.ELITE_PLAYERS['TE']
        assert 'Patrick Mahomes' in dashboard.ELITE_PLAYERS['QB']

    def test_get_player_tier_elite_wr(self):
        """get_player_tier should identify elite WRs."""
        dashboard = ResultDashboard()

        assert dashboard.get_player_tier("Ja'Marr Chase", 'WR') == 'Elite'
        assert dashboard.get_player_tier('CeeDee Lamb', 'WR') == 'Elite'
        assert dashboard.get_player_tier('Justin Jefferson', 'WR') == 'Elite'

    def test_get_player_tier_elite_rb(self):
        """get_player_tier should identify elite RBs."""
        dashboard = ResultDashboard()

        assert dashboard.get_player_tier('Derrick Henry', 'RB') == 'Elite'
        assert dashboard.get_player_tier('Saquon Barkley', 'RB') == 'Elite'

    def test_get_player_tier_elite_te(self):
        """get_player_tier should identify elite TEs."""
        dashboard = ResultDashboard()

        assert dashboard.get_player_tier('Travis Kelce', 'TE') == 'Elite'
        assert dashboard.get_player_tier('George Kittle', 'TE') == 'Elite'

    def test_get_player_tier_elite_qb(self):
        """get_player_tier should identify elite QBs."""
        dashboard = ResultDashboard()

        assert dashboard.get_player_tier('Patrick Mahomes', 'QB') == 'Elite'
        assert dashboard.get_player_tier('Josh Allen', 'QB') == 'Elite'

    def test_get_player_tier_other(self):
        """get_player_tier should return 'Other' for non-elite players."""
        dashboard = ResultDashboard()

        assert dashboard.get_player_tier('Random Player', 'WR') == 'Other'
        assert dashboard.get_player_tier('Unknown RB', 'RB') == 'Other'
        assert dashboard.get_player_tier('Backup TE', 'TE') == 'Other'
        assert dashboard.get_player_tier('Third String QB', 'QB') == 'Other'

    def test_get_player_tier_unknown_position(self):
        """get_player_tier should handle unknown positions."""
        dashboard = ResultDashboard()

        assert dashboard.get_player_tier('Anyone', 'K') == 'Other'
        assert dashboard.get_player_tier('Anyone', 'P') == 'Other'
        assert dashboard.get_player_tier('Anyone', '') == 'Other'

    def test_get_edge_bucket_20_plus(self):
        """get_edge_bucket should categorize 20%+ edges."""
        dashboard = ResultDashboard()

        assert dashboard.get_edge_bucket(20.0) == '20%+'
        assert dashboard.get_edge_bucket(25.0) == '20%+'
        assert dashboard.get_edge_bucket(50.0) == '20%+'

    def test_get_edge_bucket_15_20(self):
        """get_edge_bucket should categorize 15-20% edges."""
        dashboard = ResultDashboard()

        assert dashboard.get_edge_bucket(15.0) == '15-20%'
        assert dashboard.get_edge_bucket(17.5) == '15-20%'
        assert dashboard.get_edge_bucket(19.9) == '15-20%'

    def test_get_edge_bucket_10_15(self):
        """get_edge_bucket should categorize 10-15% edges."""
        dashboard = ResultDashboard()

        assert dashboard.get_edge_bucket(10.0) == '10-15%'
        assert dashboard.get_edge_bucket(12.0) == '10-15%'
        assert dashboard.get_edge_bucket(14.9) == '10-15%'

    def test_get_edge_bucket_5_10(self):
        """get_edge_bucket should categorize 5-10% edges."""
        dashboard = ResultDashboard()

        assert dashboard.get_edge_bucket(5.0) == '5-10%'
        assert dashboard.get_edge_bucket(7.5) == '5-10%'
        assert dashboard.get_edge_bucket(9.9) == '5-10%'

    def test_get_edge_bucket_under_5(self):
        """get_edge_bucket should categorize <5% edges."""
        dashboard = ResultDashboard()

        assert dashboard.get_edge_bucket(4.9) == '<5%'
        assert dashboard.get_edge_bucket(3.0) == '<5%'
        assert dashboard.get_edge_bucket(0.5) == '<5%'

    def test_get_edge_bucket_negative_values(self):
        """get_edge_bucket should use absolute value."""
        dashboard = ResultDashboard()

        # Should use abs() so -15% categorizes as 15-20%
        assert dashboard.get_edge_bucket(-15.0) == '15-20%'
        assert dashboard.get_edge_bucket(-3.0) == '<5%'

    def test_get_confidence_bucket_high(self):
        """get_confidence_bucket should categorize high confidence."""
        dashboard = ResultDashboard()

        assert dashboard.get_confidence_bucket(0.80) == 'High (80%+)'
        assert dashboard.get_confidence_bucket(0.85) == 'High (80%+)'
        assert dashboard.get_confidence_bucket(0.95) == 'High (80%+)'
        assert dashboard.get_confidence_bucket(1.0) == 'High (80%+)'

    def test_get_confidence_bucket_medium(self):
        """get_confidence_bucket should categorize medium confidence."""
        dashboard = ResultDashboard()

        assert dashboard.get_confidence_bucket(0.60) == 'Medium (60-80%)'
        assert dashboard.get_confidence_bucket(0.70) == 'Medium (60-80%)'
        assert dashboard.get_confidence_bucket(0.79) == 'Medium (60-80%)'

    def test_get_confidence_bucket_low(self):
        """get_confidence_bucket should categorize low confidence."""
        dashboard = ResultDashboard()

        assert dashboard.get_confidence_bucket(0.50) == 'Low (<60%)'
        assert dashboard.get_confidence_bucket(0.40) == 'Low (<60%)'
        assert dashboard.get_confidence_bucket(0.0) == 'Low (<60%)'

    def test_market_names_mapping(self):
        """MARKET_NAMES should map market keys to display names."""
        dashboard = ResultDashboard()

        assert dashboard.MARKET_NAMES['player_reception_yds'] == 'Receiving Yards'
        assert dashboard.MARKET_NAMES['player_rush_yds'] == 'Rushing Yards'
        assert dashboard.MARKET_NAMES['player_pass_yds'] == 'Passing Yards'
        assert dashboard.MARKET_NAMES['player_receptions'] == 'Receptions'


class TestResultDashboardAddToBreakdown:
    """Tests for the _add_to_breakdown helper method."""

    def test_add_to_breakdown_new_category(self):
        """_add_to_breakdown should create new category if needed."""
        dashboard = ResultDashboard()
        breakdown = {}

        # Create mock trade
        trade = MagicMock()
        trade.stake = 100.0
        trade.won = True
        trade.profit_loss = 91.0  # -110 odds win

        dashboard._add_to_breakdown(breakdown, 'OVER', trade)

        assert 'OVER' in breakdown
        assert breakdown['OVER'].category == 'OVER'
        assert breakdown['OVER'].total_bets == 1
        assert breakdown['OVER'].wins == 1
        assert breakdown['OVER'].losses == 0
        assert breakdown['OVER'].total_staked == 100.0
        assert breakdown['OVER'].total_profit == 91.0

    def test_add_to_breakdown_existing_category(self):
        """_add_to_breakdown should update existing category."""
        dashboard = ResultDashboard()
        breakdown = {
            'OVER': BreakdownStats(
                category='OVER',
                total_bets=5,
                wins=3,
                losses=2,
                total_staked=500.0,
                total_profit=100.0,
            )
        }

        # Add a loss
        trade = MagicMock()
        trade.stake = 100.0
        trade.won = False
        trade.profit_loss = -100.0

        dashboard._add_to_breakdown(breakdown, 'OVER', trade)

        assert breakdown['OVER'].total_bets == 6
        assert breakdown['OVER'].wins == 3
        assert breakdown['OVER'].losses == 3
        assert breakdown['OVER'].total_staked == 600.0
        assert breakdown['OVER'].total_profit == 0.0

    def test_add_to_breakdown_push(self):
        """_add_to_breakdown should handle pushes."""
        dashboard = ResultDashboard()
        breakdown = {}

        trade = MagicMock()
        trade.stake = 100.0
        trade.won = None  # Push
        trade.profit_loss = 0.0

        dashboard._add_to_breakdown(breakdown, 'OVER', trade)

        assert breakdown['OVER'].total_bets == 1
        assert breakdown['OVER'].wins == 0
        assert breakdown['OVER'].losses == 0
        assert breakdown['OVER'].pushes == 1

    def test_add_to_breakdown_no_profit_loss(self):
        """_add_to_breakdown should handle None profit_loss."""
        dashboard = ResultDashboard()
        breakdown = {}

        trade = MagicMock()
        trade.stake = 100.0
        trade.won = True
        trade.profit_loss = None

        dashboard._add_to_breakdown(breakdown, 'OVER', trade)

        assert breakdown['OVER'].total_bets == 1
        assert breakdown['OVER'].wins == 1
        assert breakdown['OVER'].total_profit == 0.0  # None treated as 0
