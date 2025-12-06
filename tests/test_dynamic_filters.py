"""Tests for the dynamic filters module."""

import json
import pytest
import tempfile
from pathlib import Path
from datetime import datetime

from src.sports_betting.analysis.dynamic_filters import (
    FilterRule,
    DynamicFilterConfig,
    CategoryStats,
    DynamicFilterService,
)


class TestFilterRule:
    """Tests for the FilterRule dataclass."""

    def test_filter_rule_creation(self):
        """FilterRule should store all attributes correctly."""
        rule = FilterRule(
            dimension='position_direction',
            category='WR_OVER',
            action='skip',
            win_rate=0.35,
            sample_size=25,
            total_profit=-500.0,
            reason='Win rate 35% (9-16), P&L: $-500',
        )

        assert rule.dimension == 'position_direction'
        assert rule.category == 'WR_OVER'
        assert rule.action == 'skip'
        assert rule.win_rate == 0.35
        assert rule.sample_size == 25
        assert rule.total_profit == -500.0
        assert 'Win rate' in rule.reason

    def test_filter_rule_to_dict(self):
        """FilterRule.to_dict() should serialize correctly."""
        rule = FilterRule(
            dimension='tier_direction',
            category='elite_UNDER',
            action='prioritize',
            win_rate=0.65,
            sample_size=40,
            total_profit=800.0,
            reason='Win rate 65%',
        )

        data = rule.to_dict()

        assert isinstance(data, dict)
        assert data['dimension'] == 'tier_direction'
        assert data['category'] == 'elite_UNDER'
        assert data['action'] == 'prioritize'
        assert data['win_rate'] == 0.65

    def test_filter_rule_from_dict(self):
        """FilterRule.from_dict() should deserialize correctly."""
        data = {
            'dimension': 'market',
            'category': 'player_reception_yds',
            'action': 'skip',
            'win_rate': 0.30,
            'sample_size': 50,
            'total_profit': -1000.0,
            'reason': 'Poor performance',
        }

        rule = FilterRule.from_dict(data)

        assert rule.dimension == 'market'
        assert rule.category == 'player_reception_yds'
        assert rule.action == 'skip'
        assert rule.sample_size == 50

    def test_filter_rule_roundtrip(self):
        """FilterRule should survive serialization roundtrip."""
        original = FilterRule(
            dimension='direction',
            category='OVER',
            action='skip',
            win_rate=0.45,
            sample_size=100,
            total_profit=-200.0,
            reason='test',
        )

        data = original.to_dict()
        restored = FilterRule.from_dict(data)

        assert restored.dimension == original.dimension
        assert restored.category == original.category
        assert restored.action == original.action
        assert restored.win_rate == original.win_rate
        assert restored.sample_size == original.sample_size


class TestDynamicFilterConfig:
    """Tests for the DynamicFilterConfig dataclass."""

    def test_config_defaults(self):
        """DynamicFilterConfig should have sensible defaults."""
        config = DynamicFilterConfig()

        assert config.min_sample_size == 15
        assert config.skip_below_win_rate == 0.35
        assert config.prioritize_above_win_rate == 0.55
        assert config.skip_below_profit == -300.0
        assert 'direction' in config.dimensions
        assert 'position' in config.dimensions
        assert 'tier_direction' in config.dimensions

    def test_config_custom_values(self):
        """DynamicFilterConfig should accept custom values."""
        config = DynamicFilterConfig(
            min_sample_size=20,
            skip_below_win_rate=0.40,
            prioritize_above_win_rate=0.60,
        )

        assert config.min_sample_size == 20
        assert config.skip_below_win_rate == 0.40
        assert config.prioritize_above_win_rate == 0.60


class TestCategoryStats:
    """Tests for the CategoryStats dataclass."""

    def test_category_stats_defaults(self):
        """CategoryStats should have zero defaults."""
        stats = CategoryStats()

        assert stats.wins == 0
        assert stats.losses == 0
        assert stats.pending == 0
        assert stats.total_profit == 0.0

    def test_category_stats_total(self):
        """CategoryStats.total should sum wins and losses."""
        stats = CategoryStats(wins=10, losses=15)

        assert stats.total == 25

    def test_category_stats_win_rate(self):
        """CategoryStats.win_rate should calculate correctly."""
        stats = CategoryStats(wins=3, losses=7)

        assert stats.win_rate == 0.3  # 3/10

    def test_category_stats_win_rate_zero_total(self):
        """CategoryStats.win_rate should return 0 when no bets."""
        stats = CategoryStats()

        assert stats.win_rate == 0.0

    def test_category_stats_avg_profit(self):
        """CategoryStats.avg_profit_per_bet should calculate correctly."""
        stats = CategoryStats(wins=5, losses=5, total_profit=200.0)

        assert stats.avg_profit_per_bet == 20.0  # 200/10

    def test_category_stats_avg_profit_zero_total(self):
        """CategoryStats.avg_profit_per_bet should return 0 when no bets."""
        stats = CategoryStats(total_profit=100.0)

        assert stats.avg_profit_per_bet == 0.0


class TestDynamicFilterService:
    """Tests for the DynamicFilterService class."""

    def test_service_initialization(self):
        """DynamicFilterService should initialize with default config."""
        service = DynamicFilterService()

        assert service.config is not None
        assert service.config.min_sample_size == 15
        assert service._rules == []
        assert service._last_update is None

    def test_service_custom_config(self):
        """DynamicFilterService should accept custom config."""
        config = DynamicFilterConfig(min_sample_size=30)
        service = DynamicFilterService(config=config)

        assert service.config.min_sample_size == 30

    def test_is_elite_wr(self):
        """_is_elite should identify elite WRs."""
        service = DynamicFilterService()

        assert service._is_elite("Ja'Marr Chase", 'WR') is True
        assert service._is_elite('CeeDee Lamb', 'WR') is True
        assert service._is_elite('Random Player', 'WR') is False

    def test_is_elite_rb(self):
        """_is_elite should identify elite RBs."""
        service = DynamicFilterService()

        assert service._is_elite('Derrick Henry', 'RB') is True
        assert service._is_elite('Saquon Barkley', 'RB') is True
        assert service._is_elite('Unknown RB', 'RB') is False

    def test_is_elite_te(self):
        """_is_elite should identify elite TEs."""
        service = DynamicFilterService()

        assert service._is_elite('Travis Kelce', 'TE') is True
        assert service._is_elite('George Kittle', 'TE') is True
        assert service._is_elite('Random TE', 'TE') is False

    def test_is_elite_qb(self):
        """_is_elite should identify elite QBs."""
        service = DynamicFilterService()

        assert service._is_elite('Patrick Mahomes', 'QB') is True
        assert service._is_elite('Josh Allen', 'QB') is True
        assert service._is_elite('Backup QB', 'QB') is False

    def test_is_elite_unknown_position(self):
        """_is_elite should handle unknown positions."""
        service = DynamicFilterService()

        assert service._is_elite('Anyone', 'K') is False
        assert service._is_elite('Anyone', '') is False

    def test_get_tier_elite(self):
        """_get_tier should return 'elite' for elite players."""
        service = DynamicFilterService()

        assert service._get_tier("Ja'Marr Chase", 'WR') == 'elite'
        assert service._get_tier('Derrick Henry', 'RB') == 'elite'

    def test_get_tier_other(self):
        """_get_tier should return 'other' for non-elite players."""
        service = DynamicFilterService()

        assert service._get_tier('Unknown WR', 'WR') == 'other'
        assert service._get_tier('Backup RB', 'RB') == 'other'

    def test_get_skip_categories_empty(self):
        """get_skip_categories should return empty dict when no rules."""
        service = DynamicFilterService()

        result = service.get_skip_categories()

        assert result == {}

    def test_get_skip_categories_with_rules(self):
        """get_skip_categories should return skip categories."""
        service = DynamicFilterService()
        service._rules = [
            FilterRule(
                dimension='position_direction',
                category='TE_OVER',
                action='skip',
                win_rate=0.30,
                sample_size=20,
                total_profit=-400.0,
                reason='test',
            ),
            FilterRule(
                dimension='tier_direction',
                category='elite_UNDER',
                action='skip',
                win_rate=0.35,
                sample_size=25,
                total_profit=-500.0,
                reason='test',
            ),
            FilterRule(
                dimension='direction',
                category='OVER',
                action='prioritize',  # Not a skip rule
                win_rate=0.60,
                sample_size=30,
                total_profit=600.0,
                reason='test',
            ),
        ]

        result = service.get_skip_categories()

        assert 'position_direction' in result
        assert 'TE_OVER' in result['position_direction']
        assert 'tier_direction' in result
        assert 'elite_UNDER' in result['tier_direction']
        assert 'direction' not in result  # OVER is prioritize, not skip

    def test_get_prioritize_categories(self):
        """get_prioritize_categories should return prioritize categories."""
        service = DynamicFilterService()
        service._rules = [
            FilterRule(
                dimension='direction',
                category='UNDER',
                action='prioritize',
                win_rate=0.65,
                sample_size=100,
                total_profit=2000.0,
                reason='test',
            ),
            FilterRule(
                dimension='position',
                category='RB',
                action='skip',  # Not prioritize
                win_rate=0.30,
                sample_size=20,
                total_profit=-300.0,
                reason='test',
            ),
        ]

        result = service.get_prioritize_categories()

        assert 'direction' in result
        assert 'UNDER' in result['direction']
        assert 'position' not in result

    def test_should_skip_no_rules(self):
        """should_skip should return False when no rules exist."""
        service = DynamicFilterService()

        should_skip, reason = service.should_skip(
            direction='over',
            position='WR',
            market='player_reception_yds',
            player_name='Test Player',
        )

        assert should_skip is False
        assert reason is None

    def test_should_skip_matching_rule(self):
        """should_skip should return True when matching skip rule exists."""
        service = DynamicFilterService()
        service._rules = [
            FilterRule(
                dimension='position_direction',
                category='TE_OVER',
                action='skip',
                win_rate=0.30,
                sample_size=20,
                total_profit=-400.0,
                reason='Win rate 30%',
            ),
        ]

        should_skip, reason = service.should_skip(
            direction='over',
            position='TE',
            market='player_reception_yds',
            player_name='Random TE',
        )

        assert should_skip is True
        assert reason is not None
        assert 'TE_OVER' in reason

    def test_should_skip_non_matching(self):
        """should_skip should return False when no matching rule."""
        service = DynamicFilterService()
        service._rules = [
            FilterRule(
                dimension='position_direction',
                category='TE_OVER',
                action='skip',
                win_rate=0.30,
                sample_size=20,
                total_profit=-400.0,
                reason='test',
            ),
        ]

        # WR OVER should not be skipped
        should_skip, reason = service.should_skip(
            direction='over',
            position='WR',
            market='player_reception_yds',
            player_name='Some WR',
        )

        assert should_skip is False
        assert reason is None

    def test_should_skip_tier_based(self):
        """should_skip should work with tier-based rules."""
        service = DynamicFilterService()
        service._rules = [
            FilterRule(
                dimension='tier_direction',
                category='elite_UNDER',
                action='skip',
                win_rate=0.25,
                sample_size=30,
                total_profit=-800.0,
                reason='Elite UNDERs lose',
            ),
        ]

        # Elite player UNDER should be skipped
        should_skip, reason = service.should_skip(
            direction='under',
            position='WR',
            market='player_reception_yds',
            player_name="Ja'Marr Chase",  # Elite WR
        )

        assert should_skip is True

        # Non-elite player UNDER should NOT be skipped
        should_skip, reason = service.should_skip(
            direction='under',
            position='WR',
            market='player_reception_yds',
            player_name='Random WR',
        )

        assert should_skip is False

    def test_get_confidence_boost_no_rules(self):
        """get_confidence_boost should return 1.0 when no prioritize rules."""
        service = DynamicFilterService()

        boost = service.get_confidence_boost(
            direction='over',
            position='WR',
            market='player_reception_yds',
            player_name='Test',
        )

        assert boost == 1.0

    def test_get_confidence_boost_with_match(self):
        """get_confidence_boost should return >1.0 when matching prioritize rules."""
        service = DynamicFilterService()
        service._rules = [
            FilterRule(
                dimension='direction',
                category='UNDER',
                action='prioritize',
                win_rate=0.70,
                sample_size=100,
                total_profit=3000.0,
                reason='test',
            ),
        ]

        boost = service.get_confidence_boost(
            direction='under',
            position='WR',
            market='player_reception_yds',
            player_name='Test',
        )

        assert boost > 1.0

    def test_get_confidence_boost_cap(self):
        """get_confidence_boost should be capped at 1.10."""
        service = DynamicFilterService()
        # Add many prioritize rules
        for dim in ['direction', 'position', 'market', 'position_direction', 'tier_direction']:
            service._rules.append(FilterRule(
                dimension=dim,
                category='UNDER' if dim == 'direction' else
                         'WR' if dim == 'position' else
                         'player_reception_yds' if dim == 'market' else
                         'WR_UNDER' if dim == 'position_direction' else
                         'other_UNDER',
                action='prioritize',
                win_rate=0.70,
                sample_size=50,
                total_profit=1000.0,
                reason='test',
            ))

        boost = service.get_confidence_boost(
            direction='under',
            position='WR',
            market='player_reception_yds',
            player_name='Random WR',  # other tier
        )

        assert boost <= 1.10

    def test_save_and_load_rules(self):
        """Rules should be saved to and loaded from cache file."""
        # Use a temp directory for cache
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / 'test_filters.json'

            # Create and save rules
            service = DynamicFilterService()
            service._cache_path = cache_path
            service._rules = [
                FilterRule(
                    dimension='direction',
                    category='UNDER',
                    action='prioritize',
                    win_rate=0.70,
                    sample_size=100,
                    total_profit=2000.0,
                    reason='test save',
                ),
            ]
            service._last_update = datetime.now()
            service.save_rules()

            # Verify file exists
            assert cache_path.exists()

            # Load into new service
            new_service = DynamicFilterService()
            new_service._cache_path = cache_path
            success = new_service.load_rules()

            assert success is True
            assert len(new_service._rules) == 1
            assert new_service._rules[0].category == 'UNDER'
            assert new_service._rules[0].action == 'prioritize'

    def test_load_rules_no_file(self):
        """load_rules should return False when cache file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / 'nonexistent.json'

            service = DynamicFilterService()
            service._cache_path = cache_path

            result = service.load_rules()

            assert result is False
            assert service._rules == []

    def test_format_report_no_rules(self):
        """format_report should handle empty rules."""
        service = DynamicFilterService()

        report = service.format_report()

        assert 'No dynamic filter rules' in report

    def test_format_report_with_rules(self):
        """format_report should format rules correctly."""
        service = DynamicFilterService()
        service._rules = [
            FilterRule(
                dimension='position_direction',
                category='TE_OVER',
                action='skip',
                win_rate=0.30,
                sample_size=20,
                total_profit=-400.0,
                reason='test',
            ),
            FilterRule(
                dimension='direction',
                category='UNDER',
                action='prioritize',
                win_rate=0.65,
                sample_size=50,
                total_profit=800.0,
                reason='test',
            ),
        ]
        service._last_update = datetime.now()

        report = service.format_report()

        assert 'DYNAMIC BET FILTERS' in report
        assert 'SKIP FILTERS' in report
        assert 'TE_OVER' in report
        assert 'PRIORITIZE' in report
        assert 'UNDER' in report

    def test_get_filter_summary_for_edge_calculator(self):
        """get_filter_summary_for_edge_calculator should return compat dict."""
        service = DynamicFilterService()
        service._rules = [
            FilterRule(
                dimension='position_direction',
                category='TE_OVER',
                action='skip',
                win_rate=0.30,
                sample_size=20,
                total_profit=-400.0,
                reason='test',
            ),
            FilterRule(
                dimension='tier_direction',
                category='elite_UNDER',
                action='skip',
                win_rate=0.25,
                sample_size=30,
                total_profit=-600.0,
                reason='test',
            ),
        ]
        service._last_update = datetime.now()

        summary = service.get_filter_summary_for_edge_calculator()

        assert summary['dynamic_filters_enabled'] is True
        assert summary['skip_te_over'] is True
        assert summary['skip_elite_under'] is True
        assert summary['skip_rules_count'] == 2
