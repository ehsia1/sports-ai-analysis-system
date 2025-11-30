"""Tests for Discord notifications module."""

import pytest
from unittest.mock import patch, MagicMock

from src.sports_betting.notifications.discord import (
    DiscordNotifier,
    EdgeAlert,
    WeeklyResult,
    send_edge_alert,
    send_weekly_results,
    send_health_alert,
)


class TestEdgeAlert:
    """Tests for EdgeAlert dataclass."""

    def test_edge_alert_creation(self):
        """Test creating an EdgeAlert."""
        alert = EdgeAlert(
            player_name="Ja'Marr Chase",
            stat_type="receiving_yards",
            line=85.5,
            prediction=102.3,
            edge_pct=8.5,
            direction="over",
            confidence=0.75,
            book="DraftKings",
        )
        assert alert.player_name == "Ja'Marr Chase"
        assert alert.stat_type == "receiving_yards"
        assert alert.edge_pct == 8.5
        assert alert.direction == "over"


class TestWeeklyResult:
    """Tests for WeeklyResult dataclass."""

    def test_weekly_result_creation(self):
        """Test creating a WeeklyResult."""
        result = WeeklyResult(
            total_bets=20,
            wins=12,
            losses=7,
            pushes=1,
            roi_pct=8.5,
            profit_loss=85.0,
            best_bet="Chase OVER 85.5",
            worst_bet="Hill UNDER 100.5",
        )
        assert result.total_bets == 20
        assert result.wins == 12
        assert result.roi_pct == 8.5


class TestDiscordNotifier:
    """Tests for DiscordNotifier class."""

    def test_notifier_disabled_without_webhook(self):
        """Test notifier is disabled when no webhook URL."""
        with patch(
            "src.sports_betting.notifications.discord.get_settings"
        ) as mock_settings:
            mock_settings.return_value.discord_webhook_url = None
            mock_settings.return_value.discord_notify_min_edge = 5.0

            notifier = DiscordNotifier()
            assert notifier.enabled is False

    def test_notifier_enabled_with_webhook(self):
        """Test notifier is enabled with webhook URL."""
        with patch(
            "src.sports_betting.notifications.discord.get_settings"
        ) as mock_settings:
            mock_settings.return_value.discord_webhook_url = "https://discord.com/webhook/test"
            mock_settings.return_value.discord_notify_min_edge = 5.0

            notifier = DiscordNotifier()
            assert notifier.enabled is True

    def test_send_message_disabled(self):
        """Test send_message returns False when disabled."""
        with patch(
            "src.sports_betting.notifications.discord.get_settings"
        ) as mock_settings:
            mock_settings.return_value.discord_webhook_url = None
            mock_settings.return_value.discord_notify_min_edge = 5.0

            notifier = DiscordNotifier()
            result = notifier.send_message("Test message")
            assert result is False

    @patch("src.sports_betting.notifications.discord.requests.post")
    def test_send_message_success(self, mock_post):
        """Test successful message send."""
        mock_post.return_value.status_code = 204
        mock_post.return_value.raise_for_status = MagicMock()

        notifier = DiscordNotifier(webhook_url="https://discord.com/webhook/test")
        result = notifier.send_message("Test message")

        assert result is True
        mock_post.assert_called_once()

    def test_edge_alert_below_threshold_skipped(self):
        """Test edge alert below threshold is skipped."""
        with patch(
            "src.sports_betting.notifications.discord.get_settings"
        ) as mock_settings:
            mock_settings.return_value.discord_webhook_url = "https://discord.com/webhook/test"
            mock_settings.return_value.discord_notify_min_edge = 10.0

            notifier = DiscordNotifier()

            alert = EdgeAlert(
                player_name="Test Player",
                stat_type="receiving_yards",
                line=80.0,
                prediction=85.0,
                edge_pct=5.0,  # Below 10% threshold
                direction="over",
                confidence=0.7,
            )

            result = notifier.send_edge_alert(alert)
            assert result is False


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_send_edge_alert_creates_alert(self):
        """Test send_edge_alert creates proper EdgeAlert."""
        with patch(
            "src.sports_betting.notifications.discord._get_notifier"
        ) as mock_get:
            mock_notifier = MagicMock()
            mock_notifier.send_edge_alert.return_value = True
            mock_get.return_value = mock_notifier

            result = send_edge_alert(
                player_name="Test Player",
                stat_type="rushing_yards",
                line=70.0,
                prediction=85.0,
                edge_pct=10.0,
                direction="over",
                confidence=0.8,
                book="FanDuel",
            )

            assert mock_notifier.send_edge_alert.called
            alert_arg = mock_notifier.send_edge_alert.call_args[0][0]
            assert alert_arg.player_name == "Test Player"
            assert alert_arg.stat_type == "rushing_yards"
            assert alert_arg.edge_pct == 10.0

    def test_send_weekly_results_creates_result(self):
        """Test send_weekly_results creates proper WeeklyResult."""
        with patch(
            "src.sports_betting.notifications.discord._get_notifier"
        ) as mock_get:
            mock_notifier = MagicMock()
            mock_notifier.send_weekly_results.return_value = True
            mock_get.return_value = mock_notifier

            result = send_weekly_results(
                week=13,
                season=2025,
                total_bets=15,
                wins=10,
                losses=5,
                pushes=0,
                roi_pct=12.5,
                profit_loss=125.0,
            )

            assert mock_notifier.send_weekly_results.called

    def test_send_health_alert_passes_args(self):
        """Test send_health_alert passes arguments correctly."""
        with patch(
            "src.sports_betting.notifications.discord._get_notifier"
        ) as mock_get:
            mock_notifier = MagicMock()
            mock_notifier.send_health_alert.return_value = True
            mock_get.return_value = mock_notifier

            result = send_health_alert(
                status="degraded",
                issues=["Database slow", "API credits low"],
                healthy_checks=["Models present", "Disk OK"],
            )

            mock_notifier.send_health_alert.assert_called_once_with(
                "degraded",
                ["Database slow", "API credits low"],
                ["Models present", "Disk OK"],
            )
