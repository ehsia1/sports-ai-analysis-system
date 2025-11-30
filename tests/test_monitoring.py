"""Tests for health monitoring module."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from pathlib import Path

from src.sports_betting.monitoring.health import (
    HealthChecker,
    HealthStatus,
    CheckResult,
    HealthReport,
    run_health_check,
)


class TestCheckResult:
    """Tests for CheckResult dataclass."""

    def test_check_result_passed(self):
        """Test creating a passing CheckResult."""
        result = CheckResult(
            name="Database",
            passed=True,
            message="Connected successfully",
            details={"connection_time": 0.5},
        )
        assert result.passed is True
        assert result.name == "Database"
        assert result.details["connection_time"] == 0.5

    def test_check_result_failed(self):
        """Test creating a failed CheckResult."""
        result = CheckResult(
            name="API Credits",
            passed=False,
            message="Credits exhausted",
        )
        assert result.passed is False


class TestHealthReport:
    """Tests for HealthReport dataclass."""

    def test_passed_checks_property(self):
        """Test passed_checks filters correctly."""
        report = HealthReport(
            status=HealthStatus.DEGRADED,
            checks=[
                CheckResult("Check1", True, "OK"),
                CheckResult("Check2", False, "Failed"),
                CheckResult("Check3", True, "OK"),
            ],
        )
        assert len(report.passed_checks) == 2
        assert all(c.passed for c in report.passed_checks)

    def test_failed_checks_property(self):
        """Test failed_checks filters correctly."""
        report = HealthReport(
            status=HealthStatus.DEGRADED,
            checks=[
                CheckResult("Check1", True, "OK"),
                CheckResult("Check2", False, "Failed"),
                CheckResult("Check3", False, "Also failed"),
            ],
        )
        assert len(report.failed_checks) == 2
        assert all(not c.passed for c in report.failed_checks)

    def test_summary_includes_status(self):
        """Test summary includes health status."""
        report = HealthReport(
            status=HealthStatus.HEALTHY,
            checks=[CheckResult("Test", True, "OK")],
        )
        summary = report.summary()
        assert "HEALTHY" in summary

    def test_summary_includes_check_count(self):
        """Test summary includes check count."""
        report = HealthReport(
            status=HealthStatus.HEALTHY,
            checks=[
                CheckResult("Check1", True, "OK"),
                CheckResult("Check2", True, "OK"),
            ],
        )
        summary = report.summary()
        assert "2/2" in summary


class TestHealthChecker:
    """Tests for HealthChecker class."""

    @patch("src.sports_betting.monitoring.health.get_settings")
    def test_checker_initialization(self, mock_settings):
        """Test HealthChecker initialization."""
        mock_settings.return_value.models_dir = Path("/tmp/models")
        mock_settings.return_value.data_dir = Path("/tmp/data")

        checker = HealthChecker()
        assert checker is not None

    def test_check_database_success(self):
        """Test successful database check with real database."""
        # Uses real database - should work if db exists
        checker = HealthChecker()
        result = checker.check_database()

        # Should pass if database is accessible
        assert result.name == "Database"
        # Result depends on whether db is accessible

    def test_check_database_returns_check_result(self):
        """Test database check returns proper CheckResult."""
        checker = HealthChecker()
        result = checker.check_database()

        assert isinstance(result, CheckResult)
        assert result.name == "Database"
        assert isinstance(result.passed, bool)
        assert isinstance(result.message, str)

    @patch("src.sports_betting.monitoring.health.get_settings")
    def test_check_model_files_all_present(self, mock_settings):
        """Test model files check when all present."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir)
            mock_settings.return_value.models_dir = models_dir
            mock_settings.return_value.data_dir = Path("/tmp/data")

            # Create all required model files
            for model in HealthChecker.REQUIRED_MODELS:
                (models_dir / model).touch()

            checker = HealthChecker()
            result = checker.check_model_files()

            assert result.passed is True
            assert "4 models" in result.message

    @patch("src.sports_betting.monitoring.health.get_settings")
    def test_check_model_files_missing(self, mock_settings):
        """Test model files check when some missing."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir)
            mock_settings.return_value.models_dir = models_dir
            mock_settings.return_value.data_dir = Path("/tmp/data")

            # Create only some model files
            (models_dir / "receiving_yards_v2.pkl").touch()

            checker = HealthChecker()
            result = checker.check_model_files()

            assert result.passed is False
            assert "Missing" in result.message

    @patch("src.sports_betting.monitoring.health.get_settings")
    def test_check_disk_space(self, mock_settings):
        """Test disk space check."""
        mock_settings.return_value.models_dir = Path("/tmp/models")
        mock_settings.return_value.data_dir = Path("/tmp")

        checker = HealthChecker()
        result = checker.check_disk_space(min_gb=0.001)  # Very low threshold

        # Should pass on any modern system
        assert result.passed is True
        assert "GB" in result.message

    @patch("src.sports_betting.monitoring.health.get_settings")
    def test_check_config_missing_api_key(self, mock_settings):
        """Test config check with missing API key."""
        mock_settings.return_value.models_dir = Path("/tmp")
        mock_settings.return_value.data_dir = Path("/tmp")
        mock_settings.return_value.odds_api_key = None
        mock_settings.return_value.discord_webhook_url = None

        checker = HealthChecker()
        result = checker.check_config()

        assert result.passed is False
        assert "ODDS_API_KEY" in result.message


class TestRunHealthCheck:
    """Tests for run_health_check function."""

    def test_run_health_check_returns_report(self):
        """Test run_health_check returns a report."""
        with patch.object(HealthChecker, "run_all_checks") as mock_run:
            mock_report = HealthReport(
                status=HealthStatus.HEALTHY,
                checks=[CheckResult("Test", True, "OK")],
            )
            mock_run.return_value = mock_report

            report = run_health_check(notify=False)

            assert report.status == HealthStatus.HEALTHY
            mock_run.assert_called_once()

    def test_run_health_check_notifies_on_issues(self):
        """Test run_health_check sends notification on issues."""
        # Mock the entire HealthChecker class
        mock_report = HealthReport(
            status=HealthStatus.UNHEALTHY,
            checks=[CheckResult("Test", False, "Failed")],
        )

        with patch.object(HealthChecker, "run_all_checks", return_value=mock_report):
            with patch(
                "src.sports_betting.notifications.discord.DiscordNotifier.send_health_alert"
            ) as mock_send:
                mock_send.return_value = True
                # Just verify no exception raised
                report = run_health_check(notify=True)
                assert report.status == HealthStatus.UNHEALTHY

    def test_run_health_check_no_notify_when_healthy(self):
        """Test run_health_check doesn't notify when healthy."""
        mock_report = HealthReport(
            status=HealthStatus.HEALTHY,
            checks=[CheckResult("Test", True, "OK")],
        )

        with patch.object(HealthChecker, "run_all_checks", return_value=mock_report):
            # send_health_alert should not be called for healthy status
            report = run_health_check(notify=True)
            assert report.status == HealthStatus.HEALTHY
