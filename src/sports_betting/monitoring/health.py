"""System health monitoring for NFL betting platform.

Provides comprehensive health checks for:
- Database connectivity
- API credits remaining
- Model files availability
- Recent activity verification
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional

from src.sports_betting.config import get_settings
from src.sports_betting.utils.logging import get_logger

logger = get_logger(__name__)


class HealthStatus(Enum):
    """Overall system health status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class CheckResult:
    """Result of a single health check."""

    name: str
    passed: bool
    message: str
    details: Optional[dict] = None


@dataclass
class HealthReport:
    """Complete health report."""

    status: HealthStatus
    checks: list[CheckResult] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def passed_checks(self) -> list[CheckResult]:
        """Get all passing checks."""
        return [c for c in self.checks if c.passed]

    @property
    def failed_checks(self) -> list[CheckResult]:
        """Get all failing checks."""
        return [c for c in self.checks if not c.passed]

    def summary(self) -> str:
        """Generate a summary string."""
        lines = [
            f"Health Status: {self.status.value.upper()}",
            f"Time: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Passed: {len(self.passed_checks)}/{len(self.checks)}",
            "",
        ]

        if self.passed_checks:
            lines.append("✅ Passing:")
            for check in self.passed_checks:
                lines.append(f"   • {check.name}: {check.message}")

        if self.failed_checks:
            lines.append("")
            lines.append("❌ Failing:")
            for check in self.failed_checks:
                lines.append(f"   • {check.name}: {check.message}")

        return "\n".join(lines)


class HealthChecker:
    """System health checker."""

    # Required model files
    REQUIRED_MODELS = [
        "receiving_yards_enhanced.pkl",  # Main receiving yards predictor
        "rushing_yards_v2.pkl",
        "passing_yards_v2.pkl",
        "receptions_v2.pkl",
    ]

    def __init__(self):
        """Initialize health checker."""
        self.settings = get_settings()

    def check_database(self) -> CheckResult:
        """Check database connectivity and basic integrity."""
        try:
            from src.sports_betting.database import get_session
            from src.sports_betting.database.models import Prop

            with get_session() as session:
                # Try a simple query
                count = session.query(Prop).count()

            return CheckResult(
                name="Database",
                passed=True,
                message=f"Connected, {count} props records",
                details={"props_count": count},
            )

        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return CheckResult(
                name="Database",
                passed=False,
                message=f"Connection failed: {str(e)[:50]}",
            )

    def check_api_credits(self) -> CheckResult:
        """Check remaining API credits.

        Note: The Odds API doesn't have a direct endpoint for remaining credits,
        so we estimate based on API request tracking in our database.
        """
        try:
            from src.sports_betting.database import get_session
            from src.sports_betting.database.models import ApiRequest
            from sqlalchemy import func

            # Count actual API requests this month
            first_of_month = datetime.now().replace(day=1, hour=0, minute=0, second=0)

            with get_session() as session:
                # Count API requests to odds_api this month
                monthly_requests = (
                    session.query(func.count(ApiRequest.id))
                    .filter(ApiRequest.api_source == "odds_api")
                    .filter(ApiRequest.created_at >= first_of_month)
                    .scalar()
                ) or 0

            # Each player props request uses ~4 credits
            estimated_used = monthly_requests * 4
            remaining = self.settings.odds_api_monthly_limit - estimated_used

            if remaining > 100:
                return CheckResult(
                    name="API Credits",
                    passed=True,
                    message=f"~{remaining} credits remaining (estimated)",
                    details={
                        "estimated_used": estimated_used,
                        "monthly_limit": self.settings.odds_api_monthly_limit,
                    },
                )
            elif remaining > 0:
                return CheckResult(
                    name="API Credits",
                    passed=True,  # Still passing but low
                    message=f"LOW: ~{remaining} credits remaining",
                    details={"estimated_used": estimated_used, "remaining": remaining},
                )
            else:
                return CheckResult(
                    name="API Credits",
                    passed=False,
                    message="Credits exhausted for this month",
                    details={"estimated_used": estimated_used},
                )

        except Exception as e:
            logger.error(f"API credits check failed: {e}")
            return CheckResult(
                name="API Credits",
                passed=False,
                message=f"Check failed: {str(e)[:50]}",
            )

    def check_model_files(self) -> CheckResult:
        """Check that all required model files exist."""
        missing = []
        found = []

        for model_file in self.REQUIRED_MODELS:
            model_path = self.settings.models_dir / model_file
            if model_path.exists():
                found.append(model_file)
            else:
                missing.append(model_file)

        if not missing:
            return CheckResult(
                name="Model Files",
                passed=True,
                message=f"All {len(self.REQUIRED_MODELS)} models present",
                details={"models": found},
            )
        else:
            return CheckResult(
                name="Model Files",
                passed=False,
                message=f"Missing: {', '.join(missing)}",
                details={"missing": missing, "found": found},
            )

    def check_recent_activity(self, days: int = 7) -> CheckResult:
        """Check for recent system activity.

        Args:
            days: Number of days to look back
        """
        try:
            from src.sports_betting.database import get_session
            from src.sports_betting.database.models import Prop, Prediction
            from sqlalchemy import func

            cutoff = datetime.now() - timedelta(days=days)

            with get_session() as session:
                # Check for recent props
                recent_props = (
                    session.query(func.count(Prop.id))
                    .filter(Prop.timestamp >= cutoff)
                    .scalar()
                )

                # Check for recent predictions
                recent_predictions = (
                    session.query(func.count(Prediction.id))
                    .filter(Prediction.created_at >= cutoff)
                    .scalar()
                )

            if recent_props > 0 and recent_predictions > 0:
                return CheckResult(
                    name="Recent Activity",
                    passed=True,
                    message=f"{recent_props} props, {recent_predictions} predictions in {days}d",
                    details={
                        "recent_props": recent_props,
                        "recent_predictions": recent_predictions,
                        "days": days,
                    },
                )
            elif recent_props > 0 or recent_predictions > 0:
                return CheckResult(
                    name="Recent Activity",
                    passed=True,  # Partial activity
                    message=f"Partial: {recent_props} props, {recent_predictions} predictions",
                    details={
                        "recent_props": recent_props,
                        "recent_predictions": recent_predictions,
                    },
                )
            else:
                return CheckResult(
                    name="Recent Activity",
                    passed=False,
                    message=f"No activity in the last {days} days",
                    details={"days": days},
                )

        except Exception as e:
            logger.error(f"Recent activity check failed: {e}")
            return CheckResult(
                name="Recent Activity",
                passed=False,
                message=f"Check failed: {str(e)[:50]}",
            )

    def check_disk_space(self, min_gb: float = 1.0) -> CheckResult:
        """Check available disk space.

        Args:
            min_gb: Minimum required free space in GB
        """
        try:
            import shutil

            total, used, free = shutil.disk_usage(self.settings.data_dir)
            free_gb = free / (1024**3)

            if free_gb >= min_gb:
                return CheckResult(
                    name="Disk Space",
                    passed=True,
                    message=f"{free_gb:.1f} GB free",
                    details={"free_gb": free_gb, "min_gb": min_gb},
                )
            else:
                return CheckResult(
                    name="Disk Space",
                    passed=False,
                    message=f"Low: {free_gb:.1f} GB free (need {min_gb} GB)",
                    details={"free_gb": free_gb, "min_gb": min_gb},
                )

        except Exception as e:
            logger.error(f"Disk space check failed: {e}")
            return CheckResult(
                name="Disk Space",
                passed=False,
                message=f"Check failed: {str(e)[:50]}",
            )

    def check_config(self) -> CheckResult:
        """Check that configuration is valid."""
        issues = []

        if not self.settings.odds_api_key:
            issues.append("ODDS_API_KEY not set")

        if not self.settings.discord_webhook_url:
            issues.append("DISCORD_WEBHOOK_URL not set (notifications disabled)")

        # Check paths exist
        if not self.settings.data_dir.exists():
            issues.append(f"Data dir missing: {self.settings.data_dir}")

        if not self.settings.models_dir.exists():
            issues.append(f"Models dir missing: {self.settings.models_dir}")

        if not issues:
            return CheckResult(
                name="Configuration",
                passed=True,
                message="All required settings configured",
            )
        elif "ODDS_API_KEY" in str(issues):
            return CheckResult(
                name="Configuration",
                passed=False,
                message="; ".join(issues),
                details={"issues": issues},
            )
        else:
            # Non-critical issues (like Discord not configured)
            return CheckResult(
                name="Configuration",
                passed=True,
                message=f"OK with warnings: {'; '.join(issues)}",
                details={"warnings": issues},
            )

    def run_all_checks(self) -> HealthReport:
        """Run all health checks and return report.

        Returns:
            HealthReport with all check results
        """
        logger.info("Running health checks...")

        checks = [
            self.check_database(),
            self.check_api_credits(),
            self.check_model_files(),
            self.check_recent_activity(),
            self.check_disk_space(),
            self.check_config(),
        ]

        # Determine overall status
        failed_count = sum(1 for c in checks if not c.passed)

        if failed_count == 0:
            status = HealthStatus.HEALTHY
        elif failed_count <= 2:
            status = HealthStatus.DEGRADED
        else:
            status = HealthStatus.UNHEALTHY

        report = HealthReport(status=status, checks=checks)
        logger.info(f"Health check complete: {status.value}")

        return report


def run_health_check(notify: bool = False) -> HealthReport:
    """Run health check and optionally notify.

    Args:
        notify: If True, send Discord notification on issues

    Returns:
        HealthReport
    """
    checker = HealthChecker()
    report = checker.run_all_checks()

    if notify and report.status != HealthStatus.HEALTHY:
        try:
            from src.sports_betting.notifications import send_health_alert

            send_health_alert(
                status=report.status.value,
                issues=[c.message for c in report.failed_checks],
                healthy_checks=[c.message for c in report.passed_checks],
            )
        except Exception as e:
            logger.error(f"Failed to send health notification: {e}")

    return report
