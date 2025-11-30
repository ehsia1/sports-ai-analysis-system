"""Discord notification system for NFL betting alerts.

Provides rich embeds for edge alerts and weekly results summaries.
"""

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

import requests

from src.sports_betting.config import get_settings
from src.sports_betting.utils.logging import get_logger
from src.sports_betting.utils.retry import retry_with_backoff

logger = get_logger(__name__)


@dataclass
class EdgeAlert:
    """Data class for an edge alert."""

    player_name: str
    stat_type: str
    line: float
    prediction: float
    edge_pct: float
    direction: str  # "over" or "under"
    confidence: float
    book: str = "consensus"
    game: str = ""  # e.g., "SF @ CLE"
    ev_pct: float = 0.0  # Expected value percentage
    reasoning: str = ""  # Short reasoning (1 line)
    reasoning_detailed: str = ""  # Detailed reasoning (multi-line)
    weather: Optional[str] = None  # Weather conditions
    weather_warning: Optional[str] = None  # Weather warning if applicable


@dataclass
class WeeklyResult:
    """Data class for weekly betting results."""

    total_bets: int
    wins: int
    losses: int
    pushes: int
    roi_pct: float
    profit_loss: float
    best_bet: Optional[str] = None
    worst_bet: Optional[str] = None


class DiscordNotifier:
    """Discord webhook notification handler."""

    # Colors for embeds (decimal format)
    COLOR_SUCCESS = 5763719  # Green
    COLOR_WARNING = 16776960  # Yellow
    COLOR_ERROR = 15548997  # Red
    COLOR_INFO = 5793266  # Blue
    COLOR_EDGE = 10181046  # Purple

    def __init__(self, webhook_url: Optional[str] = None):
        """Initialize Discord notifier.

        Args:
            webhook_url: Discord webhook URL. If None, uses settings.
        """
        settings = get_settings()
        self.webhook_url = webhook_url or settings.discord_webhook_url
        self.min_edge = settings.discord_notify_min_edge

        if not self.webhook_url:
            logger.warning("Discord webhook URL not configured")

    @property
    def enabled(self) -> bool:
        """Check if Discord notifications are enabled."""
        return bool(self.webhook_url)

    @retry_with_backoff(max_retries=3, initial_delay=1.0, exceptions=(requests.exceptions.RequestException,))
    def _send_webhook(self, payload: dict) -> bool:
        """Send payload to Discord webhook.

        Args:
            payload: Discord webhook payload

        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            logger.debug("Discord notifications disabled, skipping")
            return False

        try:
            response = requests.post(
                self.webhook_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=10,
            )
            response.raise_for_status()
            logger.info("Discord notification sent successfully")
            return True

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to send Discord notification: {e}")
            raise

    def send_message(self, content: str) -> bool:
        """Send a simple text message.

        Args:
            content: Message text

        Returns:
            True if sent successfully
        """
        return self._send_webhook({"content": content})

    def send_embed(
        self,
        title: str,
        description: str,
        color: int = COLOR_INFO,
        fields: Optional[list[dict]] = None,
        footer: Optional[str] = None,
        thumbnail_url: Optional[str] = None,
    ) -> bool:
        """Send a rich embed message.

        Args:
            title: Embed title
            description: Embed description
            color: Embed color (decimal)
            fields: List of field dicts with name, value, inline
            footer: Footer text
            thumbnail_url: URL for thumbnail image

        Returns:
            True if sent successfully
        """
        embed = {
            "title": title,
            "description": description,
            "color": color,
            "timestamp": datetime.utcnow().isoformat(),
        }

        if fields:
            embed["fields"] = fields

        if footer:
            embed["footer"] = {"text": footer}

        if thumbnail_url:
            embed["thumbnail"] = {"url": thumbnail_url}

        return self._send_webhook({"embeds": [embed]})

    def send_edge_alert(self, alert: EdgeAlert) -> bool:
        """Send an edge alert notification.

        Args:
            alert: EdgeAlert data

        Returns:
            True if sent successfully
        """
        # Skip if edge below threshold
        if alert.edge_pct < self.min_edge:
            logger.debug(
                f"Edge {alert.edge_pct:.1f}% below threshold {self.min_edge}%, skipping"
            )
            return False

        # Determine color based on EV (or edge if EV not provided)
        ev = alert.ev_pct if alert.ev_pct else alert.edge_pct
        if ev >= 10:
            color = self.COLOR_SUCCESS  # High edge = green
        elif ev >= 5:
            color = self.COLOR_EDGE  # Medium edge = purple
        else:
            color = self.COLOR_INFO  # Lower edge = blue

        # Direction emoji
        direction_emoji = "📈" if alert.direction == "over" else "📉"

        # Format stat type for display
        stat_display = alert.stat_type.replace("player_", "").replace("_", " ").title()

        # Build title with game if available
        title = f"{direction_emoji} {alert.player_name}"
        if alert.game:
            title += f" ({alert.game})"

        # Build description with reasoning if available
        if alert.reasoning_detailed:
            description = alert.reasoning_detailed
        else:
            diff_pct = ((alert.prediction - alert.line) / alert.line * 100) if alert.line > 0 else 0
            description = (
                f"**{alert.direction.upper()} {alert.line}** {stat_display}\n\n"
                f"Model predicts **{alert.prediction:.1f}** ({diff_pct:+.1f}% vs line)"
            )

        fields = [
            {"name": "Edge", "value": f"{alert.edge_pct:+.1f}%", "inline": True},
            {"name": "EV", "value": f"{alert.ev_pct:+.1f}%", "inline": True} if alert.ev_pct else None,
            {"name": "Confidence", "value": f"{alert.confidence:.0%}", "inline": True},
        ]
        # Filter out None fields
        fields = [f for f in fields if f is not None]

        # Add weather warning if present
        if alert.weather_warning:
            fields.append({
                "name": "🌨️ Weather Alert",
                "value": f"{alert.weather_warning} (confidence adjusted)",
                "inline": False,
            })
        elif alert.weather and alert.weather != "Dome":
            fields.append({
                "name": "Weather",
                "value": alert.weather,
                "inline": True,
            })

        # Add book
        fields.append({"name": "Book", "value": alert.book, "inline": True})

        footer = f"NFL Betting Bot | {datetime.now().strftime('%Y-%m-%d %H:%M')}"

        return self.send_embed(
            title=title,
            description=description,
            color=color,
            fields=fields,
            footer=footer,
        )

    def send_weekly_results(self, result: WeeklyResult, week: int, season: int) -> bool:
        """Send weekly results summary.

        Args:
            result: WeeklyResult data
            week: NFL week number
            season: NFL season year

        Returns:
            True if sent successfully
        """
        # Determine color based on ROI
        if result.roi_pct > 5:
            color = self.COLOR_SUCCESS
        elif result.roi_pct >= 0:
            color = self.COLOR_INFO
        else:
            color = self.COLOR_ERROR

        # Win/loss emoji
        outcome_emoji = "🏆" if result.roi_pct > 0 else "📊"

        title = f"{outcome_emoji} Week {week} Results"

        # Calculate win rate
        total_decided = result.wins + result.losses
        win_rate = result.wins / total_decided * 100 if total_decided > 0 else 0

        description = f"**{season} NFL Season - Week {week}**\n\n"
        if result.profit_loss >= 0:
            description += f"Profit: **+${result.profit_loss:.2f}** ({result.roi_pct:+.1f}% ROI)"
        else:
            description += f"Loss: **${result.profit_loss:.2f}** ({result.roi_pct:+.1f}% ROI)"

        fields = [
            {
                "name": "Record",
                "value": f"{result.wins}-{result.losses}-{result.pushes}",
                "inline": True,
            },
            {"name": "Win Rate", "value": f"{win_rate:.1f}%", "inline": True},
            {"name": "Total Bets", "value": str(result.total_bets), "inline": True},
        ]

        if result.best_bet:
            fields.append({"name": "Best Bet", "value": result.best_bet, "inline": False})

        if result.worst_bet:
            fields.append(
                {"name": "Worst Bet", "value": result.worst_bet, "inline": False}
            )

        footer = f"NFL Betting Bot | Season ROI tracking"

        return self.send_embed(
            title=title,
            description=description,
            color=color,
            fields=fields,
            footer=footer,
        )

    def send_health_alert(
        self,
        status: str,
        issues: list[str],
        healthy_checks: list[str],
    ) -> bool:
        """Send system health alert.

        Args:
            status: Overall status (healthy, degraded, unhealthy)
            issues: List of issues found
            healthy_checks: List of passing checks

        Returns:
            True if sent successfully
        """
        if status == "healthy":
            color = self.COLOR_SUCCESS
            emoji = "✅"
        elif status == "degraded":
            color = self.COLOR_WARNING
            emoji = "⚠️"
        else:
            color = self.COLOR_ERROR
            emoji = "🚨"

        title = f"{emoji} System Health: {status.upper()}"
        description = "NFL Betting System health check results"

        fields = []

        if healthy_checks:
            fields.append(
                {
                    "name": "✅ Passing",
                    "value": "\n".join(f"• {c}" for c in healthy_checks),
                    "inline": False,
                }
            )

        if issues:
            fields.append(
                {
                    "name": "❌ Issues",
                    "value": "\n".join(f"• {i}" for i in issues),
                    "inline": False,
                }
            )

        footer = f"Health Check | {datetime.now().strftime('%Y-%m-%d %H:%M')}"

        return self.send_embed(
            title=title,
            description=description,
            color=color,
            fields=fields,
            footer=footer,
        )

    def send_prediction_summary(
        self,
        season: int,
        week: int,
        prediction_counts: dict[str, int],
        top_edges: list[EdgeAlert],
    ) -> bool:
        """Send prediction summary after generating predictions.

        Args:
            season: NFL season
            week: NFL week
            prediction_counts: Dict of stat_type -> count
            top_edges: Top edge alerts to highlight

        Returns:
            True if sent successfully
        """
        title = f"🎯 Week {week} Predictions Ready"

        total = sum(prediction_counts.values())
        description = f"**{season} Season - Week {week}**\n\nGenerated {total} total predictions"

        fields = []

        # Add counts by stat type
        stat_summary = []
        for stat_type, count in prediction_counts.items():
            stat_display = stat_type.replace("_", " ").title()
            stat_summary.append(f"• {stat_display}: {count}")

        fields.append(
            {
                "name": "Predictions by Type",
                "value": "\n".join(stat_summary),
                "inline": False,
            }
        )

        # Add top edges if any
        if top_edges:
            edge_lines = []
            for alert in top_edges[:5]:  # Top 5
                direction_emoji = "📈" if alert.direction == "over" else "📉"
                edge_lines.append(
                    f"{direction_emoji} {alert.player_name} - {alert.edge_pct:.1f}% edge"
                )

            fields.append(
                {
                    "name": "Top Edges",
                    "value": "\n".join(edge_lines),
                    "inline": False,
                }
            )

        footer = f"Run 'orchestrate.py pre-game' for full analysis"

        return self.send_embed(
            title=title,
            description=description,
            color=self.COLOR_INFO,
            fields=fields,
            footer=footer,
        )


# Module-level convenience functions
_notifier: Optional[DiscordNotifier] = None


def _get_notifier() -> DiscordNotifier:
    """Get or create singleton notifier instance."""
    global _notifier
    if _notifier is None:
        _notifier = DiscordNotifier()
    return _notifier


def send_edge_alert(
    player_name: str,
    stat_type: str,
    line: float,
    prediction: float,
    edge_pct: float,
    direction: str,
    confidence: float = 0.7,
    book: str = "consensus",
    game: str = "",
    ev_pct: float = 0.0,
    reasoning: str = "",
    reasoning_detailed: str = "",
    weather: Optional[str] = None,
    weather_warning: Optional[str] = None,
) -> bool:
    """Send an edge alert notification.

    Args:
        player_name: Player name
        stat_type: Type of stat (receiving_yards, etc.)
        line: Betting line
        prediction: Model prediction
        edge_pct: Edge percentage
        direction: "over" or "under"
        confidence: Model confidence (0-1)
        book: Sportsbook name
        game: Game matchup (e.g., "SF @ CLE")
        ev_pct: Expected value percentage
        reasoning: Short reasoning (1 line)
        reasoning_detailed: Detailed reasoning (multi-line for Discord embed)
        weather: Weather conditions
        weather_warning: Weather warning if applicable

    Returns:
        True if sent successfully
    """
    alert = EdgeAlert(
        player_name=player_name,
        stat_type=stat_type,
        line=line,
        prediction=prediction,
        edge_pct=edge_pct,
        direction=direction,
        confidence=confidence,
        book=book,
        game=game,
        ev_pct=ev_pct,
        reasoning=reasoning,
        reasoning_detailed=reasoning_detailed,
        weather=weather,
        weather_warning=weather_warning,
    )
    return _get_notifier().send_edge_alert(alert)


def send_weekly_results(
    week: int,
    season: int,
    total_bets: int,
    wins: int,
    losses: int,
    pushes: int,
    roi_pct: float,
    profit_loss: float,
    best_bet: Optional[str] = None,
    worst_bet: Optional[str] = None,
) -> bool:
    """Send weekly results summary.

    Args:
        week: NFL week number
        season: NFL season year
        total_bets: Total bets made
        wins: Number of wins
        losses: Number of losses
        pushes: Number of pushes
        roi_pct: ROI percentage
        profit_loss: Dollar profit/loss
        best_bet: Best bet description
        worst_bet: Worst bet description

    Returns:
        True if sent successfully
    """
    result = WeeklyResult(
        total_bets=total_bets,
        wins=wins,
        losses=losses,
        pushes=pushes,
        roi_pct=roi_pct,
        profit_loss=profit_loss,
        best_bet=best_bet,
        worst_bet=worst_bet,
    )
    return _get_notifier().send_weekly_results(result, week, season)


def send_health_alert(
    status: str,
    issues: list[str],
    healthy_checks: list[str],
) -> bool:
    """Send system health alert.

    Args:
        status: Overall status (healthy, degraded, unhealthy)
        issues: List of issues found
        healthy_checks: List of passing checks

    Returns:
        True if sent successfully
    """
    return _get_notifier().send_health_alert(status, issues, healthy_checks)
