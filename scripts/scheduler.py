"""Automated scheduler for NFL betting workflows.

Runs pre-game, post-game, and health check workflows on a cron schedule.
Uses APScheduler with timezone-aware cron triggers.

Usage:
    docker compose up -d scheduler
    docker compose logs -f scheduler
"""

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
import subprocess
from loguru import logger


scheduler = BlockingScheduler()


def run_command(cmd: list[str]) -> None:
    """Run orchestrate command and log results."""
    full_cmd = ["python", "scripts/orchestrate.py"] + cmd
    logger.info(f"Running: {' '.join(full_cmd)}")
    result = subprocess.run(full_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Command failed: {result.stderr}")
    else:
        # Log last 500 chars of output
        output = result.stdout.strip()
        if len(output) > 500:
            output = f"...{output[-500:]}"
        logger.info(f"Command completed: {output}")


# Thursday Night Football - pre-game at 6pm ET
@scheduler.scheduled_job(
    CronTrigger(day_of_week="thu", hour=18, timezone="America/New_York")
)
def thursday_pregame():
    """Pre-game workflow for Thursday Night Football."""
    logger.info("Running Thursday pre-game workflow")
    run_command(["pre-game", "--notify"])


# Sunday games - pre-game at 11am ET (before early games)
@scheduler.scheduled_job(
    CronTrigger(day_of_week="sun", hour=11, timezone="America/New_York")
)
def sunday_pregame():
    """Pre-game workflow for Sunday games."""
    logger.info("Running Sunday pre-game workflow")
    run_command(["pre-game", "--notify"])


# Sunday night - parlay generation at 6pm ET (for SNF)
@scheduler.scheduled_job(
    CronTrigger(day_of_week="sun", hour=18, timezone="America/New_York")
)
def sunday_parlays():
    """Generate parlays for Sunday Night Football."""
    logger.info("Generating SNF parlays")
    run_command(["parlay", "--notify", "--max-legs", "10"])


# Monday Night Football - pre-game at 6pm ET
@scheduler.scheduled_job(
    CronTrigger(day_of_week="mon", hour=18, timezone="America/New_York")
)
def monday_pregame():
    """Pre-game workflow for Monday Night Football."""
    logger.info("Running Monday pre-game workflow")
    run_command(["pre-game", "--notify"])


# Tuesday morning - post-game scoring (after all games complete)
@scheduler.scheduled_job(
    CronTrigger(day_of_week="tue", hour=6, timezone="America/New_York")
)
def tuesday_postgame():
    """Post-game scoring after all weekly games complete."""
    logger.info("Running post-game scoring")
    run_command(["post-game"])


# Daily health check at 9am ET
@scheduler.scheduled_job(CronTrigger(hour=9, timezone="America/New_York"))
def daily_health():
    """Daily system health check."""
    logger.info("Running daily health check")
    run_command(["health", "--notify"])


if __name__ == "__main__":
    logger.info("Starting NFL Betting Scheduler")
    logger.info("=" * 50)
    logger.info("Scheduled jobs:")
    for job in scheduler.get_jobs():
        logger.info(f"  - {job.name}: {job.trigger}")
    logger.info("=" * 50)
    logger.info("Press Ctrl+C to exit")
    scheduler.start()
