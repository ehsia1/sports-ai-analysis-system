"""Central workflow orchestrator for the sports betting system."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from ..config import get_settings
from ..utils import get_logger, get_current_week, get_week_info, GamePhase
from .stages import (
    WorkflowStage,
    StageResult,
    StageStatus,
    STAGE_REGISTRY,
    get_stage,
    list_stages,
)

logger = get_logger(__name__)


class WorkflowType(Enum):
    """Types of predefined workflows."""

    PRE_GAME = "pre_game"  # collect_odds -> generate_predictions -> calculate_edges
    POST_GAME = "post_game"  # score_results
    FULL = "full"  # All stages in order


@dataclass
class WorkflowResult:
    """Result of executing a workflow."""

    workflow_type: str
    season: int
    week: int
    stage_results: List[StageResult] = field(default_factory=list)
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None

    @property
    def success(self) -> bool:
        """Check if all stages succeeded or were skipped."""
        return all(
            r.status in (StageStatus.SUCCESS, StageStatus.SKIPPED)
            for r in self.stage_results
        )

    @property
    def failed_stages(self) -> List[str]:
        """Get list of failed stage names."""
        return [r.stage_name for r in self.stage_results if r.status == StageStatus.FAILED]

    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate total duration in seconds."""
        if self.started_at and self.ended_at:
            return (self.ended_at - self.started_at).total_seconds()
        return None

    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            f"Workflow: {self.workflow_type}",
            f"Season: {self.season} Week {self.week}",
            f"Status: {'SUCCESS' if self.success else 'FAILED'}",
            "",
            "Stage Results:",
        ]

        for result in self.stage_results:
            status_icon = {
                StageStatus.SUCCESS: "✓",
                StageStatus.FAILED: "✗",
                StageStatus.SKIPPED: "○",
                StageStatus.PENDING: "·",
                StageStatus.RUNNING: "►",
            }.get(result.status, "?")

            duration = f" ({result.duration_seconds:.1f}s)" if result.duration_seconds else ""
            lines.append(f"  {status_icon} {result.stage_name}: {result.message}{duration}")

        if self.duration_seconds:
            lines.append("")
            lines.append(f"Total duration: {self.duration_seconds:.1f}s")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "workflow_type": self.workflow_type,
            "season": self.season,
            "week": self.week,
            "success": self.success,
            "stage_results": [r.to_dict() for r in self.stage_results],
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "duration_seconds": self.duration_seconds,
        }


class Orchestrator:
    """Central orchestrator for running workflows and stages."""

    # Define workflow stage sequences
    WORKFLOWS = {
        WorkflowType.PRE_GAME: ["collect_odds", "generate_predictions", "calculate_edges"],
        WorkflowType.POST_GAME: ["refresh_schedule", "score_results"],
        WorkflowType.FULL: ["collect_odds", "generate_predictions", "calculate_edges", "refresh_schedule", "score_results"],
    }

    def __init__(self):
        self.settings = get_settings()
        self.logger = get_logger("orchestrator")

    def get_status(self) -> Dict[str, Any]:
        """
        Get current system status.

        Returns:
            Dict with current week, phase, and available workflows
        """
        season, week = get_current_week()
        info = get_week_info(season, week)

        # Determine which workflows can run
        runnable_workflows = []
        for workflow_type, stages in self.WORKFLOWS.items():
            # Check if all stages in workflow can run
            can_run = all(
                get_stage(stage_name).can_run(info.phase)
                for stage_name in stages
            )
            if can_run:
                runnable_workflows.append(workflow_type.value)

        # Determine which individual stages can run
        runnable_stages = [
            name for name in list_stages()
            if get_stage(name).can_run(info.phase)
        ]

        return {
            "season": season,
            "week": week,
            "phase": info.phase.value,
            "games_completed": info.games_completed,
            "total_games": info.total_games,
            "first_game": info.first_game.isoformat() if info.first_game else None,
            "last_game": info.last_game.isoformat() if info.last_game else None,
            "runnable_workflows": runnable_workflows,
            "runnable_stages": runnable_stages,
            "all_stages": list_stages(),
        }

    def run_stage(
        self,
        stage_name: str,
        season: Optional[int] = None,
        week: Optional[int] = None,
        **kwargs,
    ) -> StageResult:
        """
        Run a single stage.

        Args:
            stage_name: Name of the stage to run
            season: NFL season (default: current)
            week: Week number (default: current)
            **kwargs: Additional arguments passed to stage

        Returns:
            StageResult with execution outcome
        """
        # Get current week if not specified
        if season is None or week is None:
            current_season, current_week = get_current_week()
            season = season or current_season
            week = week or current_week

        self.logger.info(f"Running stage '{stage_name}' for {season} Week {week}")

        # Get and run the stage
        stage = get_stage(stage_name)
        result = stage.execute(season, week, **kwargs)

        self.logger.info(f"Stage '{stage_name}' completed: {result.status.value}")

        return result

    def run_workflow(
        self,
        workflow_type: WorkflowType,
        season: Optional[int] = None,
        week: Optional[int] = None,
        stop_on_failure: bool = True,
        **kwargs,
    ) -> WorkflowResult:
        """
        Run a complete workflow.

        Args:
            workflow_type: Type of workflow to run
            season: NFL season (default: current)
            week: Week number (default: current)
            stop_on_failure: Stop workflow if a stage fails (default True)
            **kwargs: Additional arguments passed to all stages

        Returns:
            WorkflowResult with all stage outcomes
        """
        started_at = datetime.now()

        # Get current week if not specified
        if season is None or week is None:
            current_season, current_week = get_current_week()
            season = season or current_season
            week = week or current_week

        self.logger.info(f"Starting {workflow_type.value} workflow for {season} Week {week}")

        # Get stage sequence for this workflow
        stage_names = self.WORKFLOWS.get(workflow_type, [])
        if not stage_names:
            return WorkflowResult(
                workflow_type=workflow_type.value,
                season=season,
                week=week,
                stage_results=[],
                started_at=started_at,
                ended_at=datetime.now(),
            )

        # Run each stage in sequence
        stage_results = []
        for stage_name in stage_names:
            result = self.run_stage(stage_name, season, week, **kwargs)
            stage_results.append(result)

            # Check if we should stop
            if result.status == StageStatus.FAILED and stop_on_failure:
                self.logger.warning(f"Workflow stopped due to failure in '{stage_name}'")
                break

        ended_at = datetime.now()

        workflow_result = WorkflowResult(
            workflow_type=workflow_type.value,
            season=season,
            week=week,
            stage_results=stage_results,
            started_at=started_at,
            ended_at=ended_at,
        )

        status = "SUCCESS" if workflow_result.success else "FAILED"
        self.logger.info(f"Workflow {workflow_type.value} completed: {status}")

        return workflow_result

    def run_pre_game(
        self,
        season: Optional[int] = None,
        week: Optional[int] = None,
        **kwargs,
    ) -> WorkflowResult:
        """
        Run the pre-game workflow.

        Convenience method for running PRE_GAME workflow.
        """
        return self.run_workflow(WorkflowType.PRE_GAME, season, week, **kwargs)

    def run_post_game(
        self,
        season: Optional[int] = None,
        week: Optional[int] = None,
        **kwargs,
    ) -> WorkflowResult:
        """
        Run the post-game workflow.

        Convenience method for running POST_GAME workflow.
        """
        return self.run_workflow(WorkflowType.POST_GAME, season, week, **kwargs)

    def run_full(
        self,
        season: Optional[int] = None,
        week: Optional[int] = None,
        **kwargs,
    ) -> WorkflowResult:
        """
        Run the full workflow.

        Convenience method for running FULL workflow.
        """
        return self.run_workflow(WorkflowType.FULL, season, week, **kwargs)
