"""Tests for workflow orchestrator."""

from datetime import datetime

import pytest

from src.sports_betting.workflow import (
    Orchestrator,
    WorkflowType,
    WorkflowResult,
    StageStatus,
    StageResult,
    list_stages,
    get_stage,
)
from src.sports_betting.utils import GamePhase


class TestStageRegistry:
    """Tests for the stage registry."""

    def test_list_stages_returns_expected(self):
        """Test that list_stages returns all expected stages."""
        stages = list_stages()
        assert "collect_odds" in stages
        assert "generate_predictions" in stages
        assert "calculate_edges" in stages
        assert "refresh_schedule" in stages
        assert "score_results" in stages
        assert len(stages) == 5

    def test_get_stage_returns_instance(self):
        """Test that get_stage returns a stage instance."""
        stage = get_stage("collect_odds")
        assert stage is not None
        assert stage.name == "collect_odds"

    def test_get_stage_unknown_raises(self):
        """Test that get_stage raises for unknown stage."""
        with pytest.raises(ValueError, match="Unknown stage"):
            get_stage("nonexistent_stage")


class TestStageResult:
    """Tests for StageResult dataclass."""

    def test_stage_result_creation(self):
        """Test creating a StageResult."""
        result = StageResult(
            stage_name="test_stage",
            status=StageStatus.SUCCESS,
            message="Test passed",
            data={"key": "value"},
        )
        assert result.stage_name == "test_stage"
        assert result.status == StageStatus.SUCCESS
        assert result.message == "Test passed"
        assert result.data == {"key": "value"}

    def test_duration_calculation(self):
        """Test duration calculation."""
        started = datetime(2024, 1, 1, 10, 0, 0)
        ended = datetime(2024, 1, 1, 10, 0, 30)

        result = StageResult(
            stage_name="test",
            status=StageStatus.SUCCESS,
            message="Done",
            started_at=started,
            ended_at=ended,
        )
        assert result.duration_seconds == 30.0

    def test_duration_none_when_missing_times(self):
        """Test duration is None when times not set."""
        result = StageResult(
            stage_name="test",
            status=StageStatus.SUCCESS,
            message="Done",
        )
        assert result.duration_seconds is None

    def test_to_dict(self):
        """Test converting to dictionary."""
        result = StageResult(
            stage_name="test",
            status=StageStatus.SUCCESS,
            message="Done",
            data={"count": 5},
        )
        d = result.to_dict()
        assert d["stage_name"] == "test"
        assert d["status"] == "success"
        assert d["message"] == "Done"
        assert d["data"] == {"count": 5}


class TestWorkflowResult:
    """Tests for WorkflowResult dataclass."""

    def test_success_property_all_success(self):
        """Test success property when all stages succeed."""
        result = WorkflowResult(
            workflow_type="pre_game",
            season=2024,
            week=10,
            stage_results=[
                StageResult("stage1", StageStatus.SUCCESS, "OK"),
                StageResult("stage2", StageStatus.SUCCESS, "OK"),
            ],
        )
        assert result.success is True

    def test_success_property_with_skipped(self):
        """Test success property with skipped stages."""
        result = WorkflowResult(
            workflow_type="pre_game",
            season=2024,
            week=10,
            stage_results=[
                StageResult("stage1", StageStatus.SUCCESS, "OK"),
                StageResult("stage2", StageStatus.SKIPPED, "Skipped"),
            ],
        )
        assert result.success is True

    def test_success_property_with_failure(self):
        """Test success property when a stage fails."""
        result = WorkflowResult(
            workflow_type="pre_game",
            season=2024,
            week=10,
            stage_results=[
                StageResult("stage1", StageStatus.SUCCESS, "OK"),
                StageResult("stage2", StageStatus.FAILED, "Failed"),
            ],
        )
        assert result.success is False

    def test_failed_stages(self):
        """Test failed_stages property."""
        result = WorkflowResult(
            workflow_type="pre_game",
            season=2024,
            week=10,
            stage_results=[
                StageResult("stage1", StageStatus.SUCCESS, "OK"),
                StageResult("stage2", StageStatus.FAILED, "Error"),
                StageResult("stage3", StageStatus.FAILED, "Error"),
            ],
        )
        assert result.failed_stages == ["stage2", "stage3"]

    def test_summary_generation(self):
        """Test summary string generation."""
        result = WorkflowResult(
            workflow_type="pre_game",
            season=2024,
            week=10,
            stage_results=[
                StageResult("collect_odds", StageStatus.SUCCESS, "Collected 50 props"),
            ],
        )
        summary = result.summary()
        assert "pre_game" in summary
        assert "2024" in summary
        assert "Week 10" in summary
        assert "collect_odds" in summary


class TestOrchestrator:
    """Tests for the Orchestrator class."""

    def test_orchestrator_creation(self):
        """Test creating an orchestrator."""
        orchestrator = Orchestrator()
        assert orchestrator is not None

    def test_get_status(self):
        """Test getting system status."""
        orchestrator = Orchestrator()
        status = orchestrator.get_status()

        assert "season" in status
        assert "week" in status
        assert "phase" in status
        assert "runnable_workflows" in status
        assert "runnable_stages" in status
        assert "all_stages" in status

        assert isinstance(status["season"], int)
        assert isinstance(status["week"], int)
        assert 1 <= status["week"] <= 18

    def test_workflow_definitions(self):
        """Test that workflow definitions are correct."""
        assert WorkflowType.PRE_GAME in Orchestrator.WORKFLOWS
        assert WorkflowType.POST_GAME in Orchestrator.WORKFLOWS
        assert WorkflowType.FULL in Orchestrator.WORKFLOWS

        # Pre-game should have odds, predictions, edges
        pre_game = Orchestrator.WORKFLOWS[WorkflowType.PRE_GAME]
        assert "collect_odds" in pre_game
        assert "generate_predictions" in pre_game
        assert "calculate_edges" in pre_game

        # Post-game should have score_results
        post_game = Orchestrator.WORKFLOWS[WorkflowType.POST_GAME]
        assert "score_results" in post_game


class TestStagePhases:
    """Tests for stage phase requirements."""

    def test_odds_collection_phases(self):
        """Test odds collection stage can run in correct phases."""
        stage = get_stage("collect_odds")
        assert stage.can_run(GamePhase.PRE_GAME) is True
        assert stage.can_run(GamePhase.IN_PROGRESS) is True
        assert stage.can_run(GamePhase.POST_GAME) is False

    def test_prediction_phases(self):
        """Test prediction stage can run in correct phases."""
        stage = get_stage("generate_predictions")
        assert stage.can_run(GamePhase.PRE_GAME) is True
        assert stage.can_run(GamePhase.IN_PROGRESS) is True
        assert stage.can_run(GamePhase.POST_GAME) is False

    def test_edge_calculation_phases(self):
        """Test edge calculation stage can run in correct phases."""
        stage = get_stage("calculate_edges")
        assert stage.can_run(GamePhase.PRE_GAME) is True
        assert stage.can_run(GamePhase.IN_PROGRESS) is True
        assert stage.can_run(GamePhase.POST_GAME) is False

    def test_results_scoring_phases(self):
        """Test results scoring stage can run in correct phases."""
        stage = get_stage("score_results")
        assert stage.can_run(GamePhase.POST_GAME) is True
        assert stage.can_run(GamePhase.IN_PROGRESS) is True
        assert stage.can_run(GamePhase.PRE_GAME) is False
