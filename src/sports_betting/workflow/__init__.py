"""Workflow orchestration for the sports betting system."""

from .stages import (
    WorkflowStage,
    StageResult,
    StageStatus,
    OddsCollectionStage,
    PredictionStage,
    EdgeCalculationStage,
    ResultsScoringStage,
    STAGE_REGISTRY,
    get_stage,
    list_stages,
)
from .orchestrator import (
    Orchestrator,
    WorkflowType,
    WorkflowResult,
)

__all__ = [
    # Stage classes
    "WorkflowStage",
    "StageResult",
    "StageStatus",
    "OddsCollectionStage",
    "PredictionStage",
    "EdgeCalculationStage",
    "ResultsScoringStage",
    # Stage utilities
    "STAGE_REGISTRY",
    "get_stage",
    "list_stages",
    # Orchestrator
    "Orchestrator",
    "WorkflowType",
    "WorkflowResult",
]
