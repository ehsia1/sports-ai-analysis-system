"""Workflow stages for the sports betting system."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from ..config import get_settings
from ..utils import get_logger, GamePhase

logger = get_logger(__name__)


class StageStatus(Enum):
    """Status of a workflow stage."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StageResult:
    """Result of executing a workflow stage."""

    stage_name: str
    status: StageStatus
    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[Exception] = None
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None

    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate duration in seconds."""
        if self.started_at and self.ended_at:
            return (self.ended_at - self.started_at).total_seconds()
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "stage_name": self.stage_name,
            "status": self.status.value,
            "message": self.message,
            "data": self.data,
            "error": str(self.error) if self.error else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "duration_seconds": self.duration_seconds,
        }


class WorkflowStage(ABC):
    """Base class for workflow stages."""

    name: str = "base_stage"
    description: str = "Base workflow stage"
    required_phases: List[GamePhase] = [GamePhase.PRE_GAME]

    def __init__(self):
        self.settings = get_settings()
        self.logger = get_logger(f"stage.{self.name}")

    def can_run(self, phase: GamePhase) -> bool:
        """Check if this stage can run in the given phase."""
        return phase in self.required_phases

    @abstractmethod
    def execute(self, season: int, week: int, **kwargs) -> StageResult:
        """
        Execute the stage.

        Args:
            season: NFL season year
            week: Week number
            **kwargs: Additional arguments

        Returns:
            StageResult with execution outcome
        """
        pass

    def _create_result(
        self,
        status: StageStatus,
        message: str,
        data: Optional[Dict[str, Any]] = None,
        error: Optional[Exception] = None,
        started_at: Optional[datetime] = None,
        ended_at: Optional[datetime] = None,
    ) -> StageResult:
        """Helper to create a StageResult."""
        return StageResult(
            stage_name=self.name,
            status=status,
            message=message,
            data=data or {},
            error=error,
            started_at=started_at,
            ended_at=ended_at,
        )


class OddsCollectionStage(WorkflowStage):
    """Stage for collecting odds from The Odds API."""

    name = "collect_odds"
    description = "Fetch player props from The Odds API"
    required_phases = [GamePhase.PRE_GAME, GamePhase.IN_PROGRESS]

    def execute(self, season: int, week: int, **kwargs) -> StageResult:
        """
        Collect odds for the specified week.

        Args:
            season: NFL season
            week: Week number
            force: Force fetch even if cached (default False)
            max_events: Maximum events to fetch (default 5)
            markets: Markets to fetch (default ['player_reception_yds'])

        Returns:
            StageResult with props_stored, credits_used, credits_remaining
        """
        started_at = datetime.now()
        force = kwargs.get("force", False)
        max_events = kwargs.get("max_events", 5)
        markets = kwargs.get("markets", ["player_reception_yds"])

        self.logger.info(f"Starting odds collection for {season} Week {week}")

        try:
            from ..data.odds_api import OddsAPIClient

            client = OddsAPIClient()

            # Check API key
            if not client.api_key:
                return self._create_result(
                    status=StageStatus.FAILED,
                    message="No API key configured. Set ODDS_API_KEY in .env",
                    started_at=started_at,
                    ended_at=datetime.now(),
                )

            # Check if we should fetch
            if not force and not client.should_fetch_new_odds():
                self.logger.info("Using cached odds data")
                return self._create_result(
                    status=StageStatus.SKIPPED,
                    message="Already fetched odds today - using cached data",
                    data={"from_cache": True},
                    started_at=started_at,
                    ended_at=datetime.now(),
                )

            # Get upcoming games
            events = client.get_nfl_games()
            if not events:
                return self._create_result(
                    status=StageStatus.SKIPPED,
                    message="No upcoming NFL games found",
                    started_at=started_at,
                    ended_at=datetime.now(),
                )

            self.logger.info(f"Found {len(events)} upcoming games")

            # Limit events to conserve credits
            event_ids = [e["id"] for e in events[: min(max_events, len(events))]]

            # Fetch odds
            odds_data = client.fetch_and_cache_daily_odds(
                markets=markets, event_ids=event_ids
            )

            if not odds_data:
                return self._create_result(
                    status=StageStatus.FAILED,
                    message="Failed to fetch odds from API",
                    started_at=started_at,
                    ended_at=datetime.now(),
                )

            # Store in database
            props_stored = client.store_odds_in_database(odds_data)

            self.logger.info(f"Stored {props_stored} props")

            return self._create_result(
                status=StageStatus.SUCCESS,
                message=f"Collected {props_stored} props from {len(event_ids)} games",
                data={
                    "props_stored": props_stored,
                    "events_count": len(event_ids),
                    "credits_used": odds_data.get("total_cost", 0),
                    "credits_remaining": client.credits_remaining,
                    "markets": markets,
                },
                started_at=started_at,
                ended_at=datetime.now(),
            )

        except Exception as e:
            self.logger.error(f"Odds collection failed: {e}")
            return self._create_result(
                status=StageStatus.FAILED,
                message=f"Odds collection failed: {str(e)}",
                error=e,
                started_at=started_at,
                ended_at=datetime.now(),
            )


class PredictionStage(WorkflowStage):
    """Stage for generating ML predictions across all stat types."""

    name = "generate_predictions"
    description = "Generate ML predictions for player props (all stat types)"
    required_phases = [GamePhase.PRE_GAME, GamePhase.IN_PROGRESS]

    # Stat types to generate predictions for
    STAT_TYPES = ["receiving_yards", "rushing_yards", "passing_yards", "receptions"]

    # Map internal stat types to database market names
    MARKET_MAP = {
        "receiving_yards": "player_reception_yds",
        "rushing_yards": "player_rush_yds",
        "passing_yards": "player_pass_yds",
        "receptions": "player_receptions",
    }

    def execute(self, season: int, week: int, **kwargs) -> StageResult:
        """
        Generate predictions for the specified week across all stat types.

        Args:
            season: NFL season
            week: Week number
            stat_types: List of stat types to predict (default: all)
            save_to_db: Save predictions to database (default: True)
            output_file: Path to save JSON output (default: auto-generated)

        Returns:
            StageResult with predictions_generated count per stat type
        """
        started_at = datetime.now()
        stat_types = kwargs.get("stat_types", self.STAT_TYPES)
        save_to_db = kwargs.get("save_to_db", True)
        output_file = kwargs.get("output_file", None)

        self.logger.info(f"Generating predictions for {season} Week {week}")
        self.logger.info(f"Stat types: {stat_types}")

        all_predictions = {}
        all_metadata = {}
        errors = []

        try:
            from ..ml import (
                ReceivingYardsPredictor,
                get_predictor,
                list_predictors,
            )

            for stat_type in stat_types:
                self.logger.info(f"Processing {stat_type}...")

                try:
                    # Use specialized predictor for receiving yards
                    if stat_type == "receiving_yards":
                        predictor = ReceivingYardsPredictor()
                        preds, metadata = predictor.predict_adaptive(season, week)
                        if len(preds) > 0:
                            preds = preds.rename(
                                columns={"predicted_yards": "predicted"}
                            )
                            preds["stat_type"] = stat_type
                    else:
                        # Use v2 predictors for other stat types
                        predictor = get_predictor(stat_type)
                        preds, metadata = predictor.predict_for_week(season, week)

                    if len(preds) > 0:
                        all_predictions[stat_type] = preds
                        all_metadata[stat_type] = metadata
                        self.logger.info(
                            f"  {stat_type}: {len(preds)} predictions"
                        )
                    else:
                        self.logger.warning(f"  {stat_type}: No predictions generated")

                except FileNotFoundError as e:
                    self.logger.warning(f"  {stat_type}: Model not found - {e}")
                    errors.append(f"{stat_type}: model not found")
                except Exception as e:
                    self.logger.warning(f"  {stat_type}: Failed - {e}")
                    errors.append(f"{stat_type}: {str(e)}")

            # Calculate totals
            total_predictions = sum(len(df) for df in all_predictions.values())
            stats_with_predictions = list(all_predictions.keys())

            if total_predictions == 0:
                return self._create_result(
                    status=StageStatus.SKIPPED,
                    message=f"No predictions generated for {season} Week {week}",
                    data={"errors": errors} if errors else {},
                    started_at=started_at,
                    ended_at=datetime.now(),
                )

            self.logger.info(f"Generated {total_predictions} total predictions")

            # Save to database if requested
            db_saved = 0
            if save_to_db:
                db_saved = self._save_to_database(
                    all_predictions, season, week, started_at
                )
                self.logger.info(f"Saved {db_saved} predictions to database")

            # Save to JSON file
            output_path = self._save_to_json(
                all_predictions, all_metadata, season, week, output_file
            )
            self.logger.info(f"Saved predictions to {output_path}")

            return self._create_result(
                status=StageStatus.SUCCESS,
                message=f"Generated {total_predictions} predictions across {len(stats_with_predictions)} stat types",
                data={
                    "predictions_generated": total_predictions,
                    "predictions_saved_to_db": db_saved,
                    "output_file": str(output_path),
                    "stat_types_processed": stats_with_predictions,
                    "predictions_by_type": {
                        st: len(df) for st, df in all_predictions.items()
                    },
                    "errors": errors if errors else None,
                },
                started_at=started_at,
                ended_at=datetime.now(),
            )

        except Exception as e:
            self.logger.error(f"Prediction generation failed: {e}")
            return self._create_result(
                status=StageStatus.FAILED,
                message=f"Prediction generation failed: {str(e)}",
                error=e,
                started_at=started_at,
                ended_at=datetime.now(),
            )

    def _save_to_database(
        self, all_predictions: dict, season: int, week: int, timestamp: datetime
    ) -> int:
        """Save predictions to the Prediction table (updates existing records)."""
        from ..database import get_session
        from ..database.models import Prediction, Player, Game

        saved_count = 0
        updated_count = 0

        with get_session() as session:
            # Get games for this week (for game_id lookup)
            games = session.query(Game).filter_by(season=season, week=week).all()
            if not games:
                self.logger.warning(f"No games found for {season} Week {week}")
                return 0

            # Build player name -> id lookup
            players = session.query(Player).all()
            player_lookup = {}
            for p in players:
                # Normalize name for matching
                name_lower = p.name.lower().strip()
                player_lookup[name_lower] = p.id
                # Also try without suffixes
                for suffix in [" jr.", " jr", " sr.", " sr", " iii", " ii"]:
                    clean_name = name_lower.replace(suffix, "")
                    player_lookup[clean_name] = p.id

            # Use first game as default (predictions aren't game-specific yet)
            default_game_id = games[0].id

            for stat_type, preds_df in all_predictions.items():
                market = self.MARKET_MAP.get(stat_type, stat_type)
                model_name = f"{stat_type}_predictor"
                model_version = "v2"

                for _, row in preds_df.iterrows():
                    player_name = row.get("player_name", "")
                    if not player_name:
                        continue

                    # Look up player ID
                    name_lower = player_name.lower().strip()
                    player_id = player_lookup.get(name_lower)

                    if not player_id:
                        # Try without common suffixes
                        for suffix in [" jr.", " jr", " sr.", " sr", " iii", " ii"]:
                            clean_name = name_lower.replace(suffix, "")
                            player_id = player_lookup.get(clean_name)
                            if player_id:
                                break

                    if not player_id:
                        continue  # Skip if player not found

                    # Check if prediction already exists
                    existing = (
                        session.query(Prediction)
                        .filter_by(
                            game_id=default_game_id,
                            player_id=player_id,
                            market=market,
                            model_name=model_name,
                            model_version=model_version,
                        )
                        .first()
                    )

                    if existing:
                        # Update existing record
                        existing.prediction = float(row.get("predicted", 0))
                        existing.confidence = float(row.get("confidence", 0.7))
                        existing.created_at = timestamp
                        updated_count += 1
                    else:
                        # Create new prediction record
                        prediction = Prediction(
                            game_id=default_game_id,
                            player_id=player_id,
                            market=market,
                            model_name=model_name,
                            model_version=model_version,
                            prediction=float(row.get("predicted", 0)),
                            confidence=float(row.get("confidence", 0.7)),
                            created_at=timestamp,
                        )
                        session.add(prediction)
                        saved_count += 1

            session.commit()

        if updated_count > 0:
            self.logger.info(f"Updated {updated_count} existing predictions")

        return saved_count + updated_count

    def _save_to_json(
        self,
        all_predictions: dict,
        all_metadata: dict,
        season: int,
        week: int,
        output_file: str = None,
    ) -> str:
        """Save predictions to JSON file."""
        import json
        from pathlib import Path

        # Default output path
        if output_file is None:
            output_dir = Path(__file__).parent.parent.parent.parent / "data" / "predictions"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f"predictions_{season}_week{week}.json"
        else:
            output_file = Path(output_file)

        # Build output structure
        output = {
            "season": season,
            "week": week,
            "generated_at": datetime.now().isoformat(),
            "stat_types": list(all_predictions.keys()),
            "total_predictions": sum(len(df) for df in all_predictions.values()),
            "predictions": {},
            "metadata": {},
        }

        for stat_type, preds_df in all_predictions.items():
            # Convert DataFrame to list of dicts
            preds_list = []
            for _, row in preds_df.iterrows():
                pred_dict = {
                    "player_name": row.get("player_name", ""),
                    "player_id": row.get("player_id", ""),
                    "position": row.get("position", ""),
                    "team": row.get("recent_team", ""),
                    "predicted": float(row.get("predicted", 0)),
                    "confidence": float(row.get("confidence", 0.7)),
                }
                preds_list.append(pred_dict)

            # Sort by predicted value descending
            preds_list.sort(key=lambda x: x["predicted"], reverse=True)
            output["predictions"][stat_type] = preds_list

            # Add metadata
            if stat_type in all_metadata:
                meta = all_metadata[stat_type]
                # Convert any non-serializable values
                clean_meta = {}
                for k, v in meta.items():
                    if isinstance(v, (int, float, str, bool, list, dict, type(None))):
                        clean_meta[k] = v
                output["metadata"][stat_type] = clean_meta

        # Write JSON
        with open(output_file, "w") as f:
            json.dump(output, f, indent=2)

        return str(output_file)


class EdgeCalculationStage(WorkflowStage):
    """Stage for calculating betting edges."""

    name = "calculate_edges"
    description = "Calculate betting edges comparing predictions to odds"
    required_phases = [GamePhase.PRE_GAME, GamePhase.IN_PROGRESS]

    def execute(self, season: int, week: int, **kwargs) -> StageResult:
        """
        Calculate betting edges for the specified week.

        Args:
            season: NFL season
            week: Week number
            min_edge: Minimum edge threshold (default from settings)

        Returns:
            StageResult with edges_found count and edge details
        """
        started_at = datetime.now()
        min_edge = kwargs.get("min_edge", self.settings.min_edge)

        self.logger.info(f"Calculating edges for {season} Week {week}")

        try:
            from ..analysis.edge_calculator import EdgeCalculator

            calculator = EdgeCalculator()
            calculator.min_edge = min_edge
            calculator.min_confidence = self.settings.min_confidence

            # Find edges
            edges = calculator.find_edges_for_week(
                week=week, season=season, min_edge=min_edge
            )

            if not edges:
                return self._create_result(
                    status=StageStatus.SUCCESS,
                    message="No edges found meeting criteria",
                    data={"edges_found": 0, "min_edge_pct": min_edge * 100},
                    started_at=started_at,
                    ended_at=datetime.now(),
                )

            # Store edges in database
            calculator.store_edges_in_database(edges)

            # Format report for data
            report = calculator.format_edge_report(edges)

            self.logger.info(f"Found {len(edges)} betting edges")

            return self._create_result(
                status=StageStatus.SUCCESS,
                message=f"Found {len(edges)} edges with >{min_edge*100:.1f}% edge",
                data={
                    "edges_found": len(edges),
                    "edges": edges,
                    "report": report,
                    "min_edge_pct": min_edge * 100,
                },
                started_at=started_at,
                ended_at=datetime.now(),
            )

        except Exception as e:
            self.logger.error(f"Edge calculation failed: {e}")
            return self._create_result(
                status=StageStatus.FAILED,
                message=f"Edge calculation failed: {str(e)}",
                error=e,
                started_at=started_at,
                ended_at=datetime.now(),
            )


class ResultsScoringStage(WorkflowStage):
    """Stage for scoring predictions against actual results."""

    name = "score_results"
    description = "Score predictions against actual game results"
    required_phases = [GamePhase.POST_GAME, GamePhase.IN_PROGRESS]

    def execute(self, season: int, week: int, **kwargs) -> StageResult:
        """
        Score predictions for the specified week.

        Args:
            season: NFL season
            week: Week number
            generate_report: Generate markdown report (default True)

        Returns:
            StageResult with scoring metrics
        """
        started_at = datetime.now()
        generate_report = kwargs.get("generate_report", True)

        self.logger.info(f"Scoring results for {season} Week {week}")

        try:
            from ..database import get_session
            from ..database.models import Game, Prediction, PaperTrade

            with get_session() as session:
                # Check if games are completed
                games = (
                    session.query(Game)
                    .filter(Game.season == season, Game.week == week)
                    .all()
                )

                if not games:
                    return self._create_result(
                        status=StageStatus.SKIPPED,
                        message=f"No games found for {season} Week {week}",
                        started_at=started_at,
                        ended_at=datetime.now(),
                    )

                completed = sum(1 for g in games if g.is_completed)
                total = len(games)

                if completed == 0:
                    return self._create_result(
                        status=StageStatus.SKIPPED,
                        message=f"No completed games yet ({completed}/{total})",
                        data={"completed_games": completed, "total_games": total},
                        started_at=started_at,
                        ended_at=datetime.now(),
                    )

                # Get predictions for this week
                predictions = (
                    session.query(Prediction)
                    .join(Game)
                    .filter(Game.season == season, Game.week == week)
                    .all()
                )

                # Get paper trades for this week
                paper_trades = (
                    session.query(PaperTrade)
                    .join(Game)
                    .filter(Game.season == season, Game.week == week)
                    .all()
                )

                self.logger.info(
                    f"Scoring {len(predictions)} predictions, {len(paper_trades)} paper trades"
                )

            # For now, return basic status - full implementation would
            # fetch actual stats from nfl-data-py and compare
            return self._create_result(
                status=StageStatus.SUCCESS,
                message=f"Scored {completed}/{total} games",
                data={
                    "completed_games": completed,
                    "total_games": total,
                    "predictions_count": len(predictions),
                    "paper_trades_count": len(paper_trades),
                },
                started_at=started_at,
                ended_at=datetime.now(),
            )

        except Exception as e:
            self.logger.error(f"Results scoring failed: {e}")
            return self._create_result(
                status=StageStatus.FAILED,
                message=f"Results scoring failed: {str(e)}",
                error=e,
                started_at=started_at,
                ended_at=datetime.now(),
            )


# Registry of all available stages
STAGE_REGISTRY: Dict[str, type] = {
    "collect_odds": OddsCollectionStage,
    "generate_predictions": PredictionStage,
    "calculate_edges": EdgeCalculationStage,
    "score_results": ResultsScoringStage,
}


def get_stage(name: str) -> WorkflowStage:
    """
    Get a stage instance by name.

    Args:
        name: Stage name

    Returns:
        WorkflowStage instance

    Raises:
        ValueError: If stage name not found
    """
    if name not in STAGE_REGISTRY:
        raise ValueError(f"Unknown stage: {name}. Available: {list(STAGE_REGISTRY.keys())}")
    return STAGE_REGISTRY[name]()


def list_stages() -> List[str]:
    """Get list of available stage names."""
    return list(STAGE_REGISTRY.keys())
