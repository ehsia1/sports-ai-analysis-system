"""
Sports Betting ML Module.

Provides adaptive prediction capabilities for NFL player props.

V2.0 Features:
- Adaptive data sourcing (nfl_data_py, PFR, NGS)
- Confidence-weighted predictions
- Situational context adjustments
- Multi-stat prediction support
"""
from .predictor import ReceivingYardsPredictor
from .feature_engineering import ReceivingYardsFeatureEngineer
from .data_sources import AdaptiveDataFetcher, DataQuality
from .context import SituationalContext, get_context, set_qb_change, set_injury
from .base_predictor import BaseStatPredictor
from .stat_predictors import (
    RushingYardsPredictor,
    PassingYardsPredictor,
    PassingYardsPredictorV3,
    ReceptionsPredictor,
    PREDICTOR_REGISTRY,
    get_predictor,
    list_predictors,
)

__all__ = [
    # Receiving yards (original sophisticated predictor)
    'ReceivingYardsPredictor',
    'ReceivingYardsFeatureEngineer',
    # Multi-stat predictors
    'BaseStatPredictor',
    'RushingYardsPredictor',
    'PassingYardsPredictor',
    'PassingYardsPredictorV3',
    'ReceptionsPredictor',
    'PREDICTOR_REGISTRY',
    'get_predictor',
    'list_predictors',
    # Data and context
    'AdaptiveDataFetcher',
    'DataQuality',
    'SituationalContext',
    'get_context',
    'set_qb_change',
    'set_injury',
]
