"""
Sports Betting ML Module.

Provides adaptive prediction capabilities for NFL player props.

V2.0 Features:
- Adaptive data sourcing (nfl_data_py, PFR, NGS)
- Confidence-weighted predictions
- Situational context adjustments
"""
from .predictor import ReceivingYardsPredictor
from .feature_engineering import ReceivingYardsFeatureEngineer
from .data_sources import AdaptiveDataFetcher, DataQuality
from .context import SituationalContext, get_context, set_qb_change, set_injury

__all__ = [
    'ReceivingYardsPredictor',
    'ReceivingYardsFeatureEngineer',
    'AdaptiveDataFetcher',
    'DataQuality',
    'SituationalContext',
    'get_context',
    'set_qb_change',
    'set_injury',
]
