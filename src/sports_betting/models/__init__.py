"""Machine learning models for sports betting predictions."""

from .base.predictor import BasePredictor
from .nfl.xgboost_model import XGBoostPropsModel
from .nfl.neural_net import NeuralNetModel

__all__ = [
    "BasePredictor",
    "XGBoostPropsModel", 
    "NeuralNetModel",
]