"""Base predictor class for all ML models."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import pandas as pd
from pathlib import Path
import joblib
import logging

logger = logging.getLogger(__name__)


class BasePredictor(ABC):
    """Base class for all ML models."""
    
    def __init__(self, model_name: str, model_version: str = "1.0"):
        self.model_name = model_name
        self.model_version = model_version
        self.model = None
        self.is_trained = False
        self.feature_columns = []
        self.target_column = None
        
    @abstractmethod
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for training/prediction."""
        pass
        
    @abstractmethod
    def train(self, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Train the model on provided data."""
        pass
        
    @abstractmethod
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Make predictions on new data."""
        pass
        
    def save_model(self, model_path: Path) -> None:
        """Save trained model to disk."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
            
        model_data = {
            'model': self.model,
            'model_name': self.model_name,
            'model_version': self.model_version,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'is_trained': self.is_trained
        }
        
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model_data, model_path)
        logger.info(f"Model saved to {model_path}")
        
    def load_model(self, model_path: Path) -> None:
        """Load trained model from disk."""
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        model_data = joblib.load(model_path)
        
        self.model = model_data['model']
        self.model_name = model_data['model_name']
        self.model_version = model_data['model_version']
        self.feature_columns = model_data['feature_columns']
        self.target_column = model_data['target_column']
        self.is_trained = model_data['is_trained']
        
        logger.info(f"Model loaded from {model_path}")
        
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance if available."""
        if not self.is_trained:
            return None
            
        if hasattr(self.model, 'feature_importances_'):
            return dict(zip(self.feature_columns, self.model.feature_importances_))
        elif hasattr(self.model, 'coef_'):
            return dict(zip(self.feature_columns, abs(self.model.coef_)))
        else:
            return None
            
    def validate_features(self, df: pd.DataFrame) -> None:
        """Validate that required features are present."""
        if not self.feature_columns:
            raise ValueError("Feature columns not set")
            
        missing_features = set(self.feature_columns) - set(df.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")