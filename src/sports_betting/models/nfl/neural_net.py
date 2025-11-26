"""Neural network model for NFL edge detection."""

from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
import logging

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Install with: pip install torch")

from ..base.predictor import BasePredictor

logger = logging.getLogger(__name__)


class EdgeDetectionNet(nn.Module):
    """Neural network for edge detection."""
    
    def __init__(self, input_size: int, hidden_sizes: List[int] = [128, 64, 32]):
        super(EdgeDetectionNet, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_size = hidden_size
            
        # Output layer: [prediction, confidence]
        layers.append(nn.Linear(prev_size, 2))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)


class NeuralNetModel(BasePredictor):
    """Neural network model for edge detection and confidence estimation."""
    
    def __init__(self, prop_type: str = "receiving_yards"):
        super().__init__(f"neural_net_{prop_type}", "1.0")
        self.prop_type = prop_type
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.epochs = 100
        self.batch_size = 32
        self.learning_rate = 0.001
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for neural network training/prediction."""
        logger.info(f"Preparing neural network features for {self.prop_type}")
        
        feature_df = df.copy()
        
        # Market data features
        market_features = [
            'offered_odds', 'offered_line', 'book_margin', 'line_movement'
        ]
        
        for feature in market_features:
            if feature not in feature_df.columns:
                if feature == 'offered_odds':
                    feature_df[feature] = -110  # Default odds
                elif feature == 'book_margin':
                    feature_df[feature] = 0.05  # 5% default margin
                else:
                    feature_df[feature] = 0
        
        # Player performance features (similar to XGBoost)
        player_stats = ['receiving_yards', 'receiving_tds', 'targets', 'receptions']
        
        for stat in player_stats:
            if stat in feature_df.columns:
                # Rolling averages
                feature_df[f'{stat}_3game_avg'] = (
                    feature_df.groupby('player_id')[stat]
                    .rolling(window=3, min_periods=1)
                    .mean()
                    .reset_index(0, drop=True)
                )
                
                # Trend (last game vs average)
                feature_df[f'{stat}_trend'] = (
                    feature_df[stat] - feature_df[f'{stat}_3game_avg']
                )
        
        # Matchup features
        matchup_features = [
            'is_home', 'opp_rank', 'weather_impact', 'injury_impact'
        ]
        
        for feature in matchup_features:
            if feature not in feature_df.columns:
                if feature == 'is_home':
                    feature_df[feature] = 0.5  # Neutral
                elif feature == 'opp_rank':
                    feature_df[feature] = 16  # Middle rank
                else:
                    feature_df[feature] = 0
        
        # Market efficiency features
        efficiency_features = [
            'volume', 'public_percentage', 'sharp_money', 'closing_line_value'
        ]
        
        for feature in efficiency_features:
            if feature not in feature_df.columns:
                if feature == 'volume':
                    feature_df[feature] = 1000  # Default volume
                elif feature == 'public_percentage':
                    feature_df[feature] = 50  # 50/50 split
                else:
                    feature_df[feature] = 0
        
        # Select features for neural network
        self.feature_columns = (
            market_features + 
            [col for col in feature_df.columns if '_3game_avg' in col or '_trend' in col] +
            matchup_features + 
            efficiency_features
        )
        
        # Ensure all features exist
        for feature in self.feature_columns:
            if feature not in feature_df.columns:
                logger.warning(f"Feature {feature} not found, setting to 0")
                feature_df[feature] = 0
        
        # Handle missing values
        feature_df[self.feature_columns] = feature_df[self.feature_columns].fillna(0)
        
        logger.info(f"Prepared {len(self.feature_columns)} neural network features")
        
        return feature_df
        
    def train(self, df: pd.DataFrame, target_column: str = None) -> Dict[str, Any]:
        """Train neural network model."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available. Install with: pip install torch")
            
        if target_column is None:
            target_column = self.prop_type
            
        self.target_column = target_column
        
        logger.info(f"Training neural network for {target_column}")
        
        # Prepare features
        feature_df = self.prepare_features(df)
        
        if target_column not in feature_df.columns:
            # Create synthetic target if not available
            logger.warning(f"Target {target_column} not found, creating synthetic data")
            feature_df[target_column] = np.random.normal(60, 20, len(feature_df))
        
        # Prepare data
        X = feature_df[self.feature_columns].values
        y = feature_df[target_column].values
        
        # Remove NaN values
        valid_mask = ~np.isnan(y)
        X = X[valid_mask]
        y = y[valid_mask]
        
        if len(X) == 0:
            raise ValueError("No valid training data")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # Create synthetic confidence targets (for demonstration)
        # In practice, this would be based on historical accuracy
        conf_train = np.random.uniform(0.5, 0.9, len(y_train))
        conf_test = np.random.uniform(0.5, 0.9, len(y_test))
        
        # Combine targets [prediction, confidence]
        y_train_combined = np.column_stack([y_train, conf_train])
        y_test_combined = np.column_stack([y_test, conf_test])
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train_combined).to(self.device)
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        y_test_tensor = torch.FloatTensor(y_test_combined).to(self.device)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        # Initialize model
        input_size = X_train.shape[1]
        self.model = EdgeDetectionNet(input_size).to(self.device)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training loop
        train_losses = []
        
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            self.model.train()
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_loss)
            
            if (epoch + 1) % 20 == 0:
                logger.info(f"Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.4f}")
        
        # Evaluate model
        self.model.eval()
        with torch.no_grad():
            test_outputs = self.model(X_test_tensor)
            test_loss = criterion(test_outputs, y_test_tensor).item()
            
            # Calculate metrics for predictions only
            pred_mae = torch.mean(torch.abs(test_outputs[:, 0] - y_test_tensor[:, 0])).item()
            conf_mae = torch.mean(torch.abs(test_outputs[:, 1] - y_test_tensor[:, 1])).item()
        
        self.is_trained = True
        
        metrics = {
            'final_train_loss': train_losses[-1],
            'test_loss': test_loss,
            'prediction_mae': pred_mae,
            'confidence_mae': conf_mae,
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'features': input_size,
            'epochs': self.epochs
        }
        
        logger.info(f"Neural network training complete. Test Loss: {test_loss:.4f}")
        
        return metrics
        
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Make predictions with neural network."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
            
        logger.info(f"Making neural network predictions for {len(df)} samples")
        
        # Prepare features
        feature_df = self.prepare_features(df)
        self.validate_features(feature_df)
        
        X = feature_df[self.feature_columns].values
        X_scaled = self.scaler.transform(X)
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        # Make predictions
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            predictions = outputs[:, 0].cpu().numpy()
            confidences = outputs[:, 1].cpu().numpy()
        
        # Create results dataframe
        results = df.copy()
        results[f'{self.prop_type}_nn_prediction'] = predictions
        results[f'{self.prop_type}_nn_confidence'] = np.clip(confidences, 0.1, 0.95)
        
        # Calculate edge signals
        if 'offered_line' in feature_df.columns:
            offered_lines = feature_df['offered_line'].values
            edge_signals = (predictions - offered_lines) / offered_lines
            results[f'{self.prop_type}_edge_signal'] = edge_signals
            results[f'{self.prop_type}_edge_strength'] = np.abs(edge_signals) * confidences
        
        logger.info(f"Neural network predictions complete. Mean prediction: {predictions.mean():.2f}")
        
        return results[[col for col in results.columns if '_nn_' in col or '_edge_' in col]]
        
    def detect_edges(self, df: pd.DataFrame, min_edge: float = 0.05, min_confidence: float = 0.7) -> pd.DataFrame:
        """Detect betting edges using neural network predictions."""
        predictions = self.predict(df)
        
        # Filter for strong edges
        edge_mask = (
            (predictions[f'{self.prop_type}_edge_strength'] > min_edge) &
            (predictions[f'{self.prop_type}_nn_confidence'] > min_confidence)
        )
        
        edges = predictions[edge_mask].copy()
        
        if len(edges) > 0:
            logger.info(f"Detected {len(edges)} edges with min_edge={min_edge}, min_confidence={min_confidence}")
        
        return edges


def create_sample_market_data() -> pd.DataFrame:
    """Create sample market data for neural network training."""
    np.random.seed(42)
    n_samples = 500
    
    data = {
        'player_id': np.random.randint(1, 51, n_samples),
        'game_id': np.random.randint(1, 100, n_samples),
        'offered_odds': np.random.choice([-110, -105, -115, -120, +100], n_samples),
        'offered_line': np.random.normal(65, 15, n_samples).clip(10, 150),
        'receiving_yards': np.random.normal(62, 18, n_samples).clip(0, 200),
        'receiving_tds': np.random.poisson(0.6, n_samples),
        'targets': np.random.normal(6, 2.5, n_samples).clip(0, 15),
        'receptions': np.random.normal(4.2, 2, n_samples).clip(0, 12),
        'is_home': np.random.choice([0, 1], n_samples),
        'opp_rank': np.random.randint(1, 33, n_samples),
        'volume': np.random.lognormal(7, 1, n_samples),
        'public_percentage': np.random.normal(50, 15, n_samples).clip(10, 90),
    }
    
    return pd.DataFrame(data)