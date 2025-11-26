"""Enhanced feature engineering specifically for ML models."""

from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
import logging

from ..database import Game, InjuryReport, Player, Team, Prop, get_session
from .engineering import FeatureEngineer

logger = logging.getLogger(__name__)


class MLFeatureEngineer(FeatureEngineer):
    """ML-optimized feature engineering for player props."""
    
    def __init__(self):
        super().__init__()
        self.feature_cache = {}
        
    def create_ml_features(
        self,
        df: pd.DataFrame,
        target_column: str = 'receiving_yards',
        lookback_weeks: int = 8
    ) -> pd.DataFrame:
        """Create comprehensive ML features from raw player data."""
        logger.info(f"Creating ML features for {len(df)} samples")
        
        feature_df = df.copy()
        
        # Rolling window features
        feature_df = self._create_rolling_features(feature_df, lookback_weeks)
        
        # Trend features
        feature_df = self._create_trend_features(feature_df)
        
        # Opponent strength features  
        feature_df = self._create_opponent_features(feature_df)
        
        # Game context features
        feature_df = self._create_game_context_features_ml(feature_df)
        
        # Team performance features
        feature_df = self._create_team_features(feature_df)
        
        # Weather and external factors
        feature_df = self._create_external_features(feature_df)

        # Injury and health features
        feature_df = self._create_injury_features(feature_df)

        # Target encoding features
        feature_df = self._create_target_encoding_features(feature_df, target_column)

        # Interaction features
        feature_df = self._create_interaction_features(feature_df)
        
        logger.info(f"Created {len(feature_df.columns)} total features")
        
        return feature_df
        
    def _create_rolling_features(self, df: pd.DataFrame, window: int = 4) -> pd.DataFrame:
        """Create rolling window statistics."""
        logger.info(f"Creating {window}-game rolling features")
        
        df = df.sort_values(['player_id', 'game_date'])
        
        # Core stats to roll
        stats_to_roll = [
            'receiving_yards', 'receiving_tds', 'targets', 'receptions',
            'rushing_yards', 'rushing_tds', 'passing_yards', 'passing_tds',
            'snap_percentage', 'air_yards', 'yards_after_catch'
        ]
        
        for stat in stats_to_roll:
            if stat in df.columns:
                # Rolling mean
                df[f'{stat}_roll_{window}'] = (
                    df.groupby('player_id')[stat]
                    .rolling(window=window, min_periods=1)
                    .mean()
                    .reset_index(0, drop=True)
                )
                
                # Rolling standard deviation (consistency)
                df[f'{stat}_roll_{window}_std'] = (
                    df.groupby('player_id')[stat]
                    .rolling(window=window, min_periods=2)
                    .std()
                    .reset_index(0, drop=True)
                    .fillna(0)
                )
                
                # Rolling max (ceiling performance)
                df[f'{stat}_roll_{window}_max'] = (
                    df.groupby('player_id')[stat]
                    .rolling(window=window, min_periods=1)
                    .max()
                    .reset_index(0, drop=True)
                )
                
                # Recent vs older comparison
                if window >= 4:
                    df[f'{stat}_recent_2_avg'] = (
                        df.groupby('player_id')[stat]
                        .rolling(window=2, min_periods=1)
                        .mean()
                        .reset_index(0, drop=True)
                    )
                    
                    # Recent form vs longer average
                    df[f'{stat}_recent_vs_avg'] = (
                        df[f'{stat}_recent_2_avg'] - df[f'{stat}_roll_{window}']
                    )
        
        return df
        
    def _create_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create trend and momentum features."""
        logger.info("Creating trend and momentum features")
        
        df = df.sort_values(['player_id', 'game_date'])
        
        trend_stats = ['receiving_yards', 'targets', 'snap_percentage']
        
        for stat in trend_stats:
            if stat in df.columns:
                # Linear trend over last 4 games
                df[f'{stat}_trend_4'] = (
                    df.groupby('player_id')[stat]
                    .rolling(window=4, min_periods=3)
                    .apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 3 else 0)
                    .reset_index(0, drop=True)
                )
                
                # Momentum (last game vs 3-game average)
                df[f'{stat}_momentum'] = (
                    df[stat] - df.groupby('player_id')[stat]
                    .rolling(window=3, min_periods=2)
                    .mean()
                    .shift(1)
                    .reset_index(0, drop=True)
                )
                
                # Consistency score (inverse of coefficient of variation)
                rolling_mean = df.groupby('player_id')[stat].rolling(window=4, min_periods=2).mean()
                rolling_std = df.groupby('player_id')[stat].rolling(window=4, min_periods=2).std()
                df[f'{stat}_consistency'] = (
                    1 / (rolling_std / rolling_mean.clip(lower=1))
                ).fillna(1).reset_index(0, drop=True)
        
        return df
        
    def _create_opponent_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create opponent strength and matchup features."""
        logger.info("Creating opponent strength features")
        
        # Opponent defensive rankings
        if 'opponent' in df.columns:
            # Points allowed ranking
            opp_def_stats = df.groupby(['opponent', 'season']).agg({
                'opponent_points': 'mean',
                'opponent_passing_yards': 'mean',
                'opponent_rushing_yards': 'mean'
            }).reset_index()
            
            # Rank defenses (lower is better for defense)
            opp_def_stats['opp_def_rank'] = (
                opp_def_stats.groupby('season')['opponent_points']
                .rank(ascending=True)
            )
            
            opp_def_stats['opp_pass_def_rank'] = (
                opp_def_stats.groupby('season')['opponent_passing_yards']
                .rank(ascending=True)
            )
            
            # Merge back to main dataframe
            df = df.merge(
                opp_def_stats[['opponent', 'season', 'opp_def_rank', 'opp_pass_def_rank']],
                on=['opponent', 'season'],
                how='left'
            )
            
            # Fill missing with league average (16th)
            df['opp_def_rank'] = df['opp_def_rank'].fillna(16)
            df['opp_pass_def_rank'] = df['opp_pass_def_rank'].fillna(16)
            
        # Pace and efficiency features
        if 'opponent_plays_per_game' in df.columns:
            df['opponent_pace'] = df['opponent_plays_per_game']
        else:
            df['opponent_pace'] = 65  # Default NFL average
            
        return df
        
    def _create_game_context_features_ml(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create game context features optimized for ML."""
        logger.info("Creating game context features for ML")
        
        # Home field advantage
        if 'home_team' in df.columns and 'team' in df.columns:
            df['is_home'] = (df['home_team'] == df['team']).astype(int)
        else:
            df['is_home'] = 0
            
        # Division game indicator
        if 'opponent' in df.columns and 'team' in df.columns:
            # AFC/NFC East, North, South, West divisions
            divisions = {
                'AFC_EAST': ['BUF', 'MIA', 'NE', 'NYJ'],
                'AFC_NORTH': ['BAL', 'CIN', 'CLE', 'PIT'], 
                'AFC_SOUTH': ['HOU', 'IND', 'JAX', 'TEN'],
                'AFC_WEST': ['DEN', 'KC', 'LV', 'LAC'],
                'NFC_EAST': ['DAL', 'NYG', 'PHI', 'WAS'],
                'NFC_NORTH': ['CHI', 'DET', 'GB', 'MIN'],
                'NFC_SOUTH': ['ATL', 'CAR', 'NO', 'TB'],
                'NFC_WEST': ['ARI', 'LAR', 'SF', 'SEA']
            }
            
            team_divisions = {}
            for div, teams in divisions.items():
                for team in teams:
                    team_divisions[team] = div
                    
            df['team_division'] = df['team'].map(team_divisions).fillna('UNKNOWN')
            df['opponent_division'] = df['opponent'].map(team_divisions).fillna('UNKNOWN')
            df['is_division_game'] = (
                df['team_division'] == df['opponent_division']
            ).astype(int)
        else:
            df['is_division_game'] = 0
            
        # Primetime games
        if 'game_date' in df.columns:
            df['day_of_week'] = pd.to_datetime(df['game_date']).dt.dayofweek
            df['is_primetime'] = (
                (df['day_of_week'] == 6) |  # Sunday Night
                (df['day_of_week'] == 0) |  # Monday Night  
                (df['day_of_week'] == 3)    # Thursday Night
            ).astype(int)
        else:
            df['is_primetime'] = 0
            
        # Week number effects
        if 'week' in df.columns:
            df['is_early_season'] = (df['week'] <= 4).astype(int)
            df['is_late_season'] = (df['week'] >= 15).astype(int)
            df['is_playoffs'] = (df['week'] > 18).astype(int)
        
        # Rest days (if available)
        if 'days_rest' not in df.columns:
            df['days_rest'] = 7  # Standard week
            
        return df
        
    def _create_team_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create team-level performance features."""
        logger.info("Creating team performance features")
        
        team_stats = [
            'points_scored', 'points_allowed', 'total_yards', 
            'passing_yards', 'rushing_yards', 'turnovers'
        ]
        
        for stat in team_stats:
            if stat in df.columns:
                # Team rolling averages
                df[f'team_{stat}_4game'] = (
                    df.groupby('team')[stat]
                    .rolling(window=4, min_periods=1)
                    .mean()
                    .reset_index(0, drop=True)
                )
                
        # Team pace of play
        if 'plays_per_game' in df.columns:
            df['team_pace'] = df['plays_per_game']
        else:
            df['team_pace'] = 65  # NFL average
            
        # Team efficiency metrics
        if 'points_scored' in df.columns and 'total_yards' in df.columns:
            df['team_efficiency'] = df['points_scored'] / df['total_yards'].clip(lower=200)
        
        return df
        
    def _create_external_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create weather and external factor features."""
        logger.info("Creating external factor features")
        
        # Weather features (use defaults if not available)
        weather_features = {
            'temperature': 70,
            'wind_speed': 5,
            'precipitation': 0,
            'is_dome': 0
        }
        
        for feature, default in weather_features.items():
            if feature not in df.columns:
                df[feature] = default
                
        # Weather impact on passing
        df['weather_impact_passing'] = (
            (df['wind_speed'] > 10).astype(int) * 0.5 +
            (df['precipitation'] > 0).astype(int) * 0.3 +
            (df['temperature'] < 32).astype(int) * 0.2
        )

        return df

    def _create_injury_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create injury and health status features."""
        logger.info("Creating injury and health features")

        # Initialize injury features with defaults
        df['has_injury'] = 0
        df['injury_severity'] = 0  # 0=Healthy, 1=Questionable, 2=Doubtful, 3=Out, 4=IR
        df['days_since_injury'] = 999  # Large number = no recent injury
        df['is_returning_from_injury'] = 0
        df['practice_participation_score'] = 1.0  # 1.0 = full participation
        df['missed_games_injury'] = 0
        df['injury_risk_score'] = 0.0

        # If player_id and game_date are available, look up injury data
        if 'player_id' in df.columns and 'game_date' in df.columns:
            try:
                with get_session() as session:
                    for idx, row in df.iterrows():
                        player_id = row.get('player_id')
                        game_date = pd.to_datetime(row.get('game_date'))

                        if pd.isna(player_id) or pd.isna(game_date):
                            continue

                        # Find injury reports for this player around this game date
                        injury_reports = session.query(InjuryReport).filter(
                            InjuryReport.player_id == player_id,
                            InjuryReport.report_date <= game_date,
                            InjuryReport.report_date >= game_date - timedelta(days=14)  # Within 2 weeks
                        ).order_by(InjuryReport.report_date.desc()).all()

                        if injury_reports:
                            latest_report = injury_reports[0]

                            # Set injury status
                            df.at[idx, 'has_injury'] = 1

                            # Map injury status to severity
                            severity_map = {
                                'Healthy': 0,
                                'Questionable': 1,
                                'Doubtful': 2,
                                'Out': 3,
                                'IR': 4,
                                'PUP': 4,
                                'Suspended': 3
                            }
                            df.at[idx, 'injury_severity'] = severity_map.get(
                                latest_report.injury_status, 0
                            )

                            # Days since injury report
                            days_diff = (game_date - latest_report.report_date).days
                            df.at[idx, 'days_since_injury'] = max(0, days_diff)

                            # Returning from injury (was Out/IR in previous weeks, now Questionable/Healthy)
                            if len(injury_reports) > 1:
                                prev_report = injury_reports[1]
                                if prev_report.injury_status in ['Out', 'IR', 'PUP'] and \
                                   latest_report.injury_status in ['Questionable', 'Healthy']:
                                    df.at[idx, 'is_returning_from_injury'] = 1

                            # Practice participation score
                            practice_scores = {
                                'Full': 1.0,
                                'Limited': 0.5,
                                'DNP': 0.0,  # Did Not Participate
                                None: 1.0  # Assume full if not reported
                            }

                            wed_score = practice_scores.get(latest_report.practice_wednesday, 1.0)
                            thu_score = practice_scores.get(latest_report.practice_thursday, 1.0)
                            fri_score = practice_scores.get(latest_report.practice_friday, 1.0)

                            # Weight Friday most heavily as it's closest to game day
                            participation = (wed_score * 0.2 + thu_score * 0.3 + fri_score * 0.5)
                            df.at[idx, 'practice_participation_score'] = participation

                            # Games missed
                            df.at[idx, 'missed_games_injury'] = latest_report.games_missed

                            # Overall injury risk score (higher = more risk)
                            injury_risk = (
                                df.at[idx, 'injury_severity'] * 0.4 +
                                (1 - participation) * 0.3 +
                                df.at[idx, 'is_returning_from_injury'] * 0.2 +
                                min(df.at[idx, 'missed_games_injury'] / 5, 1.0) * 0.1
                            )
                            df.at[idx, 'injury_risk_score'] = injury_risk

            except Exception as e:
                logger.warning(f"Error creating injury features: {e}")

        # Create interaction features with injury status
        if 'receiving_yards_roll_4' in df.columns:
            # Adjust expected performance based on injury
            df['injury_adjusted_yards'] = (
                df['receiving_yards_roll_4'] * (1 - df['injury_risk_score'] * 0.3)
            )

        if 'snap_percentage_roll_4' in df.columns:
            # Expected snap reduction if injured
            df['injury_adjusted_snaps'] = (
                df['snap_percentage_roll_4'] * df['practice_participation_score']
            )

        logger.info("Injury features created successfully")
        return df

    def _create_target_encoding_features(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """Create target encoding features."""
        logger.info(f"Creating target encoding features for {target_column}")
        
        if target_column not in df.columns:
            return df
            
        # Player vs position average
        position_avg = df.groupby('position')[target_column].mean()
        df['player_vs_position_avg'] = (
            df[target_column] - df['position'].map(position_avg)
        )
        
        # Player vs team average
        team_avg = df.groupby('team')[target_column].mean()
        df['player_vs_team_avg'] = (
            df[target_column] - df['team'].map(team_avg)
        )
        
        # Opponent impact
        opp_allowed = df.groupby('opponent')[target_column].mean()
        df['opponent_avg_allowed'] = df['opponent'].map(opp_allowed).fillna(
            df[target_column].mean()
        )
        
        return df
        
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between key variables."""
        logger.info("Creating interaction features")
        
        # Home advantage interactions
        if 'is_home' in df.columns:
            if 'targets_roll_4' in df.columns:
                df['home_targets_interaction'] = df['is_home'] * df['targets_roll_4']
                
            if 'opp_def_rank' in df.columns:
                df['home_vs_defense_interaction'] = df['is_home'] * (32 - df['opp_def_rank'])
        
        # Pace interactions
        if 'team_pace' in df.columns and 'targets_roll_4' in df.columns:
            df['pace_targets_interaction'] = df['team_pace'] * df['targets_roll_4'] / 100
            
        # Weather interactions
        if 'weather_impact_passing' in df.columns and 'receiving_yards_roll_4' in df.columns:
            df['weather_receiving_interaction'] = (
                (1 - df['weather_impact_passing']) * df['receiving_yards_roll_4']
            )
        
        return df
        
    def get_feature_importance_groups(self) -> Dict[str, List[str]]:
        """Group features by importance categories for model interpretation."""
        return {
            'player_performance': [
                col for col in ['receiving_yards_roll_4', 'targets_roll_4', 'receptions_roll_4']
            ],
            'recent_form': [
                col for col in [] if 'recent' in col or 'momentum' in col
            ],
            'matchup_strength': [
                col for col in [] if 'opp_' in col
            ],
            'game_context': [
                col for col in [] if any(x in col for x in ['home', 'primetime', 'division'])
            ],
            'team_factors': [
                col for col in [] if 'team_' in col
            ],
            'external_factors': [
                col for col in [] if any(x in col for x in ['weather', 'temperature', 'wind'])
            ],
            'injury_health': [
                col for col in [] if any(x in col for x in ['injury', 'practice_participation', 'returning'])
            ]
        }


def prepare_training_data(df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, List[str]]:
    """Prepare data for ML training with comprehensive features."""
    
    ml_engineer = MLFeatureEngineer()
    
    # Create all ML features
    feature_df = ml_engineer.create_ml_features(df, target_column)
    
    # Select numeric features only
    numeric_columns = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove target column from features
    if target_column in numeric_columns:
        numeric_columns.remove(target_column)
    
    # Handle missing values
    feature_df[numeric_columns] = feature_df[numeric_columns].fillna(0)
    
    logger.info(f"Prepared training data with {len(numeric_columns)} features")
    
    return feature_df, numeric_columns