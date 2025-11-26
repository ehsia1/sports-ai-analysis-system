"""NFL historical data collector using nfl-data-py and other sources."""

from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class NFLHistoricalDataCollector:
    """Collect historical NFL data for model training and analysis."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path("data/historical")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if nfl-data-py is available
        try:
            import nfl_data_py as nfl
            self.nfl = nfl
            self.nfl_data_available = True
            logger.info("nfl-data-py library available")
        except ImportError:
            logger.warning("nfl-data-py not available. Install with: pip install nfl-data-py")
            self.nfl = None
            self.nfl_data_available = False
    
    def collect_training_data(
        self,
        seasons: List[int],
        prop_types: List[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """Collect comprehensive training data for multiple seasons."""
        
        if prop_types is None:
            prop_types = ['receiving_yards', 'receptions', 'receiving_tds', 'passing_yards', 'rushing_yards']
        
        logger.info(f"Collecting training data for seasons {seasons}")
        
        training_data = {
            'player_stats': pd.DataFrame(),
            'game_data': pd.DataFrame(),
            'team_stats': pd.DataFrame(),
            'weather_data': pd.DataFrame(),
            'injury_data': pd.DataFrame()
        }
        
        if self.nfl_data_available:
            training_data = self._collect_nfl_data_py(seasons, prop_types)
        else:
            training_data = self._create_synthetic_training_data(seasons, prop_types)
        
        # Save to cache
        self._save_training_data_cache(training_data, seasons)
        
        return training_data
    
    def _collect_nfl_data_py(
        self,
        seasons: List[int],
        prop_types: List[str]
    ) -> Dict[str, pd.DataFrame]:
        """Collect data using nfl-data-py library."""
        
        training_data = {}
        
        logger.info("Loading NFL data from nfl-data-py...")
        
        try:
            # Weekly player stats (main data for prop predictions)
            logger.info("Loading weekly player statistics...")
            weekly_stats = self.nfl.import_weekly_data(seasons)
            
            # Filter and process weekly stats
            if not weekly_stats.empty:
                # Select relevant columns for prop prediction
                prop_columns = [
                    'season', 'week', 'player_id', 'player_name', 'position', 
                    'team', 'opponent', 'fantasy_points', 'fantasy_points_ppr'
                ]
                
                # Add prop-specific columns
                if 'receiving_yards' in prop_types:
                    prop_columns.extend(['receiving_yards', 'receptions', 'targets', 'receiving_tds'])
                if 'passing_yards' in prop_types:
                    prop_columns.extend(['passing_yards', 'passing_tds', 'interceptions', 'completions', 'attempts'])
                if 'rushing_yards' in prop_types:
                    prop_columns.extend(['rushing_yards', 'rushing_tds', 'carries'])
                
                # Select available columns
                available_columns = [col for col in prop_columns if col in weekly_stats.columns]
                player_stats = weekly_stats[available_columns].copy()
                
                # Add derived features
                player_stats = self._add_derived_features(player_stats)
                
                training_data['player_stats'] = player_stats
                logger.info(f"Loaded {len(player_stats)} player-week records")
            
            # Game-level data
            logger.info("Loading play-by-play data for game context...")
            try:
                pbp_data = self.nfl.import_pbp_data(seasons[-2:])  # Last 2 seasons to avoid memory issues
                
                if not pbp_data.empty:
                    # Aggregate game-level stats
                    game_stats = pbp_data.groupby(['season', 'week', 'home_team', 'away_team']).agg({
                        'total_home_score': 'first',
                        'total_away_score': 'first',
                        'temp': 'first',
                        'wind': 'first',
                        'weather': 'first',
                        'roof': 'first',
                        'surface': 'first'
                    }).reset_index()
                    
                    training_data['game_data'] = game_stats
                    logger.info(f"Loaded {len(game_stats)} game records")
                
            except Exception as e:
                logger.warning(f"Could not load play-by-play data: {e}")
                training_data['game_data'] = pd.DataFrame()
            
            # Team stats (for opponent strength analysis)
            logger.info("Processing team-level statistics...")
            if 'player_stats' in training_data and not training_data['player_stats'].empty:
                team_stats = self._calculate_team_stats(training_data['player_stats'])
                training_data['team_stats'] = team_stats
                logger.info(f"Calculated team stats for {len(team_stats)} team-seasons")
            
            # Roster data (for player info)
            logger.info("Loading roster data...")
            try:
                rosters = self.nfl.import_rosters(seasons)
                if not rosters.empty:
                    training_data['roster_data'] = rosters[['season', 'player_id', 'position', 'team', 'jersey_number']]
                    logger.info(f"Loaded roster data for {len(rosters)} player-seasons")
            except Exception as e:
                logger.warning(f"Could not load roster data: {e}")
            
        except Exception as e:
            logger.error(f"Error loading nfl-data-py data: {e}")
            return self._create_synthetic_training_data(seasons, prop_types)
        
        # Fill empty dataframes with synthetic data
        for key in ['game_data', 'team_stats', 'weather_data', 'injury_data']:
            if key not in training_data or training_data[key].empty:
                training_data[key] = self._create_synthetic_data(key, seasons)
        
        return training_data
    
    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features for ML training."""
        
        logger.info("Adding derived features...")
        
        # Sort by player and date for rolling calculations
        df = df.sort_values(['player_id', 'season', 'week'])
        
        # Rolling averages (4-game windows)
        stat_columns = [
            'receiving_yards', 'receptions', 'targets', 'receiving_tds',
            'passing_yards', 'passing_tds', 'completions', 'attempts',
            'rushing_yards', 'rushing_tds', 'carries'
        ]
        
        for stat in stat_columns:
            if stat in df.columns:
                # 4-game rolling average
                df[f'{stat}_4game_avg'] = (
                    df.groupby('player_id')[stat]
                    .rolling(window=4, min_periods=1)
                    .mean()
                    .reset_index(0, drop=True)
                )
                
                # Season average (expanding window)
                df[f'{stat}_season_avg'] = (
                    df.groupby(['player_id', 'season'])[stat]
                    .expanding(min_periods=1)
                    .mean()
                    .reset_index([0, 1], drop=True)
                )
                
                # Recent trend (last 3 games slope)
                df[f'{stat}_trend'] = (
                    df.groupby('player_id')[stat]
                    .rolling(window=3, min_periods=2)
                    .apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 2 else 0)
                    .reset_index(0, drop=True)
                )
        
        # Efficiency metrics
        if 'receptions' in df.columns and 'targets' in df.columns:
            df['catch_rate'] = df['receptions'] / df['targets'].clip(lower=1)
        
        if 'receiving_yards' in df.columns and 'receptions' in df.columns:
            df['yards_per_reception'] = df['receiving_yards'] / df['receptions'].clip(lower=1)
        
        if 'completions' in df.columns and 'attempts' in df.columns:
            df['completion_rate'] = df['completions'] / df['attempts'].clip(lower=1)
        
        # Game context features
        df['is_home'] = 1  # Will be properly calculated when game data is available
        df['days_rest'] = 7  # Standard week
        
        # Target share (requires team-level data)
        if 'targets' in df.columns:
            team_targets = df.groupby(['team', 'season', 'week'])['targets'].sum().reset_index()
            team_targets.columns = ['team', 'season', 'week', 'team_total_targets']
            df = df.merge(team_targets, on=['team', 'season', 'week'], how='left')
            df['target_share'] = df['targets'] / df['team_total_targets'].clip(lower=1)
        
        logger.info(f"Added derived features. Dataset shape: {df.shape}")
        
        return df
    
    def _calculate_team_stats(self, player_stats: pd.DataFrame) -> pd.DataFrame:
        """Calculate team-level statistics for opponent strength analysis."""
        
        team_stats_list = []
        
        for season in player_stats['season'].unique():
            season_data = player_stats[player_stats['season'] == season]
            
            for team in season_data['team'].unique():
                team_data = season_data[season_data['team'] == team]
                
                team_stat = {
                    'season': season,
                    'team': team,
                    'games_played': team_data['week'].nunique(),
                    'total_passing_yards': team_data['passing_yards'].sum(),
                    'total_rushing_yards': team_data['rushing_yards'].sum(),
                    'total_receiving_yards': team_data['receiving_yards'].sum(),
                    'avg_passing_yards_per_game': team_data['passing_yards'].sum() / team_data['week'].nunique(),
                    'avg_rushing_yards_per_game': team_data['rushing_yards'].sum() / team_data['week'].nunique(),
                    'total_passing_tds': team_data['passing_tds'].sum(),
                    'total_rushing_tds': team_data['rushing_tds'].sum(),
                    'total_receiving_tds': team_data['receiving_tds'].sum()
                }
                
                team_stats_list.append(team_stat)
        
        return pd.DataFrame(team_stats_list)
    
    def _create_synthetic_training_data(
        self,
        seasons: List[int],
        prop_types: List[str]
    ) -> Dict[str, pd.DataFrame]:
        """Create synthetic training data when real data is not available."""
        
        logger.info("Creating synthetic training data...")
        
        # Generate synthetic player data
        np.random.seed(42)
        
        # Define player archetypes
        players = [
            {'id': 1, 'name': 'Travis Kelce', 'position': 'TE', 'team': 'KC', 'tier': 'elite'},
            {'id': 2, 'name': 'Tyreek Hill', 'position': 'WR', 'team': 'MIA', 'tier': 'elite'},
            {'id': 3, 'name': 'Josh Allen', 'position': 'QB', 'team': 'BUF', 'tier': 'elite'},
            {'id': 4, 'name': 'Stefon Diggs', 'position': 'WR', 'team': 'BUF', 'tier': 'tier1'},
            {'id': 5, 'name': 'Cooper Kupp', 'position': 'WR', 'team': 'LAR', 'tier': 'elite'},
            {'id': 6, 'name': 'Derrick Henry', 'position': 'RB', 'team': 'TEN', 'tier': 'tier1'},
            {'id': 7, 'name': 'Davante Adams', 'position': 'WR', 'team': 'LV', 'tier': 'tier1'},
            {'id': 8, 'name': 'George Kittle', 'position': 'TE', 'team': 'SF', 'tier': 'tier1'},
            {'id': 9, 'name': 'Lamar Jackson', 'position': 'QB', 'team': 'BAL', 'tier': 'tier1'},
            {'id': 10, 'name': 'Christian McCaffrey', 'position': 'RB', 'team': 'SF', 'tier': 'elite'},
        ]
        
        # Expand player pool
        for i in range(11, 101):
            position = np.random.choice(['WR', 'TE', 'RB', 'QB'], p=[0.5, 0.1, 0.2, 0.2])
            team = np.random.choice(['KC', 'BUF', 'SF', 'LAR', 'DAL', 'NE', 'GB', 'TB', 'SEA', 'NO'])
            tier = np.random.choice(['tier1', 'tier2', 'tier3'], p=[0.2, 0.4, 0.4])
            
            players.append({
                'id': i,
                'name': f'Player_{i}',
                'position': position,
                'team': team,
                'tier': tier
            })
        
        player_stats_data = []
        
        for season in seasons:
            for week in range(1, 19):  # 18 regular season weeks
                for player in players:
                    # Skip some players some weeks (injuries, etc.)
                    if np.random.random() < 0.1:
                        continue
                    
                    # Generate opponent
                    teams = ['KC', 'BUF', 'SF', 'LAR', 'DAL', 'NE', 'GB', 'TB', 'SEA', 'NO', 'DEN', 'MIA']
                    opponent = np.random.choice([t for t in teams if t != player['team']])
                    
                    # Generate stats based on position and tier
                    stats = self._generate_player_week_stats(player, week, season, opponent)
                    stats.update({
                        'season': season,
                        'week': week,
                        'player_id': player['id'],
                        'player_name': player['name'],
                        'position': player['position'],
                        'team': player['team'],
                        'opponent': opponent
                    })
                    
                    player_stats_data.append(stats)
        
        player_stats_df = pd.DataFrame(player_stats_data)
        player_stats_df = self._add_derived_features(player_stats_df)
        
        return {
            'player_stats': player_stats_df,
            'game_data': self._create_synthetic_data('game_data', seasons),
            'team_stats': self._calculate_team_stats(player_stats_df),
            'weather_data': self._create_synthetic_data('weather_data', seasons),
            'injury_data': self._create_synthetic_data('injury_data', seasons)
        }
    
    def _generate_player_week_stats(
        self,
        player: Dict[str, Any],
        week: int,
        season: int,
        opponent: str
    ) -> Dict[str, float]:
        """Generate realistic weekly stats for a player."""
        
        position = player['position']
        tier = player['tier']
        
        # Base stats by position and tier
        base_stats = {
            'WR': {
                'elite': {'receiving_yards': 85, 'receptions': 7, 'targets': 10, 'receiving_tds': 0.8},
                'tier1': {'receiving_yards': 65, 'receptions': 5, 'targets': 8, 'receiving_tds': 0.5},
                'tier2': {'receiving_yards': 45, 'receptions': 4, 'targets': 6, 'receiving_tds': 0.3},
                'tier3': {'receiving_yards': 25, 'receptions': 2, 'targets': 4, 'receiving_tds': 0.1}
            },
            'TE': {
                'elite': {'receiving_yards': 75, 'receptions': 6, 'targets': 8, 'receiving_tds': 0.7},
                'tier1': {'receiving_yards': 55, 'receptions': 4, 'targets': 6, 'receiving_tds': 0.4},
                'tier2': {'receiving_yards': 35, 'receptions': 3, 'targets': 4, 'receiving_tds': 0.2},
                'tier3': {'receiving_yards': 20, 'receptions': 2, 'targets': 3, 'receiving_tds': 0.1}
            },
            'RB': {
                'elite': {'rushing_yards': 100, 'carries': 20, 'rushing_tds': 0.9, 'receiving_yards': 30, 'receptions': 3},
                'tier1': {'rushing_yards': 75, 'carries': 16, 'rushing_tds': 0.6, 'receiving_yards': 20, 'receptions': 2},
                'tier2': {'rushing_yards': 50, 'carries': 12, 'rushing_tds': 0.3, 'receiving_yards': 10, 'receptions': 1},
                'tier3': {'rushing_yards': 25, 'carries': 8, 'rushing_tds': 0.1, 'receiving_yards': 5, 'receptions': 0.5}
            },
            'QB': {
                'elite': {'passing_yards': 290, 'passing_tds': 2.2, 'completions': 24, 'attempts': 38, 'interceptions': 0.7},
                'tier1': {'passing_yards': 250, 'passing_tds': 1.8, 'completions': 20, 'attempts': 32, 'interceptions': 0.9},
                'tier2': {'passing_yards': 220, 'passing_tds': 1.3, 'completions': 18, 'attempts': 30, 'interceptions': 1.1},
                'tier3': {'passing_yards': 180, 'passing_tds': 0.9, 'completions': 15, 'attempts': 26, 'interceptions': 1.3}
            }
        }
        
        stats = {}
        base = base_stats.get(position, {}).get(tier, {})
        
        for stat, mean_val in base.items():
            # Add variance and some weekly randomness
            std_dev = mean_val * 0.3  # 30% coefficient of variation
            value = max(0, np.random.normal(mean_val, std_dev))
            
            # Round appropriately
            if stat in ['receiving_tds', 'rushing_tds', 'passing_tds', 'interceptions']:
                stats[stat] = np.random.poisson(value) if value > 0 else 0
            elif stat in ['receptions', 'carries', 'completions', 'attempts', 'targets']:
                stats[stat] = max(0, int(np.round(value)))
            else:
                stats[stat] = max(0, np.round(value, 1))
        
        return stats
    
    def _create_synthetic_data(self, data_type: str, seasons: List[int]) -> pd.DataFrame:
        """Create synthetic data for missing data types."""
        
        if data_type == 'game_data':
            # Simple game data
            games = []
            teams = ['KC', 'BUF', 'SF', 'LAR', 'DAL', 'NE', 'GB', 'TB', 'SEA', 'NO']
            
            for season in seasons:
                for week in range(1, 19):
                    # Generate some games for this week
                    for _ in range(8):  # 16 games per week / 2
                        home_team = np.random.choice(teams)
                        away_team = np.random.choice([t for t in teams if t != home_team])
                        
                        games.append({
                            'season': season,
                            'week': week,
                            'home_team': home_team,
                            'away_team': away_team,
                            'total_home_score': np.random.randint(10, 45),
                            'total_away_score': np.random.randint(10, 45),
                            'temp': np.random.randint(20, 80),
                            'wind': np.random.randint(0, 20),
                            'weather': np.random.choice(['clear', 'rain', 'snow', 'wind']),
                            'roof': np.random.choice(['dome', 'outdoors', 'retractable']),
                            'surface': np.random.choice(['grass', 'turf'])
                        })
            
            return pd.DataFrame(games)
        
        elif data_type == 'weather_data':
            # Simple weather data
            return pd.DataFrame([{
                'season': season,
                'week': week,
                'temperature': np.random.randint(20, 80),
                'wind_speed': np.random.randint(0, 25),
                'precipitation': np.random.choice([0, 0, 0, 1])  # Mostly no precipitation
            } for season in seasons for week in range(1, 19)])
        
        elif data_type == 'injury_data':
            # Simple injury placeholder
            return pd.DataFrame([{
                'season': season,
                'week': week,
                'injured_players': np.random.randint(0, 5)
            } for season in seasons for week in range(1, 19)])
        
        return pd.DataFrame()
    
    def _save_training_data_cache(
        self,
        training_data: Dict[str, pd.DataFrame],
        seasons: List[int]
    ) -> None:
        """Save training data to cache files."""
        
        logger.info("Saving training data to cache...")
        
        seasons_str = "_".join(map(str, seasons))
        
        for data_type, df in training_data.items():
            if not df.empty:
                cache_file = self.cache_dir / f"{data_type}_{seasons_str}.csv"
                df.to_csv(cache_file, index=False)
                logger.info(f"Saved {len(df)} records to {cache_file}")
    
    def load_cached_data(self, seasons: List[int]) -> Optional[Dict[str, pd.DataFrame]]:
        """Load previously cached training data."""
        
        seasons_str = "_".join(map(str, seasons))
        cached_data = {}
        
        data_types = ['player_stats', 'game_data', 'team_stats', 'weather_data', 'injury_data']
        
        for data_type in data_types:
            cache_file = self.cache_dir / f"{data_type}_{seasons_str}.csv"
            if cache_file.exists():
                try:
                    df = pd.read_csv(cache_file)
                    cached_data[data_type] = df
                    logger.info(f"Loaded {len(df)} records from {cache_file}")
                except Exception as e:
                    logger.warning(f"Error loading {cache_file}: {e}")
        
        if len(cached_data) == len(data_types):
            logger.info("All training data loaded from cache")
            return cached_data
        else:
            logger.info("Partial or no cached data found")
            return None
    
    def get_prop_training_data(
        self,
        prop_type: str,
        seasons: List[int],
        min_games: int = 4
    ) -> pd.DataFrame:
        """Get training data specific to a prop type."""
        
        # Try to load from cache first
        cached_data = self.load_cached_data(seasons)
        
        if cached_data is None:
            # Collect fresh data
            training_data = self.collect_training_data(seasons)
        else:
            training_data = cached_data
        
        player_stats = training_data['player_stats']
        
        if player_stats.empty:
            logger.warning("No player stats data available")
            return pd.DataFrame()
        
        # Filter for prop type
        if prop_type.startswith('receiving'):
            prop_data = player_stats[player_stats['position'].isin(['WR', 'TE', 'RB'])]
        elif prop_type.startswith('passing'):
            prop_data = player_stats[player_stats['position'] == 'QB']
        elif prop_type.startswith('rushing'):
            prop_data = player_stats[player_stats['position'].isin(['RB', 'QB'])]
        else:
            prop_data = player_stats
        
        # Filter players with minimum games
        player_game_counts = prop_data.groupby('player_id').size()
        qualified_players = player_game_counts[player_game_counts >= min_games].index
        prop_data = prop_data[prop_data['player_id'].isin(qualified_players)]
        
        # Remove rows where the target stat is missing
        if prop_type in prop_data.columns:
            prop_data = prop_data.dropna(subset=[prop_type])
        
        logger.info(f"Prepared {len(prop_data)} training samples for {prop_type}")
        
        return prop_data


def demo_historical_collector():
    """Demonstrate the historical data collector."""
    
    print("📚 NFL HISTORICAL DATA COLLECTOR DEMO")
    print("=" * 50)
    
    collector = NFLHistoricalDataCollector()
    
    print("1️⃣ AVAILABILITY CHECK")
    print("-" * 30)
    print(f"nfl-data-py available: {collector.nfl_data_available}")
    
    print("\n2️⃣ COLLECTING TRAINING DATA")
    print("-" * 30)
    
    seasons = [2022, 2023]  # Last 2 seasons
    prop_types = ['receiving_yards', 'receptions', 'passing_yards']
    
    training_data = collector.collect_training_data(seasons, prop_types)
    
    print("📊 Training Data Summary:")
    for data_type, df in training_data.items():
        print(f"   {data_type}: {len(df)} records")
        if not df.empty and 'season' in df.columns:
            seasons_in_data = df['season'].unique()
            print(f"      Seasons: {list(seasons_in_data)}")
    
    print("\n3️⃣ PROP-SPECIFIC DATA")
    print("-" * 30)
    
    for prop_type in prop_types:
        prop_data = collector.get_prop_training_data(prop_type, seasons)
        print(f"📈 {prop_type}: {len(prop_data)} training samples")
        
        if not prop_data.empty and prop_type in prop_data.columns:
            mean_val = prop_data[prop_type].mean()
            std_val = prop_data[prop_type].std()
            print(f"    Mean: {mean_val:.1f}, Std: {std_val:.1f}")
    
    print(f"\n💾 Data cached in: {collector.cache_dir}")
    
    return True


if __name__ == "__main__":
    demo_historical_collector()