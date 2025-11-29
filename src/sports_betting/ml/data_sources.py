"""
Adaptive data source layer for receiving yards prediction.

Handles multiple data sources with intelligent fallbacks:
1. Primary: nfl_data_py weekly data (most granular)
2. Fallback: PFR season-to-date stats
3. Backup: NGS receiving data
4. Emergency: Position/team averages

Each source has a quality score that affects prediction confidence.
"""
import pandas as pd
import numpy as np
import nfl_data_py as nfl
from datetime import datetime
from typing import Optional, Tuple
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


class DataQuality:
    """Track data quality and source for confidence weighting."""

    # Quality scores by source (0-1, higher is better)
    SOURCE_QUALITY = {
        'nfl_weekly': 1.0,      # Best: game-by-game data
        'pfr_season': 0.75,     # Good: season totals from PFR
        'ngs_season': 0.70,     # Good: NGS advanced stats
        'historical_avg': 0.5,  # Moderate: prior season averages
        'position_avg': 0.25,   # Poor: league-wide position average
    }

    # Recency decay factor (data loses quality over time)
    RECENCY_DECAY = 0.95  # per week

    @classmethod
    def calculate_score(
        cls,
        source: str,
        games_available: int,
        weeks_old: int = 0
    ) -> float:
        """
        Calculate quality score for data.

        Args:
            source: Data source name
            games_available: Number of games in rolling window
            weeks_old: How stale the data is

        Returns:
            Quality score (0-1)
        """
        base_score = cls.SOURCE_QUALITY.get(source, 0.1)

        # Sample size adjustment (more games = more confident)
        sample_factor = min(1.0, games_available / 5)  # Full confidence at 5+ games

        # Recency adjustment (older data is less reliable)
        recency_factor = cls.RECENCY_DECAY ** weeks_old

        return base_score * sample_factor * recency_factor


class AdaptiveDataFetcher:
    """
    Fetch player stats from the best available source.

    Automatically falls back through sources based on availability:
    1. Try nfl_data_py weekly data for current season
    2. Fall back to PFR/NGS if weekly not available
    3. Use historical averages as last resort
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize data fetcher with optional cache."""
        self.cache_dir = cache_dir or Path.home() / '.sports_betting' / 'data_cache'
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Track what data we have available
        self._data_availability = {}
        self._check_data_availability()

    def _check_data_availability(self):
        """Check which data sources are available for which years."""
        current_year = datetime.now().year

        for year in range(2020, current_year + 1):
            self._data_availability[year] = {
                'nfl_weekly': self._check_weekly_data(year),
                'pfr_season': self._check_pfr_data(year),
                'ngs_season': self._check_ngs_data(year),
            }

        logger.info(f"Data availability checked: {self._data_availability}")

    def _check_weekly_data(self, year: int) -> bool:
        """Check if weekly data is available for year."""
        cache_file = self.cache_dir / f'weekly_avail_{year}.json'

        if cache_file.exists():
            with open(cache_file) as f:
                return json.load(f).get('available', False)

        try:
            df = nfl.import_weekly_data([year])
            available = len(df) > 0

            with open(cache_file, 'w') as f:
                json.dump({'available': available, 'rows': len(df)}, f)

            return available
        except Exception as e:
            logger.warning(f"Weekly data not available for {year}: {e}")
            with open(cache_file, 'w') as f:
                json.dump({'available': False, 'error': str(e)}, f)
            return False

    def _check_pfr_data(self, year: int) -> bool:
        """Check if PFR receiving data is available for year."""
        cache_file = self.cache_dir / f'pfr_avail_{year}.json'

        if cache_file.exists():
            with open(cache_file) as f:
                return json.load(f).get('available', False)

        try:
            df = nfl.import_seasonal_pfr('rec', [year])
            available = len(df) > 0

            with open(cache_file, 'w') as f:
                json.dump({'available': available, 'rows': len(df)}, f)

            return available
        except Exception as e:
            logger.warning(f"PFR data not available for {year}: {e}")
            with open(cache_file, 'w') as f:
                json.dump({'available': False, 'error': str(e)}, f)
            return False

    def _check_ngs_data(self, year: int) -> bool:
        """Check if NGS receiving data is available for year."""
        cache_file = self.cache_dir / f'ngs_avail_{year}.json'

        if cache_file.exists():
            with open(cache_file) as f:
                return json.load(f).get('available', False)

        try:
            df = nfl.import_ngs_data('receiving', [year])
            available = len(df) > 0

            with open(cache_file, 'w') as f:
                json.dump({'available': available, 'rows': len(df)}, f)

            return available
        except Exception as e:
            logger.warning(f"NGS data not available for {year}: {e}")
            with open(cache_file, 'w') as f:
                json.dump({'available': False, 'error': str(e)}, f)
            return False

    def get_best_source(self, season: int) -> str:
        """Determine the best available data source for a season."""
        avail = self._data_availability.get(season, {})

        if avail.get('nfl_weekly'):
            return 'nfl_weekly'
        elif avail.get('pfr_season'):
            return 'pfr_season'
        elif avail.get('ngs_season'):
            return 'ngs_season'
        else:
            return 'historical_avg'

    def fetch_player_stats(
        self,
        season: int,
        week: Optional[int] = None,
        force_source: Optional[str] = None
    ) -> Tuple[pd.DataFrame, str, float]:
        """
        Fetch player receiving stats from the best available source.

        Args:
            season: NFL season year
            week: Week number (if None, fetches season-to-date)
            force_source: Force a specific source (for testing)

        Returns:
            Tuple of (DataFrame, source_name, quality_score)
        """
        source = force_source or self.get_best_source(season)
        logger.info(f"Fetching {season} data using source: {source}")

        if source == 'nfl_weekly':
            df = self._fetch_weekly(season, week)
            quality = DataQuality.calculate_score('nfl_weekly', len(df), 0)
        elif source == 'pfr_season':
            df = self._fetch_pfr_season(season)
            quality = DataQuality.calculate_score('pfr_season', len(df), 0)
        elif source == 'ngs_season':
            df = self._fetch_ngs_season(season)
            quality = DataQuality.calculate_score('ngs_season', len(df), 0)
        else:
            df = self._fetch_historical_fallback(season)
            quality = DataQuality.calculate_score('historical_avg', len(df), 52)  # ~1 year old

        return df, source, quality

    def _fetch_weekly(self, season: int, week: Optional[int] = None) -> pd.DataFrame:
        """Fetch weekly data from nfl_data_py."""
        logger.info(f"Loading weekly data for {season}")
        df = nfl.import_weekly_data([season])

        # Filter to receivers
        df = df[df['position'].isin(['WR', 'TE', 'RB'])].copy()

        if week is not None:
            df = df[df['week'] <= week]

        return df

    def _fetch_pfr_season(self, season: int) -> pd.DataFrame:
        """Fetch PFR season totals and convert to per-game averages."""
        logger.info(f"Loading PFR receiving data for {season}")
        df = nfl.import_seasonal_pfr('rec', [season])

        # Calculate per-game averages
        df['yards_per_game'] = df['yds'] / df['g'].clip(lower=1)
        df['targets_per_game'] = df['tgt'] / df['g'].clip(lower=1)
        df['receptions_per_game'] = df['rec'] / df['g'].clip(lower=1)

        # Add games played for quality calculation
        df['games_played'] = df['g']

        # Rename columns to match expected format
        df = df.rename(columns={
            'player': 'player_name',
            'tm': 'recent_team',
            'pos': 'position',
        })

        return df

    def _fetch_ngs_season(self, season: int) -> pd.DataFrame:
        """Fetch NGS receiving data."""
        logger.info(f"Loading NGS receiving data for {season}")
        df = nfl.import_ngs_data('receiving', [season])

        # NGS has advanced metrics like avg_cushion, avg_separation, etc.
        return df

    def _fetch_historical_fallback(self, season: int) -> pd.DataFrame:
        """
        Fallback: Use most recent historical data available.

        Tries years in descending order until we find data.
        """
        logger.warning(f"No direct data for {season}, using historical fallback")

        for fallback_year in range(season - 1, 2019, -1):
            if self._data_availability.get(fallback_year, {}).get('nfl_weekly'):
                logger.info(f"Using {fallback_year} weekly data as fallback")
                return self._fetch_weekly(fallback_year)
            elif self._data_availability.get(fallback_year, {}).get('pfr_season'):
                logger.info(f"Using {fallback_year} PFR data as fallback")
                return self._fetch_pfr_season(fallback_year)

        # Ultimate fallback: return empty with position averages
        logger.error("No historical data available!")
        return pd.DataFrame()

    def refresh_cache(self, season: Optional[int] = None):
        """Clear cache and re-check data availability."""
        if season:
            # Clear cache for specific season
            for pattern in ['weekly_avail', 'pfr_avail', 'ngs_avail']:
                cache_file = self.cache_dir / f'{pattern}_{season}.json'
                if cache_file.exists():
                    cache_file.unlink()

            # Re-check
            self._data_availability[season] = {
                'nfl_weekly': self._check_weekly_data(season),
                'pfr_season': self._check_pfr_data(season),
                'ngs_season': self._check_ngs_data(season),
            }
        else:
            # Clear all cache
            for f in self.cache_dir.glob('*_avail_*.json'):
                f.unlink()

            self._check_data_availability()

    def get_availability_report(self) -> dict:
        """Get a report of data availability by year."""
        return self._data_availability.copy()


class PlayerStatsFetcher:
    """
    High-level interface for fetching player stats with intelligent defaults.

    Handles:
    - Combining multiple seasons of data
    - Calculating rolling averages across data sources
    - Filling gaps with appropriate fallbacks
    """

    def __init__(self):
        self.data_fetcher = AdaptiveDataFetcher()

    def get_player_features(
        self,
        season: int,
        week: int,
        player_name: Optional[str] = None
    ) -> Tuple[pd.DataFrame, dict]:
        """
        Get player features for prediction.

        Args:
            season: Target season
            week: Target week
            player_name: Optional specific player to get

        Returns:
            Tuple of (features_df, metadata_dict)
        """
        metadata = {
            'season': season,
            'week': week,
            'sources_used': [],
            'quality_scores': {},
        }

        # Determine which data sources we need
        current_year = datetime.now().year

        if season <= current_year - 1:
            # Historical season - use weekly data if available
            df, source, quality = self.data_fetcher.fetch_player_stats(season, week)
            metadata['sources_used'].append(source)
            metadata['quality_scores'][source] = quality

        else:
            # Current/future season - blend historical + current
            # 1. Get historical data for rolling baselines
            historical_dfs = []
            for hist_year in range(2020, current_year):
                df, source, quality = self.data_fetcher.fetch_player_stats(hist_year)
                if len(df) > 0:
                    historical_dfs.append(df)
                    metadata['sources_used'].append(f"{source}_{hist_year}")

            # 2. Get current season data (whatever is available)
            current_df, source, quality = self.data_fetcher.fetch_player_stats(season, week)
            metadata['sources_used'].append(f"{source}_{season}")
            metadata['quality_scores'][f'current_{source}'] = quality

            # 3. Merge and calculate features
            if len(current_df) > 0:
                df = current_df
            elif historical_dfs:
                df = pd.concat(historical_dfs, ignore_index=True)
            else:
                df = pd.DataFrame()

        if player_name and len(df) > 0:
            df = df[df['player_name'].str.contains(player_name, case=False, na=False)]

        return df, metadata

    def calculate_adaptive_features(
        self,
        df: pd.DataFrame,
        target_week: int,
        games_available: int
    ) -> pd.DataFrame:
        """
        Calculate features with adaptive windows based on available data.

        If we have 10+ games: use 3-game and 5-game windows
        If we have 5-9 games: use 2-game and 4-game windows
        If we have 1-4 games: use all available as single window
        """
        if games_available >= 10:
            windows = [3, 5]
        elif games_available >= 5:
            windows = [2, min(4, games_available)]
        else:
            windows = [games_available] if games_available > 0 else [1]

        df = df.sort_values(['player_id', 'season', 'week'])

        for window in windows:
            # Rolling averages
            df[f'rec_yards_last_{window}'] = df.groupby('player_id')['receiving_yards'].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )

            if 'targets' in df.columns:
                df[f'targets_last_{window}'] = df.groupby('player_id')['targets'].transform(
                    lambda x: x.shift(1).rolling(window, min_periods=1).mean()
                )

            if 'target_share' in df.columns:
                df[f'target_share_last_{window}'] = df.groupby('player_id')['target_share'].transform(
                    lambda x: x.shift(1).rolling(window, min_periods=1).mean()
                )

            # Consistency
            df[f'rec_yards_std_{window}'] = df.groupby('player_id')['receiving_yards'].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).std()
            )

        # Fill NaN
        for col in df.columns:
            if df[col].dtype in [np.float64, np.float32]:
                df[col] = df[col].fillna(0)

        return df
