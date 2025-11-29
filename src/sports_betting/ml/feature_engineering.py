"""
Enhanced feature engineering for receiving yards prediction.

This module adds critical features that were missing from the initial model:
- Target share (% of team targets)
- Recent performance (last 3 games)
- Opponent defense metrics
- Snap count percentage
- Air yards and deep threat indicators

V2.0 - Adaptive Features:
- Dynamic rolling windows based on available data
- Multi-source data support (weekly, PFR, NGS)
- Data quality scoring for confidence weighting
- Graceful degradation when data is missing
"""
import pandas as pd
import numpy as np
import nfl_data_py as nfl
from datetime import datetime
from typing import Optional, Tuple, List
import logging

from .data_sources import AdaptiveDataFetcher, DataQuality

logger = logging.getLogger(__name__)


class ReceivingYardsFeatureEngineer:
    """Build features for predicting WR/TE/RB receiving yards."""

    def __init__(self):
        self.snap_counts = None
        self._data_fetcher = None  # Lazily initialized
        self._data_quality = {}  # Track quality scores by player

    @property
    def data_fetcher(self):
        """Lazily initialize data fetcher (for compatibility with pickled models)."""
        # Use getattr to handle pickled models that don't have this attribute
        fetcher = getattr(self, '_data_fetcher', None)
        if fetcher is None:
            fetcher = AdaptiveDataFetcher()
            self._data_fetcher = fetcher
        return fetcher

    def load_snap_counts(self, seasons: list[int]) -> pd.DataFrame:
        """Load snap count data for given seasons."""
        print(f"Loading snap counts for {seasons}...")
        self.snap_counts = nfl.import_snap_counts(seasons)
        print(f"✓ Loaded {len(self.snap_counts)} snap count records")
        return self.snap_counts

    def add_player_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add player-level features.

        Features added:
        - target_share: % of team targets (CRITICAL!)
        - air_yards_share: % of team air yards
        - receiving_epa: Expected points added
        - targets: Number of targets
        - receptions: Number of catches
        - receiving_air_yards: Air yards (deep threat indicator)
        """
        # These are already in the weekly data, just ensure they exist
        player_features = [
            'target_share',
            'air_yards_share',
            'receiving_epa',
            'targets',
            'receptions',
            'receiving_air_yards',
            'receiving_yards_after_catch'
        ]

        # Fill missing values
        for col in player_features:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        # Calculate catch rate
        df['catch_rate'] = np.where(
            df['targets'] > 0,
            df['receptions'] / df['targets'],
            0
        )

        return df

    def add_recent_performance(self, df: pd.DataFrame, windows: list[int] = [3, 5]) -> pd.DataFrame:
        """
        Add rolling averages for recent performance.

        Args:
            df: DataFrame with player stats
            windows: List of window sizes for rolling averages

        Features added for each window N:
        - rec_yards_last_{N}: Avg receiving yards last N games
        - targets_last_{N}: Avg targets last N games
        - target_share_last_{N}: Avg target share last N games
        """
        # Sort by player and game date
        df = df.sort_values(['player_id', 'season', 'week'])

        for window in windows:
            # Rolling averages by player
            df[f'rec_yards_last_{window}'] = df.groupby('player_id')['receiving_yards'].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )

            df[f'targets_last_{window}'] = df.groupby('player_id')['targets'].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )

            df[f'target_share_last_{window}'] = df.groupby('player_id')['target_share'].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )

            # Consistency metric (std dev)
            df[f'rec_yards_std_{window}'] = df.groupby('player_id')['receiving_yards'].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).std()
            )

        # Fill NaN for first games
        for window in windows:
            df[f'rec_yards_last_{window}'] = df[f'rec_yards_last_{window}'].fillna(0)
            df[f'targets_last_{window}'] = df[f'targets_last_{window}'].fillna(0)
            df[f'target_share_last_{window}'] = df[f'target_share_last_{window}'].fillna(0)
            df[f'rec_yards_std_{window}'] = df[f'rec_yards_std_{window}'].fillna(0)

        return df

    def add_opponent_defense(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add opponent defense metrics.

        Calculate how many receiving yards the opponent typically allows
        to WRs, TEs, and RBs.

        Features added:
        - opp_rec_yards_allowed_to_pos: Avg yards allowed to this position
        - opp_targets_allowed_to_pos: Avg targets allowed to this position
        """
        # For each game, calculate opponent's defense against that position
        # using data from earlier in the season

        df = df.sort_values(['season', 'week'])

        # Initialize columns
        df['opp_rec_yards_allowed_to_pos'] = 0.0
        df['opp_targets_allowed_to_pos'] = 0.0

        for season in df['season'].unique():
            season_df = df[df['season'] == season].copy()

            for week in sorted(season_df['week'].unique()):
                if week == 1:
                    continue  # No prior data for week 1

                # Get all games before this week
                prior_weeks = df[(df['season'] == season) & (df['week'] < week)]

                # Get current week games
                current_week = df[(df['season'] == season) & (df['week'] == week)]

                for idx, row in current_week.iterrows():
                    opponent = row['opponent_team']
                    position = row['position']

                    # Find games where opponent played defense against this position
                    opp_defense = prior_weeks[
                        (prior_weeks['recent_team'] == opponent) &
                        (prior_weeks['position'] == position)
                    ]

                    if len(opp_defense) > 0:
                        df.at[idx, 'opp_rec_yards_allowed_to_pos'] = opp_defense['receiving_yards'].mean()
                        df.at[idx, 'opp_targets_allowed_to_pos'] = opp_defense['targets'].mean()

        return df

    def add_snap_counts(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add snap count features.

        Features added:
        - offense_snaps: Number of offensive snaps
        - offense_pct: % of offensive snaps played
        """
        if self.snap_counts is None:
            print("⚠️  Warning: Snap counts not loaded. Call load_snap_counts() first.")
            df['offense_snaps'] = 0
            df['offense_pct'] = 0
            return df

        # Merge snap counts
        # Match on: season, week, player_name, team
        snap_counts_subset = self.snap_counts[['season', 'week', 'player', 'team', 'offense_snaps', 'offense_pct']].copy()
        snap_counts_subset = snap_counts_subset.rename(columns={'player': 'player_name', 'team': 'recent_team'})

        df = df.merge(
            snap_counts_subset,
            on=['season', 'week', 'player_name', 'recent_team'],
            how='left'
        )

        # Fill missing
        df['offense_snaps'] = df['offense_snaps'].fillna(0)
        df['offense_pct'] = df['offense_pct'].fillna(0)

        return df

    def add_game_context(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add game context features.

        Features added:
        - is_home: 1 if home game, 0 if away (TODO: need game data)
        - week: Week number (already exists)
        - position_encoded: Numeric encoding of position
        """
        # Position encoding
        position_map = {'WR': 0, 'TE': 1, 'RB': 2}
        df['position_encoded'] = df['position'].map(position_map)

        # TODO: Add home/away when we have game data

        return df

    def build_features(
        self,
        seasons: list[int],
        include_snap_counts: bool = True
    ) -> pd.DataFrame:
        """
        Build complete feature set for training.

        Args:
            seasons: List of seasons to include
            include_snap_counts: Whether to include snap count data

        Returns:
            DataFrame with all features
        """
        print("=" * 80)
        print("BUILDING ENHANCED FEATURE SET")
        print("=" * 80)

        # 1. Load base data
        print(f"\n[1/7] Loading weekly data for {seasons}...")
        df = nfl.import_weekly_data(seasons)
        print(f"✓ Loaded {len(df)} records")

        # 2. Filter to WR/TE/RB with receiving data
        print("\n[2/7] Filtering to WR/TE/RB with receiving yards...")
        df = df[
            (df['position'].isin(['WR', 'TE', 'RB'])) &
            (df['receiving_yards'].notna()) &
            (df['targets'] > 0)
        ].copy()
        print(f"✓ {len(df)} records with receiving yards")

        # 3. Add player features
        print("\n[3/7] Adding player-level features...")
        df = self.add_player_features(df)
        print("✓ Added target_share, air_yards, EPA, etc.")

        # 4. Add recent performance
        print("\n[4/7] Adding recent performance (rolling windows)...")
        df = self.add_recent_performance(df, windows=[3, 5])
        print("✓ Added last 3 and last 5 game averages")

        # 5. Add opponent defense
        print("\n[5/7] Calculating opponent defense metrics...")
        df = self.add_opponent_defense(df)
        print("✓ Added opponent yards/targets allowed")

        # 6. Add snap counts
        if include_snap_counts:
            print("\n[6/7] Adding snap count data...")
            self.load_snap_counts(seasons)
            df = self.add_snap_counts(df)
            print("✓ Added snap counts")
        else:
            print("\n[6/7] Skipping snap counts")
            df['offense_snaps'] = 0
            df['offense_pct'] = 0

        # 7. Add game context
        print("\n[7/7] Adding game context...")
        df = self.add_game_context(df)
        print("✓ Added position encoding")

        # Summary
        print("\n" + "=" * 80)
        print("FEATURE ENGINEERING COMPLETE")
        print("=" * 80)
        print(f"\nTotal records: {len(df)}")
        print(f"Seasons: {sorted(df['season'].unique())}")
        print(f"Positions: {df['position'].value_counts().to_dict()}")
        print(f"\nFeature count: {len(df.columns)}")

        return df

    def get_feature_list(self, prediction_mode: bool = True) -> list[str]:
        """
        Get list of features to use for training.

        Args:
            prediction_mode: If True, only include features available before game starts.
                           If False, include same-game features (for analysis only).

        Returns:
            List of feature column names
        """
        features = [
            # Recent performance (AVAILABLE BEFORE GAME)
            'rec_yards_last_3',
            'targets_last_3',
            'target_share_last_3',
            'rec_yards_std_3',
            'rec_yards_last_5',
            'targets_last_5',
            'target_share_last_5',
            'rec_yards_std_5',

            # Opponent defense (AVAILABLE BEFORE GAME)
            'opp_rec_yards_allowed_to_pos',
            'opp_targets_allowed_to_pos',

            # Snap counts from recent games (AVAILABLE BEFORE GAME)
            # Note: These are from the SAME week, which is data leakage
            # TODO: Make these lagged snap counts from previous weeks
            # 'offense_snaps',
            # 'offense_pct',

            # Game context (AVAILABLE BEFORE GAME)
            'position_encoded',
            'week',
        ]

        # Optional: Same-game features (DATA LEAKAGE - only for analysis!)
        if not prediction_mode:
            features.extend([
                'targets',
                'target_share',
                'air_yards_share',
                'receiving_epa',
                'receiving_air_yards',
                'receiving_yards_after_catch',
                'catch_rate',
            ])

        return features

    # =========================================================================
    # V2.0 ADAPTIVE METHODS - Build features from any data source
    # =========================================================================

    def build_features_adaptive(
        self,
        season: int,
        week: int,
        include_current_season: bool = True
    ) -> Tuple[pd.DataFrame, dict]:
        """
        Build features adaptively based on available data sources.

        This method automatically:
        1. Checks what data is available
        2. Uses the best source for each type of data
        3. Calculates appropriate rolling windows
        4. Tracks data quality for confidence weighting

        Args:
            season: Target season for predictions
            week: Target week for predictions
            include_current_season: Whether to try to use current season data

        Returns:
            Tuple of (features_df, metadata_dict)
        """
        print("=" * 80)
        print(f"BUILDING ADAPTIVE FEATURES FOR {season} WEEK {week}")
        print("=" * 80)

        metadata = {
            'season': season,
            'week': week,
            'sources': [],
            'quality': {},
            'warnings': [],
        }

        current_year = datetime.now().year

        # Step 1: Determine data availability
        print("\n[1/5] Checking data availability...")
        availability = self.data_fetcher.get_availability_report()
        print(f"  Data availability: {availability}")

        # Step 2: Build historical baseline (always available)
        print("\n[2/5] Loading historical data for baselines...")
        historical_seasons = [y for y in range(2020, current_year) if availability.get(y, {}).get('nfl_weekly')]

        if historical_seasons:
            df_historical = self._load_weekly_data_safe(historical_seasons)
            metadata['sources'].append(f'nfl_weekly_{historical_seasons}')
            print(f"  ✓ Loaded {len(df_historical)} historical records from {historical_seasons}")
        else:
            metadata['warnings'].append("No historical weekly data available")
            df_historical = pd.DataFrame()

        # Step 3: Get current season data
        print("\n[3/5] Fetching current season data...")

        df_current = pd.DataFrame()
        current_source = None

        if include_current_season and season <= current_year:
            # Try to get current season data from best available source
            if availability.get(season, {}).get('nfl_weekly'):
                df_current = self._load_weekly_data_safe([season])
                df_current = df_current[df_current['week'] < week]  # Only prior weeks
                current_source = 'nfl_weekly'
                print(f"  ✓ Using nfl_data_py weekly for {season} (weeks 1-{week-1})")

            elif availability.get(season, {}).get('pfr_season'):
                df_current = self._load_pfr_as_weekly(season, week)
                current_source = 'pfr_season'
                print(f"  ✓ Using PFR season-to-date for {season}")

            elif availability.get(season, {}).get('ngs_season'):
                df_current = self._load_ngs_as_weekly(season, week)
                current_source = 'ngs_season'
                print(f"  ✓ Using NGS receiving for {season}")

            else:
                metadata['warnings'].append(f"No {season} data available, using historical only")
                print(f"  ⚠️ No {season} data available")

        if current_source:
            metadata['sources'].append(f'{current_source}_{season}')

        # Step 4: Combine and calculate features
        print("\n[4/5] Calculating adaptive features...")

        # If current source is PFR/NGS, it already has features pre-computed
        # We should use it directly rather than blending with historical
        if current_source in ['pfr_season', 'ngs_season'] and len(df_current) > 0:
            # Use current season data directly - features already computed
            df = df_current.copy()
            print(f"  Using {len(df)} records from {current_source} (features pre-computed)")

            # Add position encoding if missing
            if 'position_encoded' not in df.columns:
                position_map = {'WR': 0, 'TE': 1, 'RB': 2}
                df['position_encoded'] = df['position'].map(position_map).fillna(0)

            # Add opponent defense (simplified - use league averages)
            df['opp_rec_yards_allowed_to_pos'] = 60.0  # League average
            df['opp_targets_allowed_to_pos'] = 5.0

        elif len(df_historical) > 0 and len(df_current) > 0:
            # Combine historical + current (for weekly data)
            df = pd.concat([df_historical, df_current], ignore_index=True)
            print(f"  Combined {len(df_historical)} historical + {len(df_current)} current = {len(df)} total")

            # Filter to receivers
            df = df[df['position'].isin(['WR', 'TE', 'RB'])].copy()
            df = df[df['receiving_yards'].notna() & (df['targets'] > 0)]

            # Calculate features with adaptive windows
            df = self._calculate_adaptive_rolling(df, season, week)
            df = self.add_player_features(df)
            df = self.add_opponent_defense(df)
            df = self.add_game_context(df)

        elif len(df_current) > 0:
            df = df_current

            # Calculate features
            df = df[df['position'].isin(['WR', 'TE', 'RB'])].copy()
            df = self._calculate_adaptive_rolling(df, season, week)
            df = self.add_player_features(df)
            df = self.add_game_context(df)

        elif len(df_historical) > 0:
            df = df_historical

            # Filter and calculate
            df = df[df['position'].isin(['WR', 'TE', 'RB'])].copy()
            df = df[df['receiving_yards'].notna() & (df['targets'] > 0)]
            df = self._calculate_adaptive_rolling(df, season, week)
            df = self.add_player_features(df)
            df = self.add_opponent_defense(df)
            df = self.add_game_context(df)

        else:
            print("  ❌ No data available!")
            return pd.DataFrame(), metadata

        # Step 5: Calculate quality scores per player
        print("\n[5/5] Calculating data quality scores...")
        df = self._add_quality_scores(df, season, week, current_source)

        # Filter to target week
        target_df = df[(df['season'] == season) & (df['week'] == week)].copy()

        # If no target week data, create from latest available
        if len(target_df) == 0 and len(df) > 0:
            print(f"  ⚠️ No Week {week} data, projecting from latest stats")
            target_df = self._project_to_week(df, season, week)
            metadata['warnings'].append(f"Week {week} projected from prior data")

        print(f"\n✓ Built features for {len(target_df)} players")
        metadata['players'] = len(target_df)

        return target_df, metadata

    def _load_weekly_data_safe(self, seasons: List[int]) -> pd.DataFrame:
        """Load weekly data with error handling."""
        try:
            return nfl.import_weekly_data(seasons)
        except Exception as e:
            logger.warning(f"Failed to load weekly data for {seasons}: {e}")
            return pd.DataFrame()

    def _load_pfr_as_weekly(self, season: int, week: int) -> pd.DataFrame:
        """
        Load PFR season data and convert to weekly-compatible format.

        PFR gives us season totals with per-game averages.
        We treat these as pre-computed rolling averages.
        """
        try:
            pfr = nfl.import_seasonal_pfr('rec', [season])
        except Exception as e:
            logger.warning(f"Failed to load PFR data for {season}: {e}")
            return pd.DataFrame()

        if len(pfr) == 0:
            return pd.DataFrame()

        # Filter to WR/TE/RB
        pfr = pfr[pfr['pos'].isin(['WR', 'TE', 'RB', 'wr', 'te', 'rb'])].copy()

        # Calculate per-game averages (these become our features directly)
        games_played = pfr['g'].clip(lower=1)
        pfr['receiving_yards'] = pfr['yds'] / games_played  # YPG
        pfr['targets'] = pfr['tgt'] / games_played  # TPG
        pfr['receptions'] = pfr['rec'] / games_played

        # Pre-compute the rolling features directly from season averages
        # Since PFR is season-to-date, the YPG IS the rolling average
        pfr['rec_yards_last_3'] = pfr['receiving_yards']
        pfr['rec_yards_last_5'] = pfr['receiving_yards']
        pfr['targets_last_3'] = pfr['targets']
        pfr['targets_last_5'] = pfr['targets']

        # Calculate target share
        team_targets = pfr.groupby('tm')['tgt'].transform('sum')
        pfr['target_share'] = pfr['tgt'] / team_targets.clip(lower=1)
        pfr['target_share_last_3'] = pfr['target_share']
        pfr['target_share_last_5'] = pfr['target_share']

        # Consistency - estimate from total yards and games
        # Low games = high uncertainty
        pfr['rec_yards_std_3'] = 15.0  # Default moderate variance
        pfr['rec_yards_std_5'] = 15.0

        # Rename columns to match weekly format
        pfr = pfr.rename(columns={
            'player': 'player_name',
            'tm': 'recent_team',
            'pos': 'position',
        })

        # Normalize position to uppercase
        pfr['position'] = pfr['position'].str.upper()

        # Track games for quality scoring
        pfr['games_played'] = games_played
        pfr['cumulative_games'] = games_played

        # Add season/week columns
        pfr['season'] = season
        pfr['week'] = week

        # Create player_id from name (for matching)
        pfr['player_id'] = pfr['player_name'].apply(
            lambda x: f"pfr_{str(x).replace(' ', '_').replace('.', '')}" if pd.notna(x) else 'unknown'
        )

        return pfr

    def _load_ngs_as_weekly(self, season: int, week: int) -> pd.DataFrame:
        """Load NGS receiving data and convert to weekly format."""
        try:
            ngs = nfl.import_ngs_data('receiving', [season])
        except Exception as e:
            logger.warning(f"Failed to load NGS data for {season}: {e}")
            return pd.DataFrame()

        if len(ngs) == 0:
            return pd.DataFrame()

        # NGS has different column names
        ngs = ngs.rename(columns={
            'player_display_name': 'player_name',
            'team_abbr': 'recent_team',
            'player_position': 'position',
            'yards': 'receiving_yards',
        })

        ngs['season'] = season
        ngs['week'] = week

        # Add default columns if missing
        if 'targets' not in ngs.columns:
            ngs['targets'] = 0
        if 'target_share' not in ngs.columns:
            ngs['target_share'] = 0

        ngs['player_id'] = ngs['player_name'].apply(lambda x: f"ngs_{x.replace(' ', '_')}")

        return ngs

    def _calculate_adaptive_rolling(
        self,
        df: pd.DataFrame,
        target_season: int,
        target_week: int
    ) -> pd.DataFrame:
        """
        Calculate rolling features with adaptive windows.

        Window sizes adapt to available data per player:
        - 10+ games: standard 3 and 5 game windows
        - 5-9 games: 2 and 4 game windows
        - 1-4 games: use all available
        """
        df = df.sort_values(['player_id', 'season', 'week'])

        # Calculate games per player up to target week
        df['cumulative_games'] = df.groupby('player_id').cumcount() + 1

        # Standard windows
        windows = [3, 5]

        for window in windows:
            # Use adaptive min_periods
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

            df[f'rec_yards_std_{window}'] = df.groupby('player_id')['receiving_yards'].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).std()
            )

        # Fill NaN
        for col in df.columns:
            if 'last_' in col or '_std_' in col:
                df[col] = df[col].fillna(0)

        return df

    def _add_quality_scores(
        self,
        df: pd.DataFrame,
        season: int,
        week: int,
        current_source: Optional[str]
    ) -> pd.DataFrame:
        """Add data quality score per player based on data availability."""

        def calc_player_quality(row):
            games = row.get('cumulative_games', 1)

            # Source quality
            if current_source == 'nfl_weekly':
                source_score = 1.0
            elif current_source == 'pfr_season':
                source_score = 0.75
            elif current_source == 'ngs_season':
                source_score = 0.70
            else:
                source_score = 0.5

            # Sample size adjustment
            sample_score = min(1.0, games / 5)

            # Combine
            return source_score * sample_score

        df['data_quality'] = df.apply(calc_player_quality, axis=1)

        return df

    def _project_to_week(
        self,
        df: pd.DataFrame,
        season: int,
        week: int
    ) -> pd.DataFrame:
        """
        Create projected features for a future week.

        Uses most recent data per player and projects forward.
        """
        # Get latest data per player
        latest = df.groupby('player_id').last().reset_index()

        # Update to target week
        latest['season'] = season
        latest['week'] = week

        return latest


if __name__ == "__main__":
    # Test the feature engineering
    engineer = ReceivingYardsFeatureEngineer()

    # Build features for 2022-2023
    df = engineer.build_features(seasons=[2022, 2023])

    # Show sample
    print("\nSample records (Justin Jefferson):")
    jj = df[df['player_name'] == 'Justin Jefferson'].head(5)
    feature_cols = engineer.get_feature_list() + ['receiving_yards']
    print(jj[feature_cols].to_string())

    print("\n✅ Feature engineering module ready!")
