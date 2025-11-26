#!/usr/bin/env python3
"""
Explore available data from nfl_data_py for feature engineering.
"""
import nfl_data_py as nfl
import pandas as pd

print("=" * 60)
print("EXPLORING NFL DATA SOURCES")
print("=" * 60)

# 1. Weekly player data columns
print("\n1. WEEKLY PLAYER DATA")
print("-" * 60)
weekly = nfl.import_weekly_data([2024], columns=['player_id'])
print(f"Available columns ({len(weekly.columns)}):")
for col in sorted(weekly.columns):
    print(f"  - {col}")

# 2. Schedule data
print("\n2. SCHEDULE DATA")
print("-" * 60)
schedule = nfl.import_schedules([2024])
print(f"Available columns ({len(schedule.columns)}):")
for col in sorted(schedule.columns):
    print(f"  - {col}")

# 3. Check for team stats
print("\n3. TEAM STATS (from weekly rollup)")
print("-" * 60)
weekly_full = nfl.import_weekly_data([2024])
team_cols = [col for col in weekly_full.columns if 'team' in col.lower() or 'opponent' in col.lower()]
print(f"Team-related columns:")
for col in sorted(team_cols):
    print(f"  - {col}")

# 4. Sample schedule record to see game context
print("\n4. SAMPLE SCHEDULE RECORD (for game context)")
print("-" * 60)
sample_game = schedule.iloc[0]
print("Game info available:")
for col in ['gameday', 'weekday', 'gametime', 'home_team', 'away_team',
            'home_score', 'away_score', 'roof', 'surface', 'temp', 'wind']:
    if col in sample_game.index:
        print(f"  {col}: {sample_game[col]}")

# 5. Check for defensive stats by team
print("\n5. DEFENSIVE STATS ANALYSIS")
print("-" * 60)
print("Calculating team defensive stats from weekly data...")

# Group by opponent team to get defensive stats
defensive_stats = weekly_full.groupby('opponent_team').agg({
    'passing_yards': 'mean',
    'rushing_yards': 'mean',
    'receiving_yards': 'mean',
    'completions': 'sum',
    'attempts': 'sum',
}).round(2)

print("\nSample defensive stats (yards allowed per game):")
print(defensive_stats.head())

print("\n" + "=" * 60)
print("EXPLORATION COMPLETE")
print("=" * 60)
