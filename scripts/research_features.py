#!/usr/bin/env python3
"""
Research what features are available for model improvement.
"""
import nfl_data_py as nfl
import pandas as pd

print("=" * 80)
print("FEATURE RESEARCH FOR MODEL IMPROVEMENT")
print("=" * 80)

# 1. Check weekly data columns
print("\n[1/5] Exploring weekly player stats columns...")
weekly = nfl.import_weekly_data([2023])
print(f"\n✓ Loaded {len(weekly)} weekly records from 2023")
print(f"\nAvailable columns ({len(weekly.columns)}):")
for col in sorted(weekly.columns):
    print(f"  - {col}")

# Check WR/TE specific columns
print("\n[2/5] WR/TE receiving stats available:")
wr_te = weekly[weekly['position'].isin(['WR', 'TE'])].head(1)
receiving_cols = [col for col in weekly.columns if 'receiv' in col.lower() or 'target' in col.lower() or 'air' in col.lower()]
for col in receiving_cols:
    print(f"  - {col}")

# 2. Check team stats (for opponent defense)
print("\n[3/5] Exploring team-level stats...")
try:
    team_desc = nfl.import_team_desc()
    print(f"✓ Team descriptions available: {list(team_desc.columns)}")
except:
    print("✗ No team_desc dataset")

# 3. Check roster data (for snap counts, depth chart)
print("\n[4/5] Exploring roster data...")
try:
    rosters = nfl.import_rosters([2023])
    print(f"✓ Loaded {len(rosters)} roster records")
    print(f"Roster columns: {list(rosters.columns)}")
except Exception as e:
    print(f"✗ Error loading rosters: {e}")

# 4. Check snap counts
print("\n[5/5] Exploring snap count data...")
try:
    snap_counts = nfl.import_snap_counts([2023])
    print(f"✓ Loaded {len(snap_counts)} snap count records")
    print(f"Snap count columns: {list(snap_counts.columns)}")
    print("\nSample snap count data:")
    print(snap_counts.head(3))
except Exception as e:
    print(f"✗ Error loading snap counts: {e}")

# 5. Sample player to see what we have
print("\n" + "=" * 80)
print("SAMPLE PLAYER: Justin Jefferson (WR)")
print("=" * 80)
jj = weekly[(weekly['player_name'] == 'Justin Jefferson') & (weekly['season'] == 2023)].head(5)
print("\nJustin Jefferson Week 1-5 2023:")
print(jj[['week', 'opponent_team', 'targets', 'receptions', 'receiving_yards', 'target_share']].to_string())

print("\n" + "=" * 80)
print("KEY FINDINGS")
print("=" * 80)
print("""
Based on available data, we can add these features:

PLAYER-LEVEL (from weekly data):
  ✓ target_share - % of team targets (CRITICAL!)
  ✓ Recent performance - last 3 games avg
  ✓ Season averages - rolling stats
  ✓ Air yards - deep threat indicator
  ✓ Consistency - std deviation of yards

GAME-LEVEL:
  ✓ Home/away
  ✓ Opponent team

OPPONENT DEFENSE (requires calculation):
  ✓ Opponent yards allowed to position
  ✓ Opponent targets allowed to position

SNAP COUNTS (if available):
  ✓ Snap count percentage
  ✓ Offensive snaps

NOT AVAILABLE (would need external APIs):
  ✗ Vegas game totals
  ✗ Weather data
  ✗ Injury status
  ✗ Depth chart position (WR1/WR2)

RECOMMENDED PRIORITY:
  1. Target share (already in data!)
  2. Recent form (last 3 games)
  3. Opponent defense vs position
  4. Snap count percentage
  5. Air yards, target depth
""")
