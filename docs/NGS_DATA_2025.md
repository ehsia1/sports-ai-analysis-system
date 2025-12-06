# NGS Data Migration for 2025 Season

**Date**: December 2025
**Status**: Implemented

## Background

The `nfl_data_py` library's `import_weekly_data()` function returns a 404 error for the 2025 season because the yearly aggregated file isn't published until after the season ends. This created a problem for making predictions during the active 2025 season.

### Error Encountered
```
urllib.error.HTTPError: HTTP Error 404: Not Found
# When calling: nfl.import_weekly_data([2025])
```

## Solution

We discovered that **NFL Next Gen Stats (NGS)** data is available for the current season via `nfl_data_py.import_ngs_data()`. NGS provides weekly rushing, receiving, and passing stats that can substitute for the standard weekly data.

### Data Availability (Verified December 2025)
| Stat Type | Records | Weeks Available |
|-----------|---------|-----------------|
| Receiving | 761 | 1-12 |
| Rushing | 341 | 1-12 |
| Passing | 330 | 1-12 |

## Implementation Pattern

All files that load weekly data now use this pattern:

```python
def load_weekly_data(seasons: list) -> pd.DataFrame:
    """Load weekly data, using NGS for 2025+ since yearly file isn't available mid-season."""
    historical_seasons = [s for s in seasons if s < 2025]
    include_2025 = 2025 in seasons
    dfs = []

    # Load historical data (pre-2025)
    if historical_seasons:
        weekly = nfl.import_weekly_data(historical_seasons)
        dfs.append(weekly)

    # Load 2025 from NGS
    if include_2025:
        ngs_rush = nfl.import_ngs_data('rushing', [2025])
        ngs_rec = nfl.import_ngs_data('receiving', [2025])
        ngs_pass = nfl.import_ngs_data('passing', [2025])

        # Filter to regular season weekly data (exclude season totals)
        ngs_rush = ngs_rush[(ngs_rush['week'] > 0) & (ngs_rush['season_type'] == 'REG')]
        # ... transform and merge ...

        dfs.append(combined)

    return pd.concat(dfs, ignore_index=True)
```

## Column Mappings

NGS uses different column names than the standard weekly data. Here are the mappings:

### Common Columns
| NGS Column | Weekly Data Column |
|------------|-------------------|
| `player_gsis_id` | `player_id` |
| `team_abbr` | `recent_team` |
| `player_display_name` | `player_name` |

### Rushing Stats
| NGS Column | Weekly Data Column |
|------------|-------------------|
| `rush_attempts` | `carries` |
| `rush_yards` | `rushing_yards` |
| `rush_touchdowns` | `rushing_tds` |

### Receiving Stats
| NGS Column | Weekly Data Column |
|------------|-------------------|
| `yards` | `receiving_yards` |
| `rec_touchdowns` | `receiving_tds` |
| `receptions` | `receptions` (same) |
| `targets` | `targets` (same) |

### Passing Stats
| NGS Column | Weekly Data Column |
|------------|-------------------|
| `pass_yards` | `passing_yards` |
| `pass_touchdowns` | `passing_tds` |
| `attempts` | `attempts` (same) |
| `completions` | `completions` (same) |

## Files Updated

| File | Changes |
|------|---------|
| `src/sports_betting/ml/base_predictor.py` | `load_weekly_data()` method |
| `src/sports_betting/ml/feature_engineering.py` | `_load_weekly_with_ngs()` helper |
| `src/sports_betting/ml/data_sources.py` | `_check_weekly_data()`, `_fetch_weekly()` |
| `src/sports_betting/ml/stat_predictors.py` | `_load_weekly_with_ngs()` helper |
| `src/sports_betting/tracking/paper_trader.py` | `_fetch_nflverse_stats()` |
| `src/sports_betting/data/collectors/nfl_data.py` | `_load_weekly_data_with_ngs()` helper |
| `scripts/train_models_v2.py` | `load_all_weekly_data()` |
| `scripts/train_multi_stat_models.py` | `load_weekly_data()` |

## Limitations

### NGS vs Weekly Data Differences
1. **Position data**: NGS doesn't include player position in all stat types. We default receivers to "WR" when position is missing.
2. **Advanced metrics**: Some weekly_data columns (like `air_yards`, `wopr`, `racr`) are not available in NGS.
3. **Snap counts**: NGS doesn't include snap count data - must be loaded separately via `import_snap_counts()`.

### Feature Engineering Impact
Some features that depend on weekly_data-specific columns may need fallbacks:
- `target_share` - Calculated from targets and team totals
- `air_yards_share` - Not available in NGS
- `wopr` - Not available in NGS

## Testing

After implementing these changes, all 174 tests pass:
```bash
PYTHONPATH=. pytest tests/ -q
# 174 passed, 11 warnings
```

## Future Considerations

1. **Post-Season**: Once the 2025 season ends and `import_weekly_data([2025])` becomes available, we could:
   - Continue using NGS (current approach works fine)
   - Switch back to weekly_data for consistency with historical data
   - Blend both sources for richer features

2. **2026 Season**: The same pattern will apply - use NGS for current season data until the yearly file is published.

3. **Additional NGS Metrics**: NGS provides advanced metrics like `avg_cushion`, `avg_separation`, `avg_intended_air_yards` that could enhance predictions.
