# Week 13 (2025) Betting Analysis & UNDER Bias Investigation

Generated: 2025-11-29 19:51

## Executive Summary

This document analyzes the betting edges identified for Week 13 (Nov 30 - Dec 2, 2025) and investigates the systematic UNDER bias observed in our ML model predictions.

### Key Metrics
- Total props analyzed: 149
- Quality edges (10%+): 96
- OVER recommendations: 71
- UNDER recommendations: 25

---

## 1. UNDER Bias Investigation

### 1.1 Initial Observation

Our ML models produced predictions with a significant UNDER bias:
- **Receiving Yards**: 21 OVER, 12 UNDER (balanced)
- **Receptions**: 8 OVER, 36 UNDER (82% UNDER)
- **Rushing Yards**: 5 OVER, 13 UNDER (72% UNDER)
- **Passing Yards**: 6 OVER, 1 UNDER (86% OVER - after fix)

### 1.2 Root Cause Analysis

We investigated whether this bias was:
1. A model artifact (systematic under-prediction)
2. A real market inefficiency (sportsbooks setting lines too high)
3. A data/feature mismatch issue

#### Findings:

| Metric | Receptions Model | Rushing Model |
|--------|-----------------|---------------|
| Model bias vs season avg | -0.16 rec/game | +0.0 yds/game |
| % predicting under avg | 56% | 32% |
| Conclusion | Slight under-prediction | Well calibrated |

#### Line Analysis vs Season Averages:

| Stat Type | Avg Line | Avg Season Per-Game | Difference |
|-----------|----------|---------------------|------------|
| Receptions | 2.78 | 2.94 | -0.16 (lines slightly low) |
| Rushing Yards | 28.8 | 28.4 | +0.4 (lines at average) |
| Receiving Yards | 27.1 | 31.2 | -4.1 (lines significantly low) |

### 1.3 Conclusions

1. **The ML models are generally well-calibrated** - predictions are close to season averages
2. **The UNDER bias in receptions is partly a model artifact** - slight conservative bias of -0.16 rec/game
3. **Many lines are set BELOW season averages** - creating OVER value, not UNDER
4. **When comparing raw season averages to lines**: 64 OVER opportunities vs 24 UNDER
5. **Specific players have wildly mispriced lines** - both inflated (UNDER value) and deflated (OVER value)

---

## 2. Model Improvement Recommendations

### 2.1 Current Model Issues

| Model | Issue | Severity | Fix |
|-------|-------|----------|-----|
| Passing Yards | Was predicting same value for all QBs | FIXED | Switched to NGS data source |
| Receptions | Slight conservative bias (-0.16/game) | Low | Consider bias correction |
| Rushing Yards | Well calibrated | None | No fix needed |
| Receiving Yards | Using adaptive predictor | None | Best performing model |

### 2.2 Recommended Improvements

#### Short-term (Before Next Week)
1. **Add recent form weighting** - Weight last 3 games more heavily than season average
2. **Incorporate injury data** - Check for recent injuries affecting snap counts
3. **Game script adjustment** - Factor in game spread (blowouts reduce stats)

#### Medium-term
1. **Train on weekly data when available** - 2025 weekly data not yet in nfl_data_py
2. **Add opponent defense metrics** - Already in model but using defaults
3. **Ensemble approach** - Combine ML predictions with season average baseline

#### Long-term
1. **Bayesian updating** - Update predictions as season progresses
2. **Player-specific models** - For high-volume players with enough history
3. **Correlation modeling** - Same-game parlays optimization

### 2.3 Feature Engineering Gaps

Current features missing that could improve predictions:
- Home/away splits
- Weather data (outdoor games)
- Rest days between games
- Opponent ranking vs position
- Red zone opportunities (for TDs if added later)

---

## 3. Confidence-Ranked Betting Sheet

### Tier Definitions

| Tier | Edge Range | Description | Recommended Stake |
|------|------------|-------------|-------------------|
| 1 | 50%+ | Highest conviction | 2-3 units |
| 2 | 30-50% | High conviction | 1-2 units |
| 3 | 20-30% | Medium conviction | 1 unit |
| 4 | 10-20% | Low conviction | 0.5 units |

### 3.1 Tier 1 - Highest Conviction (50%+ Edge)

| Bet | Player | Stat | Matchup | Line | Avg | Edge | Odds |
|-----|--------|------|---------|------|-----|------|------|
| OVER | Tua Tagovailoa | Rush Yards | NO @ MIA | 0.5 | 3.5 | +591% | -108 |
| OVER | Aaron Rodgers | Rush Yards | BUF @ PIT | 0.5 | 2.6 | +420% | +143 |
| OVER | Bucky Irving | Rec Yards | ARI @ TB | 12.5 | 48.2 | +286% | -115 |
| OVER | Deebo Samuel Sr. | Rush Yards | DEN @ WAS | 1.5 | 5.2 | +247% | -125 |
| OVER | Baker Mayfield | Rush Yards | ARI @ TB | 7.5 | 19.6 | +162% | -125 |
| OVER | Rachaad White | Rush Yards | ARI @ TB | 14.5 | 37.6 | +160% | -110 |
| OVER | Rhamondre Stevenson | Rec Yards | NYG @ NE | 7.5 | 19.3 | +158% | -120 |
| OVER | Greg Dulcich | Rec Yards | NO @ MIA | 9.5 | 22.2 | +134% | -120 |
| OVER | Jaylen Warren | Rec Yards | BUF @ PIT | 9.5 | 21.9 | +131% | -114 |
| OVER | Sterling Shepard | Receptions | ARI @ TB | 1.5 | 3.4 | +124% | -175 |
| OVER | Jeremy McNichols | Rush Yards | DEN @ WAS | 7.5 | 16.5 | +119% | -125 |
| OVER | Nick Westbrook-Ikhine | Receptions | NO @ MIA | 0.5 | 1.0 | +100% | -250 |
| OVER | Adam Trautman | Receptions | DEN @ WAS | 0.5 | 1.0 | +100% | -210 |
| OVER | Bucky Irving | Receptions | ARI @ TB | 2.5 | 4.8 | +90% | +150 |
| UNDER | Mason Tipton | Rec Yards | NO @ MIA | 21.5 | 2.2 | -90% | -115 |
| UNDER | Mason Tipton | Receptions | NO @ MIA | 2.5 | 0.3 | -88% | -185 |
| OVER | Sterling Shepard | Rec Yards | ARI @ TB | 17.5 | 32.8 | +88% | -118 |
| OVER | Austin Hooper | Rec Yards | NYG @ NE | 10.5 | 19.4 | +84% | -115 |
| OVER | Marvin Mims Jr. | Rec Yards | DEN @ WAS | 14.5 | 26.0 | +79% | -108 |
| UNDER | Devin Neal | Rush Yards | NO @ MIA | 39.5 | 8.7 | -78% | -112 |

### 3.2 Tier 2 - High Conviction (30-50% Edge)

| Bet | Player | Stat | Matchup | Line | Avg | Edge | Odds |
|-----|--------|------|---------|------|-----|------|------|
| OVER | Ty Johnson | Rush Yards | BUF @ PIT | 5.5 | 8.2 | +49% | -120 |
| OVER | James Cook | Rec Yards | BUF @ PIT | 12.5 | 18.5 | +48% | -112 |
| OVER | Bucky Irving | Rush Yards | ARI @ TB | 40.5 | 59.2 | +46% | -115 |
| OVER | James Cook | Receptions | BUF @ PIT | 1.5 | 2.2 | +45% | -179 |
| OVER | Keon Coleman | Rec Yards | BUF @ PIT | 25.5 | 36.7 | +44% | -118 |
| OVER | Foster Moreau | Rec Yards | NO @ MIA | 2.5 | 3.6 | +43% | -115 |
| UNDER | Tyler Shough | Rush Yards | NO @ MIA | 11.5 | 6.6 | -43% | -118 |
| OVER | Keon Coleman | Receptions | BUF @ PIT | 2.5 | 3.6 | +42% | +125 |
| OVER | Elijah Higgins | Rec Yards | ARI @ TB | 10.5 | 14.7 | +40% | -115 |
| OVER | Devin Singletary | Rec Yards | NYG @ NE | 5.5 | 7.7 | +39% | -110 |
| OVER | Ty Johnson | Rec Yards | BUF @ PIT | 10.5 | 14.6 | +39% | -110 |
| OVER | Pat Freiermuth | Rec Yards | BUF @ PIT | 19.5 | 27.1 | +39% | -114 |
| UNDER | Ty Johnson | Receptions | BUF @ PIT | 1.5 | 1.0 | -33% | -110 |
| UNDER | Chris Moore | Receptions | DEN @ WAS | 1.5 | 1.0 | -33% | -188 |
| OVER | Greg Dulcich | Receptions | NO @ MIA | 1.5 | 2.0 | +33% | +145 |
| OVER | Darius Slayton | Rec Yards | NYG @ NE | 30.5 | 40.7 | +33% | -113 |
| OVER | Kayshon Boutte | Rec Yards | NYG @ NE | 33.5 | 44.6 | +33% | -110 |
| UNDER | Devin Singletary | Rush Yards | NYG @ NE | 29.5 | 19.8 | -33% | -114 |
| OVER | Roman Wilson | Rec Yards | BUF @ PIT | 11.5 | 15.1 | +31% | -115 |
| UNDER | Chris Godwin | Rec Yards | ARI @ TB | 29.5 | 20.3 | -31% | -110 |

### 3.3 Tier 3 - Medium Conviction (20-30% Edge)

| Bet | Player | Stat | Matchup | Line | Avg | Edge | Odds |
|-----|--------|------|---------|------|-----|------|------|
| OVER | Rachaad White | Rec Yards | ARI @ TB | 10.5 | 13.5 | +28% | -112 |
| UNDER | Roman Wilson | Receptions | BUF @ PIT | 1.5 | 1.1 | -27% | -165 |
| OVER | Darnell Washington | Rec Yards | BUF @ PIT | 16.5 | 21.0 | +27% | -110 |
| OVER | Michael Carter | Rec Yards | ARI @ TB | 14.5 | 18.4 | +27% | -110 |
| UNDER | TreVeyon Henderson | Rush Yards | NYG @ NE | 63.5 | 46.5 | -27% | -113 |
| OVER | Rhamondre Stevenson | Receptions | NYG @ NE | 1.5 | 1.9 | +26% | +115 |
| OVER | Deebo Samuel Sr. | Rec Yards | DEN @ WAS | 37.5 | 47.0 | +25% | -111 |
| UNDER | Chris Rodriguez Jr. | Rush Yards | DEN @ WAS | 41.5 | 31.0 | -25% | -111 |
| OVER | Bo Nix | Rush Yards | DEN @ WAS | 15.5 | 19.4 | +25% | -117 |
| OVER | Drake Maye | Rush Yards | NYG @ NE | 20.5 | 25.6 | +25% | -113 |
| OVER | Darren Waller | Rec Yards | NO @ MIA | 23.5 | 29.2 | +24% | -114 |
| OVER | Rhamondre Stevenson | Rush Yards | NYG @ NE | 25.5 | 31.6 | +24% | -110 |
| UNDER | Ollie Gordon II | Rush Yards | NO @ MIA | 19.5 | 14.9 | -24% | -110 |
| UNDER | Sean Tucker | Rush Yards | ARI @ TB | 32.5 | 25.2 | -23% | -113 |
| OVER | Darnell Washington | Receptions | BUF @ PIT | 1.5 | 1.8 | +21% | -142 |
| OVER | Malik Washington | Receptions | NO @ MIA | 2.5 | 3.0 | +20% | -161 |

### 3.4 Tier 4 - Lower Conviction (10-20% Edge)

| Bet | Player | Stat | Matchup | Line | Avg | Edge | Odds |
|-----|--------|------|---------|------|-----|------|------|
| OVER | Jaxson Dart | Rush Yards | NYG @ NE | 29.5 | 35.2 | +19% | -112 |
| UNDER | Jaylen Waddle | Receptions | NO @ MIA | 5.5 | 4.5 | -19% | -160 |
| OVER | Josh Allen | Rush Yards | BUF @ PIT | 28.5 | 33.7 | +18% | -108 |
| OVER | James Cook | Rush Yards | BUF @ PIT | 83.5 | 98.5 | +18% | -111 |
| UNDER | Tyrone Tracy Jr. | Rush Yards | NYG @ NE | 48.5 | 39.8 | -18% | -110 |
| OVER | Deebo Samuel Sr. | Receptions | DEN @ WAS | 4.5 | 5.3 | +18% | -107 |
| OVER | Tyrone Tracy Jr. | Rec Yards | NYG @ NE | 17.5 | 20.6 | +18% | -108 |
| OVER | Courtland Sutton | Receptions | DEN @ WAS | 3.5 | 4.1 | +17% | -136 |
| OVER | Stefon Diggs | Rec Yards | NYG @ NE | 48.5 | 56.6 | +17% | -114 |
| OVER | Hunter Henry | Rec Yards | NYG @ NE | 38.5 | 44.8 | +16% | -114 |
| UNDER | Tyrone Tracy Jr. | Receptions | NYG @ NE | 2.5 | 2.1 | -16% | -111 |
| OVER | Calvin Austin III | Receptions | BUF @ PIT | 2.5 | 2.9 | +16% | -127 |
| OVER | Tez Johnson | Rec Yards | ARI @ TB | 24.5 | 28.2 | +15% | -109 |
| OVER | Kenneth Gainwell | Rec Yards | BUF @ PIT | 18.5 | 21.3 | +15% | -108 |
| OVER | Chris Olave | Receptions | NO @ MIA | 5.5 | 6.3 | +14% | -130 |
| OVER | Terry McLaurin | Rec Yards | DEN @ WAS | 44.5 | 50.8 | +14% | -112 |
| OVER | Stefon Diggs | Receptions | NYG @ NE | 4.5 | 5.1 | +13% | +109 |
| OVER | Courtland Sutton | Rec Yards | DEN @ WAS | 52.5 | 59.0 | +12% | -112 |
| OVER | Calvin Austin III | Rec Yards | BUF @ PIT | 27.5 | 30.9 | +12% | -112 |
| OVER | Trey McBride | Receptions | ARI @ TB | 6.5 | 7.3 | +12% | -161 |
| OVER | Wan'Dale Robinson | Rec Yards | NYG @ NE | 59.5 | 66.2 | +11% | -110 |
| OVER | Marvin Harrison Jr. | Rec Yards | ARI @ TB | 52.5 | 58.3 | +11% | -114 |
| OVER | Emeka Egbuka | Rec Yards | ARI @ TB | 61.5 | 68.1 | +11% | -118 |
| UNDER | Jaylen Waddle | Rec Yards | NO @ MIA | 73.5 | 65.6 | -11% | -114 |
| OVER | Evan Engram | Rec Yards | DEN @ WAS | 23.5 | 26.0 | +11% | -109 |
| UNDER | Pat Bryant | Rec Yards | DEN @ WAS | 23.5 | 21.0 | -11% | -115 |
| UNDER | Greg Dortch | Rec Yards | ARI @ TB | 19.5 | 17.5 | -10% | -115 |
| OVER | Michael Carter | Rush Yards | ARI @ TB | 17.5 | 19.3 | +10% | -110 |

---

## 4. Risk Factors & Caveats

### 4.1 High-Risk Bets to Avoid

Some bets show extreme edges but carry high variance:

| Player | Stat | Line | Avg | Edge | Risk Factor |
|--------|------|------|-----|------|-------------|
| Tua Tagovailoa | Rush Yards | 0.5 | 3.5 | +591% | Could easily get 0 rush yards |
| Aaron Rodgers | Rush Yards | 0.5 | 2.6 | +420% | Same - low volume stat |
| Mason Tipton | Receptions | 2.5 | 0.3 | -88% | Low sample size (may be injured/limited) |

### 4.2 Recommended Filters

Before placing any bet, verify:
1. **Player is active** - Check injury reports
2. **Role hasn't changed** - Recent depth chart moves
3. **Reasonable sample size** - At least 5+ games this season
4. **Not a backup QB game** - Game script unpredictable

### 4.3 Betting Strategy

1. **Unit sizing by tier** - Higher conviction = larger stake
2. **Diversify across games** - Don't load up on one game
3. **Mix OVER/UNDER** - Despite edge direction, maintain balance
4. **Track results** - Update models based on performance

---

## 5. Post-Game Tracking

### Results Template

| Bet | Player | Stat | Line | Pred | Actual | Result | P/L |
|-----|--------|------|------|------|--------|--------|-----|
| | | | | | | | |

### Metrics to Calculate After Games
- Win rate by tier
- Win rate by stat type
- ROI by edge range
- Model prediction accuracy (MAE, R²)

---

## 6. Appendix: Data Sources

| Data | Source | Freshness |
|------|--------|-----------|
| Player stats | nfl_data_py (PFR seasonal) | 2025 season-to-date |
| Prop lines | The Odds API | Fetched Nov 29, 2025 |
| Passing stats | NFL Next Gen Stats | 2025 season-to-date |

### Models Used

| Stat Type | Model | Test R² | Test MAE |
|-----------|-------|---------|----------|
| Receiving Yards | Adaptive XGBoost | 0.31 | ~15 yds |
| Rushing Yards | XGBoost v2 | 0.56 | 10.1 yds |
| Passing Yards | XGBoost v2 | 0.15 | 67.4 yds |
| Receptions | XGBoost v2 | 0.34 | 1.4 rec |

---

*Document generated by sports-betting analysis system*
