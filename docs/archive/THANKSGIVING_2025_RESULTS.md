# 🦃 Thanksgiving 2025 Paper Trading Results

**Date:** November 27, 2025 (Thanksgiving Thursday games)
**Total Trades:** 30 evaluated
**Record:** 6-24 (20.0% win rate)
**ROI:** -61.4% ($-1,843.21 loss on $3,000 staked)

---

## Summary by Bet Type

| Type | Record | Win % | Notes |
|------|--------|-------|-------|
| **OVER** | 5-6 | 45.5% | Better performance, especially on RBs |
| **UNDER** | 1-18 | 5.3% | Terrible - model underestimated elite WRs |

---

## Biggest Prediction Errors

### Model Severely Underestimated Elite WR1s

1. **CeeDee Lamb (WR, DAL)**
   - Predicted: 27.7 yards
   - Actual: 112.0 yards
   - Error: -84.3 yards (305% off!)
   - Result: Lost 9 UNDER bets on him

2. **Ja'Marr Chase (WR, CIN)**
   - Predicted: 34.1 yards
   - Actual: 110.0 yards
   - Error: -75.9 yards (223% off!)
   - Result: Lost 6 UNDER bets on him

3. **Rashee Rice (WR, KC)**
   - Predicted: 30.2 yards
   - Actual: 92.0 yards
   - Error: -61.8 yards (205% off!)
   - Result: Lost 3 UNDER bets on him

### Successful Predictions (OVER bets on RBs/TEs)

1. **Kareem Hunt (RB, KC)**: Predicted 27.9 → Actual 22.0 (+$80)
2. **Isiah Pacheco (RB, KC)**: Predicted 15.9 → Actual 17.0 (+$90.91)
3. **Javonte Williams (RB, KC)**: Predicted 31.2 → Actual 21.0 (+$86.96)
4. **Keaton Mitchell (RB, BAL)**: Predicted 22.4 → Actual 12.0 (+$87.72)
5. **Luke Musgrave (TE, GB)**: Predicted 35.0 → Actual 23.0 (+$86.21)

**Amon-Ra St. Brown UNDER** only won because he got 0 yards (likely injured/DNP).

---

## Root Cause Analysis

### Why Did the Model Fail So Badly?

1. **Training Data Issues**
   - Trained on 2022-2023 data only
   - Model R² = -0.012 (worse than random guessing)
   - No validation on recent seasons

2. **Missing Critical Features**
   - No opponent defense strength
   - No target share / snap count
   - No game script / Vegas totals
   - No weather conditions
   - No injury context
   - No recency weighting (recent games matter more)

3. **Position-Specific Patterns Missed**
   - Elite WR1s have high variance and star games
   - RBs receiving yards more predictable
   - TEs more matchup-dependent

4. **Market Line Quality**
   - Sportsbooks set lines efficiently
   - Our 60-100%+ EV estimates were unrealistic
   - True edges are typically 1-5%, not 80%+

---

## What This Proves

### ✅ Workflow is Complete & Functional

The end-to-end system works:
1. ✓ Data collection (nfl_data_py integration)
2. ✓ 2025 schedule ingestion
3. ✓ Odds API integration
4. ✓ Prop collection & storage
5. ✓ Model training & predictions
6. ✓ Edge calculation
7. ✓ Paper trading system
8. ✓ Evaluation & ROI reporting

### ❌ Model is NOT Ready for Real Betting

- Predictions are highly inaccurate
- Edge calculations are meaningless without good predictions
- Would have lost 61% of bankroll in ONE day
- Needs major improvements before any real money consideration

---

## Key Learnings

1. **Elite WRs are Underestimated**
   - Current features don't capture "star power"
   - Need target share, air yards, team context
   - Thanksgiving games may have extra variance

2. **RB Receiving More Predictable**
   - Better success rate on OVER bets for RBs
   - Lower variance than WR1s
   - Might be a better market to focus on

3. **Sample Size Matters**
   - 30 bets is small but enough to show major issues
   - With 20% win rate, clear the model has problems
   - No need for 100+ bets to know this needs work

4. **Market Efficiency is Real**
   - Sportsbooks set good lines
   - Finding 80-100% EV edges is a red flag
   - Real edges are rare and small

---

## Next Steps

### Immediate (Model Improvement)

1. **Feature Engineering Overhaul**
   - Add opponent defense metrics (DVOA, yards allowed)
   - Add target share (% of team targets)
   - Add snap count percentage
   - Add recent form (last 3 games weighted)
   - Add Vegas game total (high-scoring games = more yards)
   - Add team pass/rush rate
   - Add player role (WR1, WR2, WR3)

2. **Training Data Expansion**
   - Use 2020-2024 data (more seasons)
   - Add validation split (test on 2024)
   - Consider position-specific models

3. **Model Architecture**
   - Try different XGBoost hyperparameters
   - Consider ensemble models
   - Add feature importance analysis
   - Test on validation set before production

4. **Edge Calculation Refinement**
   - Lower edge threshold (maybe 1-3% instead of 3%+)
   - Add confidence intervals
   - Require minimum prediction accuracy threshold

### Medium-Term (Testing & Validation)

1. **Backtest on 2024 Season**
   - Generate predictions for all 2024 games
   - Compare to actual results
   - Calculate ROI on full season
   - Aim for 53%+ win rate, 3-5% ROI

2. **Continue Paper Trading**
   - Track next week's games
   - Build sample size to 50-100 bets
   - Monitor if improvements work

3. **Market Analysis**
   - Compare our lines to market closing lines
   - See if we're consistently off in one direction
   - Identify which markets/positions we're better at

### Long-Term (Production Readiness)

1. **Live Pipeline**
   - Automate weekly data collection
   - Auto-generate predictions on Thursdays
   - Auto-fetch odds and calculate edges
   - Generate daily edge reports

2. **Monitoring & Alerts**
   - Track model performance over time
   - Alert when model degrades
   - Track API usage and costs

3. **Risk Management**
   - Set betting limits (% of bankroll)
   - Kelly Criterion for bet sizing
   - Stop-loss rules
   - Minimum edge thresholds

---

## Detailed Trade Results

### Game 1: GB @ DET

| Player | Bet | Line | Actual | Result | P/L |
|--------|-----|------|--------|--------|-----|
| John FitzPatrick | OVER | 7.5 | 0.0 | ✗ | -$100 |
| Josh Jacobs | OVER | 14.5 | 8.0 | ✗ | -$100 |
| Luke Musgrave | OVER | 12.5 | 23.0 | ✓ | +$86.21 |
| Josh Whyle | OVER | 1.5 | 0.0 | ✗ | -$100 |
| Amon-Ra St. Brown | UNDER | 73.5 | 0.0 | ✓ | +$125 |

**Game 1 Total:** 2-3 (-$88.79)

### Game 2: KC @ DAL

| Player | Bet | Line | Actual | Result | P/L |
|--------|-----|------|--------|--------|-----|
| CeeDee Lamb | UNDER | 82.5 | 112.0 | ✗ | -$100 |
| CeeDee Lamb | UNDER | 81.5 | 112.0 | ✗ | -$100 |
| CeeDee Lamb | UNDER | 79.5 | 112.0 | ✗ | -$100 |
| CeeDee Lamb | UNDER | 82.5 | 112.0 | ✗ | -$100 |
| CeeDee Lamb | UNDER | 81.5 | 112.0 | ✗ | -$100 |
| CeeDee Lamb | UNDER | 79.5 | 112.0 | ✗ | -$100 |
| CeeDee Lamb | UNDER | 80.5 | 112.0 | ✗ | -$100 |
| CeeDee Lamb | UNDER | 80.5 | 112.0 | ✗ | -$100 |
| CeeDee Lamb | UNDER | 79.5 | 112.0 | ✗ | -$100 |
| Rashee Rice | UNDER | 76.5 | 92.0 | ✗ | -$100 |
| Rashee Rice | UNDER | 80.5 | 92.0 | ✗ | -$100 |
| Rashee Rice | UNDER | 77.5 | 92.0 | ✗ | -$100 |
| Javonte Williams | OVER | 9.5 | 21.0 | ✓ | +$86.96 |
| Isiah Pacheco | OVER | 5.5 | 17.0 | ✓ | +$90.91 |
| Kareem Hunt | OVER | 4.5 | 22.0 | ✓ | +80.00 |
| Hunter Luepke | OVER | 0.5 | 0.0 | ✗ | -$100 |
| Luke Schoonmaker | OVER | 1.5 | 1.0 | ✗ | -$100 |

**Game 2 Total:** 3-14 (-$1,042.13)

### Game 3: CIN @ BAL

| Player | Bet | Line | Actual | Result | P/L |
|--------|-----|------|--------|--------|-----|
| Ja'Marr Chase | UNDER | 94.5 | 110.0 | ✗ | -$100 |
| Ja'Marr Chase | UNDER | 94.5 | 110.0 | ✗ | -$100 |
| Ja'Marr Chase | UNDER | 92.5 | 110.0 | ✗ | -$100 |
| Ja'Marr Chase | UNDER | 93.5 | 110.0 | ✗ | -$100 |
| Ja'Marr Chase | UNDER | 92.5 | 110.0 | ✗ | -$100 |
| Ja'Marr Chase | UNDER | 90.5 | 110.0 | ✗ | -$100 |
| Noah Fant | OVER | 10.5 | 3.0 | ✗ | -$100 |
| Keaton Mitchell | OVER | 8.5 | 12.0 | ✓ | +$87.72 |

**Game 3 Total:** 1-7 (-$712.28)

---

## Files Updated

- `/Users/evan/code/sports-betting/THANKSGIVING_2025_RESULTS.md` (this file)
- Paper trades evaluated via `scripts/evaluate_paper_trades.py`
- ROI report generated via `scripts/view_paper_trading_report.py`

---

## Conclusion

This was an **excellent learning experience** that proved:
1. ✅ The workflow is complete and works end-to-end
2. ❌ The current model is not accurate enough for betting
3. 🎯 We know exactly what to improve next

**The -61% ROI is actually GOOD NEWS** - it means we caught the problems in paper trading before risking real money. This is exactly what paper trading is for!

Next up: Feature engineering overhaul and model retraining with 2020-2024 data.
