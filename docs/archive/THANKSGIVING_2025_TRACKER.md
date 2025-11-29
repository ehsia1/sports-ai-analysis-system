# 🦃 THANKSGIVING 2025 GAME TRACKER - November 27, 2025

## Today's Games

1. **GB @ DET** (1:00 PM EST) ← 39 betting edges found!
2. **KC @ DAL** (4:30 PM EST) ← 59 betting edges found!
3. **CIN @ BAL** (8:20 PM EST) ← 64 betting edges found!

**Total: 200 betting opportunities identified**

---

## Paper Trades Recorded

### Already Recorded (5 trades from KC @ DAL):

1. **CeeDee Lamb (WR, DAL)** - UNDER 82.5 @ -103 | $100 stake
2. **CeeDee Lamb (WR, DAL)** - UNDER 81.5 @ -115 | $100 stake
3. **CeeDee Lamb (WR, DAL)** - UNDER 79.5 @ -105 | $100 stake
4. **Saquon Barkley (RB, PHI)** - OVER 18.5 @ -115 | $100 stake
5. **Rashee Rice (WR, KC)** - UNDER 76.5 @ -115 | $100 stake

---

## Top Opportunities by Game

### GB @ DET (1:00 PM EST)

**Top Edges:**
1. Amon-Ra St. Brown - UNDER 73.5 (+108.4% EV)
2. Amon-Ra St. Brown - UNDER 83.5 (+77.2% EV)
3. John FitzPatrick - OVER 7.5 (+82.7% EV)

**Players to Watch:**
- John FitzPatrick (TE, GB) - Model predicts 49.1 yards
- Josh Jacobs (RB, GB) - Model predicts 45.6 yards
- Luke Musgrave (TE, GB) - Model predicts 35.0 yards
- Amon-Ra St. Brown (WR, DET) - Model predicts 40.9 yards
- Jahmyr Gibbs (RB, DET) - Model predicts 36.2 yards

### KC @ DAL (4:30 PM EST)

**Top Edges:**
1. CeeDee Lamb - UNDER 82.5 (+91.6% EV) ✅ **RECORDED**
2. CeeDee Lamb - UNDER 81.5 (+81.7% EV) ✅ **RECORDED**
3. CeeDee Lamb - UNDER 79.5 (+89.5% EV) ✅ **RECORDED**

**Players to Watch:**
- CeeDee Lamb (WR, DAL) - Model predicts 27.7 yards
- Rashee Rice (WR, KC) - Model predicts 30.2 yards
- Travis Kelce (TE, KC) - Model predicts 37.2 yards
- Jake Ferguson (TE, DAL) - Model predicts 35.3 yards

### CIN @ BAL (8:20 PM EST)

**Top Edges:**
1. Ja'Marr Chase - UNDER 94.5 (+82.1% EV)
2. Ja'Marr Chase - UNDER 94.5 (+82.8% EV)
3. Ja'Marr Chase - UNDER 92.5 (+82.7% EV)

**Players to Watch:**
- Ja'Marr Chase (WR, CIN) - Model predicts 34.1 yards
- Mark Andrews (TE, BAL) - Model predicts 36.9 yards
- Derrick Henry (RB, BAL) - Model predicts 22.2 yards
- Zay Flowers (WR, BAL) - Model predicts 36.4 yards
- DeAndre Hopkins (WR, CIN) - Model predicts 29.4 yards

---

## How to Track Live

### During Each Game

**Watch stats live at:**
- **ESPN.com/nfl/scoreboard** - Real-time scores and stats
- **NFL.com** - Official stats
- **TheScore app** - Mobile tracking
- **Yahoo Sports** - Live updates

**What to track:**
- Player stats → Receiving section
- Look for "REC" (receptions) and "YDS" (receiving yards)

### After All Games Complete

**Evaluate your paper trades:**
```bash
python scripts/evaluate_paper_trades.py
```

Enter the actual receiving yards for each player when prompted.

**View ROI report:**
```bash
python scripts/view_paper_trading_report.py
```

---

## Quick Stats Lookup (After Games)

```python
import nfl_data_py as nfl

# Load Week 13 2025 data
weekly = nfl.import_weekly_data([2025])
week_13 = weekly[weekly['week'] == 13]

# Check specific players
players = ['CeeDee Lamb', 'Ja\'Marr Chase', 'John FitzPatrick', 'Amon-Ra St. Brown']
for player in players:
    stats = week_13[week_13['player_name'] == player]
    if len(stats) > 0:
        print(f"{player}: {stats['receiving_yards'].values[0]} yards")
```

---

## System Performance

- **Total Props Collected:** 238 (across 3 games)
- **Predictions Generated:** 53 players
- **Edges Identified:** 200 opportunities
- **Paper Trades Recorded:** 5
- **API Credits Used:** ~6 (494 remaining)

---

## Important Notes

⚠️ **Model Accuracy Warning:**
- The model was trained on 2022-2023 data (R² = -0.012)
- These edge calculations are likely **NOT accurate** for real betting
- The EV percentages (60-100%+) are unrealistic
- This is a **DEMONSTRATION ONLY** of the workflow

✅ **What This Proves:**
- ✓ Data collection pipeline works
- ✓ Odds API integration successful
- ✓ Prediction generation functional
- ✓ Edge calculation operational
- ✓ Paper trading system complete

🎯 **Next Steps After Today:**
1. Evaluate actual results vs predictions
2. Measure model accuracy
3. Identify areas for improvement
4. Refine feature engineering
5. Re-train with better features

---

## Current Hypothetical Risk

- **Total Staked:** $500 (5 trades × $100)
- **Potential Profit:** Variable based on odds
- **Actual Risk:** $0 (paper trading only!)

**This is PAPER TRADING - no real money at risk!**

Good luck tracking! 🍀🦃

