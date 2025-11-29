# 🏈 TODAY'S GAME TRACKER - November 27, 2024

## Games Today
1. **GB @ DET** ← WE HAVE PAPER TRADES FOR THIS!
2. DAL @ KC
3. CIN @ BAL

## Your Paper Trades to Track

### Game: Green Bay Packers @ Detroit Lions

**3 Active Paper Trades:**

#### 1. John FitzPatrick (TE, GB)
- **Bet**: OVER 7.5 receiving yards @ -110
- **Stake**: $100
- **Model Prediction**: 49.1 yards
- **To Win**: $90.91

**Tracking**:
- [ ] Game started
- [ ] Player active/playing
- Actual receiving yards: _____
- Result: WIN / LOSS

---

#### 2. Josh Jacobs (RB, GB)
- **Bet**: OVER 14.5 receiving yards @ -114
- **Stake**: $100
- **Model Prediction**: 45.6 yards
- **To Win**: $87.72

**Tracking**:
- [ ] Game started
- [ ] Player active/playing
- Actual receiving yards: _____
- Result: WIN / LOSS

---

#### 3. Luke Musgrave (TE, GB)
- **Bet**: OVER 12.5 receiving yards @ -116
- **Stake**: $100
- **Model Prediction**: 35.0 yards
- **To Win**: $86.21

**Tracking**:
- [ ] Game started
- [ ] Player active/playing
- Actual receiving yards: _____
- Result: WIN / LOSS

---

## How to Track Live

### During the Game
**Watch stats live at**:
- ESPN.com/nfl/game (search for GB vs DET)
- NFL.com
- TheScore app
- Yahoo Sports

**What to watch**:
- Player stats → Receiving section
- Look for "REC" (receptions) and "YDS" (yards)

### After the Game
**Get final stats from**:
- ESPN.com/nfl/boxscore
- NFL.com/scores
- Pro Football Reference

**Then evaluate**:
```bash
python scripts/evaluate_paper_trades.py
```

Enter the actual receiving yards for each player when prompted.

**Then view ROI**:
```bash
python scripts/view_paper_trading_report.py
```

---

## Quick Stats Lookup

After the game completes, you can also check stats via nfl_data_py:

```python
import nfl_data_py as nfl
weekly = nfl.import_weekly_data([2024])

# Filter for today's game
game_data = weekly[
    (weekly['week'] == 14) &
    (weekly['opponent_team'] == 'DET')
]

# Check our players
for player in ['John FitzPatrick', 'Josh Jacobs', 'Luke Musgrave']:
    stats = game_data[game_data['player_name'] == player]
    if len(stats) > 0:
        print(f"{player}: {stats['receiving_yards'].values[0]} yards")
```

---

## Game Info

**Kickoff Time**: Check ESPN for exact time
**Where to Watch**: Local broadcast / NFL Sunday Ticket / Streaming

---

## Notes

- ⚠️ Check if players are ACTIVE before game (injury reports)
- John FitzPatrick is a backup TE - may have limited snaps
- Josh Jacobs is starting RB but receiving yards can be volatile
- Luke Musgrave is TE - should see regular targets

---

## Expected Outcome

Based on our quick model:
- **Predicted**: 2-3 wins (model predicts high receiving yards)
- **Reality**: Model was trained quickly, so predictions may be off
- **Learning**: This will show us model accuracy!

**Total at Risk (hypothetical)**: $300
**Potential Win**: ~$264 if all 3 hit
**Potential Loss**: -$300 if all 3 miss

This is PAPER TRADING - no real money at risk!

Good luck tracking! 🍀
