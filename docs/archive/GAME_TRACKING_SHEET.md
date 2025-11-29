# Game Tracking Sheet - Thanksgiving 2024

## Strategy for Today

### Current Paper Trades Status
- **Game**: GB @ DET (December 5, 2024)
- **Trades Recorded**: 3 paper trades
- **Status**: Game hasn't happened yet

### Tracking Options

#### Option A: Wait for Dec 5 Game
**Best for**: Demonstrating the full system with our current paper trades

1. **Wait until December 5, 2024**
2. **Watch the GB @ DET game**
3. **Track these players**:
   - John FitzPatrick (TE) - OVER 7.5 receiving yards
   - Josh Jacobs (RB) - OVER 14.5 receiving yards
   - Luke Musgrave (TE) - OVER 12.5 receiving yards

4. **After game completes**:
   ```bash
   python scripts/evaluate_paper_trades.py
   # Enter actual receiving yards for each player

   python scripts/view_paper_trading_report.py
   # See ROI and performance
   ```

#### Option B: Track Today's Games Manually
**Best for**: Learning and practice

Watch today's Thanksgiving games and manually track stats:

**Games Today (Nov 28, 2024)**:
1. ✓ CHI @ DET (12:30 PM ET)
2. ✓ NYG @ DAL (4:30 PM ET)
3. ✓ MIA @ GB (8:20 PM ET)

**Where to watch stats**:
- ESPN.com/nfl/scoreboard
- NFL.com
- TheScore app
- Yahoo Sports

**Players to watch** (if you had their props):
- Any WR/TE with receiving yards props
- Track actual receiving yards after each game

**Manual tracking template**:
```
Player: _________________
Position: ___
Game: _____ @ _____
Prop: OVER/UNDER _____ yards
Odds: _____
Actual Result: _____ yards
Won/Lost: _____
```

#### Option C: Demo with Dummy Data
**Best for**: Testing the evaluation system now

Enter fictional results just to see the workflow:

```bash
python scripts/evaluate_paper_trades.py
```

Then enter dummy data:
- John FitzPatrick: 25 yards (would WIN the OVER 7.5)
- Josh Jacobs: 8 yards (would LOSE the OVER 14.5)
- Luke Musgrave: 45 yards (would WIN the OVER 12.5)

This demonstrates:
- ✓ Win/loss tracking
- ✓ P/L calculation
- ✓ ROI reporting
- ✓ Model accuracy analysis

## Recommended Approach for Today

**For immediate demo**: Use Option C (dummy data) to see the system work

**For real tracking**: Use Option A (wait for Dec 5) since that's when your paper trades are for

## Quick Reference Commands

```bash
# View current paper trades
python -c "
from src.sports_betting.database import get_session
from src.sports_betting.database.models import PaperTrade
with get_session() as session:
    trades = session.query(PaperTrade).filter(PaperTrade.won == None).all()
    print(f'{len(trades)} pending trades')
"

# Evaluate trades (enter results)
python scripts/evaluate_paper_trades.py

# View ROI report
python scripts/view_paper_trading_report.py

# View ROI for specific week
python scripts/view_paper_trading_report.py --week 14 --season 2024
```

## Notes

- **System is working**: You successfully recorded 3 paper trades
- **Timing mismatch**: Your trades are for Dec 5, not today's games
- **This is normal**: Paper trading is about demonstrating the workflow
- **Real usage**: You'd run this weekly as games approach

## For Next Week

To track NEXT week's games:

1. **Sunday before games**:
   ```bash
   python scripts/collect_daily_odds.py  # Get latest odds
   python scripts/record_paper_trades.py  # Record trades
   ```

2. **Sunday/Monday after games**:
   ```bash
   python scripts/evaluate_paper_trades.py  # Enter results
   python scripts/view_paper_trading_report.py  # Review performance
   ```

3. **Repeat weekly** to build a track record
