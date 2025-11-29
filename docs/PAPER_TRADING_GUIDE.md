# Paper Trading Guide - Validate the System Risk-Free

## What is Paper Trading?

**Paper trading** = Tracking hypothetical bets WITHOUT risking real money.

This lets you:
- ✅ Validate model accuracy against real markets
- ✅ Learn which edges actually win
- ✅ Build confidence before real money
- ✅ Understand variance (even good bets lose)
- ✅ Refine your strategy risk-free

## Complete Workflow

### Step 1: Collect Odds & Find Edges

```bash
# Daily: Fetch odds and identify opportunities
python scripts/collect_daily_odds.py
```

**What this script does internally:**

1. **Checks for cached odds**
   - Looks for today's odds in `~/.sports_betting/odds_cache/`
   - If found and < 24 hours old, uses cache (saves credits)
   - If not found or stale, proceeds to fetch

2. **Fetches from The Odds API** (if needed)
   - Makes request for NFL player props (receiving_yards market)
   - Cost: 1 credit per market per region = 1 credit total
   - Receives JSON with all available props from multiple bookmakers

3. **Caches the response**
   - Saves to local JSON file with timestamp
   - Next run today will use cache instead of fetching again
   - Prevents wasting API credits on multiple runs

4. **Stores props in database**
   - Parses odds data for each player
   - Finds matching Player and Game records
   - Creates/updates Prop records with:
     - Line (e.g., 84.5 yards)
     - Over odds (e.g., -110)
     - Under odds (e.g., -110)
     - Bookmaker (DraftKings, FanDuel, etc.)

5. **Finds edges by comparing to model**
   - Queries for Prediction records (from your trained models)
   - For each prop:
     - Calculates fair probability (removes vig from odds)
     - Compares model probability to market probability
     - Computes edge = model_prob - market_prob
     - Calculates expected value (EV)
   - Filters to only edges > 3% (configurable)

6. **Displays opportunities**
   - Shows best bets ranked by EV
   - Includes edge %, EV %, line, odds
   - Ready for you to review

**Output:**
```
✓ Found 3 betting opportunities!

1. Justin Jefferson - Receiving Yards
   >>> BET OVER 84.5
   >>> Edge: +5.2%
   >>> Expected Value: +8.3%
```

**Behind the scenes data:**
```python
edge = {
    'player': 'Justin Jefferson',
    'market': 'receiving_yards',
    'line': 84.5,
    'prediction': 91.3,
    'model_confidence': 0.872,
    'over': {
        'odds': -110,
        'market_probability': 0.50,  # After removing vig
        'model_probability': 0.682,   # From model prediction
        'edge': 0.182,  # 18.2%!
        'edge_pct': 18.2,
        'ev': 0.083,
        'ev_pct': 8.3,
        'should_bet': True
    },
    'under': {...}
}
```

### Step 2: Record Paper Trades

```bash
# Record hypothetical bets
python scripts/record_paper_trades.py
```

**What this script does internally:**

1. **Loads edges from Step 1**
   - Runs same edge calculation as collect_daily_odds.py
   - Gets all opportunities with edge > 3%
   - Sorts by expected value (best bets first)

2. **Displays numbered list**
   - Shows each betting opportunity
   - Includes player, prop type, side (over/under), EV

3. **Prompts for selection**
   - Interactive: "Which bets do you want to record?"
   - You enter: "1,3,5" or "all"
   - Validates your selection

4. **For each selected bet:**

   **a. Queries database for context**
   ```python
   # Gets player record
   player = session.query(Player).filter_by(name='Justin Jefferson').first()

   # Gets game record
   game = session.query(Game).filter_by(
       away_team='MIN', home_team='CHI'
   ).first()
   ```

   **b. Generates detailed reasoning**
   - Calls `PaperTrader.generate_reasoning(player, game, edge_data)`
   - Analyzes recent performance (queries last 3-5 games)
   - Calculates target share (for WR/TE) or touch share (for RB)
   - Looks up matchup data (defensive rankings)
   - Checks injury status from InjuryReport table
   - Identifies game context (weather, venue, primetime)
   - Compiles confidence factors (what supports this)
   - Compiles risk factors (what could go wrong)

   **c. Creates PaperTrade record**
   ```python
   paper_trade = PaperTrade(
       game_id=game.id,
       player_id=player.id,
       market='receiving_yards',
       bet_side='over',
       line=84.5,
       odds=-110,
       stake=100.0,  # Hypothetical $100
       model_prediction=91.3,
       model_confidence=0.872,
       edge_percentage=5.2,
       expected_value=8.3,
       reasoning={
           'primary_factor': 'Edge: +5.2%',
           'player_form': 'Last 3 games: 91.3 avg',
           'target_share': '28.5% (team leader)',
           'matchup': 'vs 28th ranked pass defense',
           'confidence_factors': ['High edge', 'Strong form'],
           'risk_factors': []
       }
   )
   session.add(paper_trade)
   session.commit()
   ```

   **d. Displays full reasoning**
   - Formats the reasoning dict into readable output
   - Shows all factors that went into the decision
   - Saves to database for later analysis

5. **Summary**
   - Shows count of recorded trades
   - Reminds you these are hypothetical
   - Tells you next steps

**Interactive selection:**
```
Which bets do you want to record?
Enter numbers: 1,3   (or 'all')

Recording paper trades...
```

**For each bet, you'll see detailed reasoning:**

```
============================================================
Justin Jefferson (WR) - Receiving Yards
Game: MIN @ CHI
Date: 2024-12-01 01:00 PM

BET: OVER 84.5
Odds: -110
Stake: $100.00

Model Prediction: 91.3
Model Confidence: 87.2%
Edge: +5.2%
Expected Value: +8.3%

REASONING:
------------------------------------------------------------
Primary Factor: Edge: +5.2%
Player Form: Last 3 games: 91.3 avg (trending up)
Target Share: Team-leading 28.5% target share
Matchup: vs 28th ranked pass defense (favorable)
Venue: Dome game (no weather concerns)
Injury Status: Healthy, full participant all week
Model Confidence: 87.2%
Prediction Vs Line: 91.3 vs 84.5 (+6.8 yards)

Confidence Factors:
  • High edge (>5%)
  • High model confidence
  • Strong recent form
  • Favorable matchup

Risk Factors:
  • (none identified)
============================================================

✓ Recorded 2 paper trades
```

### Step 3: Wait for Games

Games complete... 📺

### Step 4: Evaluate Results

```bash
# After games, enter actual results
python scripts/evaluate_paper_trades.py
```

**What this script does internally:**

1. **Queries pending trades**
   ```python
   pending = session.query(PaperTrade).filter(
       PaperTrade.won.is_(None)  # Not yet evaluated
   ).all()
   ```
   - Gets all paper trades that haven't been evaluated
   - These are bets where you haven't entered the actual result yet

2. **Loops through each pending trade**
   - Re-attaches to database session
   - Loads player and game information
   - Displays bet details for your reference

3. **Prompts for actual result**
   ```
   Enter actual receiving yards (or 'skip'):
   ```
   - You manually enter the actual stat from the game
   - e.g., if Justin Jefferson had 87 receiving yards, enter "87"
   - Validates input (must be a number)
   - Option to skip if you don't have the data yet

4. **Evaluates the bet**

   **a. Determines if bet won**
   ```python
   if bet_side == 'over':
       won = actual_result > line  # 87.0 > 84.5 = True
   else:  # under
       won = actual_result < line
   ```

   **b. Calculates profit/loss**
   ```python
   if won:
       if odds > 0:  # e.g., +150
           profit = stake * (odds / 100)
       else:  # e.g., -110
           profit = stake * (100 / abs(odds))
           # $100 * (100/110) = $90.91
   else:
       profit = -stake  # Lost $100
   ```

   **c. Calculates model error**
   ```python
   error = abs(prediction - actual_result)
   # abs(91.3 - 87.0) = 4.3 yards
   ```

5. **Updates database record**
   ```python
   trade.actual_result = 87.0
   trade.won = True
   trade.profit_loss = +90.91
   trade.evaluated_at = datetime.now()
   session.commit()
   ```

6. **Displays immediate feedback**
   - ✓ or ✗ icon for win/loss
   - Actual stat value
   - WON or LOST
   - Profit/loss in dollars
   - Model prediction error

7. **After all evaluations**
   - Generates ROI report automatically
   - Shows overall performance
   - Calculates win rate, total P/L, ROI%

**Interactive evaluation:**
```
Justin Jefferson (WR) - Receiving Yards
Game: MIN @ CHI
Bet: OVER 84.5 @ -110
Prediction: 91.3

Enter actual receiving yards: 87

✓ Actual: 87.0
   Result: WON
   P/L: +$90.91
   Model Error: 4.3 yards
```

### Step 5: Review Performance

```bash
# View ROI report
python scripts/view_paper_trading_report.py

# Optional: Filter to specific week
python scripts/view_paper_trading_report.py --week 13 --season 2024
```

**What this script does internally:**

1. **Queries evaluated trades**
   ```python
   trades = session.query(PaperTrade).filter(
       PaperTrade.won.is_not(None)  # Only evaluated trades
   ).all()

   # Optional: Filter by week
   if week:
       trades = trades.join(Game).filter(
           Game.week == week,
           Game.season == season
       )
   ```

2. **Calculates comprehensive metrics**

   **a. Basic stats**
   ```python
   total_trades = len(trades)
   wins = sum(1 for t in trades if t.won)
   losses = sum(1 for t in trades if not t.won)
   win_rate = (wins / total_trades) * 100
   ```

   **b. Financial metrics**
   ```python
   total_staked = sum(t.stake for t in trades)
   # $100 × 10 bets = $1,000

   total_profit = sum(t.profit_loss for t in trades)
   # (+90.91 +90.91 -100 +90.91 -100 +90.91 +90.91 -100 +90.91 -100)
   # = +545.45 - 400 = +145.45 (hypothetical)

   roi = (total_profit / total_staked) * 100
   # (+145.45 / 1000) * 100 = +14.5% ROI
   ```

   **c. Breakdown by bet type**
   ```python
   overs = [t for t in trades if t.bet_side == 'over']
   unders = [t for t in trades if t.bet_side == 'under']

   over_record = f"{sum(1 for t in overs if t.won)}-{sum(1 for t in overs if not t.won)}"
   under_record = f"{sum(1 for t in unders if t.won)}-{sum(1 for t in unders if not t.won)}"
   ```

3. **Formats comprehensive report**
   - Header with filtering info
   - Overall record and win rate
   - Over/under split
   - Financial summary (staked, P/L, ROI)
   - Average profit per trade
   - Individual trade details with icons

4. **Displays insights**
   - What metrics to look for
   - Target win rates (53%+ for break-even)
   - Target ROI (3-5%+)
   - Next steps based on performance

**Output:**
```
============================================================
PAPER TRADING ROI REPORT
============================================================

Total Trades: 10
Record: 6-4 (60.0%)
Over Bets: 4-2
Under Bets: 2-2

Total Staked: $1,000.00
Total Profit/Loss: +$81.82
ROI: +8.2%
Avg Profit/Trade: +$8.18

INDIVIDUAL TRADES:
------------------------------------------------------------
✓ Justin Jefferson OVER 84.5 (Actual: 87.0) +90.91
✓ CeeDee Lamb UNDER 92.5 (Actual: 88.0) +90.91
✗ Tyreek Hill OVER 74.5 (Actual: 62.0) -100.00
✓ Travis Kelce OVER 48.5 (Actual: 57.0) +90.91
...
============================================================
```

**Additional analysis shown:**
```
INSIGHTS & RECOMMENDATIONS
============================================================

What to look for:
  ✓ Win rate > 53% (break-even is 52.4% at -110 odds)
  ✓ Positive ROI (target: 3-5%)
  ✓ Model errors within expected range
  ✓ Edge% correlates with win rate

If system is profitable:
  → Track 20-30 more bets to confirm
  → Consider small real money bets
  → Continue refining models

Remember: Even with an edge, variance means losing streaks happen!
```

## Reasoning System - What Gets Tracked

For each bet, the system captures:

### 1. Primary Factor (The Edge)
- Why this bet has value
- Edge percentage vs market

### 2. Player Form
- Recent performance trend
- Last 3-5 games average
- Hot/cold streak identification

### 3. Position-Specific Metrics

**For WR/TE:**
- Target share (% of team targets)
- Recent target trend
- Red zone target share

**For RB:**
- Touch share (carries + targets)
- Snap percentage
- Goal line role

### 4. Matchup Analysis
- Opponent defensive ranking
- Position-specific matchup (WR vs CB, etc.)
- Historical performance vs opponent

### 5. Game Context
- Home/Away
- Weather (dome, outdoor, wind, temp)
- Primetime game
- Divisional matchup
- Playoff implications

### 6. Injury Status
- Current health
- Practice participation
- Injury history

### 7. Confidence Factors
- What supports this bet
- Stacking evidence
- Multiple confirmations

### 8. Risk Factors
- What could go wrong
- Concerns to monitor
- Hedge considerations

## Example Reasoning Entries

### High-Confidence Over Bet

```json
{
  "primary_factor": "Edge: +6.8%",
  "player_form": "Hot streak - 4 consecutive games over 90 yards",
  "target_share": "32.1% (team leader, up from 28% earlier in season)",
  "matchup": "vs 31st ranked pass defense (bottom 3 in league)",
  "venue": "Dome game (no weather concerns)",
  "injury_status": "Healthy, no practice limitations",
  "recent_performance": "98.3 yard avg last 4 games",
  "confidence_factors": [
    "Massive edge (>6%)",
    "Elite recent form",
    "Terrible matchup for defense",
    "Increased target share trend"
  ],
  "risk_factors": [
    "High line (88.5) leaves less margin"
  ]
}
```

### Value Under Bet

```json
{
  "primary_factor": "Edge: +4.1%",
  "player_form": "Cooling off - 48.7 yard avg last 3 games (down from 71.2)",
  "target_share": "18.2% (3rd on team, declining trend)",
  "matchup": "vs 3rd ranked pass defense (very tough)",
  "venue": "Road game in cold weather (24°F, 18mph wind)",
  "injury_status": "Questionable - limited practice Wed/Thu, full Fri",
  "confidence_factors": [
    "Strong defensive matchup",
    "Declining target share",
    "Harsh weather conditions",
    "Model predicts 68.3 vs line of 75.5"
  ],
  "risk_factors": [
    "Player could be game-time decision",
    "Moderate edge (4%) - variance could swing it"
  ]
}
```

### Marginal Edge (Might Skip)

```json
{
  "primary_factor": "Edge: +2.8%",
  "player_form": "Consistent - 64.3 yard avg (very stable)",
  "target_share": "21.5% (steady, no trend)",
  "matchup": "vs 16th ranked pass defense (league average)",
  "venue": "Neutral site",
  "injury_status": "Healthy",
  "confidence_factors": [
    "Player is very consistent",
    "Model has high confidence"
  ],
  "risk_factors": [
    "Edge below 3% threshold",
    "No strong supporting factors",
    "Line (64.5) exactly at prediction (63.8)"
  ]
}
```

## Success Metrics to Track

### Win Rate Targets
- **53%+** = Break-even at -110 odds
- **55%+** = Profitable
- **58%+** = Very strong system

### ROI Targets
- **+3%** = Good
- **+5%** = Excellent  
- **+8%** = Outstanding
- **+10%+** = Exceptional (but verify!)

### Model Accuracy
- Compare actual results to predictions
- Track average error (MAE)
- Should be within expected range:
  - WR: ±6.6 yards
  - TE: ±4.6 yards
  - RB: ±3.0 yards

### Edge Correlation
- Do higher edges win more often?
- What's the minimum profitable edge?
- Are 3% edges enough or need 5%+?

## What To Do With Results

### If Showing +ROI (After 20+ Bets)

1. ✅ **Continue tracking** - Build larger sample
2. ✅ **Analyze winners** - What factors appear most?
3. ✅ **Study losers** - Are there patterns in losses?
4. ✅ **Refine thresholds** - Adjust minimum edge if needed
5. ⚠️  **Consider real money** - Start very small

### If Break-Even (After 20+ Bets)

1. 📊 **Review reasoning** - Which factors are predictive?
2. 📊 **Check edge sizes** - Are 3% edges too marginal?
3. 📊 **Examine confidence** - Is model confidence meaningful?
4. 📊 **Test adjustments** - Higher thresholds? Different markets?
5. ⏸️  **Don't risk real money yet**

### If Losing (After 20+ Bets)

1. ❌ **Stop** - Don't risk real money
2. 🔍 **Deep analysis** - Where is model failing?
3. 🔍 **Check for bias** - Systematic over/under predictions?
4. 🔍 **Review features** - Are some misleading?
5. 🛠️  **Improve models** - Back to feature engineering

## Pro Tips

### 1. Be Honest With Yourself
- Don't cherry-pick results
- Record ALL paper trades you would actually make
- Don't skip the losers

### 2. Use Consistent Stake
- Always $100 per bet (or your chosen amount)
- Don't vary based on "feeling"
- This tests the system, not your intuition

### 3. Track Everything
- The reasoning is just as important as the result
- Learn which factors actually matter
- Build intuition for the next bet

### 4. Expect Variance
- 60% win rate can still have 4-bet losing streaks
- Judge over 20+ bets minimum
- Don't get discouraged by short-term results

### 5. Learn From Losses
- Every loss teaches something
- Was reasoning flawed?
- Was it just variance?
- Can you improve the model?

## Common Questions

**Q: How many bets before I know if it works?**
A: Minimum 20 bets, ideally 30-50. More = more confidence.

**Q: What if I'm at 55% win rate but negative ROI?**
A: You're betting too many favorites. Target higher + odds or verify edge calculations.

**Q: Should I record bets I'm not confident in?**
A: Only record bets you'd actually make. This validates YOUR process.

**Q: Can I adjust after seeing the line move?**
A: No - that's not testing the system. Record when you first see the edge.

**Q: What if model predicts 85 but actual is 125?**
A: Happens! Note what went wrong. Injury? Game script? Outlier performance?

## Ready to Start?

```bash
# 1. Collect odds
python scripts/collect_daily_odds.py

# 2. Record paper trades
python scripts/record_paper_trades.py

# 3. After games, evaluate
python scripts/evaluate_paper_trades.py

# 4. Review performance
python scripts/view_paper_trading_report.py
```

**Remember:** This is about learning and validation, not gambling!

---

## Technical Deep Dive: Database Flow

Understanding how data flows through the system:

### Database Tables Used

```
┌──────────────┐
│    Props     │  ← Odds from API/manual entry
│ (Market data)│
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Predictions  │  ← Model predictions
│ (Your models)│
└──────┬───────┘
       │
       ▼ (comparison)
┌──────────────┐
│    Edges     │  ← Where model > market
│ (Opportunities)
└──────┬───────┘
       │
       ▼ (you select)
┌──────────────┐
│ PaperTrades  │  ← Hypothetical bets with reasoning
│ (Tracking)   │
└──────┬───────┘
       │
       ▼ (after games)
┌──────────────┐
│ PaperTrades  │  ← Updated with results
│ (Evaluated)  │
└──────────────┘
```

### Data Stored in PaperTrade Table

Each record contains:

**Identifiers:**
- game_id → Links to Game (MIN @ CHI)
- player_id → Links to Player (Justin Jefferson)
- prop_id → Links to Prop (the actual market line)

**Bet Details:**
- market → 'receiving_yards'
- bet_side → 'over' or 'under'
- line → 84.5
- odds → -110
- stake → 100.0 (hypothetical)

**Model Analysis:**
- model_prediction → 91.3
- model_confidence → 0.872
- edge_percentage → 5.2
- expected_value → 8.3

**Reasoning (JSON):**
```json
{
  "primary_factor": "Edge: +5.2%",
  "player_form": "Last 3 games: 91.3 avg (trending up)",
  "target_share": "Team-leading 28.5% target share",
  "matchup": "vs 28th ranked pass defense (favorable)",
  "venue": "Dome game (no weather concerns)",
  "injury_status": "Healthy, full participant all week",
  "confidence_factors": [
    "High edge (>5%)",
    "High model confidence",
    "Strong recent form"
  ],
  "risk_factors": []
}
```

**Outcome (after evaluation):**
- actual_result → 87.0 (you enter this)
- won → True (calculated: 87.0 > 84.5)
- profit_loss → +90.91 (calculated: stake * odds)
- evaluated_at → timestamp

### Query Examples

**Get all winning bets:**
```python
winners = session.query(PaperTrade).filter(
    PaperTrade.won == True
).all()
```

**Get bets with high edges that lost:**
```python
# Learn what went wrong with high-confidence picks
high_edge_losses = session.query(PaperTrade).filter(
    PaperTrade.edge_percentage > 5.0,
    PaperTrade.won == False
).all()
```

**Analyze by reasoning factor:**
```python
# See if "favorable matchup" in reasoning correlates with wins
favorable_matchups = session.query(PaperTrade).filter(
    PaperTrade.reasoning['matchup'].astext.like('%favorable%')
).all()

win_rate = sum(1 for t in favorable_matchups if t.won) / len(favorable_matchups)
```

### What You Can Learn

**From the reasoning JSON:**
- Which factors appear in winners vs losers?
- Do "confidence_factors" actually predict wins?
- Are certain "risk_factors" red flags to avoid?
- Is target share more predictive than matchup?

**From the metrics:**
- What edge% threshold is profitable? (3%, 5%, 8%?)
- Does higher model confidence = higher win rate?
- Are overs or unders more profitable?
- What's your actual model error vs expected?

**Over time:**
- Is the system improving or degrading?
- Are you getting better at bet selection?
- Do certain players/teams over/under-perform models?
- Should you adjust edge thresholds?

This is why tracking reasoning is so valuable - it teaches you what actually works!
