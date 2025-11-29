# Odds Collection Setup Guide

## Quick Start

You have **two options** for collecting odds data:

1. **The Odds API** (Automated, 500 free credits/month) - Recommended
2. **Manual Entry** (Free, 15 min/week) - Backup option

## Option 1: The Odds API (Recommended)

### Step 1: Get Your Free API Key

1. Go to https://the-odds-api.com/
2. Click "Get Started" or "Sign Up"
3. Create a free account (no credit card required)
4. Copy your API key from the dashboard

### Step 2: Configure Your API Key

Add your API key to the `.env` file:

```bash
# Copy example if .env doesn't exist
cp .env.example .env

# Edit .env and add your key
# Change this line:
ODDS_API_KEY=your_odds_api_key_here

# To this (with your actual key):
ODDS_API_KEY=abc123xyz789...
```

### Step 3: Test the Connection

```bash
# Test API connection
python scripts/collect_daily_odds.py
```

You should see:
```
✓ Successfully fetched odds
  Credits used: 1
  Credits remaining: 499
```

### Step 4: Daily Workflow

**Run once per day** (recommend Tuesday mornings after Monday night game):

```bash
python scripts/collect_daily_odds.py
```

This will:
1. Fetch odds from The Odds API (1-2 credits)
2. Cache locally for the day
3. Store in database
4. Calculate betting edges
5. Display opportunities

**Credit Usage:**
- 1 market (receiving yards): 1 credit/day
- 30 days = 30 credits/month ✅ (well within free tier)

### Managing API Credits

**Check usage:**
```bash
# Usage is displayed after each fetch
Credits remaining: 470/500
```

**Stay within free tier:**
- Only fetch receiving yards (most accurate model)
- Run once daily (not multiple times)
- Skip weeks with no games (bye weeks, playoffs)

**If approaching limit:**
- Fall back to manual entry
- Or upgrade to paid tier ($30/month)

---

## Option 2: Manual Entry (Free Forever)

Perfect for:
- Testing without API key
- Supplementing API data
- Unlimited data at zero cost

### Step 1: Create Template

```bash
python scripts/import_manual_odds.py --create-template
```

This creates: `~/.sports_betting/manual_odds/odds_template.csv`

### Step 2: Fill In Odds

Open the template and add odds from your sportsbook:

```csv
date,player_name,team,opponent,prop_type,line,over_odds,under_odds,bookmaker
2024-12-01,Justin Jefferson,MIN,CHI,receiving_yards,84.5,-110,-110,draftkings
2024-12-01,CeeDee Lamb,DAL,NYG,receiving_yards,92.5,-115,-105,draftkings
2024-12-01,Tyreek Hill,MIA,GB,receiving_yards,74.5,-110,-110,fanduel
```

**Where to get odds:**
- DraftKings: https://sportsbook.draftkings.com/
- FanDuel: https://sportsbook.fanduel.com/
- Caesars: https://www.williamhill.com/us/

### Step 3: Import Your Odds

```bash
python scripts/import_manual_odds.py your_odds.csv
```

### Step 4: Calculate Edges

Edges are automatically calculated when you import, or run:

```bash
python scripts/collect_daily_odds.py
```

**Time Investment:** 15-20 minutes per week

---

## Hybrid Approach (Best of Both)

Use both methods to maximize coverage while staying in free tier:

**The Odds API:**
- Top 5 players (1 credit)
- Automated collection
- ~30 credits/month

**Manual Entry:**
- Additional 5-10 players
- 15 minutes/week
- Zero cost

**Total:** 10-15 props/week, completely free!

---

## Understanding the Output

When you run `collect_daily_odds.py`, you'll see:

```
BETTING EDGES REPORT - 3 opportunities found
================================================================================

1. Justin Jefferson - Receiving Yards
   Game: MIN @ CHI
   Line: 84.5
   Model Prediction: 91.3
   Confidence: 87%

   >>> BET OVER -110
   >>> Edge: +5.2%
   >>> Expected Value: +8.3%

2. CeeDee Lamb - Receiving Yards
   Game: DAL @ NYG
   Line: 92.5
   Model Prediction: 87.1
   Confidence: 82%

   >>> BET UNDER -110
   >>> Edge: +3.8%
   >>> Expected Value: +6.1%
```

**What this means:**
- **Edge**: How much better our model thinks the bet is vs the market
- **Expected Value (EV)**: Long-term profit expectation per $1 bet
- **Confidence**: Model's certainty in its prediction

**Rule of thumb:**
- Edge > 3% = Consider betting
- Edge > 5% = Strong bet
- Edge > 8% = Max bet (within bankroll management)

---

## Troubleshooting

### "No API key configured"

**Solution:** Add ODDS_API_KEY to `.env` file (see Step 2 above)

### "No edges found"

**Possible reasons:**
1. Haven't generated predictions yet → Run prediction script first
2. Market lines match model → No betting opportunities (this is normal)
3. Minimum edge threshold too high → Lower in `scripts/collect_daily_odds.py`

### "Player not found"

**Solution:** Player names must match database exactly. Check:
```bash
# Search for player in database
python -c "from src.sports_betting.database import get_session; from src.sports_betting.database.models import Player; session = get_session().__enter__(); print([p.name for p in session.query(Player).filter(Player.name.like('%Jefferson%')).all()])"
```

### "Game not found"

**Solution:** Import games first:
```bash
python scripts/collect_data.py
```

---

## Advanced: Automating Daily Collection

### macOS/Linux (crontab)

Run automatically every Tuesday at 10 AM:

```bash
# Edit crontab
crontab -e

# Add this line:
0 10 * * 2 cd /path/to/sports-betting && /path/to/venv/bin/python scripts/collect_daily_odds.py >> logs/odds_collection.log 2>&1
```

### Windows (Task Scheduler)

1. Open Task Scheduler
2. Create Basic Task
3. Trigger: Weekly, Tuesday, 10:00 AM
4. Action: Start Program
   - Program: `C:\path\to\venv\Scripts\python.exe`
   - Arguments: `scripts\collect_daily_odds.py`
   - Start in: `C:\path\to\sports-betting`

---

## Cost Analysis

### Free Tier (The Odds API)

**Monthly credits:** 500
**Your usage:** ~30 credits (1 per day × 30 days)
**Buffer:** 470 credits available
**Cost:** $0/month

### When to Upgrade ($30/month)

Only if:
- ✅ System showing +5% ROI over 6+ weeks
- ✅ Betting real money ($1,000+ bankroll)
- ✅ Want to scale to multiple markets
- ✅ Need real-time odds updates

**Break-even:** With 5% edge, need $600/month in bets to justify cost

---

## Next Steps

1. ✅ Get API key OR create manual template
2. ✅ Run daily collection script
3. → Review edges and betting opportunities
4. → Place bets on highest EV props
5. → Track results to validate model

**Need help?** Check the main README or create an issue.
