# Odds Data Strategy - Free/Low-Cost Options

## The Challenge

Without sportsbook odds, we can't:
- Calculate true edge (model prediction vs market line)
- Identify +EV betting opportunities
- Track Closing Line Value (CLV)
- Validate model accuracy against real markets

## API Options Comparison

### 1. The Odds API (Most Popular)

**Free Tier:**
- 500 credits/month
- All sports, markets, bookmakers
- Historical odds access

**Credit Usage:**
- Getting sports list: FREE (0 credits)
- Getting odds: `markets × regions` credits
  - Example: 1 region × 3 player props = 3 credits
- Getting scores: 1 credit

**NFL Season Math:**
- 16 games/week × 3 credits per game = 48 credits/week
- 17-week season = 816 credits total ❌ (exceeds free tier)

**Strategies to Stay Within Free Tier:**

✅ **Option A: Selective Games (Recommended)**
- Only fetch odds for 2-3 marquee games per week
- Focus on primetime games (TNF, SNF, MNF)
- 3 games × 3 credits = 9 credits/week
- 17 weeks × 9 = 153 credits/season ✅ (well within free tier)

✅ **Option B: Fetch Less Frequently**
- Fetch all games only 2x per week (Tuesday + Saturday)
- 48 credits × 2 = 96 credits/week
- 17 weeks × 96 = 1,632 credits ❌ Still over

✅ **Option C: Single Market Focus**
- Only fetch receiving yards props (most accurate model)
- 16 games × 1 credit = 16 credits/week
- 17 weeks × 16 = 272 credits/season ✅ (within free tier)

✅ **Option D: Top Players Only**
- Fetch odds for ~10 top players per week
- Use player-specific endpoint (more efficient)
- ~10-20 credits/week
- 17 weeks × 20 = 340 credits/season ✅ (within free tier)

**Paid Tier:**
- $30/month for 20,000 credits (overkill for personal use)

---

### 2. Sports Game Odds (SGO)

**Free Tier:**
- Amateur plan: 1,000 objects/month FREE
- Includes player props
- 14-day trial of higher tiers

**Pros:**
- More generous free tier (1,000 vs 500)
- Specifically built for player props

**Cons:**
- Less documentation
- Smaller community
- Paid tiers expensive ($99/month)

**Recommendation:** Worth trying if Odds API free tier insufficient

---

### 3. The Rundown

**Free Tier:**
- Basic free tier available
- Live odds and scores

**Cons:**
- Limited player props in free tier
- Paid from $49.99/month
- Less clear documentation

---

## Alternative Strategies (Free)

### Strategy 1: Web Scraping (Use Cautiously)

**Pros:**
- Completely free
- Real-time data
- Can target specific books

**Cons:**
- ⚠️ May violate Terms of Service
- ⚠️ Sites may block scrapers (rate limiting, CAPTCHAs)
- ⚠️ Fragile (breaks when site changes)
- Legal grey area

**Sites to Consider:**
- DraftKings (has public odds)
- ESPN Bet (displays odds publicly)
- ActionNetwork (aggregates odds)

**If You Go This Route:**
```python
# Ethical web scraping guidelines:
1. Respect robots.txt
2. Add delays between requests (2-5 seconds)
3. Use a User-Agent string
4. Only scrape public-facing data
5. Cache aggressively to minimize requests
6. Don't scrape during peak hours
```

**Legal Note:** Web scraping odds may be legally questionable. Use at your own risk.

---

### Strategy 2: Manual Data Entry (Testing Phase)

**For initial testing without APIs:**

1. Manually collect odds for 5-10 props per week from DraftKings/FanDuel
2. Store in simple CSV:
```csv
player_name,prop_type,line,over_odds,under_odds,book,timestamp
Tyreek Hill,receiving_yards,74.5,-110,-110,DraftKings,2024-12-01 10:00
CeeDee Lamb,receiving_yards,82.5,-115,-105,DraftKings,2024-12-01 10:00
```
3. Compare model predictions to manual odds
4. Build proof-of-concept before investing in API

**Pros:**
- Zero cost
- Validates approach before spending money
- Learn what data you actually need

**Cons:**
- Time-consuming
- Limited scale
- Can't automate

---

### Strategy 3: Hybrid Approach (Recommended for Starting)

**Phase 1: Proof of Concept (Weeks 1-4)**
- Manual data entry for 5-10 props/week
- Test model accuracy vs real lines
- Calculate theoretical edge and ROI
- Prove the system works

**Phase 2: Limited API (Weeks 5-12)**
- Use The Odds API free tier (500 credits)
- Strategy: Top 10 players only (Option D above)
- Still within free tier
- Scale up data collection

**Phase 3: Decide on Investment (Week 13+)**
- If showing positive results → Consider paid tier ($30/month)
- If breaking even → Stay with free tier + manual
- If losing → Refine models before spending

---

## Recommended Approach

### For Personal/Testing Use (Now):

**Use The Odds API Free Tier with "Top Players" Strategy:**

1. **Get API Key** (free signup at the-odds-api.com)

2. **Weekly Workflow:**
   ```
   Tuesday Morning:
   - Identify top 10 players from model predictions
   - Fetch odds for those specific players only
   - Cost: ~10-15 credits

   Saturday Morning:
   - Update odds for same players
   - Cost: ~10-15 credits

   Total: ~30 credits/week × 17 weeks = 510 credits (just over free tier)
   ```

3. **Stay Within Budget:**
   - Skip bye weeks (save ~60 credits)
   - Skip early season weeks while models warm up
   - Only fetch odds for model's highest-confidence predictions

4. **Alternative: Manual Supplement:**
   - Use API for 5 players (8 credits/week = 136/season)
   - Manually add 5 more players
   - Completely free

### Implementation Plan:

```python
# Smart API usage patterns:
1. Cache all API responses locally
2. Only fetch new odds when > 12 hours old
3. Fetch odds in single batch request (more efficient)
4. Track credit usage in database
5. Alert when approaching free tier limit
6. Fall back to manual entry if limit hit
```

---

## Cost-Benefit Analysis

### Option A: Stay 100% Free
- **Cost:** $0/month + 30min manual work/week
- **Coverage:** 5-10 props per week
- **Sufficient for:** Testing, learning, proof of concept

### Option B: The Odds API Free Tier Only
- **Cost:** $0/month
- **Coverage:** 10-15 props per week
- **Sufficient for:** Small-scale testing, learning edge detection

### Option C: Hybrid (Free API + Manual)
- **Cost:** $0/month + 15min manual work/week
- **Coverage:** 15-20 props per week
- **Sufficient for:** Serious testing, building track record

### Option D: Paid API ($30/month)
- **Cost:** $360/year
- **Coverage:** Unlimited (20K credits = ~2,500 prop fetches)
- **Justified when:** Showing +5% ROI on $5,000+ bankroll

**Break-Even Math:**
- Need to generate $360/year profit to justify paid tier
- With 5% edge and $5,000 bankroll, average bet $100
- Need ~72 bets/year to break even (easily achievable)
- If betting weekly: ~4 bets/week × 17 weeks = 68 bets

---

## My Recommendation

**Start with free hybrid approach:**

1. **Now:** Use The Odds API free tier
   - Focus on top 10 players/week
   - Stay within 500 credits

2. **Supplement:** Manual entry for another 5-10 players
   - Takes 15 minutes
   - Saves credits
   - Validates API data

3. **Track Everything:**
   - Log all predictions vs actual lines
   - Calculate edge on each prop
   - Measure theoretical ROI

4. **Decision Point (After 4-6 weeks):**
   - If showing +3%+ edge → Upgrade to $30/month (easily justified)
   - If break-even → Stay with free tier, refine models
   - If losing → Focus on model improvement before spending

5. **Scale When Profitable:**
   - Only invest in APIs when system is proven profitable
   - $30/month is tiny compared to potential returns
   - But don't spend until you're confident

---

## Next Steps

1. **Sign up for The Odds API free tier** (no credit card required)
2. **Create odds collection module** with smart caching
3. **Build manual entry interface** (CSV or simple web form)
4. **Start tracking:** Model predictions vs actual odds
5. **Prove the edge exists** before investing in infrastructure

Do you want me to implement the hybrid approach (free API + manual entry)?
