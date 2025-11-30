# NFL Player Props Prediction System

A machine learning system for predicting NFL player props and identifying betting edges by comparing model predictions to sportsbook lines.

## What It Does

This system:
1. **Trains ML models** on historical NFL data (2020-2024) to predict player statistics
2. **Fetches live odds** from The Odds API for current week's player props
3. **Generates predictions** for receiving yards, rushing yards, passing yards, and receptions
4. **Identifies edges** where our predictions significantly differ from sportsbook lines
5. **Generates parlays** from uncorrelated high-EV edges
6. **Tracks results** to measure model accuracy and betting ROI
7. **Sends Discord alerts** for edges and parlays

## Supported Stat Types

| Stat Type | Model | Test R² | Notes |
|-----------|-------|---------|-------|
| Receiving Yards | Adaptive XGBoost | 0.31 | Best performing, uses rolling features |
| Rushing Yards | XGBoost v2 | 0.56 | Well calibrated, no bias |
| Passing Yards | XGBoost v2 | 0.15 | Uses NGS data for predictions |
| Receptions | XGBoost v2 + Ensemble | 0.34 | 70% ML + 30% season avg blend |

## Quick Start

### 1. Install Dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Set Up Environment

```bash
cp .env.example .env
# Required: ODDS_API_KEY=your_key_here
# Optional: DISCORD_WEBHOOK_URL=your_webhook_here
```

Get a free API key at [The Odds API](https://the-odds-api.com/).

### 3. Train Models (one-time)

```bash
python scripts/train_models_v2.py
```

### 4. Weekly Workflow

```bash
# Check system status and current week
python scripts/orchestrate.py status

# BEFORE GAMES: Collect odds, generate predictions, find edges
python scripts/orchestrate.py pre-game

# AFTER GAMES: Collect results and score predictions
python scripts/orchestrate.py post-game

# Check system health
python scripts/orchestrate.py health
```

The orchestrator automatically detects the current NFL week and handles all stages.

## Project Structure

```
sports-betting/
├── src/sports_betting/
│   ├── ml/                    # ML models and predictors
│   ├── data/                  # Data collection (Odds API, NFL data)
│   ├── database/              # SQLite database models
│   ├── analysis/              # Edge calculation
│   ├── workflow/              # Orchestrator and stage management
│   ├── notifications/         # Discord alerts
│   └── monitoring/            # Health checks
├── scripts/
│   └── orchestrate.py         # Main CLI entry point
├── models/                    # Saved model files (.pkl)
├── data/                      # SQLite database
└── tests/                     # 100 tests
```

## Usage Examples

### Full Pre-Game Workflow

```bash
python scripts/orchestrate.py pre-game

# Output shows edges like:
# OVER  Keon Coleman  Rec Yards  25.5  35.2  +38.1%  -118
# UNDER Taysom Hill   Rush Yards 22.5  14.7  -34.7%  -110
```

### Run Individual Stages

```bash
# Just collect odds
python scripts/orchestrate.py stage collect_odds

# Just generate predictions
python scripts/orchestrate.py stage generate_predictions

# Override week number
python scripts/orchestrate.py pre-game --week 14
```

### Check System Health

```bash
python scripts/orchestrate.py health

# With Discord notification on issues
python scripts/orchestrate.py health --notify
```

### Generate Parlays

Build parlay combinations from your best edges:

```bash
# Generate parlays (2-5 legs)
python scripts/orchestrate.py parlay

# Send top parlays to Discord
python scripts/orchestrate.py parlay --notify

# Customize parlay generation
python scripts/orchestrate.py parlay --max-legs 4 --min-parlay-ev 20 --notify
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--top N` | 5 | Number of parlays to show per leg count |
| `--max-legs N` | 5 | Maximum legs per parlay (2-5) |
| `--min-prob` | 0.55 | Minimum probability per leg |
| `--min-leg-ev` | 5.0 | Minimum EV% per individual leg |
| `--min-parlay-ev` | 15.0 | Minimum combined EV% for parlay |
| `--max-candidates` | 30 | Max legs to consider (limits combinatorics) |
| `--notify` | false | Send top parlays to Discord |
| `--save` | false | Store parlays in database |

The parlay generator:
- Filters to high-probability, high-EV legs only
- Excludes correlated legs (same player, same-game passing stats)
- Calculates combined odds and joint probability
- Ranks by expected value

## Model Training

Models are trained on historical data using only features available before each game:

```python
# Features used (all computed from prior games):
- Rolling averages (last 3 and 5 games)
- Rolling standard deviation
- Career averages (yards per attempt, catch rate, etc.)
- Position encoding
- Week number
- Snap percentage
```

No same-game data is used to prevent data leakage.

## Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `ODDS_API_KEY` | The Odds API key | Yes |
| `DISCORD_WEBHOOK_URL` | Discord webhook for alerts | No |
| `DATABASE_URL` | SQLite path (default: data/sports_betting.db) | No |

### Discord Notifications

When configured, the system sends:
- **Edge alerts** - High-confidence betting opportunities
- **Weekly results** - Win/loss record and ROI
- **Health alerts** - System issues (when using `--notify`)

### API Usage

The Odds API free tier includes 500 requests/month. Each player props fetch uses ~4 requests. The system caches odds to minimize API usage.

Check remaining credits:
```bash
python scripts/orchestrate.py health
```

## Analysis Reports

After running predictions, reports are saved to `docs/`:

- `WEEK_13_ALL_PREDICTIONS.md` - All predictions and edges
- `WEEK_13_ANALYSIS.md` - Detailed bias investigation and betting tiers
- `WEEK_13_RESULTS.md` - Post-game performance analysis

## Development

```bash
# Run tests (100 tests)
pytest tests/

# Train models with fresh data
python scripts/train_models_v2.py

# Check system health
python scripts/orchestrate.py health
```

## Data Sources

| Source | Data | Update Frequency |
|--------|------|------------------|
| nfl-data-py | Historical stats, weekly data | Weekly |
| NFL Next Gen Stats | Advanced metrics | Weekly |
| Pro Football Reference | Seasonal stats | Daily during season |
| The Odds API | Live betting lines | Real-time |

## Limitations

- Models perform best for high-volume players with consistent roles
- Predictions don't account for injuries announced after data collection
- Game script adjustments are not modeled
- Backups and low-snap players have higher prediction variance
- Weather data is fetched from Weather.gov (US stadiums only)

## License

MIT
