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

## CLI Reference

All commands support global options:
- `--week N` - Override week number (default: auto-detect)
- `--season YYYY` - Override season year (default: auto-detect)

### Core Workflow Commands

| Command | Description |
|---------|-------------|
| `status` | Show current system status |
| `pre-game` | Run pre-game workflow (odds → predictions → edges) |
| `post-game` | Collect results and score predictions |
| `full` | Run complete pre-game + post-game workflow |

#### `pre-game` Options
```bash
python scripts/orchestrate.py pre-game
python scripts/orchestrate.py pre-game --force        # Force refresh cached data
python scripts/orchestrate.py pre-game --top 30       # Show top 30 edges (default: 20)
python scripts/orchestrate.py pre-game --no-notify    # Skip Discord notifications
python scripts/orchestrate.py pre-game --no-parlay    # Skip automatic parlay generation
```

#### `post-game` Options
```bash
python scripts/orchestrate.py post-game
python scripts/orchestrate.py post-game --show-pending  # Show trades that couldn't be auto-scored
```

### Analysis Commands

| Command | Description |
|---------|-------------|
| `results` | View and manage paper trade results |
| `history` | View historical performance summaries |
| `dashboard` | View result tracking dashboard with breakdowns |
| `filters` | Manage dynamic bet filters based on historical performance |
| `parlay` | Generate parlay recommendations |

#### `results` Options
```bash
python scripts/orchestrate.py results
python scripts/orchestrate.py results -v                    # Verbose: show all trades
python scripts/orchestrate.py results --score 123 45.5      # Manually score trade ID 123
```

#### `history` Options
```bash
python scripts/orchestrate.py history                  # View season history
python scripts/orchestrate.py history --generate      # Generate summary for current week
python scripts/orchestrate.py history --generate-all  # Generate all missing summaries
python scripts/orchestrate.py history --breakdown     # Include market breakdown
python scripts/orchestrate.py history --trends        # Show performance trends
```

#### `dashboard` Options
```bash
python scripts/orchestrate.py dashboard               # View full dashboard (all weeks)
python scripts/orchestrate.py dashboard --week 13     # Dashboard for specific week
python scripts/orchestrate.py dashboard --recommend   # Include filter recommendations
```

Shows breakdowns by: direction (OVER/UNDER), market, position, player tier (elite/other), edge %, and model confidence with actionable insights.

#### `filters` Options
```bash
python scripts/orchestrate.py filters                      # View current dynamic filters
python scripts/orchestrate.py filters --generate          # Generate filters from historical data
python scripts/orchestrate.py filters --clear             # Clear all dynamic filters
python scripts/orchestrate.py filters --generate --min-sample 20  # Require 20+ bets per category
python scripts/orchestrate.py filters --generate --skip-below 0.40  # Skip categories <40% win rate
```

| Option | Default | Description |
|--------|---------|-------------|
| `--generate` | - | Generate/refresh filters from historical paper trades |
| `--clear` | - | Clear all dynamic filters |
| `--min-sample` | 15 | Minimum bets per category to create filter |
| `--skip-below` | 0.35 | Skip categories with win rate below this |
| `--prioritize-above` | 0.55 | Prioritize categories with win rate above this |
| `--min-week` | - | Only include trades from this week onward |
| `--max-week` | - | Only include trades up to this week |

Dynamic filters automatically skip bet types with poor historical win rates (e.g., elite player UNDERs, TE OVERs) and prioritize profitable patterns (e.g., RB OVERs, non-elite OVERs).

#### `parlay` Options
```bash
python scripts/orchestrate.py parlay
python scripts/orchestrate.py parlay --notify         # Send to Discord
python scripts/orchestrate.py parlay --max-legs 10    # Up to 10-leg parlays
python scripts/orchestrate.py parlay --save           # Save to database
```

| Option | Default | Description |
|--------|---------|-------------|
| `--top N` | 5 | Parlays to show per leg count |
| `--max-legs N` | 5 | Maximum legs per parlay (2-20) |
| `--min-prob` | 0.55 | Minimum probability per leg |
| `--min-leg-ev` | 5.0 | Minimum EV% per leg |
| `--min-parlay-ev` | 15.0 | Minimum combined EV% |
| `--max-candidates` | 30 | Max legs to consider |

### Utility Commands

| Command | Description |
|---------|-------------|
| `stage` | Run a single workflow stage |
| `health` | Run system health checks |
| `weather` | Set or fetch weather for games |
| `injuries` | Show and refresh injury reports |
| `notify` | Send Discord notifications |

#### `stage` Options
Run individual workflow stages:
```bash
python scripts/orchestrate.py stage collect_odds
python scripts/orchestrate.py stage generate_predictions
python scripts/orchestrate.py stage calculate_edges
python scripts/orchestrate.py stage refresh_schedule
python scripts/orchestrate.py stage score_results
python scripts/orchestrate.py stage collect_odds --force  # Force refresh
```

#### `health` Options
```bash
python scripts/orchestrate.py health
python scripts/orchestrate.py health --notify  # Send Discord alert if unhealthy
```

#### `weather` Options
```bash
python scripts/orchestrate.py weather                           # Show current weather
python scripts/orchestrate.py weather --fetch                   # Fetch from Weather.gov
python scripts/orchestrate.py weather SF CLE rain --temp 35     # Manual override
python scripts/orchestrate.py weather SF CLE snow --wind 20     # Snow with wind
```

#### `injuries` Options
```bash
python scripts/orchestrate.py injuries
python scripts/orchestrate.py injuries --refresh  # Refresh from nfl_data_py
```

#### `notify` Options
```bash
python scripts/orchestrate.py notify --test  # Send test notification
```

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
