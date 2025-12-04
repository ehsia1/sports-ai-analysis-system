# NFL Sports Betting System - Claude Code Guide

## Virtual Environment

**IMPORTANT**: Always use the virtual environment for all Python commands.

```bash
# Activate venv before running any commands
source venv/bin/activate

# All python/pip commands should use venv
python scripts/orchestrate.py status  # Not python3!
pip install package_name              # Not pip3!
```

## Project Overview

ML-powered NFL player props prediction system that:
- Trains XGBoost models on historical NFL data
- Fetches live odds from The Odds API
- Generates predictions and identifies betting edges
- Builds parlay combinations from top edges
- Sends Discord notifications for alerts
- Tracks results and calculates ROI

## Project Structure

```
sports-betting/
├── src/sports_betting/
│   ├── analysis/           # Edge calculation & parlays
│   │   ├── edge_calculator.py    # Compare predictions to odds
│   │   └── parlay_generator.py   # Build parlay combinations
│   ├── config/             # Configuration
│   │   └── settings.py           # Pydantic settings (env vars)
│   ├── data/               # Data collection
│   │   ├── collectors/           # NFL data fetching
│   │   ├── odds_api.py           # The Odds API client
│   │   └── weather.py            # Weather.gov integration
│   ├── database/           # SQLite persistence
│   │   ├── models.py             # SQLAlchemy models
│   │   └── session.py            # DB session management
│   ├── ml/                 # Machine learning
│   │   ├── base_predictor.py     # Abstract predictor class
│   │   ├── stat_predictors.py    # Per-stat predictors
│   │   └── feature_engineering.py
│   ├── monitoring/         # Health checks
│   │   └── health.py
│   ├── notifications/      # Alerting
│   │   └── discord.py            # Discord webhook notifications
│   ├── utils/              # Shared utilities
│   │   ├── logging.py            # Loguru setup
│   │   ├── retry.py              # Exponential backoff
│   │   └── nfl_schedule.py       # Week detection
│   └── workflow/           # Orchestration
│       ├── orchestrator.py       # Central workflow runner
│       └── stages.py             # Individual stage implementations
├── scripts/
│   ├── orchestrate.py      # Main CLI entry point
│   └── train_models_v2.py  # Model training
├── models/                 # Saved .pkl model files
├── data/                   # SQLite database & predictions JSON
└── tests/                  # Pytest tests (100+)
```

## Key Files - Where to Edit

### Adding/Modifying Features

| Task | File(s) to Edit |
|------|-----------------|
| Add new CLI command | `scripts/orchestrate.py` |
| Modify edge calculation | `src/sports_betting/analysis/edge_calculator.py` |
| Change parlay logic | `src/sports_betting/analysis/parlay_generator.py` |
| Update Discord messages | `src/sports_betting/notifications/discord.py` |
| Add new config option | `src/sports_betting/config/settings.py` + `.env` |
| Modify prediction model | `src/sports_betting/ml/stat_predictors.py` |
| Change workflow stages | `src/sports_betting/workflow/stages.py` |
| Add health check | `src/sports_betting/monitoring/health.py` |
| Modify weather logic | `src/sports_betting/data/weather.py` |
| Modify injury logic | `src/sports_betting/data/injuries.py` |

### Database Models

Edit `src/sports_betting/database/models.py` for:
- `Prediction` - ML predictions
- `Edge` - Betting edges
- `Parlay` - Parlay combinations
- `ActualResult` - Post-game results
- `OddsSnapshot` - Cached odds

### Configuration

All config in `src/sports_betting/config/settings.py`:
```python
# Key settings (set via .env or environment)
ODDS_API_KEY          # Required - The Odds API key
DISCORD_WEBHOOK_URL   # Optional - Discord notifications
DATABASE_URL          # Default: data/sports_betting.db
```

## Common Commands

```bash
# Check system status
python scripts/orchestrate.py status

# Pre-game workflow (collect odds, predict, find edges)
python scripts/orchestrate.py pre-game

# Post-game workflow (collect results, score predictions)
python scripts/orchestrate.py post-game

# Generate parlays
python scripts/orchestrate.py parlay --max-legs 10 --notify

# Run specific stage
python scripts/orchestrate.py stage collect_odds
python scripts/orchestrate.py stage calculate_edges

# Health check
python scripts/orchestrate.py health --notify

# Weather management
python scripts/orchestrate.py weather --fetch

# Injury reports (fetches from ESPN if DB empty)
python scripts/orchestrate.py injuries
python scripts/orchestrate.py injuries --refresh  # Refresh from nfl_data_py
```

## Notion Documentation

### Workspace Structure
- **Parent Page**: "NFL Betting System - Production Roadmap"
- **Page ID**: `2bbac7292af381888e48fb4f4876a9a8`
- **URL**: https://www.notion.so/2bbac7292af381888e48fb4f4876a9a8

### Keeping Documentation Updated

When making significant changes, update the Notion roadmap:

```python
# Use mcp__notion__notion-update-page to update existing content
# Use mcp__notion__notion-create-pages for new documentation

# Parent page for new sports betting docs:
parent_page_id = "2bbac7292af381888e48fb4f4876a9a8"
```

### What to Document in Notion
- Phase completion status (checkboxes)
- New features added
- Architecture decisions
- Weekly results summaries
- Model performance metrics

### Documentation Requirement

**IMPORTANT**: Always document ALL findings and analysis in Notion.

After completing any analysis, backtest, or investigation, update the appropriate Notion page:

| Analysis Type | Notion Page | Page ID |
|---------------|-------------|---------|
| Weekly predictions & results | Weekly Results & Analysis | `2beac7292af3813099d4f4bc179c9caf` |
| Backtest results | Weekly Results & Analysis | `2beac7292af3813099d4f4bc179c9caf` |
| Filter effectiveness | Weekly Results & Analysis | `2beac7292af3813099d4f4bc179c9caf` |
| Model performance insights | Weekly Results & Analysis | `2beac7292af3813099d4f4bc179c9caf` |
| System changes & features | Production Roadmap | `2bbac7292af381888e48fb4f4876a9a8` |

Document these types of findings:
- Weekly edge predictions before games
- Post-game results (win/loss, ROI)
- Backtest comparisons (e.g., "with filters" vs "without")
- Counter-intuitive discoveries (e.g., confidence threshold analysis)
- Filter/model improvement recommendations
- Any data that could inform future decisions

### Creating New Pages

When adding new documentation pages, create them under the roadmap parent:
```python
mcp__notion__notion-create-pages(
    parent={"page_id": "2bbac7292af381888e48fb4f4876a9a8"},
    pages=[{"properties": {"title": "New Page Title"}, "content": "..."}]
)
```

## Roadmap Status

### ✅ Completed
- **Phase 1**: Foundation (config, logging, retry, tests)
- **Phase 2**: Dynamic week detection, orchestrator, multi-stat predictions
- **Phase 3**: Discord notifications, health monitoring
- **Bonus**: Parlay generation (2-20 legs), weather integration

### ⏳ Pending
- **Phase 4**: Docker & AWS deployment
  - Dockerfile
  - docker-compose.yml
  - APScheduler for automation
  - EC2 deployment
  - CloudWatch logging

### Week 14 Improvements (Completed)
- [x] **Elite WR Filter**: Skip UNDER bets on elite WRs (Chase, Lamb, Jefferson, etc.) for receiving yards
- [x] **Bet Deduplication**: Keep only best line per player/market/side (no more 9 bets on same player)
- [x] **Volume Threshold**: Min line thresholds filter out low-volume players (15+ rec yds, 10+ rush yds, etc.)
- [x] **Player Tier System**: Elite/Starter/Backup classification with confidence multipliers
- [x] **Diversification**: Already supported - all markets (receiving, rushing, passing, receptions) working

### Future Ideas
- More stat types (TDs, interceptions)
- Same-game parlay optimization
- Live odds monitoring
- Bankroll management
- Model auto-retraining

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src/sports_betting

# Run specific test file
pytest tests/test_edge_calculator.py -v
```

## Data Flow

```
1. Odds API → OddsSnapshot (database)
2. NFL Data (nfl_data_py) → Feature Engineering
3. Models (XGBoost) → Predictions (database + JSON)
4. Predictions + Odds → Edges (database)
5. Edges → Parlays (optional)
6. Edges/Parlays → Discord notifications
7. Post-game: Actual stats → Score predictions → Results
```

## Environment Variables

Create `.env` file:
```bash
ODDS_API_KEY=your_key_here
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
DATABASE_URL=sqlite:///data/sports_betting.db
LOG_LEVEL=INFO
```

## Caching & Storage

- **Odds cache**: `~/.sports_betting/odds_cache/`
- **Weather cache**: `~/.sports_betting/weather_overrides.json`
- **Database**: `data/sports_betting.db`
- **Predictions**: `data/predictions/predictions_{season}_week{week}.json`
- **Models**: `models/*.pkl`

## Discord Notification Types

1. **Edge Alerts** - Individual betting opportunities
2. **Parlay Alerts** - Multi-leg parlay recommendations
3. **Weekly Results** - Win/loss record and ROI
4. **Health Alerts** - System issues

## Code Style

- Use `loguru` for logging (not print)
- Type hints on all functions
- Docstrings for public methods
- Keep functions focused and small
- Use the existing patterns in codebase
