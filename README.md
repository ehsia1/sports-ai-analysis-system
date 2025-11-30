# NFL Player Props Prediction System

A machine learning system for predicting NFL player props and identifying betting edges by comparing model predictions to sportsbook lines.

## What It Does

This system:
1. **Trains ML models** on historical NFL data (2020-2024) to predict player statistics
2. **Fetches live odds** from The Odds API for current week's player props
3. **Generates predictions** for receiving yards, rushing yards, passing yards, and receptions
4. **Identifies edges** where our predictions significantly differ from sportsbook lines
5. **Tracks results** to measure model accuracy and betting ROI

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

### 2. Set Up API Key

```bash
cp .env.example .env
# Add your Odds API key: ODDS_API_KEY=your_key_here
```

Get a free API key at [The Odds API](https://the-odds-api.com/).

### 3. Train Models

```bash
python scripts/train_models_v2.py
```

### 4. Collect Odds & Generate Predictions

```bash
# Fetch this week's player props
python scripts/collect_daily_odds.py

# Generate predictions for all stat types
python scripts/predict_week13_all.py
```

### 5. After Games - Collect Results

```bash
python scripts/collect_results.py
```

## Project Structure

```
sports-betting/
├── src/sports_betting/
│   ├── ml/                    # ML models and predictors
│   │   ├── receiving_yards.py # Adaptive receiving predictor
│   │   └── __init__.py
│   ├── data/                  # Data collection
│   │   ├── odds_api.py        # The Odds API client
│   │   └── collectors/        # NFL data collectors
│   ├── database/              # SQLite database models
│   └── analysis/              # Edge calculation
├── scripts/
│   ├── train_models_v2.py     # Train all prediction models
│   ├── predict_week13_all.py  # Generate multi-stat predictions
│   ├── collect_daily_odds.py  # Fetch odds from API
│   └── collect_results.py     # Score predictions after games
├── models/                    # Saved model files (.pkl)
├── data/                      # SQLite database
└── docs/                      # Analysis reports
```

## Usage Examples

### Generate Predictions

```python
from src.sports_betting.ml import ReceivingYardsPredictor

predictor = ReceivingYardsPredictor()
predictions, diagnostics = predictor.predict_adaptive(2025, 13)
print(predictions[['player_name', 'predicted_yards', 'confidence']])
```

### Find Betting Edges

```bash
# Run the full prediction pipeline
python scripts/predict_week13_all.py

# Output shows edges like:
# OVER  Keon Coleman  Rec Yards  25.5  35.2  +38.1%  -118
# UNDER Taysom Hill   Rush Yards 22.5  14.7  -34.7%  -110
```

### Check API Usage

```python
from src.sports_betting.data.odds_api import OddsAPIClient

client = OddsAPIClient()
print(f"Remaining API credits: {client.remaining_requests}")
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
| `DATABASE_URL` | SQLite path (default: data/sports_betting.db) | No |

### API Usage

The Odds API free tier includes 500 requests/month. Each market fetch uses 1 request. The system caches odds to minimize API usage.

## Analysis Reports

After running predictions, reports are saved to `docs/`:

- `WEEK_13_ALL_PREDICTIONS.md` - All predictions and edges
- `WEEK_13_ANALYSIS.md` - Detailed bias investigation and betting tiers
- `WEEK_13_RESULTS.md` - Post-game performance analysis

## Development

```bash
# Run tests
pytest tests/

# Train models with fresh data
python scripts/train_models_v2.py

# Test results collection with previous week
python scripts/collect_results.py --test
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
- Weather and game script adjustments are not yet implemented
- Backups and low-snap players have higher prediction variance

## License

MIT
