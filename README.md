# Sports Betting AI/ML Analysis System

An advanced AI-powered sports betting analysis system focused on NFL parlay creation with comprehensive reasoning and multi-source data integration.

## Features

- **Parlay Creation Engine**: Generate optimized multi-leg and same-game parlays with correlation analysis
- **Comprehensive Reasoning**: Detailed explanations for every parlay leg with statistical analysis
- **AI/ML Predictions**: XGBoost, Neural Networks for player prop predictions and fair value estimation
- **Multi-Source Data**: Integrate live odds API, historical data, ESPN API, and manual inputs
- **Smart API Management**: Budget-protected requests within free tier limits with intelligent caching
- **Portfolio Optimization**: Kelly criterion sizing with risk management and allocation strategies
- **Same-Game Parlay Validation**: Sportsbook rule compliance and correlation constraints
- **Edge Detection**: Find profitable betting opportunities with confidence scoring
- **Performance Tracking**: Monitor model accuracy and profitability

## Quick Start

### Core System (Ready Now)
1. **Check system capabilities**:
   ```bash
   python3 system_capabilities_demo.py
   ```

2. **Generate sample parlay with reasoning**:
   ```bash
   python3 -c "
   import sys; sys.path.insert(0, 'src')
   from sports_betting.analysis.reasoning_engine import demo_reasoning_engine
   demo_reasoning_engine()
   "
   ```

### Enhanced System (ML Features)
1. **Install ML dependencies**:
   ```bash
   pip install pandas numpy scikit-learn xgboost torch nfl-data-py
   ```

2. **Set up environment**:
   ```bash
   cp .env.example .env
   # Add ODDS_API_KEY=your_actual_key to .env
   ```

3. **Run enhanced system demo**:
   ```bash
   python3 enhanced_system_demo.py
   ```

4. **Generate Week 2 parlays with reasoning**:
   ```bash
   python3 -c "
   from sports_betting.analysis.enhanced_recommender import EnhancedParlayRecommender
   from sports_betting.database import get_session
   recommender = EnhancedParlayRecommender(get_session())
   report = recommender.generate_enhanced_recommendations(
       week=2, season=2024, include_reasoning=True, train_models=True
   )
   print('Enhanced parlays with detailed reasoning generated!')
   "
   ```

## Configuration

Set up your API keys in `.env`:
- `ODDS_API_KEY`: The Odds API key
- `WEATHER_API_KEY`: Weather API key (optional)

## Project Structure

```
sports-betting/
├── src/sports_betting/
│   ├── data/              # Data collection and processing
│   ├── models/            # ML models and training
│   ├── features/          # Feature engineering
│   ├── analysis/          # Edge detection and EV calculation
│   ├── database/          # Database models and operations
│   ├── cli/               # Command line interfaces
│   └── config/            # Configuration management
├── config/                # Configuration files
├── data/                  # Local data storage
├── outputs/               # Analysis results
└── tests/                 # Test suite
```

## Usage Examples

### Parlay Generation with Reasoning
```bash
# Check all system capabilities
python3 system_capabilities_demo.py

# Generate sample parlay with detailed reasoning
python3 -c "
import sys; sys.path.insert(0, 'src')
from sports_betting.analysis.reasoning_engine import ReasoningEngine
from sports_betting.database import get_session
engine = ReasoningEngine(get_session())
reasoning = engine.generate_parlay_reasoning({
    'legs': [{'prop': 'receiving_yards', 'player': 'Travis Kelce', 'line': 72.5}],
    'correlations': {('receiving_yards', 'receptions'): 0.85}
})
print('Detailed parlay reasoning generated!')
"
```

### Data Collection and Analysis
```bash
# Test data manager capabilities
python3 -c "
import sys; sys.path.insert(0, 'src')
from sports_betting.data.data_manager import DataManager
from sports_betting.database import get_session
manager = DataManager(get_session())
data = manager.feed_week_data(week=2, season=2024, data_sources=['espn', 'historical'])
print(f'Collected data for {len(data.get(\"games\", []))} games')
"

# Test NFL historical data collection
python3 -c "
import sys; sys.path.insert(0, 'src')
from sports_betting.data.collectors.nfl_historical import demo_historical_collector
demo_historical_collector()
"
```

### Enhanced ML Recommendations
```bash
# Generate complete enhanced recommendations
python3 enhanced_system_demo.py

# Create portfolio with reasoning
python3 -c "
from sports_betting.analysis.enhanced_recommender import EnhancedParlayRecommender
from sports_betting.database import get_session
recommender = EnhancedParlayRecommender(get_session())
report = recommender.generate_enhanced_recommendations(
    week=2, season=2024, bankroll=10000, 
    risk_tolerance='moderate', include_reasoning=True
)
print('Portfolio with detailed reasoning generated!')
"
```

### API Management and Caching
```bash
# Test smart API management
python3 -c "
import sys; sys.path.insert(0, 'src')
from sports_betting.data.collectors.smart_api_manager import SmartApiManager
from sports_betting.database import get_session
manager = SmartApiManager(get_session())
status = manager.get_usage_stats()
print(f'API usage: {status[\"requests_used\"]}/{status[\"monthly_limit\"]}')
"
```

## System Capabilities

### 🎯 **Ready Now (Core System)**
- ✅ **Parlay Mathematics**: Complete correlation analysis and EV calculations
- ✅ **Reasoning Engine**: Detailed explanations for every bet recommendation
- ✅ **Portfolio Optimization**: Kelly criterion sizing with risk management
- ✅ **Same-Game Parlay Validation**: Sportsbook rule compliance
- ✅ **Smart API Management**: Budget protection within free tier limits
- ✅ **Multi-Source Data**: ESPN API, historical files, manual CSV integration

### 🚀 **Enhanced Features (With Dependencies)**
- 📦 **ML Models**: XGBoost and Neural Networks for prop predictions
- 📊 **Historical Data**: NFL statistics via nfl-data-py integration
- 🔑 **Live Odds**: Real-time market data via Odds API
- 🧠 **Advanced Reasoning**: Model-powered insights with confidence scores

### 💰 **Sample Output**
```
💎 KC Same-Game Parlay (+485, 8.7% EV)
   Recommended Bet: $285 (2.85% of $10k bankroll)

🎯 LEG 1: Travis Kelce Receiving Yards OVER 72.5 (-110)
   📊 XGBoost Prediction: 78.2 yards (83% confidence)
   📈 Reasoning: Kelce averages 78.5 yards vs teams ranked 25+ in
      pass defense. DEN ranks 28th allowing 7.8 YPT to TEs.
   ⚖️ Risk: Low risk - strong fundamentals and model agreement

🎯 LEG 2: Travis Kelce Receptions OVER 6.5 (-115)
   📊 Prediction: 7.1 receptions (79% confidence)
   📈 Reasoning: Strong 0.85 correlation with receiving yards.
   ⚖️ Risk: Low risk - highly correlated with Leg 1

🔗 CORRELATION ANALYSIS:
   Average correlation: +0.68 (optimal range)
   Joint probability boost: +12% vs independent calculation

💰 EXECUTION: Execute based on 8.7% EV, high confidence (81%), 
   optimal correlations, strong fundamentals
```

## Development

```bash
# Check core system capabilities
python3 system_capabilities_demo.py

# Run enhanced system demo (requires dependencies)
python3 enhanced_system_demo.py

# Format code (if using development environment)
black .
isort .

# Type checking
mypy src/
```