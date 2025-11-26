#!/usr/bin/env python3
"""System overview and architecture demonstration."""

from pathlib import Path
import sys

def show_project_structure():
    """Display the project structure."""
    print("🏗️  PROJECT ARCHITECTURE")
    print("=" * 50)
    
    structure = """
sports-betting/
├── 📁 src/sports_betting/           # Core application code
│   ├── 🔧 config/                   # Configuration management
│   │   ├── settings.py              # Environment-based settings
│   │   └── __init__.py
│   ├── 🗄️  database/               # Database models and operations
│   │   ├── models.py                # SQLAlchemy models (Teams, Players, Props, etc.)
│   │   ├── session.py               # Database connection management
│   │   ├── init_db.py               # Database initialization
│   │   └── __init__.py
│   ├── 📊 data/                     # Data collection and processing
│   │   ├── collectors/              # Data source integrations
│   │   │   ├── odds_api.py          # The Odds API integration
│   │   │   ├── nfl_data.py          # NFLverse data integration
│   │   │   ├── weather_api.py       # Weather data collection
│   │   │   └── __init__.py
│   │   └── __init__.py
│   ├── 🧠 features/                 # Feature engineering
│   │   ├── engineering.py           # Base feature engineering framework
│   │   ├── nfl_features.py          # NFL-specific features
│   │   └── __init__.py
│   ├── 🤖 models/                   # ML models (to be implemented)
│   │   ├── base/                    # Abstract model classes
│   │   ├── nfl/                     # NFL-specific models
│   │   └── training/                # Training pipelines
│   ├── 📈 analysis/                 # Edge detection and EV calculation (to be implemented)
│   ├── 🖥️  cli/                     # Command line interfaces
│   │   ├── analyzer.py              # Main analysis CLI
│   │   └── __init__.py
│   ├── 🛠️  utils/                   # Utility functions
│   │   ├── odds.py                  # Odds conversion and calculations
│   │   ├── logging.py               # Logging configuration
│   │   └── __init__.py
│   └── __init__.py
├── 📋 config/                       # Configuration files
│   ├── data_sources.yaml           # API endpoints and settings
│   └── model_configs/               # ML model configurations
│       ├── xgboost.yaml
│       └── neural_net.yaml
├── 📁 data/                         # Local data storage
├── 📁 outputs/                      # Analysis results
├── 📁 logs/                         # Application logs
├── 🧪 tests/                        # Test suite
├── 📄 pyproject.toml                # Poetry dependencies
├── 📄 requirements.txt              # Pip dependencies
├── 📄 README.md                     # Documentation
├── 📄 .env.example                  # Environment variables template
└── 🚀 quick_start.py                # Setup script
"""
    
    print(structure)

def show_component_status():
    """Show the status of each component."""
    print("\n🎯 COMPONENT STATUS")
    print("=" * 50)
    
    components = [
        ("✅ Configuration System", "Environment-based settings with Pydantic"),
        ("✅ Database Models", "Complete schema for teams, players, props, edges"),
        ("✅ Data Collectors", "The Odds API, NFLverse, Weather API integrations"),
        ("✅ Feature Engineering", "Rolling stats, opponent adjustments, situational features"),
        ("✅ Odds Utilities", "American/decimal conversion, de-vigging, EV, Kelly"),
        ("✅ CLI Interface", "Rich terminal interface with progress indicators"),
        ("⏳ ML Models", "XGBoost, Neural Networks, Bayesian models (next phase)"),
        ("⏳ Edge Detection", "EV calculation and opportunity identification (next phase)"),
        ("⏳ Training Pipeline", "Automated model training and validation (next phase)"),
        ("⏳ Performance Monitoring", "Backtracking, CLV analysis, P&L tracking (next phase)"),
    ]
    
    for status, description in components:
        print(f"{status} {description}")

def show_data_flow():
    """Show the data flow through the system."""
    print("\n🔄 DATA FLOW")
    print("=" * 50)
    
    flow = """
1. 📊 DATA COLLECTION
   ├── The Odds API → Player Props & Lines
   ├── NFLverse API → Historical Stats & Schedules  
   └── Weather API → Game Conditions

2. 🧹 DATA PROCESSING
   ├── Normalize team names and player IDs
   ├── De-vig odds to true probabilities
   └── Store in SQLite database

3. 🧠 FEATURE ENGINEERING
   ├── Rolling statistics (3, 5, 10 game windows)
   ├── Opponent adjustments and matchup analysis
   ├── Situational features (weather, game script)
   └── Usage and role-based metrics

4. 🤖 MODEL PREDICTIONS (Next Phase)
   ├── XGBoost → Primary predictions
   ├── Neural Networks → Pattern recognition
   ├── Bayesian Models → Uncertainty quantification
   └── Ensemble → Combined predictions

5. 📈 EDGE DETECTION (Next Phase)
   ├── Compare model predictions to market lines
   ├── Calculate expected value (EV)
   ├── Apply Kelly criterion for position sizing
   └── Filter by confidence thresholds

6. 📋 REPORTING
   ├── Terminal interface with Rich formatting
   ├── JSON output for detailed analysis
   ├── CSV export for spreadsheet analysis
   └── Performance tracking and validation
"""
    
    print(flow)

def show_ml_architecture():
    """Show the planned ML architecture."""
    print("\n🧠 ML ARCHITECTURE (NEXT PHASE)")
    print("=" * 50)
    
    ml_arch = """
🎯 PREDICTION TARGETS:
├── Receptions: Negative Binomial → Catch Rate
├── Receiving Yards: Gamma/LogNormal conditional on volume
├── Rushing Yards: Gamma distribution with game script adjustment
├── Touchdowns: Poisson with red zone allocation
└── Anytime TD: Binary classification with team TD distribution

🤖 MODEL ENSEMBLE:
├── XGBoost (Primary)
│   ├── Separate models per prop type
│   ├── Feature importance tracking
│   └── Hyperparameter optimization with Optuna
├── Neural Networks
│   ├── LSTM for sequential patterns
│   ├── Dense networks for matchup analysis
│   └── Attention mechanisms for key features
└── Bayesian Models
    ├── Beta-binomial for completion rates
    ├── Hierarchical models for player groupings
    └── Uncertainty quantification

📊 FEATURES:
├── Historical Performance (rolling windows)
├── Opponent Adjustments (strength of schedule)
├── Usage Metrics (snap share, target share, red zone role)
├── Situational Context (weather, game script, matchups)
├── Advanced Metrics (air yards, YAC, route participation)
└── Market Intelligence (line movements, steam detection)

⚖️ EDGE CALCULATION:
├── Model Predictions → Fair Lines
├── Market Lines → Implied Probabilities
├── Expected Value = (True Probability × Payout) - (False Probability × Stake)
├── Kelly Criterion → Optimal Bet Sizing
└── Portfolio Management → Risk Assessment
"""
    
    print(ml_arch)

def show_sample_workflow():
    """Show a sample analysis workflow."""
    print("\n🔬 SAMPLE WORKFLOW")
    print("=" * 50)
    
    workflow = """
📅 WEEKLY ANALYSIS PROCESS:

1. UPDATE DATA
   $ python -m sports_betting.cli.analyzer --week 5 --update-data
   
2. RUN MODELS
   ├── Load features for all active players
   ├── Generate predictions for each prop type
   ├── Calculate confidence intervals
   └── Store predictions in database

3. FIND EDGES  
   ├── Compare predictions to market lines
   ├── Calculate expected value for each opportunity
   ├── Apply minimum edge threshold (default: 2%)
   └── Rank by EV and confidence

4. GENERATE REPORT
   ├── Top opportunities table
   ├── Reasoning for each play
   ├── Portfolio risk analysis
   └── Save to JSON/CSV

5. TRACK PERFORMANCE
   ├── Monitor closing line value (CLV)
   ├── Track hit rates by model
   ├── Calculate P&L and Sharpe ratio
   └── Retrain models if performance degrades

EXAMPLE OUTPUT:
#1. Ja'Marr Chase CIN vs PIT - Receiving Yards Over 67.5
    Edge: 15.6% | EV: 8.9% | Kelly: 4.5% | Confidence: 82%
    Reasoning: Favorable slot matchup, positive game script, dome game
"""
    
    print(workflow)

def main():
    """Show complete system overview."""
    print("🏈 SPORTS BETTING AI/ML SYSTEM")
    print("🚀 Architecture Overview & Demo")
    print("═" * 60)
    
    show_project_structure()
    show_component_status()
    show_data_flow()
    show_ml_architecture()
    show_sample_workflow()
    
    print("\n" + "═" * 60)
    print("📈 CURRENT STATUS: MVP Foundation Complete")
    print("✅ Ready for ML model implementation")
    print("✅ Database schema and data pipeline working")
    print("✅ Feature engineering framework in place")
    print("✅ CLI interface functional")
    print("✅ Extensible architecture for multiple sports")
    
    print("\n🎯 NEXT STEPS:")
    print("1. Implement XGBoost models for player props")
    print("2. Build edge detection and EV calculation engine")
    print("3. Add model training and validation pipeline")
    print("4. Create performance monitoring dashboard")
    print("5. Expand to additional sports (NBA, MLB)")
    
    print("\n💡 KEY INNOVATIONS:")
    print("• Dual-mode operation: Live lines + Shadow lines")
    print("• AI-powered feature engineering")
    print("• Ensemble ML approach with uncertainty quantification")
    print("• Automated edge detection with Kelly criterion sizing")
    print("• Comprehensive performance tracking and validation")
    
    print(f"\n🏆 Total Files Created: {count_project_files()}")
    print("💾 Sample outputs available in outputs/ directory")

def count_project_files():
    """Count the number of files created."""
    src_files = len(list(Path("src").rglob("*.py"))) if Path("src").exists() else 0
    config_files = len(list(Path("config").rglob("*.yaml"))) if Path("config").exists() else 0
    root_files = len([f for f in Path(".").iterdir() if f.suffix in [".py", ".toml", ".txt", ".md"]])
    return src_files + config_files + root_files

if __name__ == "__main__":
    main()