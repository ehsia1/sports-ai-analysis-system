#!/usr/bin/env python3
"""Quick start script to set up the sports betting analysis system."""

import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ Error in {description}: {e}")
        print(f"Error output: {e.stderr}")
        return None


def main():
    """Set up the sports betting system."""
    print("🏈 Sports Betting AI/ML System Setup")
    print("=" * 40)
    
    # Check Python version
    if sys.version_info < (3, 11):
        print("❌ Python 3.11+ is required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Create directories
    print("📁 Creating directories...")
    directories = ["data", "outputs", "logs", "data/models", "data/features"]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    print("✅ Directories created")
    
    # Create .env file if it doesn't exist
    env_file = Path(".env")
    if not env_file.exists():
        print("📝 Creating .env file...")
        env_content = """# API Keys (replace with your actual keys)
ODDS_API_KEY=your_odds_api_key_here
WEATHER_API_KEY=your_weather_api_key_here

# Database
DATABASE_URL=sqlite:///data/sports_betting.db

# Model Configuration
MODEL_CACHE_DIR=data/models
FEATURE_CACHE_DIR=data/features

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/sports_betting.log

# Analysis Configuration
DEFAULT_BANKROLL=10000
MAX_BET_SIZE=0.05
MIN_EDGE_THRESHOLD=0.02
CONFIDENCE_THRESHOLD=0.7
"""
        with open(env_file, "w") as f:
            f.write(env_content)
        print("✅ .env file created")
        print("⚠️  Please edit .env file with your API keys!")
    
    # Install dependencies
    if Path("pyproject.toml").exists():
        # Try poetry first
        poetry_check = run_command("poetry --version", "Checking for Poetry")
        if poetry_check:
            run_command("poetry install", "Installing dependencies with Poetry")
        else:
            print("Poetry not found, falling back to pip...")
            run_command("pip install -e .", "Installing with pip")
    else:
        run_command("pip install -r requirements.txt", "Installing dependencies")
    
    # Initialize database
    print("🗄️  Initializing database...")
    try:
        from src.sports_betting.database import init_db
        init_db()
        print("✅ Database initialized")
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        print("You can try running: python -m sports_betting.database.init_db")
    
    # Test the CLI
    print("🧪 Testing CLI...")
    test_result = run_command("python -m sports_betting.cli.analyzer --help", "Testing CLI")
    if test_result:
        print("✅ CLI is working!")
    
    print("\n" + "=" * 40)
    print("🎉 Setup Complete!")
    print("\nNext steps:")
    print("1. Edit .env file with your API keys")
    print("2. Run: python -m sports_betting.cli.analyzer --help")
    print("3. Try: python -m sports_betting.cli.analyzer --week 5 --update-data")
    print("\nFor The Odds API key, visit: https://the-odds-api.com/")
    print("For Weather API key, visit: https://openweathermap.org/api")


if __name__ == "__main__":
    main()