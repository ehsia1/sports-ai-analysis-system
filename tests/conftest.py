"""Pytest fixtures for the sports betting test suite."""

import pytest
import tempfile
import sqlite3
from pathlib import Path
from unittest.mock import patch


@pytest.fixture
def temp_db():
    """Create a temporary SQLite database for testing."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = Path(f.name)
        yield db_path
        # Cleanup
        if db_path.exists():
            db_path.unlink()


@pytest.fixture
def sample_odds_response():
    """Sample Odds API response for testing."""
    return {
        "data": [{
            "id": "test123",
            "home_team": "Green Bay Packers",
            "away_team": "Detroit Lions",
            "commence_time": "2025-11-30T18:00:00Z",
            "bookmakers": [{
                "key": "fanduel",
                "title": "FanDuel",
                "markets": [{
                    "key": "player_reception_yds",
                    "outcomes": [
                        {"description": "Amon-Ra St. Brown", "point": 72.5, "name": "Over", "price": -115},
                        {"description": "Amon-Ra St. Brown", "point": 72.5, "name": "Under", "price": -105}
                    ]
                }]
            }]
        }],
        "events_count": 1,
        "total_cost": 1
    }


@pytest.fixture
def sample_player_props():
    """Sample player props data for testing edge calculation."""
    return {
        'player_name': 'Amon-Ra St. Brown',
        'market': 'player_reception_yds',
        'line': 72.5,
        'over_odds': -115,
        'under_odds': -105,
        'model_prediction': 85.0,
        'model_confidence': 0.75
    }


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    with patch('src.sports_betting.config.get_settings') as mock:
        mock.return_value.min_edge_pct = 3.0
        mock.return_value.min_confidence = 0.65
        mock.return_value.database_url = 'sqlite:///:memory:'
        yield mock.return_value


@pytest.fixture
def sample_ngs_receiving_data():
    """Sample NGS receiving data for testing results collection."""
    import pandas as pd
    return pd.DataFrame({
        'player_display_name': ['Amon-Ra St. Brown', 'Ja\'Marr Chase', 'CeeDee Lamb'],
        'week': [13, 13, 13],
        'yards': [95, 120, 85],
        'receptions': [8, 9, 7],
        'targets': [12, 11, 10]
    })


@pytest.fixture
def sample_ngs_rushing_data():
    """Sample NGS rushing data for testing."""
    import pandas as pd
    return pd.DataFrame({
        'player_display_name': ['Derrick Henry', 'James Cook', 'Saquon Barkley'],
        'week': [13, 13, 13],
        'rush_yards': [120, 95, 110],
        'rush_attempts': [22, 18, 20]
    })
