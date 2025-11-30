"""Weather data for NFL games."""

from dataclasses import dataclass, asdict
from typing import Dict, Optional, List, Tuple
from datetime import datetime
from pathlib import Path
import json
import logging
import math
import requests

import nfl_data_py as nfl

logger = logging.getLogger(__name__)

# Persistent storage path
WEATHER_CACHE_FILE = Path.home() / ".sports_betting" / "weather_overrides.json"

# NFL Stadium coordinates (lat, lon)
# Home team abbreviation -> (latitude, longitude, dome/outdoor)
NFL_STADIUMS = {
    "ARI": (33.5276, -112.2626, "dome"),      # State Farm Stadium
    "ATL": (33.7554, -84.4010, "dome"),       # Mercedes-Benz Stadium
    "BAL": (39.2780, -76.6227, "outdoor"),    # M&T Bank Stadium
    "BUF": (42.7738, -78.7870, "outdoor"),    # Highmark Stadium
    "CAR": (35.2258, -80.8528, "outdoor"),    # Bank of America Stadium
    "CHI": (41.8623, -87.6167, "outdoor"),    # Soldier Field
    "CIN": (39.0954, -84.5160, "outdoor"),    # Paycor Stadium
    "CLE": (41.5061, -81.6995, "outdoor"),    # Cleveland Browns Stadium
    "DAL": (32.7473, -97.0945, "dome"),       # AT&T Stadium
    "DEN": (39.7439, -105.0201, "outdoor"),   # Empower Field
    "DET": (42.3400, -83.0456, "dome"),       # Ford Field
    "GB": (44.5013, -88.0622, "outdoor"),     # Lambeau Field
    "HOU": (29.6847, -95.4107, "dome"),       # NRG Stadium
    "IND": (39.7601, -86.1639, "dome"),       # Lucas Oil Stadium
    "JAX": (30.3239, -81.6373, "outdoor"),    # TIAA Bank Field
    "KC": (39.0489, -94.4839, "outdoor"),     # Arrowhead Stadium
    "LA": (33.9535, -118.3392, "dome"),       # SoFi Stadium
    "LAC": (33.9535, -118.3392, "dome"),      # SoFi Stadium (shared)
    "LV": (36.0909, -115.1833, "dome"),       # Allegiant Stadium
    "MIA": (25.9580, -80.2389, "outdoor"),    # Hard Rock Stadium
    "MIN": (44.9737, -93.2577, "dome"),       # U.S. Bank Stadium
    "NE": (42.0909, -71.2643, "outdoor"),     # Gillette Stadium
    "NO": (29.9511, -90.0812, "dome"),        # Caesars Superdome
    "NYG": (40.8128, -74.0742, "outdoor"),    # MetLife Stadium
    "NYJ": (40.8128, -74.0742, "outdoor"),    # MetLife Stadium (shared)
    "PHI": (39.9008, -75.1675, "outdoor"),    # Lincoln Financial Field
    "PIT": (40.4468, -80.0158, "outdoor"),    # Acrisure Stadium
    "SEA": (47.5952, -122.3316, "outdoor"),   # Lumen Field
    "SF": (37.4033, -121.9695, "outdoor"),    # Levi's Stadium
    "TB": (27.9759, -82.5033, "outdoor"),     # Raymond James Stadium
    "TEN": (36.1665, -86.7713, "outdoor"),    # Nissan Stadium
    "WAS": (38.9076, -76.8645, "outdoor"),    # Commanders Field
}


def fetch_weather_gov(lat: float, lon: float) -> Optional[Dict]:
    """
    Fetch weather forecast from Weather.gov API.

    Args:
        lat: Latitude
        lon: Longitude

    Returns:
        Dict with temp_f, wind_mph, conditions or None if failed
    """
    headers = {
        "User-Agent": "(sports-betting-app, contact@example.com)",
        "Accept": "application/geo+json",
    }

    try:
        # Step 1: Get the forecast URL for this location
        points_url = f"https://api.weather.gov/points/{lat:.4f},{lon:.4f}"
        resp = requests.get(points_url, headers=headers, timeout=10)
        resp.raise_for_status()
        points_data = resp.json()

        forecast_url = points_data["properties"]["forecast"]

        # Step 2: Get the forecast
        resp = requests.get(forecast_url, headers=headers, timeout=10)
        resp.raise_for_status()
        forecast_data = resp.json()

        # Get the first period (current/upcoming)
        periods = forecast_data["properties"]["periods"]
        if not periods:
            return None

        period = periods[0]

        # Parse conditions
        forecast_text = period.get("shortForecast", "").lower()
        conditions = "clear"
        if "snow" in forecast_text:
            conditions = "snow"
        elif "rain" in forecast_text or "shower" in forecast_text:
            conditions = "rain"
        elif "wind" in forecast_text:
            conditions = "wind"
        elif "cloud" in forecast_text or "overcast" in forecast_text:
            conditions = "cloudy"

        # Parse wind speed (e.g., "10 to 15 mph" -> 12.5)
        wind_speed = period.get("windSpeed", "")
        wind_mph = None
        if wind_speed:
            import re
            numbers = re.findall(r'\d+', wind_speed)
            if numbers:
                wind_mph = sum(float(n) for n in numbers) / len(numbers)

        return {
            "temp_f": period.get("temperature"),
            "wind_mph": wind_mph,
            "conditions": conditions,
            "forecast": period.get("shortForecast"),
            "detailed": period.get("detailedForecast"),
        }

    except requests.RequestException as e:
        logger.warning(f"Weather.gov API error: {e}")
        return None
    except (KeyError, IndexError) as e:
        logger.warning(f"Weather.gov parse error: {e}")
        return None


@dataclass
class GameWeather:
    """Weather conditions for a game."""

    home_team: str
    away_team: str
    roof: str  # 'outdoors', 'dome', 'closed'
    temp_f: Optional[float] = None
    wind_mph: Optional[float] = None
    conditions: Optional[str] = None  # 'clear', 'rain', 'snow', 'wind'

    @property
    def is_dome(self) -> bool:
        return self.roof in ('dome', 'closed')

    @property
    def is_bad_weather(self) -> bool:
        """Check if weather significantly impacts passing game."""
        if self.is_dome:
            return False

        # Snow or rain
        if self.conditions in ('snow', 'rain'):
            return True

        # High wind (>15 mph)
        if self.wind_mph and self.wind_mph > 15:
            return True

        # Very cold (<25°F)
        if self.temp_f and self.temp_f < 25:
            return True

        return False

    @property
    def weather_impact(self) -> float:
        """
        Return a multiplier for prediction confidence.

        1.0 = no impact
        0.85 = moderate impact (cold, wind)
        0.75 = severe impact (snow, heavy rain)
        """
        if self.is_dome:
            return 1.0

        impact = 1.0

        # Temperature impact
        if self.temp_f:
            if self.temp_f < 20:
                impact *= 0.90
            elif self.temp_f < 32:
                impact *= 0.95

        # Wind impact
        if self.wind_mph:
            if self.wind_mph > 20:
                impact *= 0.85
            elif self.wind_mph > 15:
                impact *= 0.92

        # Precipitation impact
        if self.conditions == 'snow':
            impact *= 0.80
        elif self.conditions == 'rain':
            impact *= 0.90

        return impact

    def _is_valid_number(self, val) -> bool:
        """Check if value is a valid number (not None or NaN)."""
        if val is None:
            return False
        try:
            return not math.isnan(val)
        except (TypeError, ValueError):
            return False

    @property
    def summary(self) -> str:
        """Human-readable weather summary."""
        if self.is_dome:
            return "Dome"

        parts = []
        if self.conditions:
            parts.append(self.conditions.title())
        if self._is_valid_number(self.temp_f):
            parts.append(f"{self.temp_f:.0f}°F")
        if self._is_valid_number(self.wind_mph):
            parts.append(f"{self.wind_mph:.0f}mph wind")

        return ", ".join(parts) if parts else "Outdoor"


class WeatherService:
    """Service for fetching and managing game weather data."""

    def __init__(self):
        self._manual_weather: Dict[str, GameWeather] = {}
        self._schedule_cache: Optional[Dict] = None
        self._load_overrides()

    def _load_overrides(self):
        """Load weather overrides from file."""
        if WEATHER_CACHE_FILE.exists():
            try:
                with open(WEATHER_CACHE_FILE, 'r') as f:
                    data = json.load(f)
                    for key, w in data.items():
                        self._manual_weather[key] = GameWeather(**w)
                logger.debug(f"Loaded {len(self._manual_weather)} weather overrides")
            except Exception as e:
                logger.warning(f"Failed to load weather overrides: {e}")

    def _save_overrides(self):
        """Save weather overrides to file."""
        WEATHER_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        try:
            data = {k: asdict(v) for k, v in self._manual_weather.items()}
            with open(WEATHER_CACHE_FILE, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save weather overrides: {e}")

    def set_weather(
        self,
        home_team: str,
        away_team: str,
        conditions: str,
        temp_f: Optional[float] = None,
        wind_mph: Optional[float] = None,
    ):
        """Manually set weather for a game."""
        key = f"{away_team}@{home_team}"
        self._manual_weather[key] = GameWeather(
            home_team=home_team,
            away_team=away_team,
            roof='outdoors',
            temp_f=temp_f,
            wind_mph=wind_mph,
            conditions=conditions,
        )
        self._save_overrides()
        logger.info(f"Set weather for {key}: {conditions}, {temp_f}°F, {wind_mph}mph")

    def fetch_live_weather(self, home_team: str, away_team: str) -> Optional[GameWeather]:
        """
        Fetch live weather from Weather.gov for a game.

        Args:
            home_team: Home team abbreviation
            away_team: Away team abbreviation

        Returns:
            GameWeather with live data, or None if failed/dome
        """
        stadium = NFL_STADIUMS.get(home_team)
        if not stadium:
            logger.warning(f"Unknown stadium for team: {home_team}")
            return None

        lat, lon, roof_type = stadium

        # Skip dome games
        if roof_type == "dome":
            logger.info(f"{home_team} plays in a dome - skipping weather fetch")
            return GameWeather(
                home_team=home_team,
                away_team=away_team,
                roof="dome",
            )

        # Fetch from Weather.gov
        logger.info(f"Fetching weather for {away_team}@{home_team} ({lat}, {lon})")
        weather = fetch_weather_gov(lat, lon)

        if not weather:
            return None

        return GameWeather(
            home_team=home_team,
            away_team=away_team,
            roof="outdoors",
            temp_f=weather.get("temp_f"),
            wind_mph=weather.get("wind_mph"),
            conditions=weather.get("conditions"),
        )

    def fetch_all_outdoor_games(self, season: int, week: int) -> Dict[str, GameWeather]:
        """
        Fetch live weather for all outdoor games in a week.

        Returns:
            Dict of game_key -> GameWeather for outdoor games
        """
        results = {}

        try:
            sched = nfl.import_schedules([season])
            week_games = sched[(sched['week'] == week) & (sched['season'] == season)]

            for _, game in week_games.iterrows():
                home_team = game['home_team']
                away_team = game['away_team']
                key = f"{away_team}@{home_team}"

                # Skip if manual override exists
                if key in self._manual_weather:
                    results[key] = self._manual_weather[key]
                    continue

                # Fetch live weather
                weather = self.fetch_live_weather(home_team, away_team)
                if weather:
                    results[key] = weather
                    # Cache all outdoor weather (not just bad weather)
                    if not weather.is_dome:
                        self._manual_weather[key] = weather

            # Save any new weather data
            if results:
                self._save_overrides()

        except Exception as e:
            logger.error(f"Failed to fetch outdoor game weather: {e}")

        return results

    def get_weather_for_week(
        self,
        season: int,
        week: int,
    ) -> Dict[str, GameWeather]:
        """
        Get weather for all games in a week.

        Returns dict keyed by "AWAY@HOME" team codes.
        """
        weather_data = {}

        # Try to get from NFL schedule data
        try:
            sched = nfl.import_schedules([season])
            week_games = sched[(sched['week'] == week) & (sched['season'] == season)]

            for _, game in week_games.iterrows():
                key = f"{game['away_team']}@{game['home_team']}"

                # Check for manual override first
                if key in self._manual_weather:
                    weather_data[key] = self._manual_weather[key]
                    continue

                weather_data[key] = GameWeather(
                    home_team=game['home_team'],
                    away_team=game['away_team'],
                    roof=game.get('roof', 'outdoors') or 'outdoors',
                    temp_f=game.get('temp'),
                    wind_mph=game.get('wind'),
                )
        except Exception as e:
            logger.warning(f"Failed to fetch schedule weather: {e}")

        # Add any manual weather not in schedule
        for key, weather in self._manual_weather.items():
            if key not in weather_data:
                weather_data[key] = weather

        return weather_data

    def get_weather_for_game(
        self,
        home_team: str,
        away_team: str,
        season: int,
        week: int,
    ) -> Optional[GameWeather]:
        """Get weather for a specific game."""
        key = f"{away_team}@{home_team}"

        # Check manual override first
        if key in self._manual_weather:
            return self._manual_weather[key]

        # Get from week data
        week_weather = self.get_weather_for_week(season, week)
        return week_weather.get(key)

    def get_bad_weather_games(
        self,
        season: int,
        week: int,
    ) -> List[GameWeather]:
        """Get list of games with bad weather conditions."""
        weather_data = self.get_weather_for_week(season, week)
        return [w for w in weather_data.values() if w.is_bad_weather]

    def clear_manual_weather(self):
        """Clear all manually set weather."""
        self._manual_weather.clear()
        self._save_overrides()


# Singleton instance
_weather_service = WeatherService()


def get_weather_service() -> WeatherService:
    """Get the global weather service instance."""
    return _weather_service


def set_game_weather(
    home_team: str,
    away_team: str,
    conditions: str,
    temp_f: Optional[float] = None,
    wind_mph: Optional[float] = None,
):
    """Convenience function to set weather for a game."""
    _weather_service.set_weather(home_team, away_team, conditions, temp_f, wind_mph)
