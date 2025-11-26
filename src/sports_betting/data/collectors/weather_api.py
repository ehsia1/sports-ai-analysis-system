"""Weather data collector."""

from datetime import datetime
from typing import Dict, Optional

import requests

from ...config import get_settings


class WeatherCollector:
    """Collector for weather data using OpenWeatherMap API."""

    def __init__(self):
        self.settings = get_settings()
        self.base_url = self.settings.weather_api_base_url
        self.api_key = self.settings.weather_api_key
        self.session = requests.Session()

    def get_current_weather(self, city: str, state: str = None) -> Optional[Dict]:
        """Get current weather for a city."""
        if not self.api_key:
            return None
        
        location = f"{city},{state}" if state else city
        params = {
            "q": location,
            "appid": self.api_key,
            "units": "imperial",  # Fahrenheit
        }
        
        try:
            response = self.session.get(
                f"{self.base_url}/weather", params=params, timeout=15
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching weather for {location}: {e}")
            return None

    def get_game_weather(self, game_date: datetime, stadium_city: str, stadium_state: str = None) -> Dict:
        """Get weather conditions for a game."""
        weather_data = self.get_current_weather(stadium_city, stadium_state)
        
        if not weather_data:
            return {}
        
        try:
            main = weather_data.get("main", {})
            weather = weather_data.get("weather", [{}])[0]
            wind = weather_data.get("wind", {})
            rain = weather_data.get("rain", {})
            snow = weather_data.get("snow", {})
            
            return {
                "temperature": main.get("temp"),
                "feels_like": main.get("feels_like"),
                "humidity": main.get("humidity"),
                "pressure": main.get("pressure"),
                "description": weather.get("description", ""),
                "wind_speed": wind.get("speed", 0),
                "wind_direction": wind.get("deg"),
                "precipitation_1h": rain.get("1h", 0) + snow.get("1h", 0),
                "visibility": weather_data.get("visibility", 10000) / 1000,  # Convert to km
                "cloud_cover": weather_data.get("clouds", {}).get("all", 0),
            }
        except Exception as e:
            print(f"Error parsing weather data: {e}")
            return {}

    def get_stadium_locations(self) -> Dict[str, Dict[str, str]]:
        """Get stadium locations for NFL teams."""
        return {
            "ARI": {"city": "Glendale", "state": "AZ", "dome": True},
            "ATL": {"city": "Atlanta", "state": "GA", "dome": True},
            "BAL": {"city": "Baltimore", "state": "MD", "dome": False},
            "BUF": {"city": "Orchard Park", "state": "NY", "dome": False},
            "CAR": {"city": "Charlotte", "state": "NC", "dome": False},
            "CHI": {"city": "Chicago", "state": "IL", "dome": False},
            "CIN": {"city": "Cincinnati", "state": "OH", "dome": False},
            "CLE": {"city": "Cleveland", "state": "OH", "dome": False},
            "DAL": {"city": "Arlington", "state": "TX", "dome": True},
            "DEN": {"city": "Denver", "state": "CO", "dome": False},
            "DET": {"city": "Detroit", "state": "MI", "dome": True},
            "GB": {"city": "Green Bay", "state": "WI", "dome": False},
            "HOU": {"city": "Houston", "state": "TX", "dome": True},
            "IND": {"city": "Indianapolis", "state": "IN", "dome": True},
            "JAX": {"city": "Jacksonville", "state": "FL", "dome": False},
            "KC": {"city": "Kansas City", "state": "MO", "dome": False},
            "LV": {"city": "Las Vegas", "state": "NV", "dome": True},
            "LAC": {"city": "Los Angeles", "state": "CA", "dome": False},
            "LAR": {"city": "Los Angeles", "state": "CA", "dome": False},
            "MIA": {"city": "Miami Gardens", "state": "FL", "dome": False},
            "MIN": {"city": "Minneapolis", "state": "MN", "dome": True},
            "NE": {"city": "Foxborough", "state": "MA", "dome": False},
            "NO": {"city": "New Orleans", "state": "LA", "dome": True},
            "NYG": {"city": "East Rutherford", "state": "NJ", "dome": False},
            "NYJ": {"city": "East Rutherford", "state": "NJ", "dome": False},
            "PHI": {"city": "Philadelphia", "state": "PA", "dome": False},
            "PIT": {"city": "Pittsburgh", "state": "PA", "dome": False},
            "SF": {"city": "Santa Clara", "state": "CA", "dome": False},
            "SEA": {"city": "Seattle", "state": "WA", "dome": False},
            "TB": {"city": "Tampa", "state": "FL", "dome": False},
            "TEN": {"city": "Nashville", "state": "TN", "dome": False},
            "WAS": {"city": "Landover", "state": "MD", "dome": False},
        }

    def collect_game_weather(self, team_abbr: str, game_date: datetime) -> Dict:
        """Collect weather data for a game."""
        stadium_locations = self.get_stadium_locations()
        location = stadium_locations.get(team_abbr)
        
        if not location:
            return {}
        
        # If it's a dome, return default indoor conditions
        if location.get("dome", False):
            return {
                "temperature": 72.0,
                "feels_like": 72.0,
                "humidity": 50,
                "description": "Indoor climate controlled",
                "wind_speed": 0.0,
                "precipitation_1h": 0.0,
                "is_dome": True,
            }
        
        weather = self.get_game_weather(
            game_date, location["city"], location["state"]
        )
        weather["is_dome"] = False
        return weather