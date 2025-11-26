"""ESPN API collector for free NFL data."""

import time
from datetime import datetime
from typing import Dict, List, Optional

import requests
from sqlalchemy.orm import Session

from ...database import Game, Player, Team, get_session
from ..request_manager import CacheManager, RequestManager


class ESPNAPICollector:
    """Collector for ESPN's hidden API - completely free."""

    def __init__(self):
        self.base_url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"
        self.core_url = "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl"
        self.session = requests.Session()
        
        # Use cache manager but not request limits (ESPN is free)
        self.cache_manager = CacheManager()
        self.request_manager = RequestManager()

    def _make_request(
        self,
        url: str,
        request_type: str = "espn_data",
        cache_ttl: int = 3600,
    ) -> Optional[Dict]:
        """Make request with caching."""
        # Check cache first
        is_cached, cache_msg = self.cache_manager.is_cached_and_fresh(
            request_type, "espn", url=url
        )
        
        if is_cached:
            print(f"📋 Using cached ESPN data: {cache_msg}")
            return {"cached": True}
        
        start_time = time.time()
        
        try:
            response = self.session.get(url, timeout=30)
            response_time_ms = int((time.time() - start_time) * 1000)
            
            response.raise_for_status()
            data = response.json()
            
            # Log request (ESPN is free, so just for monitoring)
            self.request_manager.log_request(
                api_source="espn",
                endpoint=url,
                request_type=request_type,
                objects_returned=1,
                success=True,
                response_time_ms=response_time_ms,
            )
            
            # Cache the response
            self.cache_manager.cache_data(
                request_type, "espn", data, ttl_seconds=cache_ttl,
                url=url
            )
            
            print(f"✅ ESPN API request successful")
            return data
            
        except Exception as e:
            print(f"❌ ESPN API request failed: {e}")
            return None

    def get_scoreboard(self, season: int = None, week: int = None) -> Optional[Dict]:
        """Get current NFL scoreboard."""
        url = f"{self.base_url}/scoreboard"
        
        params = []
        if season:
            params.append(f"season={season}")
        if week:
            params.append(f"week={week}")
        
        if params:
            url += "?" + "&".join(params)
        
        return self._make_request(url, "scoreboard", cache_ttl=1800)  # 30 minutes

    def get_team_schedule(self, team_id: str, season: int) -> Optional[Dict]:
        """Get team schedule for the season."""
        url = f"{self.core_url}/seasons/{season}/teams/{team_id}/events"
        return self._make_request(url, "schedule", cache_ttl=24*3600)  # 24 hours

    def get_team_stats(self, team_id: str, season: int) -> Optional[Dict]:
        """Get team statistics."""
        url = f"{self.core_url}/seasons/{season}/teams/{team_id}/statistics"
        return self._make_request(url, "team_stats", cache_ttl=6*3600)  # 6 hours

    def get_player_stats(self, player_id: str, season: int) -> Optional[Dict]:
        """Get player statistics."""
        url = f"{self.core_url}/seasons/{season}/athletes/{player_id}/statistics"
        return self._make_request(url, "player_stats", cache_ttl=6*3600)  # 6 hours

    def get_game_odds(self, game_id: str) -> Optional[Dict]:
        """Get basic odds for a specific game."""
        url = f"{self.base_url}/events/{game_id}/odds"
        return self._make_request(url, "basic_odds", cache_ttl=1800)  # 30 minutes

    def collect_weekly_data(self, season: int, week: int) -> Dict:
        """Collect comprehensive weekly data from ESPN."""
        print(f"📊 Collecting ESPN data for Season {season}, Week {week}")
        
        results = {
            "season": season,
            "week": week,
            "data_collected": {},
            "errors": [],
        }
        
        # Get scoreboard
        print("🏈 Getting scoreboard...")
        scoreboard = self.get_scoreboard(season, week)
        if scoreboard and not scoreboard.get("cached"):
            results["data_collected"]["scoreboard"] = self._process_scoreboard(scoreboard)
        elif scoreboard:
            print("📋 Scoreboard data cached")
            results["data_collected"]["scoreboard"] = "cached"
        else:
            results["errors"].append("Failed to get scoreboard")
        
        # Get team data for popular teams (limit to save requests)
        popular_teams = ["DAL", "GB", "NE", "KC", "SF", "PIT", "NYG", "PHI"]
        team_data = {}
        
        for team_abbr in popular_teams[:5]:  # Limit to 5 teams for demo
            team_id = self._get_espn_team_id(team_abbr)
            if team_id:
                print(f"📈 Getting stats for {team_abbr}...")
                stats = self.get_team_stats(team_id, season)
                if stats and not stats.get("cached"):
                    team_data[team_abbr] = stats
                elif stats:
                    team_data[team_abbr] = "cached"
        
        results["data_collected"]["team_stats"] = team_data
        
        return results

    def _process_scoreboard(self, scoreboard_data: Dict) -> Dict:
        """Process scoreboard data for useful information."""
        if not scoreboard_data or "events" not in scoreboard_data:
            return {}
        
        processed_games = []
        
        for event in scoreboard_data["events"]:
            game_info = {
                "id": event.get("id"),
                "date": event.get("date"),
                "status": event.get("status", {}).get("type", {}).get("name"),
                "week": event.get("week", {}).get("number"),
                "season": event.get("season", {}).get("year"),
            }
            
            # Get teams
            competitions = event.get("competitions", [])
            if competitions:
                competitors = competitions[0].get("competitors", [])
                
                for competitor in competitors:
                    team = competitor.get("team", {})
                    is_home = competitor.get("homeAway") == "home"
                    
                    key = "home_team" if is_home else "away_team"
                    game_info[key] = {
                        "id": team.get("id"),
                        "abbreviation": team.get("abbreviation"),
                        "display_name": team.get("displayName"),
                        "score": competitor.get("score"),
                    }
                
                # Get basic odds if available
                odds = competitions[0].get("odds", [])
                if odds:
                    game_info["odds"] = {
                        "spread": odds[0].get("details"),
                        "over_under": odds[0].get("overUnder"),
                    }
            
            processed_games.append(game_info)
        
        return {
            "games_count": len(processed_games),
            "games": processed_games,
        }

    def _get_espn_team_id(self, team_abbreviation: str) -> Optional[str]:
        """Map team abbreviation to ESPN team ID."""
        # Simplified mapping - in practice you'd have a complete mapping
        espn_team_mapping = {
            "DAL": "6",   # Cowboys
            "GB": "9",    # Packers  
            "NE": "17",   # Patriots
            "KC": "12",   # Chiefs
            "SF": "25",   # 49ers
            "PIT": "23",  # Steelers
            "NYG": "19",  # Giants
            "PHI": "21",  # Eagles
            "BUF": "2",   # Bills
            "LAR": "14",  # Rams
        }
        
        return espn_team_mapping.get(team_abbreviation)

    def get_player_search(self, player_name: str) -> Optional[Dict]:
        """Search for a player (useful for getting ESPN player IDs)."""
        # ESPN search endpoint
        search_url = "https://site.api.espn.com/apis/search/v2"
        params = {
            "query": player_name,
            "type": "athlete",
            "sport": "football",
            "league": "nfl"
        }
        
        try:
            response = self.session.get(search_url, params=params, timeout=15)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"❌ Player search failed: {e}")
            return None

    def get_comprehensive_game_data(self, game_id: str) -> Dict:
        """Get comprehensive data for a specific game."""
        print(f"🎯 Getting comprehensive data for game {game_id}")
        
        data = {}
        
        # Basic game info
        game_url = f"{self.base_url}/events/{game_id}"
        game_data = self._make_request(game_url, "game_detail", cache_ttl=3600)
        if game_data and not game_data.get("cached"):
            data["game_info"] = game_data
        
        # Game odds
        odds_data = self.get_game_odds(game_id)
        if odds_data and not odds_data.get("cached"):
            data["odds"] = odds_data
        
        # Win probability (if available)
        win_prob_url = f"{self.base_url}/events/{game_id}/winprobability"
        win_prob = self._make_request(win_prob_url, "win_probability", cache_ttl=1800)
        if win_prob and not win_prob.get("cached"):
            data["win_probability"] = win_prob
        
        return data