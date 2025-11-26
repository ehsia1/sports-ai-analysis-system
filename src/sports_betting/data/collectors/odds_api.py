"""Odds API data collector."""

import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import requests
from sqlalchemy.orm import Session

from ...config import get_settings
from ...database import Book, Game, Player, Prop, get_session


class OddsAPICollector:
    """Collector for The Odds API."""

    def __init__(self):
        self.settings = get_settings()
        self.base_url = self.settings.odds_api_base_url
        self.api_key = self.settings.odds_api_key
        self.session = requests.Session()

    def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """Make a request to The Odds API."""
        if params is None:
            params = {}
        
        params["apiKey"] = self.api_key
        
        url = f"{self.base_url}{endpoint}"
        response = self.session.get(url, params=params, timeout=30)
        
        if response.status_code == 429:
            # Rate limited - wait and retry
            time.sleep(60)
            response = self.session.get(url, params=params, timeout=30)
        
        response.raise_for_status()
        return response.json()

    def get_available_sports(self) -> List[Dict]:
        """Get list of available sports."""
        return self._make_request("/sports")

    def get_odds(
        self,
        sport: str = "americanfootball_nfl",
        markets: str = "h2h,spreads,totals",
        bookmakers: Optional[str] = None,
    ) -> List[Dict]:
        """Get odds for a specific sport."""
        params = {
            "sport": sport,
            "regions": "us",
            "markets": markets,
            "oddsFormat": "american",
            "dateFormat": "iso",
        }
        
        if bookmakers:
            params["bookmakers"] = bookmakers
        
        return self._make_request(f"/sports/{sport}/odds", params)

    def get_player_props(
        self,
        sport: str = "americanfootball_nfl",
        bookmakers: Optional[str] = None,
    ) -> List[Dict]:
        """Get player prop odds."""
        params = {
            "sport": sport,
            "regions": "us",
            "markets": "player_pass_yds,player_rush_yds,player_receiving_yds,player_receptions,player_pass_tds,player_rush_tds,player_anytime_td",
            "oddsFormat": "american",
            "dateFormat": "iso",
        }
        
        if bookmakers:
            params["bookmakers"] = bookmakers
        
        return self._make_request(f"/sports/{sport}/odds", params)

    def collect_and_store_props(self, season: int, week: int) -> int:
        """Collect and store player props for a specific week."""
        props_data = self.get_player_props()
        stored_count = 0
        
        with get_session() as session:
            for game_data in props_data:
                stored_count += self._store_game_props(session, game_data, season, week)
        
        return stored_count

    def _store_game_props(self, session: Session, game_data: Dict, season: int, week: int) -> int:
        """Store props for a single game."""
        stored_count = 0
        
        # Parse teams from game data
        home_team = game_data.get("home_team")
        away_team = game_data.get("away_team")
        
        if not home_team or not away_team:
            return 0
        
        # Find the game in database
        game = self._find_or_create_game(session, home_team, away_team, season, week, game_data)
        
        if not game:
            return 0
        
        # Process each bookmaker's props
        for bookmaker_data in game_data.get("bookmakers", []):
            book = self._find_or_create_book(session, bookmaker_data["key"], bookmaker_data["title"])
            
            for market_data in bookmaker_data.get("markets", []):
                stored_count += self._store_market_props(
                    session, game, book, market_data
                )
        
        return stored_count

    def _find_or_create_game(
        self, session: Session, home_team: str, away_team: str, season: int, week: int, game_data: Dict
    ) -> Optional[Game]:
        """Find or create a game record."""
        # Convert team names to abbreviations (this is a simplified mapping)
        team_mapping = self._get_team_name_mapping()
        
        home_abbr = team_mapping.get(home_team)
        away_abbr = team_mapping.get(away_team)
        
        if not home_abbr or not away_abbr:
            return None
        
        # Find teams in database
        from ...database.models import Team
        home_team_obj = session.query(Team).filter_by(abbreviation=home_abbr).first()
        away_team_obj = session.query(Team).filter_by(abbreviation=away_abbr).first()
        
        if not home_team_obj or not away_team_obj:
            return None
        
        # Find or create game
        game = (
            session.query(Game)
            .filter_by(
                season=season,
                week=week,
                home_team_id=home_team_obj.id,
                away_team_id=away_team_obj.id,
            )
            .first()
        )
        
        if not game:
            game_date = datetime.fromisoformat(game_data.get("commence_time", "").replace("Z", "+00:00"))
            game = Game(
                season=season,
                week=week,
                game_date=game_date,
                home_team_id=home_team_obj.id,
                away_team_id=away_team_obj.id,
                external_id=game_data.get("id"),
            )
            session.add(game)
            session.flush()
        
        return game

    def _find_or_create_book(self, session: Session, book_key: str, book_title: str) -> Book:
        """Find or create a book record."""
        book = session.query(Book).filter_by(name=book_key).first()
        
        if not book:
            book = Book(name=book_key, display_name=book_title)
            session.add(book)
            session.flush()
        
        return book

    def _store_market_props(self, session: Session, game: Game, book: Book, market_data: Dict) -> int:
        """Store props for a specific market."""
        market_key = market_data.get("key", "")
        stored_count = 0
        
        # Map market keys to our internal format
        market_mapping = {
            "player_pass_yds": "passing_yards",
            "player_rush_yds": "rushing_yards", 
            "player_receiving_yds": "receiving_yards",
            "player_receptions": "receptions",
            "player_pass_tds": "passing_tds",
            "player_rush_tds": "rushing_tds",
            "player_anytime_td": "anytime_td",
        }
        
        internal_market = market_mapping.get(market_key)
        if not internal_market:
            return 0
        
        for outcome in market_data.get("outcomes", []):
            player_name = outcome.get("description", "")
            line = outcome.get("point")
            over_odds = None
            under_odds = None
            
            # Parse odds based on outcome name
            if outcome.get("name") == "Over":
                over_odds = outcome.get("price")
            elif outcome.get("name") == "Under":
                under_odds = outcome.get("price")
            
            if line is None or over_odds is None:
                continue
            
            # Find the corresponding Under outcome
            for other_outcome in market_data.get("outcomes", []):
                if (other_outcome.get("description") == player_name and 
                    other_outcome.get("point") == line and
                    other_outcome.get("name") == "Under"):
                    under_odds = other_outcome.get("price")
                    break
            
            if under_odds is None:
                continue
            
            # Find or create player
            player = self._find_or_create_player(session, player_name, game)
            if not player:
                continue
            
            # Convert American odds to decimal and probabilities
            over_price = self._american_to_decimal(over_odds)
            under_price = self._american_to_decimal(under_odds)
            
            # De-vig the probabilities
            over_prob_raw = 1 / over_price
            under_prob_raw = 1 / under_price
            total_prob = over_prob_raw + under_prob_raw
            
            over_prob = over_prob_raw / total_prob
            under_prob = under_prob_raw / total_prob
            
            # Create prop record
            prop = Prop(
                game_id=game.id,
                player_id=player.id,
                book_id=book.id,
                market=internal_market,
                line=line,
                over_odds=over_odds,
                under_odds=under_odds,
                over_price=over_price,
                under_price=under_price,
                over_probability=over_prob,
                under_probability=under_prob,
            )
            
            session.add(prop)
            stored_count += 1
        
        return stored_count

    def _find_or_create_player(self, session: Session, player_name: str, game: Game) -> Optional[Player]:
        """Find or create a player record."""
        # This is a simplified implementation
        # In practice, you'd want more sophisticated player matching
        player = session.query(Player).filter_by(name=player_name).first()
        
        if not player:
            # Try to determine team and position from game context
            # This would need more sophisticated logic in practice
            player = Player(
                name=player_name,
                position="UNKNOWN",
                team_id=game.home_team_id,  # Default assignment
            )
            session.add(player)
            session.flush()
        
        return player

    def _american_to_decimal(self, american_odds: int) -> float:
        """Convert American odds to decimal odds."""
        if american_odds > 0:
            return (american_odds / 100) + 1
        else:
            return (100 / abs(american_odds)) + 1

    def _get_team_name_mapping(self) -> Dict[str, str]:
        """Get mapping from API team names to our abbreviations."""
        # This would be more comprehensive in practice
        return {
            "Arizona Cardinals": "ARI",
            "Atlanta Falcons": "ATL",
            "Baltimore Ravens": "BAL",
            "Buffalo Bills": "BUF",
            "Carolina Panthers": "CAR",
            "Chicago Bears": "CHI",
            "Cincinnati Bengals": "CIN",
            "Cleveland Browns": "CLE",
            "Dallas Cowboys": "DAL",
            "Denver Broncos": "DEN",
            "Detroit Lions": "DET",
            "Green Bay Packers": "GB",
            "Houston Texans": "HOU",
            "Indianapolis Colts": "IND",
            "Jacksonville Jaguars": "JAX",
            "Kansas City Chiefs": "KC",
            "Las Vegas Raiders": "LV",
            "Los Angeles Chargers": "LAC",
            "Los Angeles Rams": "LAR",
            "Miami Dolphins": "MIA",
            "Minnesota Vikings": "MIN",
            "New England Patriots": "NE",
            "New Orleans Saints": "NO",
            "New York Giants": "NYG",
            "New York Jets": "NYJ",
            "Philadelphia Eagles": "PHI",
            "Pittsburgh Steelers": "PIT",
            "San Francisco 49ers": "SF",
            "Seattle Seahawks": "SEA",
            "Tampa Bay Buccaneers": "TB",
            "Tennessee Titans": "TEN",
            "Washington Commanders": "WAS",
        }