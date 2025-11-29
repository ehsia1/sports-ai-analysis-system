"""The Odds API client with smart caching and credit tracking."""

import os
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
from pathlib import Path
import json

from ..database import get_session
from ..database.models import Prop, Book, Player, Game

logger = logging.getLogger(__name__)


def american_to_decimal(american_odds: int) -> float:
    """Convert American odds to decimal odds."""
    if american_odds > 0:
        return (american_odds / 100.0) + 1.0
    else:
        return (100.0 / abs(american_odds)) + 1.0


def american_to_probability(american_odds: int) -> float:
    """Convert American odds to implied probability (with vig)."""
    if american_odds > 0:
        return 100.0 / (american_odds + 100.0)
    else:
        return abs(american_odds) / (abs(american_odds) + 100.0)


def remove_vig(over_prob: float, under_prob: float) -> tuple[float, float]:
    """
    Remove vig from probabilities to get fair probabilities.

    Uses the margin-proportional method.
    """
    total = over_prob + under_prob
    margin = total - 1.0

    if margin <= 0:
        # No vig or negative vig (shouldn't happen)
        return over_prob, under_prob

    # Remove margin proportionally
    fair_over = over_prob / total
    fair_under = under_prob / total

    return fair_over, fair_under


class OddsAPIClient:
    """Client for The Odds API with daily caching and credit tracking."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Odds API client."""
        self.api_key = api_key or os.getenv("ODDS_API_KEY")
        if not self.api_key:
            logger.warning("No ODDS_API_KEY found. Set in .env or pass to constructor.")

        self.base_url = "https://api.the-odds-api.com/v4"
        self.cache_dir = Path.home() / ".sports_betting" / "odds_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Track API usage
        self.credits_used = 0
        self.credits_remaining = None
        self.last_request_cost = 0

        # Default settings
        self.regions = "us"  # Only US books
        self.markets = "player_pass_tds,player_pass_yds,player_rush_yds,player_reception_yds"
        self.oddsFormat = "american"

    def _make_request(self, endpoint: str, params: Dict) -> Optional[Dict]:
        """Make API request and track credit usage."""
        if not self.api_key:
            logger.error("Cannot make request: No API key configured")
            return None

        url = f"{self.base_url}/{endpoint}"
        params["apiKey"] = self.api_key

        try:
            logger.info(f"Making request to {endpoint}")
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            # Track credit usage from headers
            self.credits_remaining = response.headers.get("x-requests-remaining")
            self.last_request_cost = response.headers.get("x-requests-last", 0)
            self.credits_used += int(self.last_request_cost)

            logger.info(
                f"Request successful. Cost: {self.last_request_cost} credits. "
                f"Remaining: {self.credits_remaining}"
            )

            return response.json()

        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return None

    def get_sports(self) -> List[Dict]:
        """Get list of available sports (FREE - no credits used)."""
        logger.info("Fetching available sports")
        return self._make_request("sports", {}) or []

    def get_nfl_games(self) -> List[Dict]:
        """Get upcoming NFL games (FREE - no credits used)."""
        logger.info("Fetching NFL games")
        return self._make_request("sports/americanfootball_nfl/events", {}) or []

    def get_player_props(
        self,
        markets: Optional[List[str]] = None,
        bookmakers: Optional[List[str]] = None,
        event_ids: Optional[List[str]] = None
    ) -> Optional[Dict]:
        """
        Get NFL player props odds for all events or specific events.

        Cost: number_of_markets × number_of_regions × number_of_events
        Example: 1 market × 1 region × 10 events = 10 credits
        """
        if markets is None:
            markets = ["player_reception_yds"]  # Start with just receiving yards

        # If no specific events provided, get all upcoming NFL games
        if event_ids is None:
            events = self.get_nfl_games()
            if not events:
                logger.error("No upcoming NFL games found")
                return None
            event_ids = [event['id'] for event in events[:10]]  # Limit to first 10 to save credits
            logger.info(f"Found {len(events)} games, fetching odds for first {len(event_ids)}")

        # Aggregate all props from all events
        all_props = []
        total_cost = 0

        for event_id in event_ids:
            params = {
                "regions": self.regions,
                "markets": ",".join(markets),
                "oddsFormat": self.oddsFormat,
            }

            if bookmakers:
                params["bookmakers"] = ",".join(bookmakers)

            logger.info(f"Fetching props for event {event_id}")

            event_odds = self._make_request(
                f"sports/americanfootball_nfl/events/{event_id}/odds",
                params
            )

            if event_odds:
                all_props.append(event_odds)
                total_cost += int(self.last_request_cost or 0)

        logger.info(f"Total props fetched: {len(all_props)} events")
        logger.info(f"Total cost: {total_cost} credits")

        return {
            'data': all_props,
            'events_count': len(all_props),
            'total_cost': total_cost
        }

    def get_cached_odds(self, date: Optional[datetime] = None) -> Optional[Dict]:
        """Get cached odds for a specific date."""
        if date is None:
            date = datetime.now()

        cache_file = self.cache_dir / f"odds_{date.strftime('%Y-%m-%d')}.json"

        if cache_file.exists():
            logger.info(f"Loading cached odds from {cache_file}")
            with open(cache_file, 'r') as f:
                return json.load(f)

        return None

    def cache_odds(self, odds_data: Dict, date: Optional[datetime] = None):
        """Cache odds data to disk."""
        if date is None:
            date = datetime.now()

        cache_file = self.cache_dir / f"odds_{date.strftime('%Y-%m-%d')}.json"

        logger.info(f"Caching odds to {cache_file}")
        with open(cache_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'data': odds_data,
                'credits_used': self.last_request_cost,
            }, f, indent=2)

    def should_fetch_new_odds(self, date: Optional[datetime] = None) -> bool:
        """Check if we need to fetch new odds (once per day)."""
        if date is None:
            date = datetime.now()

        cache_file = self.cache_dir / f"odds_{date.strftime('%Y-%m-%d')}.json"

        if not cache_file.exists():
            logger.info("No cached odds for today - need to fetch")
            return True

        # Check if cache is from today
        with open(cache_file, 'r') as f:
            cached = json.load(f)
            cached_time = datetime.fromisoformat(cached['timestamp'])

            # If cache is from today, don't fetch again
            if cached_time.date() == date.date():
                logger.info(f"Using today's cached odds from {cached_time}")
                return False

        return True

    def fetch_and_cache_daily_odds(
        self,
        markets: Optional[List[str]] = None,
        event_ids: Optional[List[str]] = None,
        force: bool = False
    ) -> Dict:
        """
        Fetch odds once per day and cache them.

        Args:
            markets: List of markets to fetch
            event_ids: Specific event IDs to fetch (limits API usage)
            force: Force fetch even if already cached today

        Returns:
            Dict with odds data
        """
        if not force and not self.should_fetch_new_odds():
            logger.info("Using cached odds from today")
            return self.get_cached_odds()

        logger.info("Fetching fresh odds from API...")
        odds_data = self.get_player_props(markets=markets, event_ids=event_ids)

        if odds_data:
            self.cache_odds(odds_data)
            logger.info(f"Successfully cached odds. Credits used: {odds_data.get('total_cost', 0)}")
            return odds_data
        else:
            logger.error("Failed to fetch odds from API")
            # Fall back to cached data if available
            cached = self.get_cached_odds()
            if cached:
                logger.info("Falling back to cached odds")
                return cached
            return None

    def store_odds_in_database(self, odds_data: Dict) -> int:
        """
        Store fetched odds in the database.

        Returns:
            Number of props stored
        """
        if not odds_data or 'data' not in odds_data:
            logger.error("No odds data to store")
            return 0

        stored_count = 0

        with get_session() as session:
            # Get or create bookmakers
            book_cache = {}

            for event in odds_data['data']:
                if 'bookmakers' not in event:
                    continue

                # Find the game - try by external_id first, then by team names
                game = None

                # Try external_id
                if event.get('id'):
                    game = session.query(Game).filter_by(
                        external_id=event.get('id')
                    ).first()

                # If not found, try matching by team names
                if not game:
                    home_team_name = event.get('home_team')
                    away_team_name = event.get('away_team')

                    if home_team_name and away_team_name:
                        # Find teams by name
                        from ..database.models import Team

                        home_team = session.query(Team).filter_by(name=home_team_name).first()
                        away_team = session.query(Team).filter_by(name=away_team_name).first()

                        if home_team and away_team:
                            # Query game by team IDs
                            game = session.query(Game).filter_by(
                                home_team_id=home_team.id,
                                away_team_id=away_team.id
                            ).first()

                        # If still not found, log warning
                        if not game:
                            logger.warning(
                                f"Game not found for {away_team_name} @ {home_team_name}, "
                                f"event ID: {event.get('id')}"
                            )
                            continue

                # Process each bookmaker
                for bookmaker_data in event['bookmakers']:
                    book_key = bookmaker_data['key']

                    # Get or create book
                    if book_key not in book_cache:
                        book = session.query(Book).filter_by(name=book_key).first()
                        if not book:
                            book = Book(
                                name=book_key,
                                display_name=bookmaker_data['title'],
                                region='us'
                            )
                            session.add(book)
                            session.flush()
                        book_cache[book_key] = book
                    else:
                        book = book_cache[book_key]

                    # Process markets
                    for market in bookmaker_data.get('markets', []):
                        market_key = market['key']
                        outcomes = market.get('outcomes', [])

                        # Group outcomes by player and line (to pair over/under)
                        player_lines = {}

                        for outcome in outcomes:
                            player_name = outcome.get('description')
                            line = outcome.get('point')
                            side = outcome.get('name', '').lower()  # 'over' or 'under'
                            price = outcome.get('price')

                            if not all([player_name, line is not None, side, price is not None]):
                                continue

                            # Create key to group over/under
                            key = (player_name, line)

                            if key not in player_lines:
                                player_lines[key] = {'over': None, 'under': None}

                            player_lines[key][side] = price

                        # Now store each paired over/under as a single Prop
                        for (player_name, line), odds in player_lines.items():
                            # Skip if we don't have both sides
                            if odds['over'] is None or odds['under'] is None:
                                logger.debug(
                                    f"Incomplete odds for {player_name} {line}: "
                                    f"over={odds['over']}, under={odds['under']}"
                                )
                                continue

                            # Find player
                            player = session.query(Player).filter_by(
                                name=player_name
                            ).first()

                            if not player:
                                logger.debug(f"Player not found in database: {player_name}")
                                continue

                            # Calculate decimal prices and probabilities
                            over_price = american_to_decimal(odds['over'])
                            under_price = american_to_decimal(odds['under'])

                            # Calculate implied probabilities (with vig)
                            over_prob_raw = american_to_probability(odds['over'])
                            under_prob_raw = american_to_probability(odds['under'])

                            # Remove vig to get fair probabilities
                            over_probability, under_probability = remove_vig(
                                over_prob_raw, under_prob_raw
                            )

                            # Check if prop already exists for today
                            today_start = datetime.now().replace(hour=0, minute=0, second=0)
                            existing = session.query(Prop).filter_by(
                                game_id=game.id,
                                player_id=player.id,
                                book_id=book.id,
                                market=market_key,
                                line=line
                            ).filter(
                                Prop.timestamp >= today_start
                            ).first()

                            if existing:
                                # Update existing
                                existing.over_odds = odds['over']
                                existing.under_odds = odds['under']
                                existing.over_price = over_price
                                existing.under_price = under_price
                                existing.over_probability = over_probability
                                existing.under_probability = under_probability
                                existing.timestamp = datetime.now()
                                logger.debug(f"Updated prop for {player_name}")
                            else:
                                # Create new prop
                                prop = Prop(
                                    game_id=game.id,
                                    player_id=player.id,
                                    book_id=book.id,
                                    market=market_key,
                                    line=line,
                                    over_odds=odds['over'],
                                    under_odds=odds['under'],
                                    over_price=over_price,
                                    under_price=under_price,
                                    over_probability=over_probability,
                                    under_probability=under_probability,
                                    timestamp=datetime.now()
                                )
                                session.add(prop)
                                logger.debug(f"Created prop for {player_name}")

                            stored_count += 1

            session.commit()
            logger.info(f"Stored {stored_count} props in database")

        return stored_count

    def get_usage_stats(self) -> Dict:
        """Get API usage statistics."""
        return {
            'credits_used': self.credits_used,
            'credits_remaining': self.credits_remaining,
            'last_request_cost': self.last_request_cost,
        }


def parse_market_name(api_market: str) -> str:
    """Convert API market names to our internal format."""
    mapping = {
        'player_pass_tds': 'passing_tds',
        'player_pass_yds': 'passing_yards',
        'player_rush_yds': 'rushing_yards',
        'player_reception_yds': 'receiving_yards',
        'player_receptions': 'receptions',
    }
    return mapping.get(api_market, api_market)
