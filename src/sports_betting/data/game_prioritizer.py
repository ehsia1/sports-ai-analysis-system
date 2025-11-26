"""Game prioritization system for intelligent request allocation."""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

from ..database import Game, GamePriority, Team, get_session


class GamePrioritizer:
    """Calculate and manage game priorities for betting analysis."""

    def __init__(self):
        self.primetime_slots = {
            3: ["20:15"],  # Thursday Night Football
            6: ["20:20"],  # Sunday Night Football  
            0: ["20:15"],  # Monday Night Football
        }
        
        # Divisional rivalries (higher betting interest)
        self.rivalry_games = {
            "AFC_EAST": ["BUF", "MIA", "NE", "NYJ"],
            "AFC_NORTH": ["BAL", "CIN", "CLE", "PIT"],
            "AFC_SOUTH": ["HOU", "IND", "JAX", "TEN"],
            "AFC_WEST": ["DEN", "KC", "LV", "LAC"],
            "NFC_EAST": ["DAL", "NYG", "PHI", "WAS"],
            "NFC_NORTH": ["CHI", "DET", "GB", "MIN"],
            "NFC_SOUTH": ["ATL", "CAR", "NO", "TB"],
            "NFC_WEST": ["ARI", "LAR", "SF", "SEA"],
        }

    def calculate_game_priority(self, game: Game) -> float:
        """Calculate priority score for a game (0-10 scale)."""
        priority_score = 0.0
        
        with get_session() as session:
            home_team = session.query(Team).get(game.home_team_id)
            away_team = session.query(Team).get(game.away_team_id)
            
            if not home_team or not away_team:
                return 0.0
        
        # Base score
        priority_score = 1.0
        
        # Primetime games (higher betting volume)
        if self._is_primetime_game(game):
            priority_score += 3.0
        
        # Divisional games (more predictable, higher interest)
        if self._is_divisional_game(home_team, away_team):
            priority_score += 2.0
        
        # Week importance (playoff implications)
        week_multiplier = self._get_week_importance(game.week)
        priority_score += week_multiplier
        
        # Game competitiveness (close spreads = more betting)
        # This would require actual spread data in a real implementation
        priority_score += 0.5  # Placeholder
        
        # Team popularity/market size
        popularity_bonus = self._get_team_popularity_bonus(home_team, away_team)
        priority_score += popularity_bonus
        
        return min(10.0, priority_score)

    def _is_primetime_game(self, game: Game) -> bool:
        """Check if game is in primetime slot."""
        game_weekday = game.game_date.weekday()
        game_time = game.game_date.strftime("%H:%M")
        
        primetime_times = self.primetime_slots.get(game_weekday, [])
        
        # Allow some flexibility in time matching (±30 minutes)
        for pt_time in primetime_times:
            pt_hour, pt_min = map(int, pt_time.split(":"))
            game_hour, game_min = map(int, game_time.split(":"))
            
            pt_minutes = pt_hour * 60 + pt_min
            game_minutes = game_hour * 60 + game_min
            
            if abs(pt_minutes - game_minutes) <= 30:
                return True
        
        return False

    def _is_divisional_game(self, home_team: Team, away_team: Team) -> bool:
        """Check if teams are in same division."""
        return home_team.division == away_team.division

    def _get_week_importance(self, week: int) -> float:
        """Get importance multiplier based on week of season."""
        if week <= 2:
            return 0.5  # Early season, less predictable
        elif week <= 8:
            return 1.0  # Regular importance
        elif week <= 14:
            return 1.5  # Playoff race heating up
        elif week <= 18:
            return 2.5  # Critical playoff games
        else:
            return 3.0  # Playoffs

    def _get_team_popularity_bonus(self, home_team: Team, away_team: Team) -> float:
        """Bonus for popular teams that generate more betting interest."""
        popular_teams = {
            "DAL": 1.0,  # Cowboys
            "GB": 0.8,   # Packers
            "NE": 0.8,   # Patriots
            "KC": 0.8,   # Chiefs
            "SF": 0.6,   # 49ers
            "PIT": 0.6,  # Steelers
            "NYG": 0.6,  # Giants
            "PHI": 0.5,  # Eagles
            "BUF": 0.5,  # Bills
            "LAR": 0.5,  # Rams
        }
        
        home_bonus = popular_teams.get(home_team.abbreviation, 0.0)
        away_bonus = popular_teams.get(away_team.abbreviation, 0.0)
        
        return max(home_bonus, away_bonus)

    def update_game_priorities(self, season: int, week: Optional[int] = None) -> int:
        """Update priorities for all games in a week/season."""
        with get_session() as session:
            # Get games to prioritize
            query = session.query(Game).filter_by(season=season)
            if week:
                query = query.filter_by(week=week)
            
            games = query.all()
            updated_count = 0
            
            for game in games:
                priority_score = self.calculate_game_priority(game)
                
                # Get or create priority record
                game_priority = (
                    session.query(GamePriority)
                    .filter_by(game_id=game.id)
                    .first()
                )
                
                if game_priority:
                    game_priority.priority_score = priority_score
                    game_priority.is_primetime = self._is_primetime_game(game)
                    game_priority.is_divisional = self._is_divisional_game_by_id(
                        session, game.home_team_id, game.away_team_id
                    )
                    game_priority.last_priority_update = datetime.utcnow()
                else:
                    game_priority = GamePriority(
                        game_id=game.id,
                        priority_score=priority_score,
                        is_primetime=self._is_primetime_game(game),
                        is_divisional=self._is_divisional_game_by_id(
                            session, game.home_team_id, game.away_team_id
                        ),
                    )
                    session.add(game_priority)
                
                updated_count += 1
            
            session.commit()
            return updated_count

    def _is_divisional_game_by_id(self, session: Session, home_team_id: int, away_team_id: int) -> bool:
        """Check if game is divisional by team IDs."""
        home_team = session.query(Team).get(home_team_id)
        away_team = session.query(Team).get(away_team_id)
        
        if home_team and away_team:
            return self._is_divisional_game(home_team, away_team)
        return False

    def get_prioritized_games(
        self,
        season: int,
        week: Optional[int] = None,
        min_priority: float = 0.0,
        limit: Optional[int] = None,
    ) -> List[Tuple[Game, GamePriority]]:
        """Get games sorted by priority."""
        with get_session() as session:
            query = (
                session.query(Game, GamePriority)
                .join(GamePriority, Game.id == GamePriority.game_id)
                .filter(Game.season == season)
            )
            
            if week:
                query = query.filter(Game.week == week)
            
            if min_priority > 0:
                query = query.filter(GamePriority.priority_score >= min_priority)
            
            query = query.order_by(GamePriority.priority_score.desc())
            
            if limit:
                query = query.limit(limit)
            
            return query.all()

    def get_priority_distribution(self, season: int, week: Optional[int] = None) -> Dict:
        """Get distribution of game priorities."""
        with get_session() as session:
            query = (
                session.query(GamePriority)
                .join(Game, GamePriority.game_id == Game.id)
                .filter(Game.season == season)
            )
            
            if week:
                query = query.filter(Game.week == week)
            
            priorities = [gp.priority_score for gp in query.all()]
            
            if not priorities:
                return {}
            
            return {
                "total_games": len(priorities),
                "avg_priority": sum(priorities) / len(priorities),
                "max_priority": max(priorities),
                "min_priority": min(priorities),
                "high_priority": len([p for p in priorities if p >= 7.0]),
                "medium_priority": len([p for p in priorities if 4.0 <= p < 7.0]),
                "low_priority": len([p for p in priorities if p < 4.0]),
            }

    def suggest_request_allocation(self, total_budget: int, season: int, week: int) -> Dict:
        """Suggest how to allocate API requests across games."""
        distribution = self.get_priority_distribution(season, week)
        
        if not distribution or distribution["total_games"] == 0:
            return {"error": "No games found"}
        
        # Allocate budget by priority
        high_priority_games = distribution["high_priority"]
        medium_priority_games = distribution["medium_priority"] 
        low_priority_games = distribution["low_priority"]
        
        # Allocate more budget to higher priority games
        high_budget = int(total_budget * 0.6)  # 60% for high priority
        medium_budget = int(total_budget * 0.3)  # 30% for medium
        low_budget = total_budget - high_budget - medium_budget  # Rest for low
        
        return {
            "total_budget": total_budget,
            "allocation": {
                "high_priority": {
                    "games": high_priority_games,
                    "budget": high_budget,
                    "per_game": high_budget // max(1, high_priority_games),
                },
                "medium_priority": {
                    "games": medium_priority_games,
                    "budget": medium_budget,
                    "per_game": medium_budget // max(1, medium_priority_games),
                },
                "low_priority": {
                    "games": low_priority_games,
                    "budget": low_budget,
                    "per_game": low_budget // max(1, low_priority_games),
                },
            },
        }