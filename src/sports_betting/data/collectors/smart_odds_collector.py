"""Smart Odds API collector with request management and caching."""

import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import requests
from sqlalchemy.orm import Session

from ...config import get_settings
from ...database import Book, Game, Player, Prop, get_session
from ..game_prioritizer import GamePrioritizer
from ..request_manager import CacheManager, RequestManager


class SmartOddsCollector:
    """Enhanced Odds API collector with intelligent request management."""

    def __init__(self):
        self.settings = get_settings()
        self.base_url = self.settings.odds_api_base_url
        self.api_key = self.settings.odds_api_key
        self.session = requests.Session()
        
        # Smart management components
        self.request_manager = RequestManager()
        self.cache_manager = CacheManager()
        self.game_prioritizer = GamePrioritizer()

    def _make_smart_request(
        self, 
        endpoint: str, 
        params: Dict = None,
        request_type: str = "general",
        force_refresh: bool = False
    ) -> Optional[Dict]:
        """Make a request with smart caching and rate limiting."""
        if params is None:
            params = {}
        
        params["apiKey"] = self.api_key
        
        # Generate cache key
        cache_params = {k: v for k, v in params.items() if k != "apiKey"}
        
        # Check cache first (unless forced refresh)
        if not force_refresh:
            is_cached, cache_msg = self.cache_manager.is_cached_and_fresh(
                request_type, "odds_api", endpoint=endpoint, **cache_params
            )
            
            if is_cached:
                print(f"🗂️  Using cached data: {cache_msg}")
                return {"cached": True, "cache_message": cache_msg}
        
        # Check if we can make the request
        can_request, limit_msg = self.request_manager.can_make_request("odds_api", request_type)
        
        if not can_request:
            print(f"⚠️  Request blocked: {limit_msg}")
            return None
        
        # Make the request
        url = f"{self.base_url}{endpoint}"
        start_time = time.time()
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            response_time_ms = int((time.time() - start_time) * 1000)
            
            if response.status_code == 429:
                # Rate limited - wait and retry once
                print("⏳ Rate limited, waiting 60 seconds...")
                time.sleep(60)
                response = self.session.get(url, params=params, timeout=30)
                response_time_ms += 60000
            
            response.raise_for_status()
            data = response.json()
            
            # Log successful request
            self.request_manager.log_request(
                api_source="odds_api",
                endpoint=endpoint,
                request_type=request_type,
                objects_returned=len(data) if isinstance(data, list) else 1,
                success=True,
                response_time_ms=response_time_ms,
            )
            
            # Cache the response
            ttl = 24 * 60 * 60 if request_type == "props" else 12 * 60 * 60  # 24h for props, 12h for odds
            self.cache_manager.cache_data(
                request_type, "odds_api", data, ttl_seconds=ttl,
                endpoint=endpoint, **cache_params
            )
            
            print(f"✅ API request successful: {len(data) if isinstance(data, list) else 1} objects")
            return data
            
        except Exception as e:
            response_time_ms = int((time.time() - start_time) * 1000)
            
            # Log failed request
            self.request_manager.log_request(
                api_source="odds_api",
                endpoint=endpoint,
                request_type=request_type,
                success=False,
                error_message=str(e),
                response_time_ms=response_time_ms,
            )
            
            print(f"❌ API request failed: {e}")
            return None

    def get_prioritized_props(
        self,
        season: int,
        week: int,
        priority_threshold: float = 5.0,
        bookmakers: Optional[str] = None,
    ) -> Dict:
        """Get player props for prioritized games only."""
        print(f"🎯 Getting props for priority games (threshold: {priority_threshold})")
        
        # Update game priorities first
        self.game_prioritizer.update_game_priorities(season, week)
        
        # Get prioritized games
        prioritized_games = self.game_prioritizer.get_prioritized_games(
            season, week, min_priority=priority_threshold, limit=10
        )
        
        if not prioritized_games:
            print("⚠️  No high-priority games found")
            return {"games": [], "total_props": 0}
        
        print(f"🏈 Found {len(prioritized_games)} priority games")
        
        # Get request budget allocation
        budget_info = self.request_manager.get_priority_budget()
        available_budget = budget_info["high_priority"]
        
        print(f"💰 Available budget: {available_budget} requests")
        
        all_props = []
        requests_used = 0
        
        for game, priority in prioritized_games:
            if requests_used >= available_budget:
                print(f"💸 Budget exhausted ({requests_used}/{available_budget})")
                break
            
            print(f"\n🏈 Processing: Week {game.week} - Priority {priority.priority_score:.1f}")
            
            # Get props for this specific game (this would need game-specific endpoint)
            props_data = self._get_game_props(game, bookmakers)
            
            if props_data and not props_data.get("cached"):
                requests_used += 1
                all_props.extend(props_data.get("props", []))
            elif props_data and props_data.get("cached"):
                print("📋 Using cached props data")
                # In a real implementation, we'd load cached props here
        
        return {
            "games": len(prioritized_games),
            "total_props": len(all_props),
            "requests_used": requests_used,
            "budget_remaining": available_budget - requests_used,
            "props": all_props,
        }

    def _get_game_props(self, game: Game, bookmakers: Optional[str] = None) -> Optional[Dict]:
        """Get props for a specific game."""
        # In a real implementation, you'd need game-specific endpoints
        # For now, we'll simulate by getting all props and filtering
        
        params = {
            "sport": "americanfootball_nfl",
            "regions": "us",
            "markets": "player_pass_yds,player_rush_yds,player_receiving_yds,player_receptions,player_anytime_td",
            "oddsFormat": "american",
            "dateFormat": "iso",
        }
        
        if bookmakers:
            params["bookmakers"] = bookmakers
        
        # Make request
        data = self._make_smart_request(
            f"/sports/americanfootball_nfl/odds",
            params=params,
            request_type="props"
        )
        
        if not data or data.get("cached"):
            return data
        
        # Filter for this specific game (simplified)
        game_props = []
        for game_data in data:
            # Match by team names (simplified matching)
            if self._matches_game(game_data, game):
                game_props.append(game_data)
        
        return {"props": game_props}

    def _matches_game(self, api_game_data: Dict, db_game: Game) -> bool:
        """Check if API game data matches database game."""
        # Simplified matching - in practice you'd need better team name mapping
        api_home = api_game_data.get("home_team", "").upper()
        api_away = api_game_data.get("away_team", "").upper()
        
        # This would need proper team name mapping
        return True  # Placeholder

    def weekly_update_strategy(self, season: int, week: int) -> Dict:
        """Execute the weekly update strategy."""
        print(f"🗓️  Starting weekly update for Season {season}, Week {week}")
        
        # Check available budget
        budget = self.request_manager.get_priority_budget()
        print(f"📊 Budget analysis: {budget}")
        
        results = {
            "season": season,
            "week": week,
            "budget_before": budget,
            "operations": [],
        }
        
        # Step 1: Update game priorities (free operation)
        print("\n1️⃣ Updating game priorities...")
        priority_count = self.game_prioritizer.update_game_priorities(season, week)
        results["operations"].append({
            "step": "priority_update",
            "games_updated": priority_count,
            "requests_used": 0,
        })
        
        # Step 2: Get distribution and allocation plan
        distribution = self.game_prioritizer.get_priority_distribution(season, week)
        allocation = self.game_prioritizer.suggest_request_allocation(
            budget["daily_budget"], season, week
        )
        
        print(f"\n📊 Priority distribution: {distribution}")
        print(f"💰 Suggested allocation: {allocation}")
        
        # Step 3: Collect high-priority props
        print("\n2️⃣ Collecting high-priority props...")
        high_priority_result = self.get_prioritized_props(
            season, week, priority_threshold=7.0
        )
        results["operations"].append({
            "step": "high_priority_props",
            **high_priority_result,
        })
        
        # Step 4: Collect medium-priority props if budget allows
        remaining_budget = budget["daily_budget"] - high_priority_result["requests_used"]
        
        if remaining_budget > 0:
            print(f"\n3️⃣ Collecting medium-priority props (budget: {remaining_budget})...")
            medium_priority_result = self.get_prioritized_props(
                season, week, priority_threshold=4.0
            )
            results["operations"].append({
                "step": "medium_priority_props", 
                **medium_priority_result,
            })
        
        # Final budget check
        final_budget = self.request_manager.get_priority_budget()
        results["budget_after"] = final_budget
        
        return results

    def get_smart_usage_report(self) -> Dict:
        """Generate a comprehensive usage report."""
        usage_stats = self.request_manager.get_usage_stats("odds_api")
        cache_stats = self.cache_manager.get_cache_stats()
        budget = self.request_manager.get_priority_budget()
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "api_usage": usage_stats,
            "cache_performance": cache_stats,
            "budget_analysis": budget,
            "optimization_tips": self._get_optimization_tips(usage_stats, budget),
        }

    def _get_optimization_tips(self, usage_stats: Dict, budget: Dict) -> List[str]:
        """Generate optimization tips based on usage patterns."""
        tips = []
        
        if usage_stats["usage_percentage"] > 80:
            tips.append("⚠️ High API usage - consider increasing cache TTL")
        
        if budget["days_left"] < 5 and budget["total_remaining"] < 50:
            tips.append("🚨 Low budget remaining - focus on highest priority games only")
        
        if len(usage_stats["recent_requests"]) < 5:
            tips.append("💡 Low recent activity - good time for bulk updates")
        
        return tips