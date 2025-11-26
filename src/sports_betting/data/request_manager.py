"""Smart API request management for free tier optimization."""

import hashlib
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

from ..config import get_settings
from ..database import ApiRequest, DataCache, get_session


class RequestManager:
    """Manage API requests to stay within free tier limits."""

    def __init__(self):
        self.settings = get_settings()
        self.monthly_limits = {
            "odds_api": 500,
            "espn": 10000,  # Effectively unlimited
        }
        
    def can_make_request(self, api_source: str, request_type: str) -> Tuple[bool, str]:
        """Check if we can make a request within limits."""
        with get_session() as session:
            # Get current month's usage
            month_start = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            
            current_usage = (
                session.query(ApiRequest)
                .filter(
                    ApiRequest.api_source == api_source,
                    ApiRequest.created_at >= month_start,
                    ApiRequest.success == True
                )
                .count()
            )
            
            limit = self.monthly_limits.get(api_source, 100)
            
            if current_usage >= limit:
                return False, f"Monthly limit exceeded: {current_usage}/{limit}"
            
            # Check daily budget (monthly limit / 30)
            daily_budget = limit // 30
            today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            
            today_usage = (
                session.query(ApiRequest)
                .filter(
                    ApiRequest.api_source == api_source,
                    ApiRequest.created_at >= today_start,
                    ApiRequest.success == True
                )
                .count()
            )
            
            if today_usage >= daily_budget * 2:  # Allow some flexibility
                return False, f"Daily budget exceeded: {today_usage}/{daily_budget * 2}"
            
            return True, f"OK: {current_usage}/{limit} monthly, {today_usage}/{daily_budget * 2} daily"

    def log_request(
        self,
        api_source: str,
        endpoint: str,
        request_type: str,
        objects_returned: int = 0,
        success: bool = True,
        error_message: Optional[str] = None,
        response_time_ms: Optional[int] = None,
    ) -> int:
        """Log an API request for tracking."""
        with get_session() as session:
            request_log = ApiRequest(
                api_source=api_source,
                endpoint=endpoint,
                request_type=request_type,
                objects_returned=objects_returned,
                success=success,
                error_message=error_message,
                response_time_ms=response_time_ms,
            )
            session.add(request_log)
            session.flush()
            return request_log.id

    def get_usage_stats(self, api_source: str) -> Dict:
        """Get current usage statistics."""
        with get_session() as session:
            month_start = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            
            monthly_usage = (
                session.query(ApiRequest)
                .filter(
                    ApiRequest.api_source == api_source,
                    ApiRequest.created_at >= month_start,
                    ApiRequest.success == True
                )
                .count()
            )
            
            monthly_limit = self.monthly_limits.get(api_source, 100)
            
            # Recent successful requests
            recent_requests = (
                session.query(ApiRequest)
                .filter(
                    ApiRequest.api_source == api_source,
                    ApiRequest.created_at >= datetime.now() - timedelta(days=7),
                    ApiRequest.success == True
                )
                .order_by(ApiRequest.created_at.desc())
                .limit(10)
                .all()
            )
            
            return {
                "monthly_usage": monthly_usage,
                "monthly_limit": monthly_limit,
                "remaining": monthly_limit - monthly_usage,
                "usage_percentage": (monthly_usage / monthly_limit) * 100,
                "recent_requests": [
                    {
                        "endpoint": req.endpoint,
                        "type": req.request_type,
                        "objects": req.objects_returned,
                        "time": req.created_at,
                    }
                    for req in recent_requests
                ],
            }

    def get_priority_budget(self) -> Dict[str, int]:
        """Calculate request budget allocation by priority."""
        stats = self.get_usage_stats("odds_api")
        remaining = stats["remaining"]
        
        # Days left in month
        today = datetime.now()
        if today.month == 12:
            next_month = today.replace(year=today.year + 1, month=1, day=1)
        else:
            next_month = today.replace(month=today.month + 1, day=1)
        
        days_left = (next_month - today).days
        daily_budget = max(1, remaining // max(1, days_left))
        
        return {
            "total_remaining": remaining,
            "days_left": days_left,
            "daily_budget": daily_budget,
            "high_priority": max(1, daily_budget // 2),
            "medium_priority": max(1, daily_budget // 3),
            "low_priority": max(1, daily_budget // 6),
        }


class CacheManager:
    """Manage intelligent data caching with TTL."""

    def __init__(self):
        self.default_ttl = {
            "props": 24 * 60 * 60,  # 24 hours
            "odds": 12 * 60 * 60,   # 12 hours
            "scores": 6 * 60 * 60,  # 6 hours
            "schedule": 7 * 24 * 60 * 60,  # 7 days
        }

    def _generate_cache_key(self, data_type: str, **kwargs) -> str:
        """Generate a unique cache key."""
        key_parts = [data_type]
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}={v}")
        
        key_string = "|".join(key_parts)
        return hashlib.sha256(key_string.encode()).hexdigest()[:16]

    def _calculate_data_hash(self, data: any) -> str:
        """Calculate hash of data for change detection."""
        data_str = str(data)
        return hashlib.sha256(data_str.encode()).hexdigest()

    def is_cached_and_fresh(
        self,
        data_type: str,
        data_source: str,
        game_id: Optional[int] = None,
        player_id: Optional[int] = None,
        **kwargs
    ) -> Tuple[bool, Optional[str]]:
        """Check if data is cached and still fresh."""
        cache_key = self._generate_cache_key(data_type, game_id=game_id, player_id=player_id, **kwargs)
        
        with get_session() as session:
            cache_entry = (
                session.query(DataCache)
                .filter_by(cache_key=cache_key)
                .first()
            )
            
            if not cache_entry:
                return False, "Not cached"
            
            if datetime.utcnow() > cache_entry.expires_at:
                cache_entry.is_stale = True
                session.commit()
                return False, f"Expired at {cache_entry.expires_at}"
            
            return True, f"Fresh until {cache_entry.expires_at}"

    def cache_data(
        self,
        data_type: str,
        data_source: str,
        data: any,
        ttl_seconds: Optional[int] = None,
        game_id: Optional[int] = None,
        player_id: Optional[int] = None,
        **kwargs
    ) -> str:
        """Cache data with TTL."""
        cache_key = self._generate_cache_key(data_type, game_id=game_id, player_id=player_id, **kwargs)
        data_hash = self._calculate_data_hash(data)
        
        if ttl_seconds is None:
            ttl_seconds = self.default_ttl.get(data_type, 3600)  # Default 1 hour
        
        expires_at = datetime.utcnow() + timedelta(seconds=ttl_seconds)
        
        with get_session() as session:
            # Check if cache entry exists
            cache_entry = (
                session.query(DataCache)
                .filter_by(cache_key=cache_key)
                .first()
            )
            
            if cache_entry:
                # Update existing entry
                cache_entry.data_hash = data_hash
                cache_entry.expires_at = expires_at
                cache_entry.last_updated = datetime.utcnow()
                cache_entry.request_count += 1
                cache_entry.is_stale = False
            else:
                # Create new entry
                cache_entry = DataCache(
                    cache_key=cache_key,
                    data_type=data_type,
                    data_source=data_source,
                    game_id=game_id,
                    player_id=player_id,
                    data_hash=data_hash,
                    expires_at=expires_at,
                )
                session.add(cache_entry)
            
            session.commit()
            return cache_key

    def invalidate_cache(self, cache_key: str):
        """Manually invalidate a cache entry."""
        with get_session() as session:
            cache_entry = (
                session.query(DataCache)
                .filter_by(cache_key=cache_key)
                .first()
            )
            
            if cache_entry:
                cache_entry.is_stale = True
                cache_entry.expires_at = datetime.utcnow()
                session.commit()

    def cleanup_expired_cache(self) -> int:
        """Remove expired cache entries."""
        with get_session() as session:
            expired_count = (
                session.query(DataCache)
                .filter(DataCache.expires_at < datetime.utcnow())
                .count()
            )
            
            session.query(DataCache).filter(
                DataCache.expires_at < datetime.utcnow()
            ).delete()
            
            session.commit()
            return expired_count

    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        with get_session() as session:
            total_entries = session.query(DataCache).count()
            fresh_entries = session.query(DataCache).filter(
                DataCache.expires_at > datetime.utcnow(),
                DataCache.is_stale == False
            ).count()
            stale_entries = session.query(DataCache).filter(
                DataCache.is_stale == True
            ).count()
            
            return {
                "total_entries": total_entries,
                "fresh_entries": fresh_entries,
                "stale_entries": stale_entries,
                "hit_rate": (fresh_entries / max(1, total_entries)) * 100,
            }