"""Comprehensive data feeding and management system."""

from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from sqlalchemy.orm import Session
from pathlib import Path
import json

from ..config import get_settings
from ..database import Game, Player, Team, Prop, get_session
from .collectors.smart_odds_collector import SmartOddsCollector
from .collectors.espn_api import ESPNAPICollector
from .request_manager import RequestManager

logger = logging.getLogger(__name__)


class DataManager:
    """Central manager for all data feeding operations."""
    
    def __init__(self, session: Optional[Session] = None):
        self.session = session or get_session()
        self.settings = get_settings()
        
        # Initialize collectors
        self.odds_collector = SmartOddsCollector()
        self.espn_collector = ESPNAPICollector()
        self.request_manager = RequestManager()
        
        # Data sources configuration
        self.data_sources = {
            'live_api': {
                'enabled': True,
                'collector': self.odds_collector,
                'cost_per_request': 1,
                'description': 'Live market odds from The Odds API'
            },
            'espn_free': {
                'enabled': True,
                'collector': self.espn_collector,
                'cost_per_request': 0,
                'description': 'Free NFL data from ESPN API'
            },
            'historical_files': {
                'enabled': True,
                'collector': None,
                'cost_per_request': 0,
                'description': 'Historical data from local files'
            },
            'nfl_data_py': {
                'enabled': False,  # Enable when library is available
                'collector': None,
                'cost_per_request': 0,
                'description': 'NFL statistics from nfl_data_py library'
            }
        }
        
    def feed_week_data(
        self,
        week: int,
        season: int,
        data_sources: Optional[List[str]] = None,
        force_refresh: bool = False
    ) -> Dict[str, Any]:
        """Feed comprehensive data for a specific week."""
        
        logger.info(f"Feeding data for NFL Week {week}, Season {season}")
        
        if data_sources is None:
            data_sources = ['live_api', 'espn_free', 'historical_files']
        
        results = {
            'week': week,
            'season': season,
            'data_collected': {},
            'total_api_cost': 0,
            'status': 'success',
            'errors': []
        }
        
        # Check API budget before starting
        budget_status = self.request_manager.get_priority_budget()
        if budget_status['daily_budget'] < 10 and 'live_api' in data_sources:
            logger.warning(f"Low API budget: {budget_status['daily_budget']} requests remaining")
        
        # Feed from each source
        for source in data_sources:
            if source not in self.data_sources:
                results['errors'].append(f"Unknown data source: {source}")
                continue
                
            if not self.data_sources[source]['enabled']:
                logger.info(f"Data source '{source}' is disabled")
                continue
            
            try:
                logger.info(f"Feeding data from {source}...")
                source_result = self._feed_from_source(source, week, season, force_refresh)
                
                results['data_collected'][source] = source_result
                results['total_api_cost'] += source_result.get('api_cost', 0)
                
            except Exception as e:
                error_msg = f"Error feeding from {source}: {e}"
                logger.error(error_msg)
                results['errors'].append(error_msg)
                results['status'] = 'partial' if results['data_collected'] else 'failed'
        
        # Summary
        total_records = sum(
            result.get('records_collected', 0) 
            for result in results['data_collected'].values()
        )
        
        logger.info(f"Data feeding complete: {total_records} records, {results['total_api_cost']} API cost")
        
        return results
    
    def _feed_from_source(
        self,
        source: str,
        week: int,
        season: int,
        force_refresh: bool = False
    ) -> Dict[str, Any]:
        """Feed data from a specific source."""
        
        source_config = self.data_sources[source]
        
        if source == 'live_api':
            return self._feed_live_api_data(week, season, force_refresh)
        elif source == 'espn_free':
            return self._feed_espn_data(week, season, force_refresh)
        elif source == 'historical_files':
            return self._feed_historical_files(week, season)
        elif source == 'nfl_data_py':
            return self._feed_nfl_data_py(week, season)
        else:
            raise ValueError(f"Unknown data source: {source}")
    
    def _feed_live_api_data(
        self,
        week: int,
        season: int,
        force_refresh: bool = False
    ) -> Dict[str, Any]:
        """Feed live market data from Odds API."""
        
        result = {
            'source': 'live_api',
            'records_collected': 0,
            'api_cost': 0,
            'data_types': []
        }
        
        # Check if we can make requests
        can_request, msg = self.request_manager.can_make_request("odds_api", "props")
        if not can_request and not force_refresh:
            logger.warning(f"Cannot make API requests: {msg}")
            result['warning'] = msg
            return result
        
        try:
            # Get NFL games for the week
            props_data = self.odds_collector.get_prioritized_props(
                season, week, priority_threshold=6.0
            )
            
            if props_data and 'operations' in props_data:
                for operation in props_data['operations']:
                    result['records_collected'] += operation.get('props_collected', 0)
                    result['api_cost'] += operation.get('requests_used', 0)
                
                result['data_types'].append('player_props')
            
            # Also collect basic odds (lower priority)
            if result['api_cost'] < 50:  # Save budget
                odds_data = self.odds_collector.get_games_odds('americanfootball_nfl')
                if odds_data:
                    result['records_collected'] += len(odds_data.get('games', []))
                    result['api_cost'] += 1
                    result['data_types'].append('game_odds')
            
        except Exception as e:
            logger.error(f"Error collecting live API data: {e}")
            result['error'] = str(e)
        
        return result
    
    def _feed_espn_data(
        self,
        week: int,
        season: int,
        force_refresh: bool = False
    ) -> Dict[str, Any]:
        """Feed free data from ESPN API."""
        
        result = {
            'source': 'espn_free',
            'records_collected': 0,
            'api_cost': 0,
            'data_types': []
        }
        
        try:
            # Collect weekly data from ESPN
            espn_data = self.espn_collector.collect_weekly_data(season, week)
            
            if espn_data and 'data_collected' in espn_data:
                data_collected = espn_data['data_collected']
                
                # Count records from each data type
                if 'scoreboard' in data_collected:
                    scoreboard = data_collected['scoreboard']
                    if isinstance(scoreboard, dict):
                        result['records_collected'] += scoreboard.get('games_count', 0)
                    result['data_types'].append('scoreboard')
                
                if 'standings' in data_collected:
                    standings = data_collected['standings']
                    if isinstance(standings, dict):
                        result['records_collected'] += standings.get('teams_count', 0)
                    result['data_types'].append('standings')
                
                if 'injuries' in data_collected:
                    result['data_types'].append('injuries')
                    result['records_collected'] += 10  # Estimate
        
        except Exception as e:
            logger.error(f"Error collecting ESPN data: {e}")
            result['error'] = str(e)
        
        return result
    
    def _feed_historical_files(
        self,
        week: int,
        season: int
    ) -> Dict[str, Any]:
        """Feed data from historical files."""
        
        result = {
            'source': 'historical_files',
            'records_collected': 0,
            'api_cost': 0,
            'data_types': []
        }
        
        # Look for historical data files
        data_dir = Path(self.settings.feature_cache_dir) / "historical"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Sample historical data patterns to look for
        file_patterns = [
            f"nfl_{season}_week_{week}_props.csv",
            f"nfl_{season}_week_{week}_odds.json",
            f"player_stats_{season}.csv",
            f"team_stats_{season}_week_{week}.json"
        ]
        
        for pattern in file_patterns:
            file_path = data_dir / pattern
            if file_path.exists():
                try:
                    if pattern.endswith('.csv'):
                        df = pd.read_csv(file_path)
                        result['records_collected'] += len(df)
                    elif pattern.endswith('.json'):
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                        result['records_collected'] += len(data) if isinstance(data, list) else 1
                    
                    result['data_types'].append(pattern.split('.')[0])
                    
                except Exception as e:
                    logger.warning(f"Error reading {file_path}: {e}")
        
        # If no historical files exist, create sample data
        if result['records_collected'] == 0:
            sample_data = self._create_sample_historical_data(week, season)
            result.update(sample_data)
        
        return result
    
    def _feed_nfl_data_py(
        self,
        week: int,
        season: int
    ) -> Dict[str, Any]:
        """Feed data using nfl_data_py library."""
        
        result = {
            'source': 'nfl_data_py',
            'records_collected': 0,
            'api_cost': 0,
            'data_types': []
        }
        
        try:
            import nfl_data_py as nfl
            
            # Get play-by-play data
            logger.info("Loading NFL play-by-play data...")
            pbp = nfl.import_pbp_data([season])
            week_pbp = pbp[pbp.week == week]
            
            if not week_pbp.empty:
                result['records_collected'] += len(week_pbp)
                result['data_types'].append('play_by_play')
            
            # Get weekly stats
            logger.info("Loading NFL weekly stats...")
            weekly = nfl.import_weekly_data([season])
            week_stats = weekly[weekly.week == week]
            
            if not week_stats.empty:
                result['records_collected'] += len(week_stats)
                result['data_types'].append('weekly_stats')
            
            # Store data to cache for later use
            cache_dir = Path(self.settings.feature_cache_dir) / "nfl_data_py"
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            week_pbp.to_csv(cache_dir / f"pbp_{season}_week_{week}.csv", index=False)
            week_stats.to_csv(cache_dir / f"stats_{season}_week_{week}.csv", index=False)
            
        except ImportError:
            logger.warning("nfl_data_py not available - install with: pip install nfl_data_py")
            result['error'] = 'nfl_data_py not installed'
        except Exception as e:
            logger.error(f"Error with nfl_data_py: {e}")
            result['error'] = str(e)
        
        return result
    
    def _create_sample_historical_data(
        self,
        week: int,
        season: int
    ) -> Dict[str, Any]:
        """Create sample historical data for testing."""
        
        logger.info("Creating sample historical data for testing")
        
        # Create sample player prop data
        np.random.seed(week + season)
        
        players = [
            {'id': 1, 'name': 'Travis Kelce', 'team': 'KC', 'position': 'TE'},
            {'id': 2, 'name': 'Tyreek Hill', 'team': 'MIA', 'position': 'WR'},
            {'id': 3, 'name': 'Josh Allen', 'team': 'BUF', 'position': 'QB'},
            {'id': 4, 'name': 'Stefon Diggs', 'team': 'BUF', 'position': 'WR'},
            {'id': 5, 'name': 'Derrick Henry', 'team': 'TEN', 'position': 'RB'},
            {'id': 6, 'name': 'Cooper Kupp', 'team': 'LAR', 'position': 'WR'},
        ]
        
        sample_props = []
        for player in players:
            # Create realistic historical props for this player
            if player['position'] in ['WR', 'TE']:
                base_yards = {'WR': 75, 'TE': 65}[player['position']]
                for past_week in range(max(1, week - 4), week):
                    sample_props.append({
                        'player_id': player['id'],
                        'player_name': player['name'],
                        'team': player['team'],
                        'position': player['position'],
                        'week': past_week,
                        'season': season,
                        'receiving_yards': max(0, np.random.normal(base_yards, 20)),
                        'receptions': max(0, np.random.normal(5.5, 2)),
                        'targets': max(0, np.random.normal(8, 3)),
                        'receiving_tds': np.random.poisson(0.6)
                    })
            
            elif player['position'] == 'QB':
                for past_week in range(max(1, week - 4), week):
                    sample_props.append({
                        'player_id': player['id'],
                        'player_name': player['name'],
                        'team': player['team'],
                        'position': player['position'],
                        'week': past_week,
                        'season': season,
                        'passing_yards': max(0, np.random.normal(275, 50)),
                        'passing_tds': max(0, np.random.poisson(2.1)),
                        'completions': max(0, np.random.normal(22, 5)),
                        'interceptions': np.random.poisson(0.8)
                    })
            
            elif player['position'] == 'RB':
                for past_week in range(max(1, week - 4), week):
                    sample_props.append({
                        'player_id': player['id'],
                        'player_name': player['name'],
                        'team': player['team'],
                        'position': player['position'],
                        'week': past_week,
                        'season': season,
                        'rushing_yards': max(0, np.random.normal(85, 25)),
                        'rushing_tds': np.random.poisson(0.8),
                        'carries': max(0, np.random.normal(18, 4)),
                        'receiving_yards': max(0, np.random.normal(25, 10))
                    })
        
        # Save sample data
        data_dir = Path(self.settings.feature_cache_dir) / "historical"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        df = pd.DataFrame(sample_props)
        sample_file = data_dir / f"sample_nfl_{season}_historical.csv"
        df.to_csv(sample_file, index=False)
        
        logger.info(f"Created sample data: {len(sample_props)} records saved to {sample_file}")
        
        return {
            'records_collected': len(sample_props),
            'data_types': ['sample_historical_stats'],
            'note': 'Sample data created for testing'
        }
    
    def get_data_status(self) -> Dict[str, Any]:
        """Get current data availability status."""
        
        status = {
            'api_budget': self.request_manager.get_priority_budget(),
            'data_sources': {},
            'cache_status': {},
            'recommendations': []
        }
        
        # Check each data source
        for source, config in self.data_sources.items():
            status['data_sources'][source] = {
                'enabled': config['enabled'],
                'available': self._check_source_availability(source),
                'cost_per_request': config['cost_per_request'],
                'description': config['description']
            }
        
        # Check cache directories
        cache_dirs = [
            'historical',
            'nfl_data_py', 
            'models',
            'features'
        ]
        
        for cache_dir in cache_dirs:
            dir_path = Path(self.settings.feature_cache_dir) / cache_dir
            status['cache_status'][cache_dir] = {
                'exists': dir_path.exists(),
                'file_count': len(list(dir_path.glob('*'))) if dir_path.exists() else 0
            }
        
        # Generate recommendations
        if status['api_budget']['daily_budget'] > 50:
            status['recommendations'].append("Good API budget - can collect live data")
        elif status['api_budget']['daily_budget'] > 10:
            status['recommendations'].append("Moderate API budget - use selectively")
        else:
            status['recommendations'].append("Low API budget - rely on cached/historical data")
        
        if not status['data_sources']['nfl_data_py']['available']:
            status['recommendations'].append("Install nfl_data_py for historical stats: pip install nfl_data_py")
        
        return status
    
    def _check_source_availability(self, source: str) -> bool:
        """Check if a data source is currently available."""
        
        if source == 'live_api':
            return bool(self.settings.odds_api_key and self.settings.odds_api_key != 'your_api_key_here')
        elif source == 'espn_free':
            return True  # ESPN API is always available
        elif source == 'historical_files':
            data_dir = Path(self.settings.feature_cache_dir) / "historical"
            return data_dir.exists()
        elif source == 'nfl_data_py':
            try:
                import nfl_data_py
                return True
            except ImportError:
                return False
        
        return False
    
    def create_manual_data_template(self, week: int, season: int) -> str:
        """Create a template CSV for manual data entry."""
        
        template_data = [
            {
                'player_id': 1,
                'player_name': 'Travis Kelce',
                'team': 'KC',
                'position': 'TE',
                'prop_type': 'receiving_yards',
                'market_line': 72.5,
                'over_odds': -110,
                'under_odds': -110,
                'week': week,
                'season': season,
                'opponent': 'DEN',
                'is_home': True,
                'game_date': '2024-09-15',
                'notes': 'Sample entry - replace with actual data'
            },
            {
                'player_id': 1,
                'player_name': 'Travis Kelce',
                'team': 'KC',
                'position': 'TE',
                'prop_type': 'receptions',
                'market_line': 6.5,
                'over_odds': -115,
                'under_odds': -105,
                'week': week,
                'season': season,
                'opponent': 'DEN',
                'is_home': True,
                'game_date': '2024-09-15',
                'notes': 'Sample entry - replace with actual data'
            }
        ]
        
        # Save template
        data_dir = Path(self.settings.feature_cache_dir) / "manual"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        template_file = data_dir / f"manual_data_template_week_{week}.csv"
        df = pd.DataFrame(template_data)
        df.to_csv(template_file, index=False)
        
        logger.info(f"Manual data template created: {template_file}")
        return str(template_file)


def create_comprehensive_data_test():
    """Create a comprehensive test of the data feeding system."""
    
    print("🏈 COMPREHENSIVE DATA FEEDING SYSTEM TEST")
    print("=" * 60)
    
    try:
        # Initialize data manager
        data_manager = DataManager()
        
        print("1️⃣ CHECKING DATA SOURCE AVAILABILITY")
        print("-" * 40)
        
        status = data_manager.get_data_status()
        
        print("📊 Data Sources Status:")
        for source, info in status['data_sources'].items():
            status_icon = "✅" if info['available'] else "❌"
            enabled_icon = "🟢" if info['enabled'] else "⚪"
            print(f"   {status_icon} {enabled_icon} {source}: {info['description']}")
        
        print(f"\n💰 API Budget: {status['api_budget']['daily_budget']} requests remaining")
        
        print(f"\n💡 Recommendations:")
        for rec in status['recommendations']:
            print(f"   • {rec}")
        
        print("\n2️⃣ TESTING DATA FEEDING")
        print("-" * 40)
        
        # Test feeding data for Week 2
        week = 2
        season = 2024
        
        # Use available sources only
        available_sources = [
            source for source, info in status['data_sources'].items() 
            if info['enabled'] and info['available']
        ]
        
        print(f"📥 Feeding data from sources: {available_sources}")
        
        results = data_manager.feed_week_data(
            week=week,
            season=season,
            data_sources=available_sources[:2]  # Limit to first 2 to avoid API overuse
        )
        
        print(f"\n✅ Data feeding results:")
        print(f"   Status: {results['status']}")
        print(f"   API cost: {results['total_api_cost']} requests")
        
        total_records = sum(
            result.get('records_collected', 0) 
            for result in results['data_collected'].values()
        )
        print(f"   Total records: {total_records}")
        
        for source, result in results['data_collected'].items():
            print(f"   📊 {source}: {result['records_collected']} records, types: {result.get('data_types', [])}")
        
        if results['errors']:
            print(f"   ⚠️ Errors: {len(results['errors'])}")
            for error in results['errors']:
                print(f"     • {error}")
        
        print("\n3️⃣ CREATING MANUAL DATA TEMPLATE")
        print("-" * 40)
        
        template_file = data_manager.create_manual_data_template(week, season)
        print(f"✅ Manual data template created: {template_file}")
        print("   Edit this file to add your own data, then feed it to the system")
        
        return True
        
    except Exception as e:
        print(f"❌ Data feeding test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    create_comprehensive_data_test()