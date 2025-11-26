"""Automated parlay recommendation system."""

from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from sqlalchemy.orm import Session

from .edge_detector import EdgeDetector
from .parlay_builder import ParlayBuilder
from .sgp_validator import SGPValidator
from ..database import Game, Player, Prop
from ..config import get_settings

logger = logging.getLogger(__name__)


class ParlayRecommender:
    """Automated system for generating optimized parlay recommendations."""
    
    def __init__(self, session: Session):
        self.session = session
        self.settings = get_settings()
        
        # Initialize components
        self.edge_detector = EdgeDetector(session)
        self.parlay_builder = ParlayBuilder(session)
        self.sgp_validator = SGPValidator()
        
        # Recommendation parameters
        self.min_edges_for_parlays = 3
        self.max_recommendations = 10
        self.diversification_target = 0.3  # Target correlation for diversification
        
        # Risk management
        self.max_parlay_risk = 0.02  # 2% of bankroll per parlay
        self.max_total_risk = 0.10   # 10% total parlay allocation
        
        # Recommendation tiers
        self.confidence_tiers = {
            'premium': {'min_confidence': 0.80, 'min_ev': 0.08, 'max_risk': 0.03},
            'standard': {'min_confidence': 0.70, 'min_ev': 0.05, 'max_risk': 0.02},
            'value': {'min_confidence': 0.60, 'min_ev': 0.04, 'max_risk': 0.01}
        }
        
    def generate_weekly_recommendations(
        self,
        week: int,
        season: int,
        bankroll: float = 10000,
        risk_tolerance: str = 'moderate'
    ) -> Dict[str, Any]:
        """Generate comprehensive weekly parlay recommendations."""
        
        logger.info(f"Generating weekly recommendations for week {week}, season {season}")
        
        try:
            # Step 1: Detect edges for the week
            edges_df = self._collect_weekly_edges(week, season)
            
            if edges_df.empty:
                return self._empty_recommendation_response("No edges detected for the week")
            
            # Step 2: Generate parlay recommendations
            recommendations = self._generate_parlay_recommendations(
                edges_df, bankroll, risk_tolerance
            )
            
            # Step 3: Create same-game parlay recommendations
            sgp_recommendations = self._generate_sgp_recommendations(edges_df, bankroll)
            
            # Step 4: Portfolio optimization
            optimized_portfolio = self._optimize_recommendation_portfolio(
                recommendations, sgp_recommendations, bankroll, risk_tolerance
            )
            
            # Step 5: Generate comprehensive report
            final_report = self._generate_recommendation_report(
                optimized_portfolio, edges_df, week, season, bankroll
            )
            
            logger.info(f"Generated {len(optimized_portfolio.get('parlays', []))} parlay recommendations")
            
            return final_report
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return self._error_recommendation_response(str(e))
            
    def _collect_weekly_edges(self, week: int, season: int) -> pd.DataFrame:
        """Collect all edges for the week."""
        
        # In production, this would query real market data
        # For demo, create sample market data
        sample_market_data = self._generate_sample_market_data(week, season)
        
        # Detect edges
        edges_df = self.edge_detector.detect_edges(
            sample_market_data,
            week=week,
            season=season,
            prop_types=['receiving_yards', 'receptions', 'receiving_tds', 'passing_yards', 'rushing_yards']
        )
        
        return edges_df
        
    def _generate_sample_market_data(self, week: int, season: int) -> pd.DataFrame:
        """Generate sample market data for demonstration."""
        
        np.random.seed(week + season)  # Reproducible results
        
        players = [
            {'id': 1, 'name': 'Travis Kelce', 'team': 'KC', 'position': 'TE'},
            {'id': 2, 'name': 'Tyreek Hill', 'team': 'MIA', 'position': 'WR'},
            {'id': 3, 'name': 'Josh Allen', 'team': 'BUF', 'position': 'QB'},
            {'id': 4, 'name': 'Stefon Diggs', 'team': 'BUF', 'position': 'WR'},
            {'id': 5, 'name': 'Derrick Henry', 'team': 'TEN', 'position': 'RB'},
            {'id': 6, 'name': 'Cooper Kupp', 'team': 'LAR', 'position': 'WR'},
            {'id': 7, 'name': 'Justin Jefferson', 'team': 'MIN', 'position': 'WR'},
            {'id': 8, 'name': 'Lamar Jackson', 'team': 'BAL', 'position': 'QB'},
        ]
        
        prop_types = ['receiving_yards', 'receptions', 'receiving_tds', 'passing_yards', 'rushing_yards']
        
        market_data = []
        game_id = 1
        
        for i, player in enumerate(players):
            # Assign games (2 players per game for SGP potential)
            if i % 2 == 0:
                game_id += 1
                
            for prop_type in prop_types:
                # Skip irrelevant props
                if prop_type.startswith('receiving') and player['position'] in ['QB', 'RB']:
                    continue
                if prop_type.startswith('passing') and player['position'] != 'QB':
                    continue
                if prop_type.startswith('rushing') and player['position'] in ['WR', 'TE']:
                    continue
                    
                # Generate realistic market lines
                if prop_type == 'receiving_yards':
                    base_line = {'WR': 75, 'TE': 65, 'RB': 25}.get(player['position'], 50)
                elif prop_type == 'receptions':
                    base_line = {'WR': 6, 'TE': 5.5, 'RB': 3}.get(player['position'], 4)
                elif prop_type == 'receiving_tds':
                    base_line = 0.5
                elif prop_type == 'passing_yards':
                    base_line = 270
                elif prop_type == 'rushing_yards':
                    base_line = 80
                else:
                    continue
                    
                # Add some variance
                market_line = base_line + np.random.normal(0, base_line * 0.1)
                
                market_data.append({
                    'player_id': player['id'],
                    'player_name': player['name'],
                    'game_id': game_id,
                    'prop_type': prop_type,
                    'market_line': round(market_line, 1),
                    'market_over_odds': np.random.choice([-105, -110, -115]),
                    'market_under_odds': np.random.choice([-105, -110, -115]),
                    'team': player['team'],
                    'position': player['position']
                })
        
        return pd.DataFrame(market_data)
        
    def _generate_parlay_recommendations(
        self,
        edges_df: pd.DataFrame,
        bankroll: float,
        risk_tolerance: str
    ) -> List[Dict[str, Any]]:
        """Generate multi-game parlay recommendations."""
        
        if len(edges_df) < self.min_edges_for_parlays:
            return []
        
        # Build parlays (multi-game)
        parlays_df = self.parlay_builder.build_parlays(
            edges_df,
            max_parlays=15,
            same_game_only=False
        )
        
        if parlays_df.empty:
            return []
        
        # Optimize parlay selection
        optimized_parlays = self.parlay_builder.optimize_parlay_selection(
            parlays_df, bankroll, self._get_risk_multiplier(risk_tolerance)
        )
        
        return optimized_parlays.to_dict('records')[:8]  # Top 8 multi-game parlays
        
    def _generate_sgp_recommendations(
        self,
        edges_df: pd.DataFrame,
        bankroll: float
    ) -> List[Dict[str, Any]]:
        """Generate same-game parlay recommendations."""
        
        sgp_recommendations = []
        
        # Group edges by game
        games = edges_df['game_id'].unique()
        
        for game_id in games:
            game_edges = edges_df[edges_df['game_id'] == game_id]
            
            if len(game_edges) < 2:
                continue
            
            # Try to build 2-3 leg SGPs from this game
            for target_legs in [2, 3, 4]:
                if len(game_edges) < target_legs:
                    continue
                    
                # Get available edges as list of dicts
                available_edges = game_edges.to_dict('records')
                
                # Optimize SGP construction
                sgp_result = self.sgp_validator.optimize_sgp_construction(
                    available_edges,
                    target_legs=target_legs,
                    max_correlation=0.6
                )
                
                if sgp_result and 'sgp' in sgp_result and sgp_result['sgp']:
                    sgp_data = sgp_result['sgp']
                    sgp_data['game_id'] = game_id
                    sgp_data['parlay_type'] = 'same_game'
                    sgp_recommendations.append(sgp_data)
        
        # Sort by score and return top SGPs
        sgp_recommendations.sort(key=lambda x: x.get('score', 0), reverse=True)
        return sgp_recommendations[:5]  # Top 5 SGPs
        
    def _optimize_recommendation_portfolio(
        self,
        parlays: List[Dict[str, Any]],
        sgps: List[Dict[str, Any]],
        bankroll: float,
        risk_tolerance: str
    ) -> Dict[str, Any]:
        """Optimize the complete recommendation portfolio."""
        
        all_recommendations = parlays + sgps
        
        if not all_recommendations:
            return {'parlays': [], 'sgps': [], 'portfolio_metrics': {}}
        
        # Risk parameters based on tolerance
        risk_params = self._get_risk_parameters(risk_tolerance)
        
        # Tier recommendations
        tiered_recommendations = self._tier_recommendations(all_recommendations)
        
        # Portfolio allocation
        portfolio = self._allocate_portfolio(
            tiered_recommendations, bankroll, risk_params
        )
        
        return portfolio
        
    def _tier_recommendations(self, recommendations: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Tier recommendations by quality."""
        
        tiers = {'premium': [], 'standard': [], 'value': []}
        
        for rec in recommendations:
            confidence = rec.get('confidence_score', rec.get('combined_confidence', 0.6))
            ev = rec.get('expected_value', rec.get('combined_ev', 0.03))
            
            if confidence >= 0.80 and ev >= 0.08:
                tiers['premium'].append(rec)
            elif confidence >= 0.70 and ev >= 0.05:
                tiers['standard'].append(rec)
            elif confidence >= 0.60 and ev >= 0.04:
                tiers['value'].append(rec)
        
        return tiers
        
    def _allocate_portfolio(
        self,
        tiered_recs: Dict[str, List[Dict[str, Any]]],
        bankroll: float,
        risk_params: Dict[str, float]
    ) -> Dict[str, Any]:
        """Allocate bankroll across recommendation tiers."""
        
        portfolio = {
            'parlays': [],
            'sgps': [],
            'portfolio_metrics': {
                'total_allocation': 0,
                'expected_return': 0,
                'risk_level': 'moderate'
            }
        }
        
        total_allocation = 0
        total_ev = 0
        
        # Allocation percentages by tier
        tier_allocations = {
            'premium': risk_params['premium_allocation'],
            'standard': risk_params['standard_allocation'], 
            'value': risk_params['value_allocation']
        }
        
        for tier, allocation_pct in tier_allocations.items():
            tier_recs = tiered_recs.get(tier, [])
            if not tier_recs:
                continue
                
            tier_budget = bankroll * allocation_pct
            
            # Sort by optimization score
            tier_recs.sort(key=lambda x: x.get('optimization_score', x.get('score', 0)), reverse=True)
            
            # Allocate within tier
            per_bet = tier_budget / min(len(tier_recs), 3)  # Max 3 bets per tier
            
            for i, rec in enumerate(tier_recs[:3]):
                bet_amount = min(per_bet, bankroll * self.max_parlay_risk)
                
                rec['tier'] = tier
                rec['recommended_bet_amount'] = bet_amount
                rec['allocation_percentage'] = bet_amount / bankroll
                
                if rec.get('parlay_type') == 'same_game':
                    portfolio['sgps'].append(rec)
                else:
                    portfolio['parlays'].append(rec)
                    
                total_allocation += bet_amount
                total_ev += rec.get('expected_value', rec.get('combined_ev', 0)) * bet_amount
        
        # Portfolio metrics
        portfolio['portfolio_metrics'] = {
            'total_allocation': total_allocation,
            'allocation_percentage': total_allocation / bankroll,
            'expected_return': total_ev,
            'expected_return_percentage': total_ev / bankroll if bankroll > 0 else 0,
            'risk_level': self._calculate_portfolio_risk_level(portfolio),
            'diversification_score': self._calculate_diversification_score(portfolio)
        }
        
        return portfolio
        
    def _get_risk_parameters(self, risk_tolerance: str) -> Dict[str, float]:
        """Get risk parameters based on tolerance."""
        
        risk_profiles = {
            'conservative': {
                'premium_allocation': 0.06,    # 6% in premium
                'standard_allocation': 0.03,   # 3% in standard
                'value_allocation': 0.01,      # 1% in value
                'max_single_bet': 0.015        # 1.5% max single bet
            },
            'moderate': {
                'premium_allocation': 0.08,
                'standard_allocation': 0.05,
                'value_allocation': 0.02,
                'max_single_bet': 0.02
            },
            'aggressive': {
                'premium_allocation': 0.10,
                'standard_allocation': 0.07,
                'value_allocation': 0.03,
                'max_single_bet': 0.025
            }
        }
        
        return risk_profiles.get(risk_tolerance, risk_profiles['moderate'])
        
    def _get_risk_multiplier(self, risk_tolerance: str) -> float:
        """Get risk multiplier for parlay building."""
        
        multipliers = {
            'conservative': 0.03,
            'moderate': 0.05,
            'aggressive': 0.07
        }
        
        return multipliers.get(risk_tolerance, 0.05)
        
    def _calculate_portfolio_risk_level(self, portfolio: Dict[str, Any]) -> str:
        """Calculate overall portfolio risk level."""
        
        allocation_pct = portfolio['portfolio_metrics']['allocation_percentage']
        
        if allocation_pct > 0.15:
            return 'high'
        elif allocation_pct > 0.08:
            return 'moderate'
        else:
            return 'low'
            
    def _calculate_diversification_score(self, portfolio: Dict[str, Any]) -> float:
        """Calculate portfolio diversification score."""
        
        all_recs = portfolio['parlays'] + portfolio['sgps']
        
        if not all_recs:
            return 0.0
        
        # Check game diversity
        games = set()
        for rec in all_recs:
            if 'legs' in rec:
                # For parlays, get games from legs
                leg_games = set(leg.get('game_id', 1) for leg in rec.get('legs_summary', []))
                games.update(leg_games)
            else:
                games.add(rec.get('game_id', 1))
        
        game_diversity = len(games) / max(len(all_recs), 1)
        
        # Check parlay type diversity
        parlay_types = [rec.get('parlay_type', 'multi_game') for rec in all_recs]
        type_diversity = len(set(parlay_types)) / len(parlay_types) if parlay_types else 0
        
        # Combined diversification score
        return (game_diversity * 0.7 + type_diversity * 0.3)
        
    def _generate_recommendation_report(
        self,
        portfolio: Dict[str, Any],
        edges_df: pd.DataFrame,
        week: int,
        season: int,
        bankroll: float
    ) -> Dict[str, Any]:
        """Generate comprehensive recommendation report."""
        
        report = {
            'week': week,
            'season': season,
            'generated_at': datetime.now().isoformat(),
            'bankroll': bankroll,
            'summary': {
                'total_edges_found': len(edges_df),
                'total_parlays': len(portfolio['parlays']),
                'total_sgps': len(portfolio['sgps']),
                'portfolio_allocation': portfolio['portfolio_metrics']['allocation_percentage'],
                'expected_return': portfolio['portfolio_metrics']['expected_return_percentage'],
                'risk_level': portfolio['portfolio_metrics']['risk_level']
            },
            'recommendations': {
                'premium_parlays': [p for p in portfolio['parlays'] if p.get('tier') == 'premium'],
                'standard_parlays': [p for p in portfolio['parlays'] if p.get('tier') == 'standard'],
                'value_parlays': [p for p in portfolio['parlays'] if p.get('tier') == 'value'],
                'same_game_parlays': portfolio['sgps']
            },
            'portfolio_analysis': portfolio['portfolio_metrics'],
            'risk_management': {
                'max_single_bet': max([r.get('recommended_bet_amount', 0) for r in portfolio['parlays'] + portfolio['sgps']] + [0]),
                'total_risk_exposure': portfolio['portfolio_metrics']['total_allocation'],
                'diversification_score': portfolio['portfolio_metrics']['diversification_score']
            },
            'execution_guide': self._generate_execution_guide(portfolio)
        }
        
        return report
        
    def _generate_execution_guide(self, portfolio: Dict[str, Any]) -> List[str]:
        """Generate step-by-step execution guide."""
        
        guide = []
        
        all_recs = portfolio['parlays'] + portfolio['sgps']
        
        if not all_recs:
            return ["No recommendations to execute this week."]
        
        # Sort by tier and expected value
        all_recs.sort(key=lambda x: (
            {'premium': 3, 'standard': 2, 'value': 1}.get(x.get('tier', 'value'), 1),
            x.get('expected_value', x.get('combined_ev', 0))
        ), reverse=True)
        
        guide.append("📋 WEEKLY PARLAY EXECUTION PLAN")
        guide.append("=" * 50)
        
        for i, rec in enumerate(all_recs[:5], 1):  # Top 5 recommendations
            rec_type = "SGP" if rec.get('parlay_type') == 'same_game' else "Parlay"
            tier = rec.get('tier', 'standard').title()
            bet_amount = rec.get('recommended_bet_amount', 0)
            
            guide.append(f"\n{i}. {tier} {rec_type} - ${bet_amount:.0f}")
            
            if 'legs_summary' in rec:
                guide.append("   Legs:")
                for leg in rec['legs_summary'][:3]:  # Show first 3 legs
                    guide.append(f"   • {leg['player_name']} {leg['prop_type']} {leg['best_side']} {leg['market_line']}")
            
            ev = rec.get('expected_value', rec.get('combined_ev', 0))
            confidence = rec.get('confidence_score', rec.get('combined_confidence', 0))
            
            guide.append(f"   Expected Value: {ev:.1%}, Confidence: {confidence:.1%}")
        
        guide.append(f"\n💰 Total Week Allocation: ${portfolio['portfolio_metrics']['total_allocation']:.0f}")
        guide.append(f"📈 Expected Return: {portfolio['portfolio_metrics']['expected_return_percentage']:.2%}")
        
        return guide
        
    def _empty_recommendation_response(self, message: str) -> Dict[str, Any]:
        """Return empty recommendation response."""
        return {
            'message': message,
            'recommendations': {'premium_parlays': [], 'standard_parlays': [], 'value_parlays': [], 'same_game_parlays': []},
            'summary': {'total_parlays': 0, 'total_sgps': 0, 'portfolio_allocation': 0, 'expected_return': 0},
            'execution_guide': [message]
        }
        
    def _error_recommendation_response(self, error: str) -> Dict[str, Any]:
        """Return error recommendation response."""
        return {
            'error': error,
            'recommendations': {'premium_parlays': [], 'standard_parlays': [], 'value_parlays': [], 'same_game_parlays': []},
            'summary': {'total_parlays': 0, 'total_sgps': 0, 'portfolio_allocation': 0, 'expected_return': 0},
            'execution_guide': [f"Error generating recommendations: {error}"]
        }


def demo_parlay_recommendations() -> Dict[str, Any]:
    """Demonstrate parlay recommendation system."""
    
    logger.info("Running parlay recommendation demo")
    
    # Mock recommender (no database session needed)
    class MockParlayRecommender(ParlayRecommender):
        def __init__(self):
            from ..config import Settings
            self.settings = Settings()
            self.edge_detector = None  # Will override methods
            self.parlay_builder = None
            self.sgp_validator = SGPValidator()
    
    recommender = MockParlayRecommender()
    
    # Generate recommendations
    recommendations = recommender.generate_weekly_recommendations(
        week=2,
        season=2024,
        bankroll=10000,
        risk_tolerance='moderate'
    )
    
    logger.info(f"Demo complete: Generated recommendations for Week 2")
    
    return recommendations