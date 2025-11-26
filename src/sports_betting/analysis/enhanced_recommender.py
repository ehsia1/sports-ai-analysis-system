"""Enhanced parlay recommender with comprehensive reasoning and data integration."""

from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from sqlalchemy.orm import Session

from .parlay_recommender import ParlayRecommender
from .reasoning_engine import ReasoningEngine
from ..data.data_manager import DataManager
from ..data.collectors.nfl_historical import NFLHistoricalDataCollector
from ..models import XGBoostPropsModel, NeuralNetModel
from ..features.ml_features import MLFeatureEngineer, prepare_training_data

logger = logging.getLogger(__name__)


class EnhancedParlayRecommender(ParlayRecommender):
    """Enhanced parlay recommender with reasoning and comprehensive data integration."""
    
    def __init__(self, session: Session):
        super().__init__(session)
        
        # Enhanced components
        self.reasoning_engine = ReasoningEngine()
        self.data_manager = DataManager(session)
        self.historical_collector = NFLHistoricalDataCollector()
        self.ml_engineer = MLFeatureEngineer()
        
        # Trained models cache
        self.trained_models = {}
        self.model_training_status = {}
        
    def generate_enhanced_recommendations(
        self,
        week: int,
        season: int,
        bankroll: float = 10000,
        risk_tolerance: str = 'moderate',
        include_reasoning: bool = True,
        train_models: bool = True
    ) -> Dict[str, Any]:
        """Generate enhanced recommendations with reasoning and real data."""
        
        logger.info(f"Generating enhanced recommendations for Week {week}, Season {season}")
        
        try:
            # Phase 1: Data Collection
            data_status = self._collect_comprehensive_data(week, season)
            
            # Phase 2: Model Training (if requested and needed)
            if train_models:
                model_status = self._ensure_models_trained([season - 1, season - 2])
            else:
                model_status = {'status': 'skipped', 'models_available': len(self.trained_models)}
            
            # Phase 3: Enhanced Edge Detection
            enhanced_edges = self._detect_enhanced_edges(week, season, include_reasoning)
            
            # Phase 4: Enhanced Parlay Construction
            enhanced_parlays = self._build_enhanced_parlays(
                enhanced_edges, bankroll, risk_tolerance, include_reasoning
            )
            
            # Phase 5: Comprehensive Reporting
            final_report = self._create_enhanced_report(
                enhanced_parlays, enhanced_edges, data_status, model_status,
                week, season, bankroll, include_reasoning
            )
            
            logger.info(f"Enhanced recommendations complete: {len(enhanced_parlays.get('parlays', []))} parlays with reasoning")
            
            return final_report
            
        except Exception as e:
            logger.error(f"Error in enhanced recommendation generation: {e}")
            return self._error_enhanced_response(str(e), week, season)
    
    def _collect_comprehensive_data(self, week: int, season: int) -> Dict[str, Any]:
        """Collect comprehensive data from all available sources."""
        
        logger.info("Collecting comprehensive data...")
        
        # Check data availability
        data_status = self.data_manager.get_data_status()
        
        # Determine available sources
        available_sources = [
            source for source, info in data_status['data_sources'].items()
            if info['enabled'] and info['available']
        ]
        
        # Prioritize sources based on budget and quality
        source_priority = []
        
        if data_status['api_budget']['daily_budget'] > 20:
            source_priority.extend(['live_api', 'espn_free'])
        elif data_status['api_budget']['daily_budget'] > 5:
            source_priority.extend(['espn_free', 'live_api'])
        else:
            source_priority.extend(['espn_free', 'historical_files'])
        
        # Add always available sources
        source_priority.extend(['historical_files', 'nfl_data_py'])
        
        # Use first 3 available prioritized sources
        selected_sources = []
        for source in source_priority:
            if source in available_sources and len(selected_sources) < 3:
                selected_sources.append(source)
        
        # Collect data
        collection_results = self.data_manager.feed_week_data(
            week=week,
            season=season,
            data_sources=selected_sources
        )
        
        return {
            'status': collection_results['status'],
            'sources_used': selected_sources,
            'total_records': sum(
                result.get('records_collected', 0)
                for result in collection_results['data_collected'].values()
            ),
            'api_cost': collection_results['total_api_cost'],
            'data_quality': self._assess_data_quality(collection_results),
            'collection_details': collection_results
        }
    
    def _ensure_models_trained(self, training_seasons: List[int]) -> Dict[str, Any]:
        """Ensure ML models are trained and ready."""
        
        logger.info(f"Ensuring models are trained for seasons: {training_seasons}")
        
        prop_types = ['receiving_yards', 'receptions', 'receiving_tds']
        training_status = {
            'models_trained': 0,
            'models_loaded': 0,
            'training_errors': [],
            'model_performance': {}
        }
        
        for prop_type in prop_types:
            try:
                # Check if model already exists and is trained
                if f'xgboost_{prop_type}' in self.trained_models:
                    training_status['models_loaded'] += 1
                    continue
                
                # Get training data
                training_data = self.historical_collector.get_prop_training_data(
                    prop_type, training_seasons, min_games=6
                )
                
                if training_data.empty:
                    logger.warning(f"No training data for {prop_type}")
                    continue
                
                # Prepare features
                feature_df, feature_columns = prepare_training_data(training_data, prop_type)
                
                if len(feature_df) < 100:
                    logger.warning(f"Insufficient training data for {prop_type}: {len(feature_df)} samples")
                    continue
                
                # Train XGBoost model
                xgb_model = XGBoostPropsModel(prop_type)
                metrics = xgb_model.train(feature_df, prop_type)
                
                # Store trained model
                self.trained_models[f'xgboost_{prop_type}'] = xgb_model
                training_status['models_trained'] += 1
                training_status['model_performance'][prop_type] = {
                    'mae': metrics.get('mae', 0),
                    'r2': metrics.get('r2', 0),
                    'training_samples': metrics.get('training_samples', 0)
                }
                
                logger.info(f"Trained {prop_type} model: MAE={metrics.get('mae', 0):.2f}, R²={metrics.get('r2', 0):.3f}")
                
            except Exception as e:
                error_msg = f"Error training {prop_type} model: {e}"
                logger.error(error_msg)
                training_status['training_errors'].append(error_msg)
        
        return training_status
    
    def _detect_enhanced_edges(
        self,
        week: int,
        season: int,
        include_reasoning: bool = True
    ) -> pd.DataFrame:
        """Detect edges with enhanced ML predictions and reasoning."""
        
        logger.info("Detecting enhanced edges with ML predictions...")
        
        # Get market data (from data collection phase)
        market_data = self._get_current_market_data(week, season)
        
        if market_data.empty:
            logger.warning("No market data available for edge detection")
            return pd.DataFrame()
        
        # Enhanced edge detection with ML predictions
        enhanced_edges = []
        
        for _, market_row in market_data.iterrows():
            try:
                # Get player historical context
                player_context = self._get_player_context(
                    market_row['player_id'], week, season
                )
                
                # Get ML predictions if models are available
                ml_predictions = self._get_ml_predictions(
                    market_row, player_context
                )
                
                # Calculate enhanced fair value
                enhanced_fair_value = self._calculate_enhanced_fair_value(
                    market_row, player_context, ml_predictions
                )
                
                # Generate reasoning if requested
                if include_reasoning:
                    reasoning = self.reasoning_engine.generate_leg_reasoning(
                        player_context,
                        market_row.to_dict(),
                        ml_predictions,
                        historical_context=None
                    )
                else:
                    reasoning = None
                
                # Create enhanced edge record
                enhanced_edge = {
                    **market_row.to_dict(),
                    'ml_prediction': ml_predictions.get('prediction', market_row['market_line']),
                    'ml_confidence': ml_predictions.get('confidence', 0.6),
                    'enhanced_ev': enhanced_fair_value.get('expected_value', 0.03),
                    'fair_line': enhanced_fair_value.get('fair_line', market_row['market_line']),
                    'edge_strength': enhanced_fair_value.get('edge_strength', 0.05),
                    'reasoning': reasoning
                }
                
                # Only include if meets threshold
                if enhanced_edge['enhanced_ev'] >= 0.03:
                    enhanced_edges.append(enhanced_edge)
                
            except Exception as e:
                logger.warning(f"Error processing edge for player {market_row.get('player_id', 'unknown')}: {e}")
        
        if enhanced_edges:
            edges_df = pd.DataFrame(enhanced_edges)
            logger.info(f"Detected {len(edges_df)} enhanced edges")
            return edges_df
        else:
            logger.info("No enhanced edges detected")
            return pd.DataFrame()
    
    def _get_current_market_data(self, week: int, season: int) -> pd.DataFrame:
        """Get current market data for edge detection."""
        
        # This would normally query the database or recent API data
        # For demo purposes, create sample market data
        np.random.seed(week + season)
        
        sample_market = [
            {
                'player_id': 1,
                'player_name': 'Travis Kelce',
                'team': 'KC',
                'position': 'TE',
                'prop_type': 'receiving_yards',
                'market_line': 72.5,
                'market_over_odds': -110,
                'market_under_odds': -110,
                'volume': 2500,
                'book': 'DraftKings'
            },
            {
                'player_id': 1,
                'player_name': 'Travis Kelce',
                'team': 'KC',
                'position': 'TE',
                'prop_type': 'receptions',
                'market_line': 6.5,
                'market_over_odds': -115,
                'market_under_odds': -105,
                'volume': 1800,
                'book': 'DraftKings'
            },
            {
                'player_id': 2,
                'player_name': 'Tyreek Hill',
                'team': 'MIA',
                'position': 'WR',
                'prop_type': 'receiving_yards',
                'market_line': 85.5,
                'market_over_odds': -108,
                'market_under_odds': -112,
                'volume': 3200,
                'book': 'FanDuel'
            },
            {
                'player_id': 3,
                'player_name': 'Josh Allen',
                'team': 'BUF',
                'position': 'QB',
                'prop_type': 'passing_yards',
                'market_line': 267.5,
                'market_over_odds': -110,
                'market_under_odds': -110,
                'volume': 2800,
                'book': 'DraftKings'
            }
        ]
        
        return pd.DataFrame(sample_market)
    
    def _get_player_context(
        self,
        player_id: int,
        week: int,
        season: int
    ) -> Dict[str, Any]:
        """Get comprehensive player context for analysis."""
        
        # This would query historical stats, matchup data, etc.
        # For demo, create realistic context
        
        player_contexts = {
            1: {  # Travis Kelce
                'player_name': 'Travis Kelce',
                'position': 'TE',
                'team': 'KC',
                'receiving_yards_4game_avg': 78.5,
                'receiving_yards_season_avg': 72.1,
                'receiving_yards_trend_4': 2.3,
                'receptions_4game_avg': 6.8,
                'targets_4game_avg': 9.2,
                'is_home': True,
                'opponent': 'DEN',
                'opp_def_rank': 28,
                'weather_impact_passing': 0.0,
                'team_pace': 68.2
            },
            2: {  # Tyreek Hill
                'player_name': 'Tyreek Hill',
                'position': 'WR',
                'team': 'MIA',
                'receiving_yards_4game_avg': 89.2,
                'receiving_yards_season_avg': 84.6,
                'receiving_yards_trend_4': 1.8,
                'receptions_4game_avg': 7.5,
                'targets_4game_avg': 11.1,
                'is_home': False,
                'opponent': 'NYJ',
                'opp_def_rank': 15,
                'weather_impact_passing': 0.1,
                'team_pace': 71.5
            },
            3: {  # Josh Allen
                'player_name': 'Josh Allen',
                'position': 'QB',
                'team': 'BUF',
                'passing_yards_4game_avg': 285.3,
                'passing_yards_season_avg': 278.9,
                'passing_yards_trend_4': 3.2,
                'passing_tds_4game_avg': 2.3,
                'completions_4game_avg': 22.8,
                'is_home': True,
                'opponent': 'MIA',
                'opp_def_rank': 22,
                'weather_impact_passing': 0.0,
                'team_pace': 66.8
            }
        }
        
        return player_contexts.get(player_id, {})
    
    def _get_ml_predictions(
        self,
        market_row: pd.Series,
        player_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get ML model predictions if available."""
        
        prop_type = market_row['prop_type']
        model_key = f'xgboost_{prop_type}'
        
        if model_key in self.trained_models:
            try:
                model = self.trained_models[model_key]
                prediction, confidence = model.get_prop_line_prediction(player_context)
                
                return {
                    'prediction': prediction,
                    'confidence': confidence,
                    'model_type': 'xgboost',
                    'has_ml_prediction': True
                }
            except Exception as e:
                logger.warning(f"Error getting ML prediction: {e}")
        
        # Fallback to heuristic prediction
        return {
            'prediction': market_row['market_line'],
            'confidence': 0.6,
            'model_type': 'heuristic',
            'has_ml_prediction': False
        }
    
    def _calculate_enhanced_fair_value(
        self,
        market_row: pd.Series,
        player_context: Dict[str, Any],
        ml_predictions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate enhanced fair value using ML predictions."""
        
        market_line = market_row['market_line']
        ml_prediction = ml_predictions['prediction']
        confidence = ml_predictions['confidence']
        
        # Calculate line difference
        line_diff = ml_prediction - market_line
        
        # Calculate edge strength based on confidence and line difference
        edge_strength = abs(line_diff) * confidence / market_line if market_line > 0 else 0
        
        # Calculate expected value (simplified)
        if line_diff > 0:
            # Model predicts higher than market (over bet)
            ev = edge_strength * 0.9  # Discount for uncertainty
        else:
            # Model predicts lower than market (under bet)
            ev = edge_strength * 0.9
        
        return {
            'fair_line': ml_prediction,
            'line_difference': line_diff,
            'edge_strength': edge_strength,
            'expected_value': max(0, ev),
            'recommended_side': 'over' if line_diff > 0 else 'under'
        }
    
    def _build_enhanced_parlays(
        self,
        enhanced_edges: pd.DataFrame,
        bankroll: float,
        risk_tolerance: str,
        include_reasoning: bool = True
    ) -> Dict[str, Any]:
        """Build enhanced parlays with reasoning."""
        
        if enhanced_edges.empty:
            return {'parlays': [], 'sgps': [], 'total_parlays': 0}
        
        logger.info(f"Building enhanced parlays from {len(enhanced_edges)} edges...")
        
        # Use parent class parlay building
        basic_parlays = super().generate_weekly_recommendations(
            week=2,  # Placeholder
            season=2024,  # Placeholder
            bankroll=bankroll,
            risk_tolerance=risk_tolerance
        )
        
        # Enhance with reasoning
        if include_reasoning and basic_parlays.get('recommendations'):
            enhanced_parlays = self._add_reasoning_to_parlays(
                basic_parlays['recommendations'], enhanced_edges
            )
        else:
            enhanced_parlays = basic_parlays.get('recommendations', {})
        
        return enhanced_parlays
    
    def _add_reasoning_to_parlays(
        self,
        parlay_recommendations: Dict[str, Any],
        enhanced_edges: pd.DataFrame
    ) -> Dict[str, Any]:
        """Add detailed reasoning to parlay recommendations."""
        
        enhanced_recommendations = parlay_recommendations.copy()
        
        # Add reasoning to each parlay type
        for parlay_type in ['premium_parlays', 'standard_parlays', 'same_game_parlays']:
            if parlay_type in enhanced_recommendations:
                for i, parlay in enumerate(enhanced_recommendations[parlay_type]):
                    try:
                        # Get legs reasoning
                        legs_reasoning = []
                        
                        for leg in parlay.get('legs_summary', []):
                            # Find matching edge
                            matching_edge = enhanced_edges[
                                (enhanced_edges['player_name'] == leg.get('player_name', '')) &
                                (enhanced_edges['prop_type'] == leg.get('prop_type', ''))
                            ]
                            
                            if not matching_edge.empty:
                                edge_row = matching_edge.iloc[0]
                                if edge_row.get('reasoning'):
                                    legs_reasoning.append(edge_row['reasoning'])
                        
                        # Add parlay-level reasoning
                        if legs_reasoning:
                            parlay_reasoning = self.reasoning_engine.generate_parlay_reasoning(
                                legs_reasoning,
                                {'average_correlation': 0.45, 'max_correlation': 0.65},
                                parlay
                            )
                            
                            enhanced_recommendations[parlay_type][i]['detailed_reasoning'] = parlay_reasoning
                            enhanced_recommendations[parlay_type][i]['execution_guide'] = (
                                parlay_reasoning.get('execution_reasoning', 'Execute based on analysis')
                            )
                    
                    except Exception as e:
                        logger.warning(f"Error adding reasoning to parlay: {e}")
        
        return enhanced_recommendations
    
    def _create_enhanced_report(
        self,
        enhanced_parlays: Dict[str, Any],
        enhanced_edges: pd.DataFrame,
        data_status: Dict[str, Any],
        model_status: Dict[str, Any],
        week: int,
        season: int,
        bankroll: float,
        include_reasoning: bool
    ) -> Dict[str, Any]:
        """Create comprehensive enhanced report."""
        
        # Count parlays
        total_parlays = sum(
            len(enhanced_parlays.get(key, []))
            for key in ['premium_parlays', 'standard_parlays', 'value_parlays']
        )
        total_sgps = len(enhanced_parlays.get('same_game_parlays', []))
        
        # Calculate portfolio metrics
        all_parlays = []
        for parlay_type in ['premium_parlays', 'standard_parlays', 'value_parlays', 'same_game_parlays']:
            all_parlays.extend(enhanced_parlays.get(parlay_type, []))
        
        total_allocation = sum(p.get('recommended_bet_amount', 0) for p in all_parlays)
        avg_ev = np.mean([p.get('expected_value', 0) for p in all_parlays]) if all_parlays else 0
        
        enhanced_report = {
            'week': week,
            'season': season,
            'generated_at': datetime.now().isoformat(),
            'system_version': '2.0_enhanced',
            'includes_reasoning': include_reasoning,
            
            'data_summary': {
                'sources_used': data_status.get('sources_used', []),
                'total_records_collected': data_status.get('total_records', 0),
                'api_requests_used': data_status.get('api_cost', 0),
                'data_quality': data_status.get('data_quality', 'good')
            },
            
            'model_summary': {
                'models_trained': model_status.get('models_trained', 0),
                'models_loaded': model_status.get('models_loaded', 0),
                'model_performance': model_status.get('model_performance', {}),
                'training_errors': len(model_status.get('training_errors', []))
            },
            
            'analysis_summary': {
                'edges_detected': len(enhanced_edges),
                'avg_edge_strength': enhanced_edges['edge_strength'].mean() if not enhanced_edges.empty else 0,
                'ml_predictions_used': sum(1 for _, row in enhanced_edges.iterrows() if row.get('reasoning', {}).get('has_ml_prediction', False)) if not enhanced_edges.empty else 0
            },
            
            'recommendations': enhanced_parlays,
            
            'portfolio_summary': {
                'total_parlays': total_parlays,
                'total_sgps': total_sgps,
                'total_allocation': total_allocation,
                'allocation_percentage': total_allocation / bankroll if bankroll > 0 else 0,
                'expected_return': avg_ev,
                'risk_level': self._assess_portfolio_risk(total_allocation / bankroll if bankroll > 0 else 0)
            },
            
            'execution_priority': self._create_execution_priority(all_parlays),
            
            'system_insights': self._generate_system_insights(
                data_status, model_status, enhanced_edges, all_parlays
            )
        }
        
        return enhanced_report
    
    def _assess_data_quality(self, collection_results: Dict[str, Any]) -> str:
        """Assess overall data quality."""
        
        total_records = sum(
            result.get('records_collected', 0)
            for result in collection_results['data_collected'].values()
        )
        
        api_cost = collection_results.get('total_api_cost', 0)
        sources_used = len(collection_results['data_collected'])
        
        if total_records > 100 and api_cost > 0 and sources_used >= 2:
            return 'excellent'
        elif total_records > 50 and sources_used >= 2:
            return 'good'
        elif total_records > 20:
            return 'fair'
        else:
            return 'limited'
    
    def _assess_portfolio_risk(self, allocation_pct: float) -> str:
        """Assess portfolio risk level."""
        
        if allocation_pct > 0.15:
            return 'high'
        elif allocation_pct > 0.08:
            return 'moderate'
        else:
            return 'conservative'
    
    def _create_execution_priority(self, all_parlays: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create execution priority list."""
        
        if not all_parlays:
            return []
        
        # Sort by expected value and confidence
        prioritized = sorted(
            all_parlays,
            key=lambda p: (p.get('expected_value', 0) * p.get('confidence_score', 0.5)),
            reverse=True
        )
        
        execution_list = []
        for i, parlay in enumerate(prioritized[:5], 1):
            execution_list.append({
                'priority': i,
                'parlay_summary': f"{parlay.get('num_legs', 2)}-leg {parlay.get('parlay_type', 'parlay')}",
                'expected_value': parlay.get('expected_value', 0),
                'bet_amount': parlay.get('recommended_bet_amount', 0),
                'reasoning_available': 'detailed_reasoning' in parlay
            })
        
        return execution_list
    
    def _generate_system_insights(
        self,
        data_status: Dict[str, Any],
        model_status: Dict[str, Any],
        enhanced_edges: pd.DataFrame,
        all_parlays: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate system insights and recommendations."""
        
        insights = []
        
        # Data insights
        if data_status.get('api_cost', 0) == 0:
            insights.append("✅ No API budget used - running on free data sources")
        elif data_status.get('api_cost', 0) < 10:
            insights.append(f"💰 Conservative API usage: {data_status['api_cost']} requests")
        
        # Model insights
        models_trained = model_status.get('models_trained', 0)
        if models_trained > 0:
            insights.append(f"🤖 {models_trained} ML models trained and active")
        else:
            insights.append("📊 Using heuristic models - install ML libraries for enhanced predictions")
        
        # Edge insights
        if not enhanced_edges.empty:
            high_confidence_edges = len(enhanced_edges[enhanced_edges['ml_confidence'] > 0.75])
            insights.append(f"🎯 {len(enhanced_edges)} edges detected, {high_confidence_edges} high-confidence")
        
        # Portfolio insights
        if all_parlays:
            with_reasoning = sum(1 for p in all_parlays if 'detailed_reasoning' in p)
            insights.append(f"🧠 {with_reasoning}/{len(all_parlays)} parlays include detailed reasoning")
        
        return insights
    
    def _error_enhanced_response(self, error: str, week: int, season: int) -> Dict[str, Any]:
        """Create error response for enhanced recommendations."""
        
        return {
            'error': error,
            'week': week,
            'season': season,
            'system_version': '2.0_enhanced',
            'recommendations': {'premium_parlays': [], 'standard_parlays': [], 'same_game_parlays': []},
            'portfolio_summary': {'total_parlays': 0, 'total_allocation': 0, 'expected_return': 0},
            'system_insights': [f"❌ System error: {error}"]
        }


def demo_enhanced_recommender():
    """Demonstrate the enhanced parlay recommender."""
    
    print("🚀 ENHANCED PARLAY RECOMMENDER DEMO")
    print("=" * 60)
    
    # Mock enhanced recommender
    class MockEnhancedRecommender(EnhancedParlayRecommender):
        def __init__(self):
            # Initialize with mock components
            self.reasoning_engine = ReasoningEngine()
            self.trained_models = {}
    
    recommender = MockEnhancedRecommender()
    
    print("1️⃣ GENERATING ENHANCED RECOMMENDATIONS")
    print("-" * 40)
    
    enhanced_report = recommender.generate_enhanced_recommendations(
        week=2,
        season=2024,
        bankroll=10000,
        risk_tolerance='moderate',
        include_reasoning=True,
        train_models=False  # Skip training for demo
    )
    
    print("📊 Enhanced Recommendation Report:")
    print(f"   System Version: {enhanced_report.get('system_version', 'unknown')}")
    print(f"   Reasoning Included: {enhanced_report.get('includes_reasoning', False)}")
    
    # Data summary
    data_summary = enhanced_report.get('data_summary', {})
    print(f"\n📥 Data Collection:")
    print(f"   Sources: {data_summary.get('sources_used', [])}")
    print(f"   Records: {data_summary.get('total_records_collected', 0)}")
    print(f"   API Cost: {data_summary.get('api_requests_used', 0)} requests")
    print(f"   Quality: {data_summary.get('data_quality', 'unknown').title()}")
    
    # Model summary
    model_summary = enhanced_report.get('model_summary', {})
    print(f"\n🤖 ML Models:")
    print(f"   Trained: {model_summary.get('models_trained', 0)}")
    print(f"   Loaded: {model_summary.get('models_loaded', 0)}")
    print(f"   Errors: {model_summary.get('training_errors', 0)}")
    
    # Analysis summary
    analysis_summary = enhanced_report.get('analysis_summary', {})
    print(f"\n🔍 Analysis Results:")
    print(f"   Edges Detected: {analysis_summary.get('edges_detected', 0)}")
    print(f"   Avg Edge Strength: {analysis_summary.get('avg_edge_strength', 0):.1%}")
    print(f"   ML Predictions: {analysis_summary.get('ml_predictions_used', 0)}")
    
    # Portfolio summary
    portfolio = enhanced_report.get('portfolio_summary', {})
    print(f"\n💰 Portfolio Summary:")
    print(f"   Total Parlays: {portfolio.get('total_parlays', 0)}")
    print(f"   SGPs: {portfolio.get('total_sgps', 0)}")
    print(f"   Allocation: ${portfolio.get('total_allocation', 0):.0f} ({portfolio.get('allocation_percentage', 0):.1%})")
    print(f"   Expected Return: {portfolio.get('expected_return', 0):.1%}")
    print(f"   Risk Level: {portfolio.get('risk_level', 'unknown').title()}")
    
    # System insights
    insights = enhanced_report.get('system_insights', [])
    if insights:
        print(f"\n💡 System Insights:")
        for insight in insights:
            print(f"   {insight}")
    
    # Execution priority
    execution_priority = enhanced_report.get('execution_priority', [])
    if execution_priority:
        print(f"\n🎯 Execution Priority:")
        for item in execution_priority[:3]:
            print(f"   {item['priority']}. {item['parlay_summary']}: {item['expected_value']:.1%} EV, ${item['bet_amount']:.0f}")
    
    print(f"\n✅ Enhanced recommendation demo complete!")
    
    return enhanced_report


if __name__ == "__main__":
    demo_enhanced_recommender()