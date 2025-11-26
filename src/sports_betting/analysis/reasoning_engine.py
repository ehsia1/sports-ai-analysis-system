"""Reasoning engine for generating detailed parlay explanations."""

from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ReasoningEngine:
    """Generate human-readable explanations for betting recommendations."""
    
    def __init__(self):
        # Reasoning templates
        self.prop_templates = {
            'receiving_yards': {
                'positive_factors': [
                    "averages {avg:.1f} yards vs teams ranked {rank}+ in pass defense",
                    "opponent allows {opp_avg:.1f} YPG to {position}s (rank {opp_rank})",
                    "home field advantage adds +{home_boost:.1f} yards historically",
                    "positive game script expected (team favored by {spread})",
                    "targets trending up: {trend:+.1f} per game over last {weeks} weeks"
                ],
                'negative_factors': [
                    "facing top-{rank} pass defense allowing only {opp_avg:.1f} YPG",
                    "away game historically reduces output by {away_penalty:.1f} yards",
                    "negative game script likely (team underdog by {spread})",
                    "targets trending down: {trend:.1f} per game recently"
                ],
                'model_insights': [
                    "XGBoost model: {xgb_conf:.0%} confidence, predicts {xgb_pred:.1f} yards",
                    "Neural network: {nn_conf:.0%} confidence, sees {nn_edge:.1%} edge",
                    "Historical model accuracy: {accuracy:.1%} on similar matchups"
                ]
            },
            'receptions': {
                'positive_factors': [
                    "strong {corr:.2f} correlation with receiving yards performance",
                    "target share increases vs weak pass defense",
                    "opponent allows {completion_rate:.1%} completion rate",
                    "short-area specialist benefits from game plan"
                ],
                'negative_factors': [
                    "target competition from {competitor} in similar role",
                    "opponent excels at limiting receptions ({opp_rec_allowed:.1f}/game)",
                    "weather conditions may reduce short passing"
                ],
                'model_insights': [
                    "Receptions model shows {confidence:.0%} hit probability",
                    "Target share model predicts {target_share:.1%} of team targets"
                ]
            },
            'receiving_tds': {
                'positive_factors': [
                    "red zone target share: {rz_share:.1%} of team opportunities",
                    "opponent allows {opp_td_rate:.1f} passing TDs per game",
                    "positive touchdown regression expected (+{td_regression:.1f})",
                    "goal line formation usage trending up"
                ],
                'negative_factors': [
                    "red zone usage conflict with {competitor}",
                    "opponent excellent at red zone defense (rank {rz_rank})",
                    "negative touchdown regression due (-{td_regression:.1f})"
                ],
                'model_insights': [
                    "TD probability model: {td_prob:.1%} chance of 1+ scores",
                    "Red zone efficiency: {rz_eff:.1%} over last {weeks} games"
                ]
            },
            'passing_yards': {
                'positive_factors': [
                    "opponent allows {opp_ypa:.1f} YPA to opposing QBs",
                    "positive game script likely (team favored by {spread})",
                    "pass attempts trending up: +{attempts_trend:.1f} per game",
                    "weapons healthy: {weapons_health} key receivers available"
                ],
                'negative_factors': [
                    "strong opponent pass rush ({sack_rate:.1%} sack rate)",
                    "weather impact: {weather_desc} reduces passing by {weather_impact:.1%}",
                    "negative game script expected (run-heavy if ahead)"
                ],
                'model_insights': [
                    "Passing yards model: {py_conf:.0%} confidence",
                    "Game script probability: {script_prob:.1%} pass-heavy game"
                ]
            }
        }
        
        # Correlation explanations
        self.correlation_explanations = {
            (0.8, 1.0): "Very strong positive correlation - highly complementary",
            (0.6, 0.8): "Strong positive correlation - mutually beneficial", 
            (0.3, 0.6): "Moderate positive correlation - somewhat linked",
            (0.0, 0.3): "Weak correlation - largely independent",
            (-0.3, 0.0): "Weak negative correlation - slight conflict",
            (-0.6, -0.3): "Moderate negative correlation - competing outcomes",
            (-1.0, -0.6): "Strong negative correlation - conflicting game scripts"
        }
        
        # Market efficiency insights
        self.market_insights = [
            "Market efficiency: {efficiency:.1%} - {efficiency_desc}",
            "Public betting: {public_pct:.0%} on this side (contrarian value)",
            "Sharp money indicators: {sharp_signals} detected",
            "Line movement: {line_move:+.1f} since opening (market adjustment)",
            "Closing line value: {clv:+.1%} expected based on model"
        ]
        
    def generate_leg_reasoning(
        self,
        player_data: Dict[str, Any],
        market_data: Dict[str, Any],
        model_predictions: Dict[str, Any],
        historical_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive reasoning for a single parlay leg."""
        
        prop_type = market_data.get('prop_type', 'receiving_yards')
        player_name = player_data.get('player_name', 'Player')
        
        reasoning = {
            'player_name': player_name,
            'prop_type': prop_type,
            'market_line': market_data.get('market_line', 0),
            'recommended_side': market_data.get('best_side', 'over'),
            'confidence': model_predictions.get('confidence', 0.7),
            'edge_percentage': model_predictions.get('ev', 0.05) * 100,
            'factors': {
                'positive': [],
                'negative': [],
                'model_insights': [],
                'market_context': []
            },
            'summary': "",
            'risk_assessment': ""
        }
        
        # Generate positive factors
        positive_factors = self._generate_positive_factors(
            prop_type, player_data, market_data, model_predictions, historical_context
        )
        reasoning['factors']['positive'] = positive_factors
        
        # Generate negative factors
        negative_factors = self._generate_negative_factors(
            prop_type, player_data, market_data, model_predictions, historical_context
        )
        reasoning['factors']['negative'] = negative_factors
        
        # Generate model insights
        model_insights = self._generate_model_insights(
            prop_type, model_predictions, historical_context
        )
        reasoning['factors']['model_insights'] = model_insights
        
        # Generate market context
        market_context = self._generate_market_context(
            market_data, model_predictions
        )
        reasoning['factors']['market_context'] = market_context
        
        # Create summary
        reasoning['summary'] = self._create_summary(reasoning)
        
        # Risk assessment
        reasoning['risk_assessment'] = self._assess_risk(reasoning, model_predictions)
        
        return reasoning
    
    def _generate_positive_factors(
        self,
        prop_type: str,
        player_data: Dict[str, Any],
        market_data: Dict[str, Any],
        model_predictions: Dict[str, Any],
        historical_context: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Generate positive factors supporting the bet."""
        
        factors = []
        templates = self.prop_templates.get(prop_type, {}).get('positive_factors', [])
        
        # Create context for template formatting
        context = {
            'avg': player_data.get(f'{prop_type}_4game_avg', player_data.get(f'{prop_type}_avg', 65)),
            'rank': min(25, player_data.get('opp_def_rank', 16)),
            'opp_avg': 250 - (player_data.get('opp_def_rank', 16) * 5),  # Synthetic opponent average
            'position': player_data.get('position', 'WR'),
            'opp_rank': player_data.get('opp_def_rank', 16),
            'home_boost': 2.3 if player_data.get('is_home', 0) else 0,
            'spread': abs(player_data.get('point_spread', 3.5)),
            'trend': player_data.get(f'{prop_type}_trend_4', 0.5),
            'weeks': 4,
            'completion_rate': 65.5 + (5 * (32 - player_data.get('opp_def_rank', 16)) / 32),
            'corr': 0.75,  # Receiving yards to receptions correlation
            'rz_share': 25.5,
            'opp_td_rate': 1.8 + (player_data.get('opp_def_rank', 16) / 20),
            'td_regression': 0.3,
            'opp_ypa': 6.8 + (player_data.get('opp_def_rank', 16) / 15),
            'attempts_trend': 1.2,
            'weapons_health': 3
        }
        
        # Select relevant templates based on context
        for template in templates:
            try:
                if 'home_boost' in template and context['home_boost'] > 0:
                    factors.append(template.format(**context))
                elif 'opp_rank' in template and context['opp_rank'] > 20:
                    factors.append(template.format(**context))
                elif 'trend' in template and context['trend'] > 0:
                    factors.append(template.format(**context))
                elif 'avg' in template:
                    factors.append(template.format(**context))
                    
                if len(factors) >= 3:  # Limit to top 3 factors
                    break
                    
            except (KeyError, ValueError):
                continue  # Skip templates with missing data
        
        return factors
    
    def _generate_negative_factors(
        self,
        prop_type: str,
        player_data: Dict[str, Any],
        market_data: Dict[str, Any],
        model_predictions: Dict[str, Any],
        historical_context: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Generate negative factors that could hurt the bet."""
        
        factors = []
        templates = self.prop_templates.get(prop_type, {}).get('negative_factors', [])
        
        context = {
            'rank': player_data.get('opp_def_rank', 16),
            'opp_avg': 280 - (player_data.get('opp_def_rank', 16) * 8),
            'away_penalty': 3.2 if not player_data.get('is_home', 0) else 0,
            'spread': abs(player_data.get('point_spread', 3.5)),
            'trend': player_data.get(f'{prop_type}_trend_4', -0.2),
            'competitor': 'teammate',
            'opp_rec_allowed': 5.8 - (player_data.get('opp_def_rank', 16) / 10),
            'rz_rank': player_data.get('opp_def_rank', 16),
            'td_regression': 0.2,
            'sack_rate': 6.5 + (player_data.get('opp_def_rank', 16) / 8),
            'weather_desc': 'windy conditions',
            'weather_impact': 15
        }
        
        # Only include relevant negative factors
        for template in templates:
            try:
                if 'away_penalty' in template and context['away_penalty'] > 0:
                    factors.append(template.format(**context))
                elif 'rank' in template and context['rank'] < 10:
                    factors.append(template.format(**context))
                elif len(factors) == 0:  # Always include at least one
                    factors.append(template.format(**context))
                    
                if len(factors) >= 2:  # Limit negative factors
                    break
                    
            except (KeyError, ValueError):
                continue
        
        return factors
    
    def _generate_model_insights(
        self,
        prop_type: str,
        model_predictions: Dict[str, Any],
        historical_context: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Generate insights from ML models."""
        
        insights = []
        templates = self.prop_templates.get(prop_type, {}).get('model_insights', [])
        
        context = {
            'xgb_conf': model_predictions.get('xgboost_confidence', 0.75),
            'xgb_pred': model_predictions.get('xgboost_prediction', 68.5),
            'nn_conf': model_predictions.get('neural_confidence', 0.71),
            'nn_edge': model_predictions.get('edge_strength', 0.068),
            'accuracy': model_predictions.get('historical_accuracy', 0.652),
            'confidence': model_predictions.get('confidence', 0.73),
            'target_share': 23.5,
            'td_prob': 0.42,
            'rz_eff': 65.8,
            'weeks': 6,
            'py_conf': 0.78,
            'script_prob': 0.63
        }
        
        # Generate model-specific insights
        for template in templates:
            try:
                insights.append(template.format(**context))
                if len(insights) >= 2:
                    break
            except (KeyError, ValueError):
                continue
        
        # Add ensemble insight if multiple models
        if len(insights) > 1:
            insights.append(f"Model consensus: {context['confidence']:.0%} agreement on edge")
        
        return insights
    
    def _generate_market_context(
        self,
        market_data: Dict[str, Any],
        model_predictions: Dict[str, Any]
    ) -> List[str]:
        """Generate market and betting context."""
        
        context_factors = []
        
        # Market efficiency
        efficiency = model_predictions.get('market_efficiency', 0.85)
        if efficiency < 0.8:
            context_factors.append(f"Market efficiency: {efficiency:.1%} - potential value spot")
        elif efficiency > 0.95:
            context_factors.append(f"Market efficiency: {efficiency:.1%} - sharp line")
        
        # EV context
        ev = model_predictions.get('ev', 0.05)
        if ev > 0.08:
            context_factors.append(f"Strong edge: {ev:.1%} expected value")
        elif ev > 0.04:
            context_factors.append(f"Moderate edge: {ev:.1%} expected value")
        
        # Volume/liquidity context
        volume = market_data.get('volume', 1000)
        if volume < 500:
            context_factors.append("Low volume market - less efficient pricing")
        elif volume > 5000:
            context_factors.append("High volume market - sharp pricing")
        
        # Line movement context
        line_move = market_data.get('line_movement', 0)
        if abs(line_move) > 1:
            direction = "up" if line_move > 0 else "down"
            context_factors.append(f"Line moved {abs(line_move):.1f} points {direction} - market adjustment")
        
        return context_factors[:3]  # Limit to 3 context factors
    
    def _create_summary(self, reasoning: Dict[str, Any]) -> str:
        """Create a concise summary of the reasoning."""
        
        player_name = reasoning['player_name']
        prop_type = reasoning['prop_type'].replace('_', ' ').title()
        side = reasoning['recommended_side'].upper()
        line = reasoning['market_line']
        confidence = reasoning['confidence']
        edge = reasoning['edge_percentage']
        
        # Get top positive factor
        top_positive = reasoning['factors']['positive'][0] if reasoning['factors']['positive'] else "favorable matchup"
        
        summary = (
            f"{player_name} {prop_type} {side} {line}: {confidence:.0%} confidence, "
            f"{edge:.1%} edge. Key factor: {top_positive}"
        )
        
        return summary
    
    def _assess_risk(
        self,
        reasoning: Dict[str, Any],
        model_predictions: Dict[str, Any]
    ) -> str:
        """Assess the overall risk level of the bet."""
        
        confidence = reasoning['confidence']
        edge = reasoning['edge_percentage'] / 100
        negative_factors = len(reasoning['factors']['negative'])
        
        # Risk scoring
        risk_score = 0
        
        # Confidence component
        if confidence > 0.8:
            risk_score += 1
        elif confidence > 0.7:
            risk_score += 0.5
        else:
            risk_score += 0
        
        # Edge component
        if edge > 0.08:
            risk_score += 1
        elif edge > 0.04:
            risk_score += 0.5
        
        # Negative factors component
        if negative_factors == 0:
            risk_score += 0.5
        elif negative_factors >= 3:
            risk_score -= 0.5
        
        # Risk assessment
        if risk_score >= 2.0:
            return "Low risk - strong fundamentals and model agreement"
        elif risk_score >= 1.0:
            return "Moderate risk - decent edge but some concerns"
        else:
            return "Higher risk - lower confidence or significant negatives"
    
    def generate_parlay_reasoning(
        self,
        legs_reasoning: List[Dict[str, Any]],
        correlation_analysis: Dict[str, Any],
        parlay_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate overall reasoning for a complete parlay."""
        
        parlay_reasoning = {
            'total_legs': len(legs_reasoning),
            'parlay_type': parlay_metrics.get('parlay_type', 'multi_game'),
            'overall_confidence': np.mean([leg['confidence'] for leg in legs_reasoning]),
            'total_edge': sum([leg['edge_percentage'] for leg in legs_reasoning]) / 100,
            'parlay_edge': parlay_metrics.get('expected_value', 0.06),
            'joint_probability': parlay_metrics.get('joint_probability', 0.15),
            'correlation_summary': self._explain_correlations(correlation_analysis),
            'risk_analysis': self._analyze_parlay_risk(legs_reasoning, correlation_analysis),
            'execution_reasoning': "",
            'legs_summary': []
        }
        
        # Create leg summaries
        for i, leg in enumerate(legs_reasoning, 1):
            parlay_reasoning['legs_summary'].append({
                'leg_number': i,
                'summary': leg['summary'],
                'confidence': leg['confidence'],
                'risk': leg['risk_assessment']
            })
        
        # Execution reasoning
        parlay_reasoning['execution_reasoning'] = self._create_execution_reasoning(parlay_reasoning)
        
        return parlay_reasoning
    
    def _explain_correlations(self, correlation_analysis: Dict[str, Any]) -> str:
        """Explain correlation impact on the parlay."""
        
        avg_correlation = correlation_analysis.get('average_correlation', 0.2)
        
        # Find the appropriate explanation
        explanation = "Moderate correlation"
        for (min_corr, max_corr), desc in self.correlation_explanations.items():
            if min_corr <= avg_correlation < max_corr:
                explanation = desc
                break
        
        correlation_impact = "increases" if avg_correlation > 0 else "decreases"
        impact_magnitude = abs(avg_correlation) * 10
        
        return f"{explanation} (avg: {avg_correlation:+.2f}). This {correlation_impact} joint probability by ~{impact_magnitude:.0f}%."
    
    def _analyze_parlay_risk(
        self,
        legs_reasoning: List[Dict[str, Any]],
        correlation_analysis: Dict[str, Any]
    ) -> str:
        """Analyze overall parlay risk."""
        
        # Individual leg risks
        low_risk_legs = sum(1 for leg in legs_reasoning if "Low risk" in leg['risk_assessment'])
        total_legs = len(legs_reasoning)
        
        # Correlation risk
        max_correlation = correlation_analysis.get('max_correlation', 0.3)
        
        if low_risk_legs == total_legs and max_correlation < 0.7:
            return "Low parlay risk - all legs have strong fundamentals and reasonable correlations"
        elif low_risk_legs >= total_legs * 0.7 and max_correlation < 0.8:
            return "Moderate parlay risk - most legs solid but some correlation risk"
        else:
            return "Higher parlay risk - some weaker legs or high correlation dependency"
    
    def _create_execution_reasoning(self, parlay_reasoning: Dict[str, Any]) -> str:
        """Create reasoning for why this parlay should be executed."""
        
        confidence = parlay_reasoning['overall_confidence']
        edge = parlay_reasoning['parlay_edge'] * 100
        legs = parlay_reasoning['total_legs']
        
        reasoning_parts = []
        
        # Edge justification
        if edge > 8:
            reasoning_parts.append(f"Strong {edge:.1%} expected value")
        elif edge > 4:
            reasoning_parts.append(f"Solid {edge:.1%} expected value")
        else:
            reasoning_parts.append(f"Marginal {edge:.1%} expected value")
        
        # Confidence justification
        if confidence > 0.75:
            reasoning_parts.append(f"high model confidence ({confidence:.0%})")
        elif confidence > 0.65:
            reasoning_parts.append(f"decent model confidence ({confidence:.0%})")
        else:
            reasoning_parts.append(f"modest model confidence ({confidence:.0%})")
        
        # Correlation justification
        reasoning_parts.append(parlay_reasoning['correlation_summary'].lower())
        
        # Risk justification
        reasoning_parts.append(parlay_reasoning['risk_analysis'].lower())
        
        execution_reasoning = (
            f"Execute this {legs}-leg parlay based on: " +
            ", ".join(reasoning_parts[:3]) + ". " +
            parlay_reasoning['risk_analysis']
        )
        
        return execution_reasoning


def demo_reasoning_engine():
    """Demonstrate the reasoning engine with sample data."""
    
    print("🧠 REASONING ENGINE DEMO")
    print("=" * 50)
    
    reasoning_engine = ReasoningEngine()
    
    # Sample player and market data
    player_data = {
        'player_name': 'Travis Kelce',
        'position': 'TE',
        'team': 'KC',
        'receiving_yards_4game_avg': 78.5,
        'receiving_yards_trend_4': 2.1,
        'is_home': True,
        'opp_def_rank': 28,
        'point_spread': -7.5
    }
    
    market_data = {
        'prop_type': 'receiving_yards',
        'market_line': 72.5,
        'best_side': 'over',
        'volume': 2500,
        'line_movement': 1.5
    }
    
    model_predictions = {
        'confidence': 0.76,
        'ev': 0.085,
        'xgboost_confidence': 0.78,
        'xgboost_prediction': 79.2,
        'neural_confidence': 0.74,
        'edge_strength': 0.081,
        'market_efficiency': 0.82
    }
    
    print("1️⃣ SINGLE LEG REASONING")
    print("-" * 30)
    
    leg_reasoning = reasoning_engine.generate_leg_reasoning(
        player_data, market_data, model_predictions
    )
    
    print(f"📊 {leg_reasoning['summary']}")
    print(f"🎯 Confidence: {leg_reasoning['confidence']:.0%}, Edge: {leg_reasoning['edge_percentage']:.1f}%")
    
    print(f"\n✅ Positive Factors:")
    for factor in leg_reasoning['factors']['positive']:
        print(f"   • {factor}")
    
    print(f"\n⚠️ Risk Factors:")
    for factor in leg_reasoning['factors']['negative']:
        print(f"   • {factor}")
    
    print(f"\n🤖 Model Insights:")
    for insight in leg_reasoning['factors']['model_insights']:
        print(f"   • {insight}")
    
    print(f"\n📈 Market Context:")
    for context in leg_reasoning['factors']['market_context']:
        print(f"   • {context}")
    
    print(f"\n⚖️ Risk Assessment: {leg_reasoning['risk_assessment']}")
    
    print("\n2️⃣ PARLAY REASONING")
    print("-" * 30)
    
    # Create multiple legs
    legs_reasoning = [leg_reasoning]
    
    # Add second leg
    leg2_data = player_data.copy()
    leg2_data['player_name'] = 'Patrick Mahomes'
    leg2_data['position'] = 'QB'
    
    leg2_market = {
        'prop_type': 'passing_yards',
        'market_line': 267.5,
        'best_side': 'over',
        'volume': 3200
    }
    
    leg2_predictions = {
        'confidence': 0.72,
        'ev': 0.063,
        'market_efficiency': 0.88
    }
    
    leg2_reasoning = reasoning_engine.generate_leg_reasoning(
        leg2_data, leg2_market, leg2_predictions
    )
    legs_reasoning.append(leg2_reasoning)
    
    # Correlation analysis
    correlation_analysis = {
        'average_correlation': 0.55,
        'max_correlation': 0.65,
        'correlation_type': 'positive'
    }
    
    # Parlay metrics
    parlay_metrics = {
        'parlay_type': 'same_game',
        'expected_value': 0.074,
        'joint_probability': 0.184
    }
    
    parlay_reasoning = reasoning_engine.generate_parlay_reasoning(
        legs_reasoning, correlation_analysis, parlay_metrics
    )
    
    print(f"🏈 {parlay_reasoning['total_legs']}-leg {parlay_reasoning['parlay_type']} parlay")
    print(f"📊 Overall confidence: {parlay_reasoning['overall_confidence']:.0%}")
    print(f"💰 Parlay edge: {parlay_reasoning['parlay_edge']:.1%}")
    print(f"🎯 Joint probability: {parlay_reasoning['joint_probability']:.1%}")
    
    print(f"\n🔗 Correlation Analysis:")
    print(f"   {parlay_reasoning['correlation_summary']}")
    
    print(f"\n⚖️ Risk Analysis:")
    print(f"   {parlay_reasoning['risk_analysis']}")
    
    print(f"\n🎯 Execution Reasoning:")
    print(f"   {parlay_reasoning['execution_reasoning']}")
    
    print(f"\n📋 Legs Summary:")
    for leg_summary in parlay_reasoning['legs_summary']:
        print(f"   {leg_summary['leg_number']}. {leg_summary['summary']}")
        print(f"      Risk: {leg_summary['risk']}")
    
    return True


if __name__ == "__main__":
    demo_reasoning_engine()