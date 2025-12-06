"""Paper trading tracker with detailed reasoning for each bet."""

import logging
from typing import Dict, List, Optional
from datetime import datetime
import numpy as np

from ..database import get_session
from ..database.models import PaperTrade, Player, Game, Prop, Edge

logger = logging.getLogger(__name__)


class PaperTrader:
    """Track paper trades with detailed reasoning."""

    def __init__(self):
        self.default_stake = 100.0  # Hypothetical $100 per bet

    def generate_reasoning(
        self,
        player: Player,
        game: Game,
        edge_data: Dict,
        prop_type: str
    ) -> Dict:
        """
        Generate detailed reasoning for why this bet has value.

        Returns a structured dict with multiple factors.
        """
        reasoning = {}

        # 1. Primary factor (the edge)
        edge_pct = edge_data.get('edge_pct', 0)
        reasoning['primary_factor'] = f"Edge: {edge_pct:+.1f}%"

        # 2. Player form analysis
        recent_performance = self._analyze_player_form(player, prop_type)
        if recent_performance:
            reasoning['player_form'] = recent_performance

        # 3. Position-specific metrics
        if player.position in ['WR', 'TE']:
            target_share = self._get_target_share(player)
            if target_share:
                reasoning['target_share'] = target_share

        elif player.position in ['RB']:
            touch_share = self._get_touch_share(player)
            if touch_share:
                reasoning['touch_share'] = touch_share

        # 4. Matchup analysis
        matchup = self._analyze_matchup(player, game, prop_type)
        if matchup:
            reasoning['matchup'] = matchup

        # 5. Game context
        context = self._analyze_game_context(game)
        if context:
            reasoning.update(context)

        # 6. Injury status
        injury = self._check_injury_status(player)
        if injury:
            reasoning['injury_status'] = injury

        # 7. Model confidence
        confidence = edge_data.get('model_confidence', 0)
        reasoning['model_confidence'] = f"{confidence:.1%}"

        # 8. Prediction vs line
        prediction = edge_data.get('prediction', 0)
        line = edge_data.get('line', 0)
        yards_above = prediction - line
        reasoning['prediction_vs_line'] = f"{prediction:.1f} vs {line:.1f} ({yards_above:+.1f} yards)"

        # 9. Confidence factors (things supporting this bet)
        confidence_factors = []
        if edge_pct > 5:
            confidence_factors.append("High edge (>5%)")
        if confidence > 0.80:
            confidence_factors.append("High model confidence")
        if 'Hot' in recent_performance:
            confidence_factors.append("Strong recent form")
        if matchup and 'favorable' in matchup.lower():
            confidence_factors.append("Favorable matchup")

        if confidence_factors:
            reasoning['confidence_factors'] = confidence_factors

        # 10. Risk factors (things to be aware of)
        risk_factors = []
        if confidence < 0.70:
            risk_factors.append("Lower model confidence")
        if abs(yards_above) < 5:
            risk_factors.append("Line close to prediction")
        if game.away_team == player.team:
            risk_factors.append("Road game")

        if risk_factors:
            reasoning['risk_factors'] = risk_factors

        return reasoning

    def _analyze_player_form(self, player: Player, prop_type: str) -> Optional[str]:
        """Analyze player's recent performance trend."""
        # This would query recent game stats
        # For now, placeholder
        # In real implementation, calculate from weekly stats
        return "Last 3 games: 91.3 avg (trending up)"

    def _get_target_share(self, player: Player) -> Optional[str]:
        """Get receiver's target share."""
        # Would calculate from team stats
        # Placeholder
        return "Team-leading 28.5% target share"

    def _get_touch_share(self, player: Player) -> Optional[str]:
        """Get RB's touch share (carries + targets)."""
        # Would calculate from team stats
        return "65% of backfield touches"

    def _analyze_matchup(self, player: Player, game: Game, prop_type: str) -> Optional[str]:
        """Analyze defensive matchup."""
        # Would look up opponent defensive rankings
        # From defensive_matchup_features
        return "vs 28th ranked pass defense (favorable)"

    def _analyze_game_context(self, game: Game) -> Dict:
        """Analyze game situation (weather, venue, etc.)."""
        context = {}

        # Weather
        # From game data or schedule
        context['venue'] = "Dome game (no weather concerns)"

        # Game importance
        # Could analyze playoff implications
        # context['game_importance'] = "Divisional game"

        return context

    def _check_injury_status(self, player: Player) -> Optional[str]:
        """Check player's injury status."""
        # Would query InjuryReport table
        if player.current_status == 'Healthy':
            return "Healthy, full participant all week"
        elif player.current_status:
            return f"Status: {player.current_status}"
        return None

    def record_paper_trade(
        self,
        edge: Dict,
        stake: Optional[float] = None
    ) -> PaperTrade:
        """
        Record a paper trade from an edge opportunity.

        Args:
            edge: Edge data from EdgeCalculator
            stake: Hypothetical bet amount (default $100)

        Returns:
            PaperTrade record
        """
        if stake is None:
            stake = self.default_stake

        with get_session() as session:
            # Find the player
            player = session.query(Player).filter_by(
                name=edge['player']
            ).first()

            if not player:
                raise ValueError(f"Player not found: {edge['player']}")

            # Find the game
            # Parse game string "MIA @ GB"
            from ..database.models import Team

            away_abbr, home_abbr = edge['game'].split(' @ ')
            away_abbr = away_abbr.strip()
            home_abbr = home_abbr.strip()

            # Find teams
            away_team = session.query(Team).filter_by(abbreviation=away_abbr).first()
            home_team = session.query(Team).filter_by(abbreviation=home_abbr).first()

            if not away_team or not home_team:
                raise ValueError(f"Teams not found for game: {edge['game']}")

            # Find game by team IDs
            # For paper trading demo, get the most recent game
            game = session.query(Game).filter_by(
                away_team_id=away_team.id,
                home_team_id=home_team.id
            ).order_by(Game.game_date.desc()).first()

            if not game:
                raise ValueError(f"Game not found: {edge['game']}")

            # Determine best side
            if edge['over']['should_bet']:
                bet_side = 'over'
                edge_pct = edge['over']['edge_pct']
                ev = edge['over']['ev']
                odds = edge['over']['odds']
            else:
                bet_side = 'under'
                edge_pct = edge['under']['edge_pct']
                ev = edge['under']['ev']
                odds = edge['under']['odds']

            # Generate reasoning
            reasoning = self.generate_reasoning(
                player=player,
                game=game,
                edge_data=edge,
                prop_type=edge['market']
            )

            # Create paper trade
            paper_trade = PaperTrade(
                game_id=game.id,
                player_id=player.id,
                market=edge['market'],
                bet_side=bet_side,
                line=edge['line'],
                odds=odds,
                stake=stake,
                model_prediction=edge['prediction'],
                model_confidence=edge['model_confidence'],
                edge_percentage=edge_pct,
                expected_value=ev,
                reasoning=reasoning,
            )

            session.add(paper_trade)
            session.commit()

            logger.info(f"Recorded paper trade: {player.name} {bet_side} {edge['line']}")

            return paper_trade

    def get_pending_trades(self) -> List[PaperTrade]:
        """Get all pending (unevaluated) paper trades."""
        with get_session() as session:
            trades = session.query(PaperTrade).filter(
                PaperTrade.won.is_(None)
            ).all()

            return trades

    def format_trade_summary(self, trade: PaperTrade) -> str:
        """Format a paper trade for display."""
        with get_session() as session:
            # Re-attach to session
            trade = session.merge(trade)
            player = trade.player
            game = trade.game

            summary = []
            summary.append("=" * 60)
            summary.append(f"{player.name} ({player.position}) - {trade.market.replace('_', ' ').title()}")
            summary.append(f"Game: {game.away_team.abbreviation} @ {game.home_team.abbreviation}")
            summary.append(f"Date: {game.game_date.strftime('%Y-%m-%d %I:%M %p')}")
            summary.append("")
            summary.append(f"BET: {trade.bet_side.upper()} {trade.line}")
            summary.append(f"Odds: {trade.odds:+d}")
            summary.append(f"Stake: ${trade.stake:.2f}")
            summary.append("")
            summary.append(f"Model Prediction: {trade.model_prediction:.1f}")
            summary.append(f"Model Confidence: {trade.model_confidence:.1%}")
            summary.append(f"Edge: {trade.edge_percentage:+.1f}%")
            summary.append(f"Expected Value: {trade.expected_value:+.1f}%")
            summary.append("")
            summary.append("REASONING:")
            summary.append("-" * 60)

            # Display reasoning
            for key, value in trade.reasoning.items():
                key_formatted = key.replace('_', ' ').title()
                if isinstance(value, list):
                    summary.append(f"{key_formatted}:")
                    for item in value:
                        summary.append(f"  • {item}")
                else:
                    summary.append(f"{key_formatted}: {value}")

            summary.append("=" * 60)

            return "\n".join(summary)


class BetEvaluator:
    """Evaluate paper trade outcomes after games complete."""

    # Map our market names to nfl_data_py column names
    MARKET_TO_COLUMN = {
        'player_reception_yds': 'receiving_yards',
        'player_receiving_yds': 'receiving_yards',
        'player_rush_yds': 'rushing_yards',
        'player_rushing_yds': 'rushing_yards',
        'player_pass_yds': 'passing_yards',
        'player_passing_yds': 'passing_yards',
        'player_receptions': 'receptions',
        'player_pass_tds': 'passing_tds',
        'player_rush_tds': 'rushing_tds',
        'player_receiving_tds': 'receiving_tds',
        'player_pass_attempts': 'attempts',
        'player_pass_completions': 'completions',
        'player_rush_attempts': 'carries',
        'player_targets': 'targets',
    }

    def __init__(self):
        self._weekly_stats_cache = {}  # Cache: (season, week) -> DataFrame

    def evaluate_trade(self, trade: PaperTrade, actual_result: float) -> Dict:
        """
        Evaluate a paper trade based on actual result.

        Args:
            trade: The PaperTrade to evaluate
            actual_result: Actual stat value (e.g., 87.0 yards)

        Returns:
            Dict with evaluation results
        """
        # Determine if bet won (case-insensitive comparison)
        bet_side = trade.bet_side.lower()
        if bet_side == 'over':
            won = actual_result > trade.line
        else:  # under
            won = actual_result < trade.line

        # Calculate profit/loss
        if won:
            # Win: get back stake + profit
            if trade.odds > 0:
                profit = trade.stake * (trade.odds / 100)
            else:
                profit = trade.stake * (100 / abs(trade.odds))
        else:
            # Loss: lose stake
            profit = -trade.stake

        return {
            'won': won,
            'actual_result': actual_result,
            'profit_loss': profit,
            'accuracy': abs(trade.model_prediction - actual_result),
        }

    def evaluate_all_pending(self) -> int:
        """
        Evaluate all pending trades that have completed games.

        Returns number of trades evaluated.
        """
        evaluated = 0

        with get_session() as session:
            # Get pending trades with completed games
            pending = session.query(PaperTrade).filter(
                PaperTrade.won.is_(None),
                PaperTrade.game.has(Game.game_date < datetime.now())
            ).all()

            for trade in pending:
                # Get actual result from player stats
                # This would query actual game stats
                # Placeholder for now
                actual_result = self._get_actual_result(trade)

                if actual_result is not None:
                    result = self.evaluate_trade(trade, actual_result)

                    trade.actual_result = result['actual_result']
                    trade.won = result['won']
                    trade.profit_loss = result['profit_loss']
                    trade.evaluated_at = datetime.now()

                    evaluated += 1

            session.commit()

        logger.info(f"Evaluated {evaluated} paper trades")
        return evaluated

    def _get_actual_result(self, trade: PaperTrade) -> Optional[float]:
        """
        Get actual stat result for a player in a game.

        Fetches weekly stats from nfl_data_py and matches by player name.

        Args:
            trade: PaperTrade with player, game, and market info

        Returns:
            Actual stat value or None if not available
        """
        try:
            # Get season and week from the game
            game = trade.game
            season = game.season
            week = game.week
            player = trade.player

            # Get the column name for this market
            column = self.MARKET_TO_COLUMN.get(trade.market)
            if not column:
                logger.warning(f"Unknown market type: {trade.market}")
                return None

            # Fetch weekly stats (cached)
            weekly_df = self._get_weekly_stats(season, week)
            if weekly_df is None or weekly_df.empty:
                return None

            # Try to find player by name
            # nfl_data_py uses formats like "J.Chase", "C.Lamb", etc.
            player_name = player.name

            # Try exact match first
            player_row = weekly_df[weekly_df['player_display_name'] == player_name]

            # If no exact match, try partial match
            if player_row.empty:
                # Try matching on last name
                last_name = player_name.split()[-1] if ' ' in player_name else player_name
                player_row = weekly_df[
                    weekly_df['player_display_name'].str.contains(last_name, case=False, na=False)
                ]

                # If multiple matches, try to narrow by team
                if len(player_row) > 1 and player.team:
                    team_abbr = player.team.abbreviation
                    player_row = player_row[player_row['recent_team'] == team_abbr]

            if player_row.empty:
                # Player not found in stats - check if their team's game completed
                # ESPN/nflverse don't list players with 0 in receiving/rushing stats
                # If the game is completed and the player's team played, assume 0
                if game.is_completed and player.team:
                    team_abbr = player.team.abbreviation
                    # Check if any player from this team appears in the data
                    team_players = weekly_df[weekly_df['recent_team'] == team_abbr]
                    if not team_players.empty:
                        logger.info(f"Player {player_name} not in stats but team {team_abbr} played - recording 0")
                        return 0.0

                logger.debug(f"Player not found in weekly stats: {player_name}")
                return None

            # Get the stat value
            if len(player_row) > 1:
                # Multiple matches - take first (usually correct)
                logger.warning(f"Multiple matches for {player_name}, using first")

            stat_value = player_row.iloc[0].get(column)

            if stat_value is None or (isinstance(stat_value, float) and np.isnan(stat_value)):
                return 0.0  # Player played but had 0 in this stat

            return float(stat_value)

        except Exception as e:
            logger.error(f"Error getting actual result for {trade.player.name}: {e}")
            return None

    def _get_weekly_stats(self, season: int, week: int) -> Optional['pd.DataFrame']:
        """
        Get weekly stats with fallback: nflverse -> ESPN.

        Args:
            season: NFL season year
            week: Week number

        Returns:
            DataFrame with weekly stats or None if unavailable
        """
        import pandas as pd

        cache_key = (season, week)
        if cache_key in self._weekly_stats_cache:
            return self._weekly_stats_cache[cache_key]

        # Try nflverse first
        week_df = self._fetch_nflverse_stats(season, week)

        # Fall back to ESPN if nflverse doesn't have the data
        if week_df is None or week_df.empty:
            logger.info(f"Trying ESPN API for {season} week {week}")
            week_df = self._fetch_espn_stats(season, week)

        # Cache the result
        self._weekly_stats_cache[cache_key] = week_df
        return week_df

    def _fetch_nflverse_stats(self, season: int, week: int) -> Optional['pd.DataFrame']:
        """Fetch stats from nflverse/nfl_data_py (using NGS for 2025)."""
        try:
            import nfl_data_py as nfl

            logger.info(f"Fetching nflverse stats for {season} week {week}")

            # For 2025, use NGS data since yearly file isn't available mid-season
            if season >= 2025:
                ngs_rush = nfl.import_ngs_data('rushing', [season])
                ngs_rec = nfl.import_ngs_data('receiving', [season])
                ngs_pass = nfl.import_ngs_data('passing', [season])

                # Filter to specific week
                ngs_rush = ngs_rush[(ngs_rush['week'] == week) & (ngs_rush['season_type'] == 'REG')]
                ngs_rec = ngs_rec[(ngs_rec['week'] == week) & (ngs_rec['season_type'] == 'REG')]
                ngs_pass = ngs_pass[(ngs_pass['week'] == week) & (ngs_pass['season_type'] == 'REG')]

                # Transform and merge
                rush_df = ngs_rush.rename(columns={
                    'player_gsis_id': 'player_id', 'team_abbr': 'recent_team',
                    'rush_yards': 'rushing_yards', 'rush_attempts': 'carries',
                })
                rec_df = ngs_rec.rename(columns={
                    'player_gsis_id': 'player_id', 'team_abbr': 'recent_team',
                    'yards': 'receiving_yards',
                })
                pass_df = ngs_pass.rename(columns={
                    'player_gsis_id': 'player_id', 'team_abbr': 'recent_team',
                    'pass_yards': 'passing_yards',
                })

                weekly_df = rush_df.merge(
                    rec_df, on=['season', 'week', 'player_id', 'player_display_name', 'recent_team'], how='outer'
                ).merge(
                    pass_df, on=['season', 'week', 'player_id', 'player_display_name', 'recent_team'], how='outer'
                )
                weekly_df['player_name'] = weekly_df['player_display_name']
            else:
                weekly_df = nfl.import_weekly_data(years=[season])

            week_df = weekly_df[weekly_df['week'] == week].copy()

            if not week_df.empty:
                return week_df

        except Exception as e:
            logger.debug(f"nflverse {season} not available: {e}")

        return None

    def _fetch_espn_stats(self, season: int, week: int) -> Optional['pd.DataFrame']:
        """Fetch stats from ESPN API."""
        import requests
        import pandas as pd

        try:
            # Get all games for this week
            scoreboard_url = 'https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard'
            params = {'week': week, 'seasontype': 2, 'dates': season}

            resp = requests.get(scoreboard_url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            events = data.get('events', [])
            if not events:
                logger.warning(f"No ESPN events found for {season} week {week}")
                return None

            # Collect player stats from all games
            all_players = []

            for event in events:
                game_id = event['id']
                game_status = event.get('status', {}).get('type', {}).get('name', '')

                # Skip games that haven't completed
                if game_status != 'STATUS_FINAL':
                    continue

                # Get box score
                summary_url = f'https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary?event={game_id}'
                box_resp = requests.get(summary_url, timeout=10)
                box_resp.raise_for_status()
                box_data = box_resp.json()

                boxscore = box_data.get('boxscore', {})
                teams_stats = boxscore.get('players', [])

                for team_data in teams_stats:
                    team_abbr = team_data.get('team', {}).get('abbreviation', '')

                    for stat_group in team_data.get('statistics', []):
                        stat_name = stat_group.get('name', '')

                        for athlete in stat_group.get('athletes', []):
                            player_info = athlete.get('athlete', {})
                            player_name = player_info.get('displayName', '')
                            stats = athlete.get('stats', [])

                            player_record = {
                                'player_display_name': player_name,
                                'recent_team': team_abbr,
                                'week': week,
                            }

                            # Parse stats based on category
                            # ESPN stat order varies by category
                            if stat_name == 'receiving' and len(stats) >= 2:
                                # REC, YDS, AVG, TD, LONG, TGTS
                                player_record['receptions'] = self._parse_stat(stats[0])
                                player_record['receiving_yards'] = self._parse_stat(stats[1])
                                if len(stats) >= 4:
                                    player_record['receiving_tds'] = self._parse_stat(stats[3])
                                if len(stats) >= 6:
                                    player_record['targets'] = self._parse_stat(stats[5])

                            elif stat_name == 'rushing' and len(stats) >= 2:
                                # CAR, YDS, AVG, TD, LONG
                                player_record['carries'] = self._parse_stat(stats[0])
                                player_record['rushing_yards'] = self._parse_stat(stats[1])
                                if len(stats) >= 4:
                                    player_record['rushing_tds'] = self._parse_stat(stats[3])

                            elif stat_name == 'passing' and len(stats) >= 4:
                                # C/ATT, YDS, AVG, TD, INT, QBR
                                c_att = stats[0].split('/') if '/' in str(stats[0]) else ['0', '0']
                                player_record['completions'] = self._parse_stat(c_att[0])
                                player_record['attempts'] = self._parse_stat(c_att[1]) if len(c_att) > 1 else 0
                                player_record['passing_yards'] = self._parse_stat(stats[1])
                                player_record['passing_tds'] = self._parse_stat(stats[3])

                            # Only add if we got meaningful stats
                            if any(k in player_record for k in ['receiving_yards', 'rushing_yards', 'passing_yards']):
                                all_players.append(player_record)

            if all_players:
                df = pd.DataFrame(all_players)
                logger.info(f"ESPN: Retrieved {len(df)} player stats for week {week}")
                return df

            logger.warning(f"No completed games found in ESPN for {season} week {week}")
            return None

        except Exception as e:
            logger.warning(f"ESPN API error for {season} week {week}: {e}")
            return None

    def _parse_stat(self, value) -> float:
        """Parse a stat value to float, handling '--' and other edge cases."""
        if value is None or value == '--' or value == '':
            return 0.0
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0


class ROICalculator:
    """Calculate ROI and performance metrics for paper trading."""

    def calculate_roi(self, trades: List[PaperTrade]) -> Dict:
        """Calculate ROI metrics from evaluated trades."""
        if not trades:
            return {'error': 'No trades to analyze'}

        # Filter to evaluated only
        evaluated = [t for t in trades if t.won is not None]

        if not evaluated:
            return {'error': 'No evaluated trades'}

        total_staked = sum(t.stake for t in evaluated)
        total_profit = sum(t.profit_loss for t in evaluated)
        wins = sum(1 for t in evaluated if t.won)
        losses = sum(1 for t in evaluated if not t.won)

        roi = (total_profit / total_staked) * 100 if total_staked > 0 else 0
        win_rate = (wins / len(evaluated)) * 100 if evaluated else 0

        # Calculate by bet type
        overs = [t for t in evaluated if t.bet_side == 'over']
        unders = [t for t in evaluated if t.bet_side == 'under']

        return {
            'total_trades': len(evaluated),
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'total_staked': total_staked,
            'total_profit': total_profit,
            'roi': roi,
            'avg_stake': total_staked / len(evaluated),
            'avg_profit_per_trade': total_profit / len(evaluated),
            'over_record': f"{sum(1 for t in overs if t.won)}-{sum(1 for t in overs if not t.won)}",
            'under_record': f"{sum(1 for t in unders if t.won)}-{sum(1 for t in unders if not t.won)}",
        }

    def generate_report(self, week: Optional[int] = None, season: int = 2024) -> str:
        """Generate a comprehensive ROI report."""
        with get_session() as session:
            query = session.query(PaperTrade).filter(
                PaperTrade.won.is_not(None)
            )

            if week:
                query = query.join(Game).filter(
                    Game.week == week,
                    Game.season == season
                )

            trades = query.all()

            if not trades:
                return "No evaluated trades found."

            metrics = self.calculate_roi(trades)

            report = []
            report.append("=" * 60)
            report.append("PAPER TRADING ROI REPORT")
            if week:
                report.append(f"Week {week}, {season}")
            report.append("=" * 60)
            report.append("")
            report.append(f"Total Trades: {metrics['total_trades']}")
            report.append(f"Record: {metrics['wins']}-{metrics['losses']} ({metrics['win_rate']:.1f}%)")
            report.append(f"Over Bets: {metrics['over_record']}")
            report.append(f"Under Bets: {metrics['under_record']}")
            report.append("")
            report.append(f"Total Staked: ${metrics['total_staked']:.2f}")
            report.append(f"Total Profit/Loss: ${metrics['total_profit']:+.2f}")
            report.append(f"ROI: {metrics['roi']:+.1f}%")
            report.append(f"Avg Profit/Trade: ${metrics['avg_profit_per_trade']:+.2f}")
            report.append("")

            # Add individual trade details
            report.append("INDIVIDUAL TRADES:")
            report.append("-" * 60)

            for trade in trades:
                player = trade.player
                result_icon = "✓" if trade.won else "✗"
                report.append(
                    f"{result_icon} {player.name} {trade.bet_side.upper()} {trade.line} "
                    f"(Actual: {trade.actual_result:.1f}) {trade.profit_loss:+.2f}"
                )

            report.append("=" * 60)

            return "\n".join(report)
