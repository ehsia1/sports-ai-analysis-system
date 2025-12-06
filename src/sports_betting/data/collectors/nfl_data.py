"""NFL data collector using nfl_data_py."""

import warnings
from datetime import datetime
from typing import Dict, List, Optional

import nfl_data_py as nfl
import pandas as pd
from sqlalchemy.orm import Session

from ...database import Game, InjuryReport, Player, RosterChange, Team, get_session

# Suppress nfl_data_py warnings
warnings.filterwarnings("ignore", module="nfl_data_py")


class NFLDataCollector:
    """Collector for NFL data using nfl_data_py."""

    def __init__(self):
        self.current_season = datetime.now().year
        if datetime.now().month < 9:  # Before September
            self.current_season -= 1

    def _load_weekly_data_with_ngs(self, year: int) -> pd.DataFrame:
        """Load weekly data, using NGS for 2025+ since yearly file isn't available mid-season."""
        if year >= 2025:
            # Use NGS data for current season
            ngs_rush = nfl.import_ngs_data('rushing', [year])
            ngs_rec = nfl.import_ngs_data('receiving', [year])
            ngs_pass = nfl.import_ngs_data('passing', [year])

            # Filter out season totals (week 0)
            ngs_rush = ngs_rush[(ngs_rush['week'] > 0) & (ngs_rush['season_type'] == 'REG')]
            ngs_rec = ngs_rec[(ngs_rec['week'] > 0) & (ngs_rec['season_type'] == 'REG')]
            ngs_pass = ngs_pass[(ngs_pass['week'] > 0) & (ngs_pass['season_type'] == 'REG')]

            # Transform and merge
            rush_df = ngs_rush.rename(columns={
                'player_gsis_id': 'player_id', 'team_abbr': 'recent_team',
                'rush_yards': 'rushing_yards', 'rush_attempts': 'carries',
                'rush_touchdowns': 'rushing_tds',
            })[['season', 'week', 'player_id', 'player_display_name', 'recent_team',
                'carries', 'rushing_yards', 'rushing_tds']].copy()

            rec_df = ngs_rec.rename(columns={
                'player_gsis_id': 'player_id', 'team_abbr': 'recent_team',
                'yards': 'receiving_yards', 'rec_touchdowns': 'receiving_tds',
            })[['season', 'week', 'player_id', 'player_display_name', 'recent_team',
                'receptions', 'targets', 'receiving_yards', 'receiving_tds']].copy()

            pass_df = ngs_pass.rename(columns={
                'player_gsis_id': 'player_id', 'team_abbr': 'recent_team',
                'pass_yards': 'passing_yards', 'pass_touchdowns': 'passing_tds',
            })[['season', 'week', 'player_id', 'player_display_name', 'recent_team',
                'attempts', 'completions', 'passing_yards', 'passing_tds']].copy()

            combined = rush_df.merge(
                rec_df, on=['season', 'week', 'player_id', 'player_display_name', 'recent_team'], how='outer'
            ).merge(
                pass_df, on=['season', 'week', 'player_id', 'player_display_name', 'recent_team'], how='outer'
            )

            # Fill NaN stats with 0
            stat_cols = ['carries', 'rushing_yards', 'rushing_tds', 'receptions', 'targets',
                         'receiving_yards', 'receiving_tds', 'attempts', 'completions',
                         'passing_yards', 'passing_tds']
            for col in stat_cols:
                if col in combined.columns:
                    combined[col] = combined[col].fillna(0)

            combined['player_name'] = combined['player_display_name']
            combined['season_type'] = 'REG'
            return combined
        else:
            return nfl.import_weekly_data(years=[year])

    def collect_teams(self) -> int:
        """Collect and store NFL team data."""
        teams_df = nfl.import_team_desc()
        stored_count = 0

        with get_session() as session:
            for _, team_row in teams_df.iterrows():
                # Check if team already exists
                existing_team = session.query(Team).filter_by(
                    abbreviation=team_row['team_abbr']
                ).first()

                if not existing_team:
                    # Extract city and team name
                    full_name = team_row['team_name']  # e.g., "Arizona Cardinals"
                    nick_name = team_row['team_nick']  # e.g., "Cardinals"

                    # City is everything before the nickname
                    city = full_name.replace(nick_name, '').strip()

                    team = Team(
                        name=full_name,
                        abbreviation=team_row['team_abbr'],
                        city=city,
                        conference=team_row['team_conf'],
                        division=team_row['team_division']
                    )
                    session.add(team)
                    stored_count += 1

            session.commit()

        return stored_count

    def collect_schedule(self, years: List[int] = None) -> int:
        """Collect and store NFL schedule data."""
        if years is None:
            years = [self.current_season]

        stored_count = 0
        
        for year in years:
            schedule_df = nfl.import_schedules(years=[year])
            
            with get_session() as session:
                for _, game_row in schedule_df.iterrows():
                    if self._store_game(session, game_row, year):
                        stored_count += 1
        
        return stored_count

    def collect_player_stats(self, years: List[int] = None, stat_type: str = "weekly") -> int:
        """Collect and store player statistics (using NGS for 2025)."""
        if years is None:
            years = [self.current_season]

        stored_count = 0

        for year in years:
            if stat_type == "weekly":
                stats_df = self._load_weekly_data_with_ngs(year)
            elif stat_type == "seasonal":
                stats_df = nfl.import_seasonal_data(years=[year])
            else:
                raise ValueError(f"Invalid stat_type: {stat_type}")

            with get_session() as session:
                stored_count += self._store_player_stats(session, stats_df, year)

        return stored_count

    def collect_rosters(self, years: List[int] = None) -> int:
        """Collect and store roster data."""
        if years is None:
            years = [self.current_season]

        stored_count = 0

        for year in years:
            rosters_df = nfl.import_seasonal_rosters([year])

            with get_session() as session:
                for _, player_row in rosters_df.iterrows():
                    if self._store_player(session, player_row):
                        stored_count += 1

        return stored_count

    def collect_injury_reports(self, years: List[int] = None) -> int:
        """Collect and store injury report data."""
        if years is None:
            years = [self.current_season]

        stored_count = 0

        try:
            injury_df = nfl.import_injuries(years=years)

            with get_session() as session:
                for _, injury_row in injury_df.iterrows():
                    if self._store_injury_report(session, injury_row):
                        stored_count += 1

            # Update player current_status based on latest injury reports
            self._update_player_statuses()

            return stored_count
        except Exception as e:
            print(f"Error collecting injury data: {e}")
            return 0

    def get_team_stats(self, year: int, week: int = None) -> pd.DataFrame:
        """Get team statistics for analysis (using NGS for 2025)."""
        if week:
            # Get weekly team stats up to specified week
            weekly_df = self._load_weekly_data_with_ngs(year)
            team_stats = weekly_df[weekly_df['week'] <= week].groupby('recent_team').agg({
                'passing_yards': 'mean',
                'rushing_yards': 'mean',
                'receiving_yards': 'mean',
                'passing_tds': 'mean',
                'rushing_tds': 'mean',
                'receiving_tds': 'mean',
                'targets': 'mean',
                'receptions': 'mean',
            }).reset_index()
        else:
            # Get seasonal team stats
            team_stats = nfl.import_team_desc(years=[year])

        return team_stats

    def get_player_weekly_stats(self, year: int, week: int = None) -> pd.DataFrame:
        """Get player weekly statistics (using NGS for 2025)."""
        weekly_df = self._load_weekly_data_with_ngs(year)

        if week:
            return weekly_df[weekly_df['week'] == week]

        return weekly_df

    def get_advanced_stats(self, year: int) -> Dict[str, pd.DataFrame]:
        """Get advanced NFL statistics."""
        stats = {}
        
        try:
            # Next Gen Stats
            stats['ngs_passing'] = nfl.import_ngs_data('passing', years=[year])
            stats['ngs_rushing'] = nfl.import_ngs_data('rushing', years=[year])
            stats['ngs_receiving'] = nfl.import_ngs_data('receiving', years=[year])
        except Exception as e:
            print(f"Error collecting NGS data: {e}")
        
        try:
            # PFF data (if available)
            stats['pff_passing'] = nfl.import_pff_data('passing', years=[year])
            stats['pff_rushing'] = nfl.import_pff_data('rushing', years=[year])
            stats['pff_receiving'] = nfl.import_pff_data('receiving', years=[year])
        except Exception as e:
            print(f"Error collecting PFF data: {e}")
        
        return stats

    def _store_game(self, session: Session, game_row: pd.Series, year: int) -> bool:
        """Store a single game record."""
        try:
            # Find home and away teams
            home_team = session.query(Team).filter_by(abbreviation=game_row['home_team']).first()
            away_team = session.query(Team).filter_by(abbreviation=game_row['away_team']).first()
            
            if not home_team or not away_team:
                return False
            
            # Check if game already exists
            existing_game = session.query(Game).filter_by(
                season=year,
                week=game_row['week'],
                home_team_id=home_team.id,
                away_team_id=away_team.id
            ).first()
            
            if existing_game:
                # Update existing game with any new data
                if pd.notna(game_row['home_score']):
                    existing_game.home_score = int(game_row['home_score'])
                if pd.notna(game_row['away_score']):
                    existing_game.away_score = int(game_row['away_score'])
                if pd.notna(game_row['temp']):
                    existing_game.temperature = float(game_row['temp'])
                if pd.notna(game_row['wind']):
                    existing_game.wind_speed = float(game_row['wind'])
                
                existing_game.is_completed = pd.notna(game_row['home_score'])
                existing_game.is_dome = game_row.get('roof', '') in ['dome', 'closed']
                
                return False  # Didn't create new record
            
            # Create new game
            game = Game(
                external_id=game_row.get('game_id'),
                season=year,
                week=int(game_row['week']),
                season_type=game_row.get('season_type', 'REG'),
                game_date=pd.to_datetime(game_row['gameday']),
                home_team_id=home_team.id,
                away_team_id=away_team.id,
                home_score=int(game_row['home_score']) if pd.notna(game_row['home_score']) else None,
                away_score=int(game_row['away_score']) if pd.notna(game_row['away_score']) else None,
                is_completed=pd.notna(game_row['home_score']),
                temperature=float(game_row['temp']) if pd.notna(game_row['temp']) else None,
                wind_speed=float(game_row['wind']) if pd.notna(game_row['wind']) else None,
                is_dome=game_row.get('roof', '') in ['dome', 'closed'],
            )
            
            session.add(game)
            return True
            
        except Exception as e:
            print(f"Error storing game: {e}")
            return False

    def _store_player(self, session: Session, player_row: pd.Series) -> bool:
        """Store a single player record."""
        try:
            # Find team
            team = session.query(Team).filter_by(abbreviation=player_row['team']).first()
            if not team:
                return False
            
            # Check if player already exists
            existing_player = session.query(Player).filter_by(
                external_id=player_row['player_id']
            ).first()
            
            if existing_player:
                # Update existing player
                existing_player.team_id = team.id
                existing_player.position = player_row['position']
                existing_player.jersey_number = int(player_row['jersey_number']) if pd.notna(player_row['jersey_number']) else None
                existing_player.height = int(player_row['height']) if pd.notna(player_row['height']) else None
                existing_player.weight = int(player_row['weight']) if pd.notna(player_row['weight']) else None
                existing_player.experience = int(player_row['years_exp']) if pd.notna(player_row['years_exp']) else None
                return False
            
            # Create new player
            player = Player(
                external_id=player_row['player_id'],
                name=f"{player_row['player_name']}",
                position=player_row['position'],
                team_id=team.id,
                jersey_number=int(player_row['jersey_number']) if pd.notna(player_row['jersey_number']) else None,
                height=int(player_row['height']) if pd.notna(player_row['height']) else None,
                weight=int(player_row['weight']) if pd.notna(player_row['weight']) else None,
                experience=int(player_row['years_exp']) if pd.notna(player_row['years_exp']) else None,
            )
            
            session.add(player)
            return True
            
        except Exception as e:
            print(f"Error storing player: {e}")
            return False

    def _store_player_stats(self, session: Session, stats_df: pd.DataFrame, year: int) -> int:
        """Store player statistics (this would need a separate stats table)."""
        # For now, we'll just return the count
        # In practice, you'd want a separate PlayerStats table
        return len(stats_df)

    def _store_injury_report(self, session: Session, injury_row: pd.Series) -> bool:
        """Store a single injury report record."""
        try:
            # Find player by external_id (gsis_id in nfl-data-py)
            player = session.query(Player).filter_by(
                external_id=injury_row['gsis_id']
            ).first()

            if not player:
                # Try to find by name if external_id doesn't match
                player = session.query(Player).filter_by(
                    name=injury_row.get('full_name', '')
                ).first()

            if not player:
                return False

            # Parse report date (date_modified in nfl-data-py)
            report_date = pd.to_datetime(injury_row['date_modified']) if pd.notna(injury_row.get('date_modified')) else datetime.utcnow()

            # Get season and week from row
            season = int(injury_row['season'])
            week = int(injury_row['week'])

            # Parse practice status
            practice_status = injury_row.get('practice_status', '')
            if pd.isna(practice_status):
                practice_status = ''

            # Map practice status to simplified format
            # nfl-data-py uses: "Full Participation", "Limited Participation", "Did Not Participate"
            if 'Full' in practice_status:
                practice_participation = 'Full'
            elif 'Limited' in practice_status:
                practice_participation = 'Limited'
            elif 'Did Not Participate' in practice_status or 'DNP' in practice_status:
                practice_participation = 'DNP'
            else:
                practice_participation = None

            # Check if injury report already exists for this player/season/week
            existing_report = session.query(InjuryReport).filter_by(
                player_id=player.id,
                season=season,
                week=week
            ).first()

            # Handle injury_status - must not be None
            injury_status = injury_row.get('report_status')
            if pd.isna(injury_status) or injury_status is None:
                injury_status = 'Injured'  # Default for records without status

            # Handle injury fields that might be NaN
            primary_injury = injury_row.get('report_primary_injury')
            if pd.isna(primary_injury):
                primary_injury = None

            secondary_injury = injury_row.get('report_secondary_injury')
            if pd.isna(secondary_injury):
                secondary_injury = None

            if existing_report:
                # Update existing report
                existing_report.injury_status = injury_status
                existing_report.primary_injury = primary_injury
                existing_report.secondary_injury = secondary_injury
                existing_report.report_date = report_date
                # Store practice status in the Friday field (most recent)
                existing_report.practice_friday = practice_participation
                existing_report.is_active_report = True
                return False

            # Create new injury report
            injury_report = InjuryReport(
                player_id=player.id,
                report_date=report_date,
                season=season,
                week=week,
                injury_status=injury_status,
                primary_injury=primary_injury,
                secondary_injury=secondary_injury,
                # Store practice status in the Friday field (most recent)
                practice_friday=practice_participation,
                is_active_report=True,
            )

            session.add(injury_report)
            return True

        except Exception as e:
            print(f"Error storing injury report: {e}")
            return False

    def _update_player_statuses(self):
        """Update player current_status based on latest injury reports."""
        try:
            with get_session() as session:
                # Get all players with injury reports
                players_with_injuries = session.query(Player).join(InjuryReport).distinct().all()

                for player in players_with_injuries:
                    # Get the most recent injury report
                    latest_report = session.query(InjuryReport).filter_by(
                        player_id=player.id,
                        is_active_report=True
                    ).order_by(InjuryReport.report_date.desc()).first()

                    if latest_report:
                        player.current_status = latest_report.injury_status
                    else:
                        player.current_status = "Healthy"

                # Mark all players without injury reports as Healthy
                players_without_injuries = session.query(Player).filter(
                    ~Player.injury_reports.any()
                ).all()

                for player in players_without_injuries:
                    if not player.current_status:
                        player.current_status = "Healthy"

        except Exception as e:
            print(f"Error updating player statuses: {e}")

    def update_current_week_data(self) -> Dict[str, int]:
        """Update data for the current week."""
        results = {}

        # Update schedule
        results['games'] = self.collect_schedule([self.current_season])

        # Update rosters (with change tracking)
        results['players'], results['roster_changes'] = self.weekly_roster_refresh()

        # Update player stats
        results['stats'] = self.collect_player_stats([self.current_season])

        # Update injury reports
        results['injuries'] = self.collect_injury_reports([self.current_season])

        return results

    def weekly_roster_refresh(self, week: Optional[int] = None) -> tuple[int, int]:
        """
        Refresh rosters and track any team changes.

        Returns:
            tuple: (players_updated_count, roster_changes_count)
        """
        if week is None:
            # Determine current week (simplified - you may want more sophisticated logic)
            week = ((datetime.now() - datetime(self.current_season, 9, 1)).days // 7) + 1
            week = max(1, min(week, 18))  # Clamp between 1-18

        print(f"Refreshing rosters for {self.current_season} Week {week}...")

        players_updated = 0
        roster_changes = 0

        try:
            # Load current rosters from nfl-data-py
            current_rosters_df = nfl.import_seasonal_rosters([self.current_season])

            with get_session() as session:
                # Track all active player IDs in current rosters
                current_active_players = set()

                for _, player_row in current_rosters_df.iterrows():
                    player_external_id = player_row['player_id']
                    current_active_players.add(player_external_id)

                    # Find team
                    team = session.query(Team).filter_by(
                        abbreviation=player_row['team']
                    ).first()

                    if not team:
                        continue

                    # Find existing player
                    existing_player = session.query(Player).filter_by(
                        external_id=player_external_id
                    ).first()

                    if existing_player:
                        # Check if team changed
                        if existing_player.team_id != team.id:
                            old_team_id = existing_player.team_id

                            # Record roster change
                            roster_change = RosterChange(
                                player_id=existing_player.id,
                                change_date=datetime.now(),
                                season=self.current_season,
                                week=week,
                                change_type="Traded",  # Could be Trade/Signed/etc
                                from_team_id=old_team_id,
                                to_team_id=team.id,
                                notes=f"Team change detected during weekly roster refresh"
                            )
                            session.add(roster_change)
                            roster_changes += 1

                            # Update player team
                            existing_player.team_id = team.id
                            players_updated += 1

                        # Update player details
                        existing_player.position = player_row['position']
                        existing_player.jersey_number = int(player_row['jersey_number']) if pd.notna(player_row['jersey_number']) else None
                        existing_player.is_active = True

                    else:
                        # New player - create record
                        new_player = Player(
                            external_id=player_external_id,
                            name=player_row['player_name'],
                            position=player_row['position'],
                            team_id=team.id,
                            jersey_number=int(player_row['jersey_number']) if pd.notna(player_row['jersey_number']) else None,
                            height=int(player_row['height']) if pd.notna(player_row['height']) else None,
                            weight=int(player_row['weight']) if pd.notna(player_row['weight']) else None,
                            experience=int(player_row['years_exp']) if pd.notna(player_row['years_exp']) else None,
                            is_active=True
                        )
                        session.add(new_player)

                        # Record as new signing
                        session.flush()  # Get the new player ID
                        roster_change = RosterChange(
                            player_id=new_player.id,
                            change_date=datetime.now(),
                            season=self.current_season,
                            week=week,
                            change_type="Signed",
                            to_team_id=team.id,
                            notes="New player added to roster"
                        )
                        session.add(roster_change)
                        roster_changes += 1
                        players_updated += 1

                # Mark players not on current rosters as inactive
                all_players = session.query(Player).filter_by(is_active=True).all()
                for player in all_players:
                    if player.external_id and player.external_id not in current_active_players:
                        player.is_active = False

                        # Record release
                        roster_change = RosterChange(
                            player_id=player.id,
                            change_date=datetime.now(),
                            season=self.current_season,
                            week=week,
                            change_type="Released",
                            from_team_id=player.team_id,
                            notes="Player no longer on active roster"
                        )
                        session.add(roster_change)
                        roster_changes += 1
                        players_updated += 1

            print(f"Roster refresh complete: {players_updated} players updated, {roster_changes} roster changes tracked")
            return players_updated, roster_changes

        except Exception as e:
            print(f"Error during roster refresh: {e}")
            return 0, 0