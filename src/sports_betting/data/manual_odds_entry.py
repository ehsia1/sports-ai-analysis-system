"""Manual odds entry from CSV for when API limits are reached."""

import csv
from datetime import datetime
from pathlib import Path
from typing import List, Dict
import logging

from ..database import get_session
from ..database.models import Prop, Book, Player, Game

logger = logging.getLogger(__name__)


class ManualOddsImporter:
    """Import odds from manually entered CSV files."""

    def __init__(self):
        self.manual_dir = Path.home() / ".sports_betting" / "manual_odds"
        self.manual_dir.mkdir(parents=True, exist_ok=True)

    def get_template_path(self) -> Path:
        """Get path to CSV template file."""
        return self.manual_dir / "odds_template.csv"

    def create_template(self):
        """Create a CSV template for manual entry."""
        template_path = self.get_template_path()

        if template_path.exists():
            logger.info(f"Template already exists at {template_path}")
            return template_path

        headers = [
            'date',  # YYYY-MM-DD
            'player_name',  # Exact name as in database
            'team',  # Team abbreviation
            'opponent',  # Opponent abbreviation
            'prop_type',  # receiving_yards, rushing_yards, etc.
            'line',  # Prop line (e.g., 74.5)
            'over_odds',  # American odds for over (e.g., -110)
            'under_odds',  # American odds for under (e.g., -110)
            'bookmaker',  # draftkings, fanduel, etc.
        ]

        example_rows = [
            {
                'date': '2024-12-01',
                'player_name': 'Tyreek Hill',
                'team': 'MIA',
                'opponent': 'GB',
                'prop_type': 'receiving_yards',
                'line': '74.5',
                'over_odds': '-110',
                'under_odds': '-110',
                'bookmaker': 'draftkings',
            },
            {
                'date': '2024-12-01',
                'player_name': 'CeeDee Lamb',
                'team': 'DAL',
                'opponent': 'NYG',
                'prop_type': 'receiving_yards',
                'line': '82.5',
                'over_odds': '-115',
                'under_odds': '-105',
                'bookmaker': 'draftkings',
            },
        ]

        with open(template_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(example_rows)

        logger.info(f"Created template at {template_path}")
        return template_path

    def import_from_csv(self, csv_path: Path) -> int:
        """
        Import manually entered odds from CSV.

        CSV Format:
            date,player_name,team,opponent,prop_type,line,over_odds,under_odds,bookmaker

        Returns:
            Number of props imported
        """
        if not csv_path.exists():
            logger.error(f"CSV file not found: {csv_path}")
            return 0

        imported_count = 0

        with get_session() as session:
            book_cache = {}
            player_cache = {}

            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)

                for row in reader:
                    try:
                        # Skip example rows
                        if row.get('player_name') in ['Tyreek Hill', 'CeeDee Lamb']:
                            continue

                        # Parse data
                        game_date = datetime.strptime(row['date'], '%Y-%m-%d')
                        player_name = row['player_name'].strip()
                        prop_type = row['prop_type'].strip()
                        line = float(row['line'])
                        over_odds = int(row['over_odds'])
                        under_odds = int(row['under_odds'])
                        bookmaker = row['bookmaker'].strip().lower()

                        # Get or create bookmaker
                        if bookmaker not in book_cache:
                            book = session.query(Book).filter_by(name=bookmaker).first()
                            if not book:
                                book = Book(
                                    name=bookmaker,
                                    display_name=bookmaker.title(),
                                    region='us'
                                )
                                session.add(book)
                                session.flush()
                            book_cache[bookmaker] = book
                        else:
                            book = book_cache[bookmaker]

                        # Find player
                        if player_name not in player_cache:
                            player = session.query(Player).filter_by(
                                name=player_name
                            ).first()
                            if not player:
                                logger.warning(f"Player not found: {player_name}")
                                continue
                            player_cache[player_name] = player
                        else:
                            player = player_cache[player_name]

                        # Find game (by team, opponent, and date)
                        team_abbr = row['team'].strip().upper()
                        opponent_abbr = row['opponent'].strip().upper()

                        game = session.query(Game).filter(
                            Game.game_date >= game_date.replace(hour=0, minute=0),
                            Game.game_date < game_date.replace(hour=23, minute=59),
                        ).filter(
                            ((Game.home_team == team_abbr) & (Game.away_team == opponent_abbr)) |
                            ((Game.home_team == opponent_abbr) & (Game.away_team == team_abbr))
                        ).first()

                        if not game:
                            logger.warning(
                                f"Game not found: {team_abbr} vs {opponent_abbr} on {game_date.date()}"
                            )
                            continue

                        # Check if prop already exists
                        existing = session.query(Prop).filter_by(
                            game_id=game.id,
                            player_id=player.id,
                            book_id=book.id,
                            market=prop_type,
                        ).filter(
                            Prop.timestamp >= datetime.now().replace(hour=0, minute=0, second=0)
                        ).first()

                        if existing:
                            # Update existing
                            existing.line = line
                            existing.over_odds = over_odds
                            existing.under_odds = under_odds
                            existing.timestamp = datetime.now()
                            logger.info(f"Updated prop for {player_name} {prop_type}")
                        else:
                            # Create new prop
                            prop = Prop(
                                game_id=game.id,
                                player_id=player.id,
                                book_id=book.id,
                                market=prop_type,
                                line=line,
                                over_odds=over_odds,
                                under_odds=under_odds,
                                timestamp=datetime.now()
                            )
                            session.add(prop)
                            logger.info(f"Created prop for {player_name} {prop_type}")

                        imported_count += 1

                    except Exception as e:
                        logger.error(f"Error importing row {row}: {e}")
                        continue

            session.commit()

        logger.info(f"Imported {imported_count} manual props from CSV")
        return imported_count

    def export_template_for_week(self, week: int, season: int) -> Path:
        """
        Export a template CSV with top players for a specific week.

        This makes manual entry easier by pre-filling player names.
        """
        export_path = self.manual_dir / f"week_{week}_{season}_template.csv"

        with get_session() as session:
            # Get games for this week
            games = session.query(Game).filter_by(
                season=season,
                week=week,
                season_type='REG'
            ).all()

            if not games:
                logger.warning(f"No games found for week {week} season {season}")
                return None

            # Prepare template rows (empty odds for user to fill in)
            rows = []

            for game in games:
                game_date = game.game_date.strftime('%Y-%m-%d')

                # You could query top players by targets/touches here
                # For now, just include basic game info
                rows.append({
                    'date': game_date,
                    'player_name': '',  # User fills in
                    'team': game.home_team,
                    'opponent': game.away_team,
                    'prop_type': 'receiving_yards',
                    'line': '',
                    'over_odds': '-110',
                    'under_odds': '-110',
                    'bookmaker': 'draftkings',
                })

        # Write CSV
        headers = ['date', 'player_name', 'team', 'opponent', 'prop_type',
                  'line', 'over_odds', 'under_odds', 'bookmaker']

        with open(export_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)

        logger.info(f"Exported template to {export_path}")
        return export_path


def create_manual_entry_template():
    """Create the manual odds entry template."""
    importer = ManualOddsImporter()
    template_path = importer.create_template()
    print(f"✓ Manual entry template created at: {template_path}")
    print("\nTo use:")
    print("1. Copy the template to a new file (e.g., odds_2024-12-01.csv)")
    print("2. Fill in player names and odds from sportsbook")
    print("3. Run: python -m scripts.import_manual_odds <your_file>.csv")
    return template_path


if __name__ == "__main__":
    create_manual_entry_template()
