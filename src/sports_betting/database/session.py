"""Database session management."""

from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from ..config import get_settings
from .models import Base

# Global engine instance
_engine = None
_session_factory = None


def get_engine():
    """Get the database engine."""
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_engine(
            settings.database_url,
            echo=False,  # Set to True for SQL debugging
            pool_pre_ping=True,
        )
    return _engine


def get_session_factory():
    """Get the session factory."""
    global _session_factory
    if _session_factory is None:
        engine = get_engine()
        _session_factory = sessionmaker(bind=engine)
    return _session_factory


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Get a database session with automatic cleanup."""
    session_factory = get_session_factory()
    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db():
    """Initialize the database by creating all tables."""
    engine = get_engine()
    Base.metadata.create_all(bind=engine)
    
    # Insert default data
    with get_session() as session:
        _insert_default_teams(session)
        _insert_default_books(session)


def _insert_default_teams(session: Session):
    """Insert default NFL teams."""
    from .models import Team
    
    # Check if teams already exist
    if session.query(Team).count() > 0:
        return
    
    nfl_teams = [
        {"name": "Arizona Cardinals", "abbreviation": "ARI", "city": "Arizona", "conference": "NFC", "division": "West"},
        {"name": "Atlanta Falcons", "abbreviation": "ATL", "city": "Atlanta", "conference": "NFC", "division": "South"},
        {"name": "Baltimore Ravens", "abbreviation": "BAL", "city": "Baltimore", "conference": "AFC", "division": "North"},
        {"name": "Buffalo Bills", "abbreviation": "BUF", "city": "Buffalo", "conference": "AFC", "division": "East"},
        {"name": "Carolina Panthers", "abbreviation": "CAR", "city": "Carolina", "conference": "NFC", "division": "South"},
        {"name": "Chicago Bears", "abbreviation": "CHI", "city": "Chicago", "conference": "NFC", "division": "North"},
        {"name": "Cincinnati Bengals", "abbreviation": "CIN", "city": "Cincinnati", "conference": "AFC", "division": "North"},
        {"name": "Cleveland Browns", "abbreviation": "CLE", "city": "Cleveland", "conference": "AFC", "division": "North"},
        {"name": "Dallas Cowboys", "abbreviation": "DAL", "city": "Dallas", "conference": "NFC", "division": "East"},
        {"name": "Denver Broncos", "abbreviation": "DEN", "city": "Denver", "conference": "AFC", "division": "West"},
        {"name": "Detroit Lions", "abbreviation": "DET", "city": "Detroit", "conference": "NFC", "division": "North"},
        {"name": "Green Bay Packers", "abbreviation": "GB", "city": "Green Bay", "conference": "NFC", "division": "North"},
        {"name": "Houston Texans", "abbreviation": "HOU", "city": "Houston", "conference": "AFC", "division": "South"},
        {"name": "Indianapolis Colts", "abbreviation": "IND", "city": "Indianapolis", "conference": "AFC", "division": "South"},
        {"name": "Jacksonville Jaguars", "abbreviation": "JAX", "city": "Jacksonville", "conference": "AFC", "division": "South"},
        {"name": "Kansas City Chiefs", "abbreviation": "KC", "city": "Kansas City", "conference": "AFC", "division": "West"},
        {"name": "Las Vegas Raiders", "abbreviation": "LV", "city": "Las Vegas", "conference": "AFC", "division": "West"},
        {"name": "Los Angeles Chargers", "abbreviation": "LAC", "city": "Los Angeles", "conference": "AFC", "division": "West"},
        {"name": "Los Angeles Rams", "abbreviation": "LAR", "city": "Los Angeles", "conference": "NFC", "division": "West"},
        {"name": "Miami Dolphins", "abbreviation": "MIA", "city": "Miami", "conference": "AFC", "division": "East"},
        {"name": "Minnesota Vikings", "abbreviation": "MIN", "city": "Minnesota", "conference": "NFC", "division": "North"},
        {"name": "New England Patriots", "abbreviation": "NE", "city": "New England", "conference": "AFC", "division": "East"},
        {"name": "New Orleans Saints", "abbreviation": "NO", "city": "New Orleans", "conference": "NFC", "division": "South"},
        {"name": "New York Giants", "abbreviation": "NYG", "city": "New York", "conference": "NFC", "division": "East"},
        {"name": "New York Jets", "abbreviation": "NYJ", "city": "New York", "conference": "AFC", "division": "East"},
        {"name": "Philadelphia Eagles", "abbreviation": "PHI", "city": "Philadelphia", "conference": "NFC", "division": "East"},
        {"name": "Pittsburgh Steelers", "abbreviation": "PIT", "city": "Pittsburgh", "conference": "AFC", "division": "North"},
        {"name": "San Francisco 49ers", "abbreviation": "SF", "city": "San Francisco", "conference": "NFC", "division": "West"},
        {"name": "Seattle Seahawks", "abbreviation": "SEA", "city": "Seattle", "conference": "NFC", "division": "West"},
        {"name": "Tampa Bay Buccaneers", "abbreviation": "TB", "city": "Tampa Bay", "conference": "NFC", "division": "South"},
        {"name": "Tennessee Titans", "abbreviation": "TEN", "city": "Tennessee", "conference": "AFC", "division": "South"},
        {"name": "Washington Commanders", "abbreviation": "WAS", "city": "Washington", "conference": "NFC", "division": "East"},
    ]
    
    for team_data in nfl_teams:
        team = Team(**team_data)
        session.add(team)


def _insert_default_books(session: Session):
    """Insert default sportsbooks."""
    from .models import Book
    
    # Check if books already exist
    if session.query(Book).count() > 0:
        return
    
    books = [
        {"name": "pinnacle", "display_name": "Pinnacle"},
        {"name": "draftkings", "display_name": "DraftKings"},
        {"name": "fanduel", "display_name": "FanDuel"},
        {"name": "betmgm", "display_name": "BetMGM"},
        {"name": "caesars", "display_name": "Caesars"},
        {"name": "unibet", "display_name": "Unibet"},
        {"name": "bovada", "display_name": "Bovada"},
        {"name": "betonlineag", "display_name": "BetOnline"},
    ]
    
    for book_data in books:
        book = Book(**book_data)
        session.add(book)