"""Database models for the sports betting system."""

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Mapped, mapped_column, relationship

Base = declarative_base()


class Team(Base):
    """NFL team information."""

    __tablename__ = "teams"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    abbreviation: Mapped[str] = mapped_column(String(10), nullable=False, unique=True)
    city: Mapped[str] = mapped_column(String(50), nullable=False)
    conference: Mapped[str] = mapped_column(String(10), nullable=False)  # AFC/NFC
    division: Mapped[str] = mapped_column(String(20), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    home_games: Mapped[list["Game"]] = relationship(
        "Game", back_populates="home_team", foreign_keys="Game.home_team_id"
    )
    away_games: Mapped[list["Game"]] = relationship(
        "Game", back_populates="away_team", foreign_keys="Game.away_team_id"
    )
    players: Mapped[list["Player"]] = relationship("Player", back_populates="team")


class Player(Base):
    """NFL player information."""

    __tablename__ = "players"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    external_id: Mapped[Optional[str]] = mapped_column(String(50), unique=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    position: Mapped[str] = mapped_column(String(10), nullable=False)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"))
    jersey_number: Mapped[Optional[int]] = mapped_column(Integer)
    height: Mapped[Optional[int]] = mapped_column(Integer)  # inches
    weight: Mapped[Optional[int]] = mapped_column(Integer)  # pounds
    experience: Mapped[Optional[int]] = mapped_column(Integer)  # years
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    current_status: Mapped[Optional[str]] = mapped_column(String(20))  # Healthy, Questionable, Doubtful, Out, IR, PUP, Suspended
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    team: Mapped["Team"] = relationship("Team", back_populates="players")
    features: Mapped[list["PlayerFeature"]] = relationship(
        "PlayerFeature", back_populates="player"
    )
    predictions: Mapped[list["Prediction"]] = relationship(
        "Prediction", back_populates="player"
    )
    props: Mapped[list["Prop"]] = relationship("Prop", back_populates="player")
    injury_reports: Mapped[list["InjuryReport"]] = relationship(
        "InjuryReport", back_populates="player", order_by="desc(InjuryReport.report_date)"
    )


class InjuryReport(Base):
    """NFL player injury reports and status tracking."""

    __tablename__ = "injury_reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    player_id: Mapped[int] = mapped_column(ForeignKey("players.id"), nullable=False)
    report_date: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    season: Mapped[int] = mapped_column(Integer, nullable=False)
    week: Mapped[int] = mapped_column(Integer, nullable=False)

    # Injury details
    injury_status: Mapped[str] = mapped_column(String(20), nullable=False)  # Questionable, Doubtful, Out, IR, etc.
    primary_injury: Mapped[Optional[str]] = mapped_column(String(50))  # Ankle, Knee, Shoulder, etc.
    secondary_injury: Mapped[Optional[str]] = mapped_column(String(50))
    injury_description: Mapped[Optional[str]] = mapped_column(Text)

    # Practice participation
    practice_wednesday: Mapped[Optional[str]] = mapped_column(String(20))  # Full, Limited, DNP
    practice_thursday: Mapped[Optional[str]] = mapped_column(String(20))
    practice_friday: Mapped[Optional[str]] = mapped_column(String(20))

    # Additional tracking
    date_of_injury: Mapped[Optional[datetime]] = mapped_column(DateTime)
    expected_return_date: Mapped[Optional[datetime]] = mapped_column(DateTime)
    games_missed: Mapped[int] = mapped_column(Integer, default=0)
    is_active_report: Mapped[bool] = mapped_column(Boolean, default=True)  # Latest report for the week
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    player: Mapped["Player"] = relationship("Player", back_populates="injury_reports")

    __table_args__ = (
        UniqueConstraint("player_id", "season", "week", "report_date"),
    )


class RosterChange(Base):
    """Track player roster changes (trades, signings, releases)."""

    __tablename__ = "roster_changes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    player_id: Mapped[int] = mapped_column(ForeignKey("players.id"), nullable=False)
    change_date: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    season: Mapped[int] = mapped_column(Integer, nullable=False)
    week: Mapped[Optional[int]] = mapped_column(Integer)

    # Change details
    change_type: Mapped[str] = mapped_column(String(20), nullable=False)  # Traded, Signed, Released, Retired
    from_team_id: Mapped[Optional[int]] = mapped_column(ForeignKey("teams.id"))
    to_team_id: Mapped[Optional[int]] = mapped_column(ForeignKey("teams.id"))
    notes: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    player: Mapped["Player"] = relationship("Player")
    from_team: Mapped[Optional["Team"]] = relationship("Team", foreign_keys=[from_team_id])
    to_team: Mapped[Optional["Team"]] = relationship("Team", foreign_keys=[to_team_id])


class Game(Base):
    """NFL game information."""

    __tablename__ = "games"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    external_id: Mapped[Optional[str]] = mapped_column(String(50), unique=True)
    season: Mapped[int] = mapped_column(Integer, nullable=False)
    week: Mapped[int] = mapped_column(Integer, nullable=False)
    season_type: Mapped[str] = mapped_column(String(20), default="REG")  # REG, POST
    game_date: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    home_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"))
    away_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"))
    home_score: Mapped[Optional[int]] = mapped_column(Integer)
    away_score: Mapped[Optional[int]] = mapped_column(Integer)
    is_completed: Mapped[bool] = mapped_column(Boolean, default=False)
    weather_conditions: Mapped[Optional[str]] = mapped_column(Text)
    temperature: Mapped[Optional[float]] = mapped_column(Float)
    wind_speed: Mapped[Optional[float]] = mapped_column(Float)
    precipitation: Mapped[Optional[float]] = mapped_column(Float)
    is_dome: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    home_team: Mapped["Team"] = relationship(
        "Team", back_populates="home_games", foreign_keys=[home_team_id]
    )
    away_team: Mapped["Team"] = relationship(
        "Team", back_populates="away_games", foreign_keys=[away_team_id]
    )
    props: Mapped[list["Prop"]] = relationship("Prop", back_populates="game")
    features: Mapped[list["PlayerFeature"]] = relationship(
        "PlayerFeature", back_populates="game"
    )
    predictions: Mapped[list["Prediction"]] = relationship(
        "Prediction", back_populates="game"
    )

    __table_args__ = (UniqueConstraint("season", "week", "home_team_id", "away_team_id"),)


class Book(Base):
    """Sportsbook information."""

    __tablename__ = "books"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(50), nullable=False, unique=True)
    display_name: Mapped[str] = mapped_column(String(100), nullable=False)
    region: Mapped[str] = mapped_column(String(10), default="us")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    props: Mapped[list["Prop"]] = relationship("Prop", back_populates="book")
    edges: Mapped[list["Edge"]] = relationship("Edge", back_populates="book")


class Prop(Base):
    """Player prop betting lines."""

    __tablename__ = "props"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    game_id: Mapped[int] = mapped_column(ForeignKey("games.id"))
    player_id: Mapped[int] = mapped_column(ForeignKey("players.id"))
    book_id: Mapped[int] = mapped_column(ForeignKey("books.id"))
    market: Mapped[str] = mapped_column(String(50), nullable=False)  # receiving_yards, etc.
    line: Mapped[float] = mapped_column(Float, nullable=False)
    over_odds: Mapped[int] = mapped_column(Integer, nullable=False)  # American odds
    under_odds: Mapped[int] = mapped_column(Integer, nullable=False)
    over_price: Mapped[float] = mapped_column(Float)  # Decimal odds
    under_price: Mapped[float] = mapped_column(Float)
    over_probability: Mapped[float] = mapped_column(Float)  # De-vigged probability
    under_probability: Mapped[float] = mapped_column(Float)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    # Relationships
    game: Mapped["Game"] = relationship("Game", back_populates="props")
    player: Mapped["Player"] = relationship("Player", back_populates="props")
    book: Mapped["Book"] = relationship("Book", back_populates="props")

    __table_args__ = (
        UniqueConstraint("game_id", "player_id", "book_id", "market", "timestamp"),
    )


class PlayerFeature(Base):
    """Computed features for players in specific games."""

    __tablename__ = "player_features"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    game_id: Mapped[int] = mapped_column(ForeignKey("games.id"))
    player_id: Mapped[int] = mapped_column(ForeignKey("players.id"))
    features: Mapped[dict] = mapped_column(JSON, nullable=False)
    feature_version: Mapped[str] = mapped_column(String(20), default="v1")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    game: Mapped["Game"] = relationship("Game", back_populates="features")
    player: Mapped["Player"] = relationship("Player", back_populates="features")

    __table_args__ = (UniqueConstraint("game_id", "player_id", "feature_version"),)


class Prediction(Base):
    """Model predictions for player props."""

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    game_id: Mapped[int] = mapped_column(ForeignKey("games.id"))
    player_id: Mapped[int] = mapped_column(ForeignKey("players.id"))
    market: Mapped[str] = mapped_column(String(50), nullable=False)
    model_name: Mapped[str] = mapped_column(String(50), nullable=False)
    model_version: Mapped[str] = mapped_column(String(20), nullable=False)
    prediction: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    distribution_params: Mapped[Optional[dict]] = mapped_column(JSON)
    p10: Mapped[Optional[float]] = mapped_column(Float)
    p50: Mapped[Optional[float]] = mapped_column(Float)
    p90: Mapped[Optional[float]] = mapped_column(Float)
    probability_over: Mapped[Optional[dict]] = mapped_column(JSON)  # {line: prob}
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    game: Mapped["Game"] = relationship("Game", back_populates="predictions")
    player: Mapped["Player"] = relationship("Player", back_populates="predictions")

    __table_args__ = (
        UniqueConstraint("game_id", "player_id", "market", "model_name", "model_version"),
    )


class ShadowLine(Base):
    """Shadow lines (fair value estimates) when no market lines exist."""

    __tablename__ = "shadow_lines"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    game_id: Mapped[int] = mapped_column(ForeignKey("games.id"))
    player_id: Mapped[int] = mapped_column(ForeignKey("players.id"))
    market: Mapped[str] = mapped_column(String(50), nullable=False)
    shadow_line: Mapped[float] = mapped_column(Float, nullable=False)
    fair_over_price: Mapped[float] = mapped_column(Float, nullable=False)
    fair_under_price: Mapped[float] = mapped_column(Float, nullable=False)
    exploit_score: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    reasoning: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (UniqueConstraint("game_id", "player_id", "market"),)


class Edge(Base):
    """Identified betting edges with EV calculations."""

    __tablename__ = "edges"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    game_id: Mapped[int] = mapped_column(ForeignKey("games.id"))
    player_id: Mapped[int] = mapped_column(ForeignKey("players.id"))
    book_id: Mapped[int] = mapped_column(ForeignKey("books.id"))
    market: Mapped[str] = mapped_column(String(50), nullable=False)
    side: Mapped[str] = mapped_column(String(10), nullable=False)  # over/under
    offered_line: Mapped[float] = mapped_column(Float, nullable=False)
    offered_odds: Mapped[int] = mapped_column(Integer, nullable=False)
    fair_line: Mapped[float] = mapped_column(Float, nullable=False)
    fair_probability: Mapped[float] = mapped_column(Float, nullable=False)
    expected_value: Mapped[float] = mapped_column(Float, nullable=False)
    kelly_fraction: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    reasoning: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    book: Mapped["Book"] = relationship("Book", back_populates="edges")

    __table_args__ = (
        UniqueConstraint("game_id", "player_id", "book_id", "market", "side"),
    )


class Parlay(Base):
    """Parlay combinations and pricing."""

    __tablename__ = "parlays"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    legs: Mapped[list] = mapped_column(JSON, nullable=False)  # List of edge IDs
    parlay_type: Mapped[str] = mapped_column(String(20), default="same_game")
    offered_odds: Mapped[Optional[int]] = mapped_column(Integer)
    fair_odds: Mapped[float] = mapped_column(Float, nullable=False)
    joint_probability: Mapped[float] = mapped_column(Float, nullable=False)
    expected_value: Mapped[float] = mapped_column(Float, nullable=False)
    correlation_matrix: Mapped[Optional[dict]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class ApiRequest(Base):
    """Track API requests for rate limiting and monitoring."""

    __tablename__ = "api_requests"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    api_source: Mapped[str] = mapped_column(String(50), nullable=False)  # odds_api, espn, etc.
    endpoint: Mapped[str] = mapped_column(String(200), nullable=False)
    request_type: Mapped[str] = mapped_column(String(50), nullable=False)  # props, odds, scores
    objects_returned: Mapped[int] = mapped_column(Integer, default=0)
    success: Mapped[bool] = mapped_column(Boolean, default=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    response_time_ms: Mapped[Optional[int]] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class GamePriority(Base):
    """Track game priority for intelligent request scheduling."""

    __tablename__ = "game_priorities"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    game_id: Mapped[int] = mapped_column(ForeignKey("games.id"), unique=True)
    priority_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    is_primetime: Mapped[bool] = mapped_column(Boolean, default=False)
    is_divisional: Mapped[bool] = mapped_column(Boolean, default=False)
    affects_playoffs: Mapped[bool] = mapped_column(Boolean, default=False)
    has_line_movement: Mapped[bool] = mapped_column(Boolean, default=False)
    betting_volume_estimate: Mapped[Optional[float]] = mapped_column(Float)
    last_priority_update: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    # Relationships
    game: Mapped["Game"] = relationship("Game")


class DataCache(Base):
    """Cache metadata for intelligent data refresh decisions."""

    __tablename__ = "data_cache"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    cache_key: Mapped[str] = mapped_column(String(200), nullable=False, unique=True)
    data_type: Mapped[str] = mapped_column(String(50), nullable=False)  # props, odds, scores
    data_source: Mapped[str] = mapped_column(String(50), nullable=False)  # odds_api, espn
    game_id: Mapped[Optional[int]] = mapped_column(ForeignKey("games.id"))
    player_id: Mapped[Optional[int]] = mapped_column(ForeignKey("players.id"))
    data_hash: Mapped[str] = mapped_column(String(64))  # SHA-256 hash of data
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    last_updated: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    request_count: Mapped[int] = mapped_column(Integer, default=1)
    is_stale: Mapped[bool] = mapped_column(Boolean, default=False)
    
    # Relationships
    game: Mapped[Optional["Game"]] = relationship("Game")
    player: Mapped[Optional["Player"]] = relationship("Player")