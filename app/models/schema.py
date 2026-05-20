"""
Schema de base de datos. Todo lo que necesitamos para guardar histórico
y calcular predicciones.
"""
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Index,
    UniqueConstraint
)
from sqlalchemy.orm import relationship
from datetime import datetime
from app.database import Base


class League(Base):
    __tablename__ = "leagues"
    
    id = Column(Integer, primary_key=True)  # ID de API-Football
    name = Column(String(120), nullable=False)
    country = Column(String(80))
    logo = Column(String(255))
    flag = Column(String(255))
    season = Column(Integer, nullable=False)
    
    teams = relationship("Team", back_populates="league")
    fixtures = relationship("Fixture", back_populates="league")


class Team(Base):
    __tablename__ = "teams"
    
    id = Column(Integer, primary_key=True)  # ID de API-Football
    name = Column(String(120), nullable=False)
    code = Column(String(10))
    logo = Column(String(255))
    league_id = Column(Integer, ForeignKey("leagues.id"))
    
    league = relationship("League", back_populates="teams")
    
    __table_args__ = (Index("ix_teams_name", "name"),)


class Fixture(Base):
    __tablename__ = "fixtures"
    
    id = Column(Integer, primary_key=True)  # ID de API-Football
    league_id = Column(Integer, ForeignKey("leagues.id"), nullable=False)
    season = Column(Integer, nullable=False)
    date = Column(DateTime, nullable=False, index=True)
    status = Column(String(20))  # NS, 1H, HT, 2H, FT, etc
    
    home_team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)
    away_team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)
    
    home_goals = Column(Integer)
    away_goals = Column(Integer)
    
    venue_name = Column(String(160))
    referee = Column(String(120))
    
    league = relationship("League", back_populates="fixtures")
    home_team = relationship("Team", foreign_keys=[home_team_id])
    away_team = relationship("Team", foreign_keys=[away_team_id])
    stats = relationship("MatchStat", back_populates="fixture", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index("ix_fixtures_date_status", "date", "status"),
        Index("ix_fixtures_league_season", "league_id", "season"),
    )


class MatchStat(Base):
    """
    Una fila por equipo por partido. Acá guardamos TODAS las métricas
    que después usamos para predecir.
    """
    __tablename__ = "match_stats"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    fixture_id = Column(Integer, ForeignKey("fixtures.id"), nullable=False)
    team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)
    is_home = Column(Boolean, nullable=False)
    
    # Tiros
    shots_total = Column(Integer)
    shots_on = Column(Integer)
    shots_off = Column(Integer)
    shots_blocked = Column(Integer)
    shots_inside_box = Column(Integer)
    shots_outside_box = Column(Integer)
    
    # Faltas y tarjetas
    fouls = Column(Integer)
    yellow_cards = Column(Integer)
    red_cards = Column(Integer)
    offsides = Column(Integer)
    
    # Set pieces
    corner_kicks = Column(Integer)
    
    # Posesión y precisión
    ball_possession = Column(Float)  # 0..100
    passes_total = Column(Integer)
    passes_accurate = Column(Integer)
    passes_accuracy = Column(Float)  # 0..100
    
    # Portero
    goalkeeper_saves = Column(Integer)
    
    fixture = relationship("Fixture", back_populates="stats")
    
    __table_args__ = (
        UniqueConstraint("fixture_id", "team_id", name="uq_fixture_team"),
    )


class TeamForm(Base):
    """
    Cache de promedios calculados por equipo. Se actualiza cada vez que
    el worker baja stats nuevos. Lectura rápida para el predictor.
    """
    __tablename__ = "team_form"
    
    team_id = Column(Integer, ForeignKey("teams.id"), primary_key=True)
    last_updated = Column(DateTime, default=datetime.utcnow)
    matches_played = Column(Integer, default=0)
    
    # Promedios general (ponderado por recencia)
    avg_goals_scored = Column(Float, default=0)
    avg_goals_conceded = Column(Float, default=0)
    avg_corners_for = Column(Float, default=0)
    avg_corners_against = Column(Float, default=0)
    avg_cards_for = Column(Float, default=0)
    avg_cards_against = Column(Float, default=0)
    avg_fouls_committed = Column(Float, default=0)
    avg_fouls_drawn = Column(Float, default=0)
    avg_shots_total = Column(Float, default=0)
    avg_shots_on = Column(Float, default=0)
    avg_possession = Column(Float, default=50)
    
    # Splits local/visitante (clave para predicciones precisas)
    home_goals_scored = Column(Float, default=0)
    home_goals_conceded = Column(Float, default=0)
    home_corners_for = Column(Float, default=0)
    home_corners_against = Column(Float, default=0)
    home_cards_for = Column(Float, default=0)
    
    away_goals_scored = Column(Float, default=0)
    away_goals_conceded = Column(Float, default=0)
    away_corners_for = Column(Float, default=0)
    away_corners_against = Column(Float, default=0)
    away_cards_for = Column(Float, default=0)


class Prediction(Base):
    """
    Predicciones calculadas para partidos próximos. Cache para no
    recalcular en cada request.
    """
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    fixture_id = Column(Integer, ForeignKey("fixtures.id"), nullable=False)
    metric = Column(String(40), nullable=False)  # goals, corners, cards, fouls, shots
    market = Column(String(40), nullable=False)  # over_2_5, btts_yes, etc
    probability = Column(Float, nullable=False)  # 0..1
    expected_value = Column(Float)  # lambda de Poisson o valor esperado
    confidence = Column(String(10))  # low/medium/high según muestra
    calculated_at = Column(DateTime, default=datetime.utcnow)
    
    fixture = relationship("Fixture")
    
    __table_args__ = (
        Index("ix_predictions_fixture", "fixture_id"),
        UniqueConstraint("fixture_id", "metric", "market", name="uq_pred_fixture_market"),
    )
