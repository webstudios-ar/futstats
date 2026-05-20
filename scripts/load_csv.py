"""
Carga datasets históricos en formato CSV para tener histórico masivo
SIN gastar requests de API-Football.

Formato soportado: football-data.co.uk (el dataset más usado en papers
académicos de football betting). Cada CSV tiene una temporada de una liga
con stats completas: goles, córners, tarjetas, faltas, tiros.

Cómo descargarlo:
  https://www.football-data.co.uk/data.php
  o desde Kaggle: "European Soccer Database" / "Football Data UK"

Columnas relevantes que esperamos:
  Date, HomeTeam, AwayTeam, FTHG (full time home goals), FTAG,
  HS (home shots), AS, HST (shots on target), AST,
  HC (corners), AC, HF (fouls), AF, HY (yellow), AY, HR (red), AR,
  Referee

Uso:
  python -m scripts.load_csv data/E0_2023.csv 39 2023
  
  Donde:
    E0_2023.csv = archivo CSV
    39 = league_id de API-Football (Premier League)
    2023 = temporada
"""
import sys
import os
import csv
import logging
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import SessionLocal, engine, Base
from app.models import League, Team, Fixture, MatchStat

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# Mapeo de nombres comunes football-data.co.uk -> API-Football
# Si tu dataset usa otros nombres, ampliá este dict
TEAM_NAME_MAP = {
    # Premier
    "Man United": "Manchester United",
    "Man City": "Manchester City",
    "Newcastle": "Newcastle United",
    "Wolves": "Wolverhampton Wanderers",
    "Spurs": "Tottenham",
    "Nott'm Forest": "Nottingham Forest",
    # La Liga
    "Ath Madrid": "Atletico Madrid",
    "Ath Bilbao": "Athletic Bilbao",
    "Sociedad": "Real Sociedad",
    # Serie A
    "Inter": "Inter Milan",
    "Milan": "AC Milan",
    # Bundesliga
    "Bayern Munich": "Bayern München",
    "Dortmund": "Borussia Dortmund",
    "Leverkusen": "Bayer Leverkusen",
}


def normalize_team(name: str) -> str:
    return TEAM_NAME_MAP.get(name, name).strip()


def parse_date(s: str) -> datetime:
    """football-data.co.uk usa DD/MM/YY o DD/MM/YYYY."""
    for fmt in ("%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    raise ValueError(f"Formato de fecha desconocido: {s}")


def get_or_create_team(db, name: str, league_id: int, team_id_counter: list) -> int:
    """Busca o crea un equipo. Devuelve su ID."""
    normalized = normalize_team(name)
    existing = db.query(Team).filter(Team.name == normalized).first()
    if existing:
        return existing.id
    
    # Usar IDs negativos para equipos creados desde CSV (no chocan con API-Football)
    team_id_counter[0] += 1
    new_id = -team_id_counter[0]
    db.add(Team(id=new_id, name=normalized, league_id=league_id))
    db.commit()
    return new_id


def safe_int(val) -> int | None:
    if val is None or val == "":
        return None
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return None


def load_csv(csv_path: str, league_id: int, season: int, league_name: str | None = None):
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    
    try:
        # Asegurar liga
        league = db.query(League).get(league_id)
        if not league:
            db.add(League(
                id=league_id,
                name=league_name or f"League {league_id}",
                season=season,
            ))
            db.commit()
            log.info(f"Liga creada: {league_id}")
        
        # Counter para IDs negativos
        existing_negative = db.query(Team).filter(Team.id < 0).count()
        team_counter = [existing_negative]
        
        # Counter para fixture IDs (también negativos para no chocar)
        existing_neg_fix = db.query(Fixture).filter(Fixture.id < 0).count()
        fix_counter = [existing_neg_fix]
        
        loaded = 0
        skipped = 0
        
        with open(csv_path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    date = parse_date(row.get("Date", ""))
                    home_name = row.get("HomeTeam", "").strip()
                    away_name = row.get("AwayTeam", "").strip()
                    if not home_name or not away_name:
                        skipped += 1
                        continue
                    
                    home_g = safe_int(row.get("FTHG"))
                    away_g = safe_int(row.get("FTAG"))
                    if home_g is None or away_g is None:
                        skipped += 1
                        continue
                    
                    home_id = get_or_create_team(db, home_name, league_id, team_counter)
                    away_id = get_or_create_team(db, away_name, league_id, team_counter)
                    
                    # Chequear duplicado (mismo home/away/date)
                    dup = (
                        db.query(Fixture)
                        .filter(Fixture.home_team_id == home_id)
                        .filter(Fixture.away_team_id == away_id)
                        .filter(Fixture.date == date)
                        .first()
                    )
                    if dup:
                        skipped += 1
                        continue
                    
                    fix_counter[0] += 1
                    fix_id = -fix_counter[0] - 1000000  # offset para no chocar con teams negativos
                    
                    fix = Fixture(
                        id=fix_id,
                        league_id=league_id, season=season,
                        date=date, status="FT",
                        home_team_id=home_id, away_team_id=away_id,
                        home_goals=home_g, away_goals=away_g,
                        referee=row.get("Referee"),
                    )
                    db.add(fix)
                    
                    # Stats por equipo
                    home_stat = MatchStat(
                        fixture_id=fix_id, team_id=home_id, is_home=True,
                        shots_total=safe_int(row.get("HS")),
                        shots_on=safe_int(row.get("HST")),
                        corner_kicks=safe_int(row.get("HC")),
                        fouls=safe_int(row.get("HF")),
                        yellow_cards=safe_int(row.get("HY")),
                        red_cards=safe_int(row.get("HR")),
                    )
                    away_stat = MatchStat(
                        fixture_id=fix_id, team_id=away_id, is_home=False,
                        shots_total=safe_int(row.get("AS")),
                        shots_on=safe_int(row.get("AST")),
                        corner_kicks=safe_int(row.get("AC")),
                        fouls=safe_int(row.get("AF")),
                        yellow_cards=safe_int(row.get("AY")),
                        red_cards=safe_int(row.get("AR")),
                    )
                    db.add_all([home_stat, away_stat])
                    
                    loaded += 1
                    if loaded % 100 == 0:
                        db.commit()
                        log.info(f"  ... {loaded} partidos cargados")
                
                except Exception as e:
                    log.warning(f"Fila ignorada por error: {e}")
                    skipped += 1
                    continue
            
            db.commit()
        
        log.info(f"✅ {loaded} partidos cargados, {skipped} ignorados, archivo {csv_path}")
        
    finally:
        db.close()


def main():
    if len(sys.argv) < 4:
        print("Uso: python -m scripts.load_csv <csv_path> <league_id> <season> [league_name]")
        print("Ejemplo: python -m scripts.load_csv data/E0_2324.csv 39 2023 'Premier League'")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    league_id = int(sys.argv[2])
    season = int(sys.argv[3])
    league_name = sys.argv[4] if len(sys.argv) > 4 else None
    
    if not Path(csv_path).exists():
        log.error(f"Archivo no encontrado: {csv_path}")
        sys.exit(1)
    
    load_csv(csv_path, league_id, season, league_name)


if __name__ == "__main__":
    main()
