"""
Carga histórica de partidos desde football-data.co.uk.

Este servicio descarga CSVs públicos con histórico de partidos y los carga
en la DB. Se ejecuta una sola vez desde el endpoint admin.

Las URLs son las "oficiales" de football-data.co.uk, el dataset estándar
en literatura académica de pronóstico de fútbol.
"""
import logging
import csv
import io
from datetime import datetime
from pathlib import Path

import httpx
from sqlalchemy.orm import Session

from app.models import League, Team, Fixture, MatchStat

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuración: temporadas y ligas a descargar
# ---------------------------------------------------------------------------

# Cada tupla: (url, league_id, season, league_name)
# league_id es el ID de API-Football (para consistencia con el resto del sistema)
DATASETS_TO_LOAD = [
    # Premier League (England) - league_id 39
    ("https://www.football-data.co.uk/mmz4281/2425/E0.csv", 39, 2024, "Premier League"),
    ("https://www.football-data.co.uk/mmz4281/2324/E0.csv", 39, 2023, "Premier League"),
    ("https://www.football-data.co.uk/mmz4281/2223/E0.csv", 39, 2022, "Premier League"),
    ("https://www.football-data.co.uk/mmz4281/2122/E0.csv", 39, 2021, "Premier League"),
    ("https://www.football-data.co.uk/mmz4281/2021/E0.csv", 39, 2020, "Premier League"),
    # La Liga (Spain) - league_id 140
    ("https://www.football-data.co.uk/mmz4281/2425/SP1.csv", 140, 2024, "La Liga"),
    ("https://www.football-data.co.uk/mmz4281/2324/SP1.csv", 140, 2023, "La Liga"),
    ("https://www.football-data.co.uk/mmz4281/2223/SP1.csv", 140, 2022, "La Liga"),
    # Serie A (Italy) - league_id 135
    ("https://www.football-data.co.uk/mmz4281/2425/I1.csv", 135, 2024, "Serie A"),
    ("https://www.football-data.co.uk/mmz4281/2324/I1.csv", 135, 2023, "Serie A"),
    # Bundesliga (Germany) - league_id 78
    ("https://www.football-data.co.uk/mmz4281/2425/D1.csv", 78, 2024, "Bundesliga"),
    ("https://www.football-data.co.uk/mmz4281/2324/D1.csv", 78, 2023, "Bundesliga"),
    # Ligue 1 (France) - league_id 61
    ("https://www.football-data.co.uk/mmz4281/2425/F1.csv", 61, 2024, "Ligue 1"),
    ("https://www.football-data.co.uk/mmz4281/2324/F1.csv", 61, 2023, "Ligue 1"),
]


# Mapeo de nombres de equipos: football-data.co.uk -> nombres estándar
TEAM_NAME_MAP = {
    "Man United": "Manchester United",
    "Man City": "Manchester City",
    "Newcastle": "Newcastle United",
    "Wolves": "Wolverhampton Wanderers",
    "Spurs": "Tottenham",
    "Nott'm Forest": "Nottingham Forest",
    "Ath Madrid": "Atletico Madrid",
    "Ath Bilbao": "Athletic Bilbao",
    "Sociedad": "Real Sociedad",
    "Inter": "Inter Milan",
    "Milan": "AC Milan",
    "Bayern Munich": "Bayern München",
    "Dortmund": "Borussia Dortmund",
    "Leverkusen": "Bayer Leverkusen",
    # Argentina (formato football-data.co.uk extra league)
    "Estudiantes L.P.": "Estudiantes",
    "Newells Old Boys": "Newell's Old Boys",
    "Central Cordoba (SdE)": "Central Córdoba",
    "Velez Sarsfield": "Vélez Sarsfield",
}


def normalize_team(name: str) -> str:
    return TEAM_NAME_MAP.get(name, name).strip()


def parse_date(s: str) -> datetime | None:
    """football-data.co.uk usa DD/MM/YY o DD/MM/YYYY."""
    if not s:
        return None
    for fmt in ("%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d"):
        try:
            return datetime.strptime(s.strip(), fmt)
        except ValueError:
            continue
    return None


def safe_int(val) -> int | None:
    if val is None or val == "":
        return None
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------

def _ensure_league(db: Session, league_id: int, name: str, season: int):
    league = db.query(League).get(league_id)
    if not league:
        db.add(League(id=league_id, name=name, season=season))
        db.commit()
        logger.info(f"Liga creada: {league_id} - {name}")


def _get_or_create_team(db: Session, name: str, league_id: int, counter: list) -> int:
    normalized = normalize_team(name)
    existing = db.query(Team).filter(Team.name == normalized).first()
    if existing:
        return existing.id
    counter[0] += 1
    new_id = -counter[0]
    db.add(Team(id=new_id, name=normalized, league_id=league_id))
    db.commit()
    return new_id


async def _download_csv(url: str) -> str | None:
    """Descarga un CSV. Devuelve el texto o None si falló."""
    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            headers = {"User-Agent": "Mozilla/5.0 (compatible; FutStats/1.0)"}
            response = await client.get(url, headers=headers)
            if response.status_code != 200:
                logger.warning(f"HTTP {response.status_code} para {url}")
                return None
            # football-data.co.uk usa encoding latin-1
            return response.content.decode("latin-1", errors="replace")
    except Exception as e:
        logger.error(f"Error descargando {url}: {e}")
        return None


def _load_csv_text(db: Session, csv_text: str, league_id: int, season: int,
                   team_counter: list, fix_counter: list) -> tuple[int, int]:
    """Carga un CSV en memoria. Devuelve (cargados, omitidos)."""
    loaded = 0
    skipped = 0
    
    reader = csv.DictReader(io.StringIO(csv_text))
    for row in reader:
        try:
            date = parse_date(row.get("Date", ""))
            home_name = (row.get("HomeTeam") or "").strip()
            away_name = (row.get("AwayTeam") or "").strip()
            
            if not date or not home_name or not away_name:
                skipped += 1
                continue
            
            home_g = safe_int(row.get("FTHG"))
            away_g = safe_int(row.get("FTAG"))
            if home_g is None or away_g is None:
                skipped += 1
                continue
            
            home_id = _get_or_create_team(db, home_name, league_id, team_counter)
            away_id = _get_or_create_team(db, away_name, league_id, team_counter)
            
            # Chequear duplicado
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
            fix_id = -(fix_counter[0] + 1_000_000)  # offset para no chocar con teams
            
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
        
        except Exception as e:
            logger.warning(f"Fila ignorada: {e}")
            skipped += 1
            continue
    
    db.commit()
    return loaded, skipped


def _arg_season_to_year(season_str: str) -> int | None:
    """
    El ARG.csv usa 'Season' como '2023', '2024', '2024/2025', etc.
    Devolvemos el año de inicio como int.
    """
    if not season_str:
        return None
    s = season_str.strip()
    # Formato '2024/2025' -> 2024
    if "/" in s:
        s = s.split("/")[0]
    try:
        return int(s)
    except ValueError:
        return None


def _load_extra_league_csv(db: Session, csv_text: str, league_id: int,
                           min_season: int, league_name: str,
                           team_counter: list, fix_counter: list) -> tuple[int, int]:
    """
    Carga un CSV en formato 'extra league' de football-data.co.uk.
    Columnas: Country, League, Season, Date, Time, Home, Away, HG, AG, Res, (PH, PD, PA...)
    A diferencia del formato europeo, NO trae corners/cards/fouls.
    Solo carga temporadas >= min_season para no meter datos antiguos.
    """
    loaded = 0
    skipped = 0

    reader = csv.DictReader(io.StringIO(csv_text))
    for row in reader:
        try:
            season_year = _arg_season_to_year(row.get("Season", ""))
            if season_year is None or season_year < min_season:
                skipped += 1
                continue

            date = parse_date(row.get("Date", ""))
            home_name = (row.get("Home") or "").strip()
            away_name = (row.get("Away") or "").strip()

            if not date or not home_name or not away_name:
                skipped += 1
                continue

            home_g = safe_int(row.get("HG"))
            away_g = safe_int(row.get("AG"))
            if home_g is None or away_g is None:
                skipped += 1
                continue

            # Aseguramos que la liga exista con la season de este partido
            _ensure_league(db, league_id, league_name, season_year)

            home_id = _get_or_create_team(db, home_name, league_id, team_counter)
            away_id = _get_or_create_team(db, away_name, league_id, team_counter)

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
            fix_id = -(fix_counter[0] + 1_000_000)

            fix = Fixture(
                id=fix_id,
                league_id=league_id, season=season_year,
                date=date, status="FT",
                home_team_id=home_id, away_team_id=away_id,
                home_goals=home_g, away_goals=away_g,
            )
            db.add(fix)
            # Sin stats detalladas en extra-leagues, pero creamos los registros
            # base para que el predictor no falle (corners/cards quedan None).
            db.add_all([
                MatchStat(fixture_id=fix_id, team_id=home_id, is_home=True),
                MatchStat(fixture_id=fix_id, team_id=away_id, is_home=False),
            ])
            loaded += 1

            if loaded % 100 == 0:
                db.commit()
        except Exception as e:
            logger.warning(f"Fila ARG ignorada: {e}")
            skipped += 1
            continue

    db.commit()
    return loaded, skipped


async def load_all_historical_data(db: Session) -> dict:
    """
    Carga histórico completo. Llamar una sola vez desde el endpoint admin.
    Devuelve estadísticas por dataset.
    """
    # Contadores globales para IDs negativos (no colisionar con API-Football)
    existing_neg_teams = db.query(Team).filter(Team.id < 0).count()
    team_counter = [existing_neg_teams]
    
    existing_neg_fix = db.query(Fixture).filter(Fixture.id < -1_000_000).count()
    fix_counter = [existing_neg_fix]
    
    results = []
    total_loaded = 0
    
    for url, league_id, season, league_name in DATASETS_TO_LOAD:
        logger.info(f"Descargando {url}")
        _ensure_league(db, league_id, league_name, season)
        
        csv_text = await _download_csv(url)
        if not csv_text:
            results.append({
                "url": url, "league": league_name, "season": season,
                "loaded": 0, "skipped": 0, "error": "download failed"
            })
            continue
        
        loaded, skipped = _load_csv_text(
            db, csv_text, league_id, season, team_counter, fix_counter
        )
        total_loaded += loaded
        results.append({
            "url": url, "league": league_name, "season": season,
            "loaded": loaded, "skipped": skipped,
        })
        logger.info(f"  {league_name} {season}: {loaded} partidos cargados, {skipped} ignorados")

    # -----------------------------------------------------------------------
    # Liga Argentina (formato extra-league, todas las temporadas en un archivo)
    # league_id 128 = Liga Profesional Argentina en API-Football
    # -----------------------------------------------------------------------
    ARG_URL = "https://www.football-data.co.uk/new/ARG.csv"
    ARG_MIN_SEASON = 2020  # solo desde 2020 en adelante
    logger.info(f"Descargando liga argentina: {ARG_URL}")
    _ensure_league(db, 128, "Liga Profesional Argentina", ARG_MIN_SEASON)

    arg_text = await _download_csv(ARG_URL)
    if arg_text:
        arg_loaded, arg_skipped = _load_extra_league_csv(
            db, arg_text, 128, ARG_MIN_SEASON,
            "Liga Profesional Argentina", team_counter, fix_counter
        )
        total_loaded += arg_loaded
        results.append({
            "url": ARG_URL, "league": "Liga Profesional Argentina",
            "season": f">={ARG_MIN_SEASON}",
            "loaded": arg_loaded, "skipped": arg_skipped,
        })
        logger.info(f"  Argentina: {arg_loaded} partidos cargados, {arg_skipped} ignorados")
    else:
        results.append({
            "url": ARG_URL, "league": "Liga Profesional Argentina",
            "season": f">={ARG_MIN_SEASON}",
            "loaded": 0, "skipped": 0, "error": "download failed",
        })

    return {
        "total_loaded": total_loaded,
        "total_teams": db.query(Team).count(),
        "datasets": results,
    }
