"""
Worker de sincronización. Ejecuta:
  1. Sync de próximos partidos + recientes
  2. Stats de partidos terminados sin stats
  3. Re-ajuste de Dixon-Coles
  4. Recalcular predicciones de próximos partidos
"""
import logging
import asyncio
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from app.database import SessionLocal
from app.config import get_settings
from app.models import League, Team, Fixture, MatchStat
from app.services.api_football import get_client, APIFootballError
from app.services.predictor import predict_match, save_predictions, fit_and_cache

logger = logging.getLogger(__name__)
settings = get_settings()


STAT_FIELD_MAP = {
    "Shots on Goal": "shots_on",
    "Shots off Goal": "shots_off",
    "Total Shots": "shots_total",
    "Blocked Shots": "shots_blocked",
    "Shots insidebox": "shots_inside_box",
    "Shots outsidebox": "shots_outside_box",
    "Fouls": "fouls",
    "Corner Kicks": "corner_kicks",
    "Offsides": "offsides",
    "Ball Possession": "ball_possession",
    "Yellow Cards": "yellow_cards",
    "Red Cards": "red_cards",
    "Goalkeeper Saves": "goalkeeper_saves",
    "Total passes": "passes_total",
    "Passes accurate": "passes_accurate",
    "Passes %": "passes_accuracy",
}


def _parse_stat_value(value):
    if value is None:
        return None
    if isinstance(value, str):
        v = value.replace("%", "").strip()
        try:
            return float(v) if "." in v else int(v)
        except ValueError:
            return None
    return value


async def sync_leagues_and_teams(db: Session):
    client = get_client()
    for league_id in settings.league_ids:
        try:
            league_data = await client.get_league(league_id)
            if not league_data:
                continue
            l = league_data["league"]
            country = league_data["country"]
            existing = db.query(League).get(league_id)
            if not existing:
                db.add(League(
                    id=league_id, name=l["name"],
                    country=country.get("name"), logo=l.get("logo"),
                    flag=country.get("flag"), season=settings.season,
                ))
                db.commit()
                logger.info(f"League {l['name']} sincronizada")
            
            teams = await client.get_teams(league_id, settings.season)
            for t in teams:
                team_info = t["team"]
                if not db.query(Team).get(team_info["id"]):
                    db.add(Team(
                        id=team_info["id"], name=team_info["name"],
                        code=team_info.get("code"), logo=team_info.get("logo"),
                        league_id=league_id,
                    ))
            db.commit()
            logger.info(f"  - {len(teams)} equipos para league {league_id}")
        except APIFootballError as e:
            logger.error(f"Error sync liga {league_id}: {e}")
            continue


async def sync_recent_fixtures(db: Session, days_back: int = 7, days_forward: int = 7):
    client = get_client()
    from_date = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    to_date = (datetime.utcnow() + timedelta(days=days_forward)).strftime("%Y-%m-%d")
    
    for league_id in settings.league_ids:
        try:
            fixtures = await client.get_fixtures(
                league_id=league_id, season=settings.season,
                from_date=from_date, to_date=to_date,
            )
            for f in fixtures:
                fix_id = f["fixture"]["id"]
                existing = db.query(Fixture).get(fix_id)
                date = datetime.fromisoformat(f["fixture"]["date"].replace("Z", "+00:00"))
                status = f["fixture"]["status"]["short"]
                home_id = f["teams"]["home"]["id"]
                away_id = f["teams"]["away"]["id"]
                home_goals = f["goals"]["home"]
                away_goals = f["goals"]["away"]
                
                for tid, tinfo in [(home_id, f["teams"]["home"]), (away_id, f["teams"]["away"])]:
                    if not db.query(Team).get(tid):
                        db.add(Team(id=tid, name=tinfo["name"],
                                    logo=tinfo.get("logo"), league_id=league_id))
                
                if existing:
                    existing.status = status
                    existing.home_goals = home_goals
                    existing.away_goals = away_goals
                    existing.date = date
                else:
                    db.add(Fixture(
                        id=fix_id, league_id=league_id, season=settings.season,
                        date=date, status=status,
                        home_team_id=home_id, away_team_id=away_id,
                        home_goals=home_goals, away_goals=away_goals,
                        venue_name=(f["fixture"].get("venue") or {}).get("name"),
                        referee=f["fixture"].get("referee"),
                    ))
            db.commit()
            logger.info(f"League {league_id}: {len(fixtures)} fixtures sincronizados")
        except APIFootballError as e:
            logger.error(f"Error sync fixtures liga {league_id}: {e}")
            continue


async def sync_match_stats_for_finished(db: Session, max_fixtures: int = 20):
    client = get_client()
    pending = (
        db.query(Fixture)
        .outerjoin(MatchStat, Fixture.id == MatchStat.fixture_id)
        .filter(Fixture.status == "FT")
        .filter(MatchStat.id.is_(None))
        .order_by(Fixture.date.desc())
        .limit(max_fixtures)
        .all()
    )
    for fixture in pending:
        try:
            stats = await client.get_fixture_stats(fixture.id)
            if not stats:
                continue
            for team_stats in stats:
                team_id = team_stats["team"]["id"]
                is_home = team_id == fixture.home_team_id
                stat_dict = {"fixture_id": fixture.id, "team_id": team_id, "is_home": is_home}
                for s in team_stats.get("statistics", []):
                    field = STAT_FIELD_MAP.get(s["type"])
                    if field:
                        stat_dict[field] = _parse_stat_value(s["value"])
                db.add(MatchStat(**stat_dict))
            db.commit()
            logger.info(f"Stats fixture {fixture.id}")
        except APIFootballError as e:
            logger.error(f"Error stats fixture {fixture.id}: {e}")
            continue


def refit_model(db: Session):
    """Re-ajusta Dixon-Coles con todos los datos disponibles."""
    try:
        fit_and_cache(db, min_matches=50)
    except ValueError as e:
        logger.warning(f"Skip refit: {e}")
    except Exception as e:
        logger.exception(f"Error refit: {e}")


def recalculate_predictions(db: Session):
    upcoming = (
        db.query(Fixture)
        .filter(Fixture.status == "NS")
        .filter(Fixture.date > datetime.utcnow())
        .filter(Fixture.date < datetime.utcnow() + timedelta(days=14))
        .all()
    )
    count = 0
    for fixture in upcoming:
        pred = predict_match(db, fixture)
        if pred:
            save_predictions(db, pred)
            count += 1
    logger.info(f"Recalculadas predicciones para {count} fixtures")


async def full_sync_cycle():
    db = SessionLocal()
    try:
        logger.info("=== INICIANDO CICLO DE SYNC ===")
        await sync_recent_fixtures(db)
        await sync_match_stats_for_finished(db, max_fixtures=15)
        refit_model(db)
        recalculate_predictions(db)
        logger.info("=== CICLO COMPLETADO ===")
    finally:
        db.close()


def run_sync_in_background():
    asyncio.run(full_sync_cycle())
