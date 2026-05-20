"""Endpoints JSON: búsqueda, predicciones, admin, backtest."""
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from sqlalchemy.orm import Session
from sqlalchemy import or_, func
from datetime import datetime, timedelta
import re

from app.database import get_db, SessionLocal
from app.models import Fixture, League, Team
from app.services.predictor import predict_match, save_predictions, fit_and_cache, get_cached_params
from app.services.data_sync import full_sync_cycle, sync_leagues_and_teams

router = APIRouter(prefix="/api", tags=["api"])


# ---------------------------------------------------------------------------
# Búsqueda
# ---------------------------------------------------------------------------

@router.get("/search")
def search(
    q: str = Query(..., min_length=2, description="Texto a buscar: nombre de equipo, 'X vs Y', etc"),
    limit: int = 10,
    db: Session = Depends(get_db),
):
    """
    Búsqueda inteligente:
      - "boca" → todos los próximos partidos de Boca
      - "real madrid vs" → próximos partidos de Real Madrid
      - "real madrid vs barcelona" → matchup específico
      - "boca river" → matchup entre los dos
    
    Devuelve fixtures próximos o recientes ordenados por proximidad temporal.
    """
    q_clean = q.strip().lower()
    
    # Detectar el separador "vs" / "v" / " - "
    separators = [" vs ", " v ", " - ", " contra "]
    parts = [q_clean]
    for sep in separators:
        if sep in q_clean:
            parts = [p.strip() for p in q_clean.split(sep) if p.strip()]
            break
    
    # Buscar equipos por nombre (LIKE insensitive)
    def find_teams(token: str) -> list[Team]:
        if not token:
            return []
        like = f"%{token}%"
        return (
            db.query(Team)
            .filter(func.lower(Team.name).like(like))
            .limit(10)
            .all()
        )
    
    matches = []
    
    if len(parts) >= 2 and parts[1]:
        # Caso "X vs Y" con ambos términos
        teams_a = find_teams(parts[0])
        teams_b = find_teams(parts[1])
        ids_a = {t.id for t in teams_a}
        ids_b = {t.id for t in teams_b}
        
        if ids_a and ids_b:
            # Buscar fixtures entre cualquiera de A vs cualquiera de B
            results = (
                db.query(Fixture)
                .filter(
                    or_(
                        Fixture.home_team_id.in_(ids_a) & Fixture.away_team_id.in_(ids_b),
                        Fixture.home_team_id.in_(ids_b) & Fixture.away_team_id.in_(ids_a),
                    )
                )
                .order_by(
                    # Próximos primero, después los recientes
                    func.abs(func.extract("epoch", Fixture.date) -
                            func.extract("epoch", func.now()))
                )
                .limit(limit)
                .all()
            )
            matches.extend(results)
    
    elif len(parts) == 1 or (len(parts) >= 2 and not parts[1]):
        # Caso "X" o "X vs" — todos los partidos del equipo X
        teams = find_teams(parts[0])
        if teams:
            team_ids = {t.id for t in teams}
            results = (
                db.query(Fixture)
                .filter(
                    or_(
                        Fixture.home_team_id.in_(team_ids),
                        Fixture.away_team_id.in_(team_ids),
                    )
                )
                .filter(Fixture.date >= datetime.utcnow() - timedelta(days=2))
                .order_by(Fixture.date.asc())
                .limit(limit)
                .all()
            )
            matches.extend(results)
    
    return [{
        "id": f.id,
        "date": f.date.isoformat(),
        "home": {"id": f.home_team_id, "name": f.home_team.name, "logo": f.home_team.logo},
        "away": {"id": f.away_team_id, "name": f.away_team.name, "logo": f.away_team.logo},
        "league": f.league.name if f.league else None,
        "status": f.status,
    } for f in matches]


@router.get("/search/teams")
def search_teams(q: str = Query(..., min_length=1), limit: int = 8, db: Session = Depends(get_db)):
    """Autocomplete de equipos. Devuelve lista corta para sugerencias."""
    like = f"%{q.lower()}%"
    teams = (
        db.query(Team)
        .filter(func.lower(Team.name).like(like))
        .order_by(func.length(Team.name))  # prioriza nombres cortos = más populares
        .limit(limit)
        .all()
    )
    return [{
        "id": t.id, "name": t.name, "logo": t.logo,
        "league": t.league.name if t.league else None,
    } for t in teams]


# ---------------------------------------------------------------------------
# Fixtures y predicciones
# ---------------------------------------------------------------------------

@router.get("/fixtures/upcoming")
def upcoming_fixtures(
    db: Session = Depends(get_db),
    league: int | None = None,
    limit: int = 50,
):
    q = (
        db.query(Fixture)
        .filter(Fixture.status == "NS")
        .filter(Fixture.date >= datetime.utcnow())
    )
    if league:
        q = q.filter(Fixture.league_id == league)
    fixtures = q.order_by(Fixture.date).limit(limit).all()
    return [{
        "id": f.id, "date": f.date.isoformat(),
        "home": f.home_team.name, "away": f.away_team.name,
        "league": f.league.name if f.league else None,
    } for f in fixtures]


@router.get("/match/{fixture_id}/prediction")
def get_prediction(fixture_id: int, db: Session = Depends(get_db)):
    fixture = db.query(Fixture).get(fixture_id)
    if not fixture:
        raise HTTPException(404, "Fixture no encontrado")
    pred = predict_match(db, fixture)
    if not pred:
        raise HTTPException(422, "Sin datos suficientes para predecir")
    save_predictions(db, pred)
    return pred.__dict__


@router.get("/leagues")
def list_leagues(db: Session = Depends(get_db)):
    leagues = db.query(League).order_by(League.name).all()
    return [{"id": l.id, "name": l.name, "country": l.country, "logo": l.logo} for l in leagues]


# ---------------------------------------------------------------------------
# Admin
# ---------------------------------------------------------------------------

@router.post("/admin/sync")
async def trigger_sync(background_tasks: BackgroundTasks):
    async def _run():
        await full_sync_cycle()
    background_tasks.add_task(_run)
    return {"status": "sync started"}


@router.post("/admin/seed-leagues")
async def trigger_seed(background_tasks: BackgroundTasks):
    async def _run():
        db = SessionLocal()
        try:
            await sync_leagues_and_teams(db)
        finally:
            db.close()
    background_tasks.add_task(_run)
    return {"status": "seed started"}


@router.post("/admin/fit-model")
def fit_model(db: Session = Depends(get_db), min_matches: int = 100, xi: float = 0.0019):
    """Re-ajusta los parámetros Dixon-Coles con todos los datos disponibles."""
    try:
        params = fit_and_cache(db, min_matches=min_matches, xi=xi)
        return {
            "status": "ok",
            "matches_used": params.matches_used,
            "n_teams": len(params.teams),
            "home_advantage": params.home_advantage,
            "rho": params.rho,
            "log_likelihood": params.log_likelihood,
            "fitted_at": params.fitted_at.isoformat(),
        }
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.get("/admin/model-status")
def model_status():
    p = get_cached_params()
    if not p:
        return {"fitted": False}
    return {
        "fitted": True,
        "matches_used": p.matches_used,
        "n_teams": len(p.teams),
        "home_advantage": round(p.home_advantage, 4),
        "rho": round(p.rho, 4),
        "log_likelihood": round(p.log_likelihood, 2),
        "fitted_at": p.fitted_at.isoformat(),
    }


@router.get("/health")
def health():
    return {"status": "ok", "ts": datetime.utcnow().isoformat()}
