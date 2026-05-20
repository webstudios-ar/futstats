"""Rutas que renderizan HTML."""
from fastapi import APIRouter, Request, Depends, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from sqlalchemy import asc
from datetime import datetime, timedelta
import json
from pathlib import Path

from app.database import get_db
from app.models import Fixture, League
from app.services.predictor import predict_match

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")


@router.get("/", response_class=HTMLResponse)
def home(request: Request, db: Session = Depends(get_db), league: int | None = None):
    """Listado de próximos partidos + buscador prominente."""
    q = (
        db.query(Fixture)
        .filter(Fixture.status == "NS")
        .filter(Fixture.date >= datetime.utcnow())
        .filter(Fixture.date <= datetime.utcnow() + timedelta(days=10))
    )
    if league:
        q = q.filter(Fixture.league_id == league)
    
    fixtures = q.order_by(asc(Fixture.date)).limit(60).all()
    leagues = db.query(League).order_by(League.name).all()
    
    by_date: dict[str, list] = {}
    for f in fixtures:
        key = f.date.strftime("%Y-%m-%d")
        by_date.setdefault(key, []).append(f)
    
    return templates.TemplateResponse("home.html", {
        "request": request,
        "by_date": by_date,
        "leagues": leagues,
        "current_league": league,
    })


@router.get("/match/{fixture_id}", response_class=HTMLResponse)
def match_detail(fixture_id: int, request: Request, db: Session = Depends(get_db)):
    fixture = db.query(Fixture).get(fixture_id)
    if not fixture:
        raise HTTPException(404, "Fixture no encontrado")
    pred = predict_match(db, fixture)
    return templates.TemplateResponse("match.html", {
        "request": request,
        "fixture": fixture,
        "pred": pred,
    })


@router.get("/about", response_class=HTMLResponse)
def about(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})


@router.get("/backtest", response_class=HTMLResponse)
def backtest_view(request: Request):
    """Muestra el último reporte de backtest si existe."""
    results_dir = Path("results")
    latest = None
    if results_dir.exists():
        files = sorted(results_dir.glob("backtest_*.json"), reverse=True)
        if files:
            with open(files[0]) as f:
                latest = json.load(f)
    
    return templates.TemplateResponse("backtest.html", {
        "request": request,
        "results": latest,
    })
