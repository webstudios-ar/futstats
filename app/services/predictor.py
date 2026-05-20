"""
Predictor de partido. Combina:
  - Dixon-Coles para goles, 1X2 y BTTS (modelo paramétrico completo)
  - Poisson sobre promedios ponderados por recencia para córners, tarjetas,
    faltas, tiros y posesión (no hay literatura tan consolidada para estos)

El modelo Dixon-Coles se entrena una vez (cacheamos los parámetros) y se
reutiliza para todas las predicciones. Los demás se calculan on-demand
por equipo.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
import logging
import math
import threading

from sqlalchemy.orm import Session
from sqlalchemy import desc

from app.models import Fixture, MatchStat, Prediction
from app.services.dixon_coles import (
    DixonColesParams, fit_dixon_coles, predict_score,
    outcome_probs, btts_prob, total_goals_over, poisson_over,
    GOAL_LINES, CORNER_LINES, CARD_LINES, FOUL_LINES, SHOT_LINES,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cache global de parámetros Dixon-Coles
# ---------------------------------------------------------------------------

_params_cache: DixonColesParams | None = None
_params_lock = threading.Lock()


def get_cached_params() -> DixonColesParams | None:
    return _params_cache


def set_cached_params(p: DixonColesParams):
    global _params_cache
    with _params_lock:
        _params_cache = p


def fit_and_cache(db: Session, min_matches: int = 100, xi: float = 0.0019) -> DixonColesParams:
    """
    Entrena Dixon-Coles con todos los partidos terminados disponibles
    y guarda el resultado en cache.
    """
    finished = (
        db.query(Fixture)
        .filter(Fixture.status == "FT")
        .filter(Fixture.home_goals.isnot(None))
        .filter(Fixture.away_goals.isnot(None))
        .order_by(Fixture.date)
        .all()
    )
    
    if len(finished) < min_matches:
        raise ValueError(
            f"Solo hay {len(finished)} partidos terminados. "
            f"Necesitamos al menos {min_matches} para un ajuste razonable."
        )
    
    matches = [
        {
            "home_id": f.home_team_id,
            "away_id": f.away_team_id,
            "home_goals": f.home_goals,
            "away_goals": f.away_goals,
            "date": f.date,
        }
        for f in finished
    ]
    
    params = fit_dixon_coles(matches, xi=xi)
    set_cached_params(params)
    logger.info(
        f"Dixon-Coles ajustado: {params.matches_used} partidos, "
        f"{len(params.teams)} equipos, home_adv={params.home_advantage:.3f}, "
        f"rho={params.rho:.3f}, log-lik={params.log_likelihood:.1f}"
    )
    return params


# ---------------------------------------------------------------------------
# Métricas no-goles (corners, cards, fouls, shots, possession)
# ---------------------------------------------------------------------------

@dataclass
class NonGoalMetrics:
    matches: int
    corners_for_home: float; corners_against_home: float
    corners_for_away: float; corners_against_away: float
    cards_for_home: float; cards_for_away: float
    fouls_home: float; fouls_away: float
    shots_home: float; shots_away: float
    possession_avg: float


def _weighted_mean(values: list[float], xi_days: float = 0.012, today: datetime | None = None,
                   dates: list[datetime] | None = None) -> float:
    """
    Media ponderada por recencia. xi_days=0.012 implica que un partido de
    hace 60 días pesa la mitad que uno de hoy (decay exponencial).
    """
    if not values:
        return 0.0
    if not dates or not today:
        # Fallback: peso lineal por posición (más recientes primero)
        n = len(values)
        weights = [2.0 - (i / max(n - 1, 1)) for i in range(n)]
    else:
        weights = [math.exp(-xi_days * max(0, (today - d).days)) for d in dates]
    
    total_w = sum(weights)
    if total_w == 0:
        return sum(values) / len(values)
    return sum(v * w for v, w in zip(values, weights)) / total_w


def compute_non_goal_metrics(db: Session, team_id: int, last_n: int = 20) -> NonGoalMetrics | None:
    """Promedios ponderados de córners, tarjetas, etc para un equipo."""
    stats_rows = (
        db.query(MatchStat, Fixture)
        .join(Fixture, MatchStat.fixture_id == Fixture.id)
        .filter(MatchStat.team_id == team_id)
        .filter(Fixture.status == "FT")
        .order_by(desc(Fixture.date))
        .limit(last_n)
        .all()
    )
    if not stats_rows:
        return None
    
    today = datetime.utcnow()
    home_records, away_records = [], []
    home_dates, away_dates = [], []
    
    for stat, fixture in stats_rows:
        opp_stat = next((s for s in fixture.stats if s.team_id != team_id), None)
        is_home = stat.is_home
        rec = {
            "corners_for": stat.corner_kicks or 0,
            "corners_against": (opp_stat.corner_kicks if opp_stat else 0) or 0,
            "cards_for": (stat.yellow_cards or 0) + (stat.red_cards or 0),
            "fouls": stat.fouls or 0,
            "shots": stat.shots_total or 0,
            "possession": stat.ball_possession or 50,
        }
        if is_home:
            home_records.append(rec)
            home_dates.append(fixture.date)
        else:
            away_records.append(rec)
            away_dates.append(fixture.date)
    
    def get(records, dates_list, key, default=0.0):
        vals = [r[key] for r in records]
        return _weighted_mean(vals, today=today, dates=dates_list) if vals else default
    
    all_records = home_records + away_records
    all_dates = home_dates + away_dates
    
    return NonGoalMetrics(
        matches=len(all_records),
        corners_for_home=get(home_records, home_dates, "corners_for"),
        corners_against_home=get(home_records, home_dates, "corners_against"),
        corners_for_away=get(away_records, away_dates, "corners_for"),
        corners_against_away=get(away_records, away_dates, "corners_against"),
        cards_for_home=get(home_records, home_dates, "cards_for"),
        cards_for_away=get(away_records, away_dates, "cards_for"),
        fouls_home=get(home_records, home_dates, "fouls"),
        fouls_away=get(away_records, away_dates, "fouls"),
        shots_home=get(home_records, home_dates, "shots"),
        shots_away=get(away_records, away_dates, "shots"),
        possession_avg=_weighted_mean(
            [r["possession"] for r in all_records],
            today=today, dates=all_dates,
        ) or 50,
    )


# ---------------------------------------------------------------------------
# Predicción completa
# ---------------------------------------------------------------------------

@dataclass
class MatchPrediction:
    fixture_id: int
    home_team: str
    away_team: str
    model: str  # "dixon-coles" o "fallback"
    
    expected_goals_home: float
    expected_goals_away: float
    expected_corners: float
    expected_cards: float
    expected_fouls: float
    expected_shots: float
    
    prob_home_win: float
    prob_draw: float
    prob_away_win: float
    
    prob_btts_yes: float
    prob_btts_no: float
    
    goals_over: dict[float, float]
    corners_over: dict[float, float]
    cards_over: dict[float, float]
    fouls_over: dict[float, float]
    shots_over: dict[float, float]
    
    possession_home: float
    possession_away: float
    
    most_likely_scores: list[tuple[int, int, float]]  # top 5 scores con prob
    
    confidence: str
    matches_analyzed: int


def _confidence_from(non_goal_matches: int, dc_has_team: bool) -> str:
    if not dc_has_team:
        return "low"
    if non_goal_matches >= 15:
        return "high"
    if non_goal_matches >= 8:
        return "medium"
    return "low"


def predict_match(db: Session, fixture: Fixture) -> MatchPrediction | None:
    """
    Predicción completa para un fixture próximo.
    Requiere parámetros Dixon-Coles ya entrenados en cache.
    """
    params = get_cached_params()
    
    home = compute_non_goal_metrics(db, fixture.home_team_id)
    away = compute_non_goal_metrics(db, fixture.away_team_id)
    
    if not home or not away or home.matches < 3 or away.matches < 3:
        return None
    
    # --- Dixon-Coles para goles ---
    if params and params.has_team(fixture.home_team_id) and params.has_team(fixture.away_team_id):
        lam_h, lam_a, matrix = predict_score(params, fixture.home_team_id, fixture.away_team_id)
        p_home, p_draw, p_away = outcome_probs(matrix)
        p_btts = btts_prob(matrix)
        goals_over_map = {l: total_goals_over(matrix, l) for l in GOAL_LINES}
        
        # Top 5 scores más probables
        n = matrix.shape[0]
        flat = [(x, y, matrix[x, y]) for x in range(n) for y in range(n)]
        flat.sort(key=lambda t: -t[2])
        top_scores = [(x, y, round(p, 4)) for x, y, p in flat[:5]]
        
        model_used = "dixon-coles"
    else:
        # Fallback: Poisson simple si DC no tiene a estos equipos todavía
        from app.services.dixon_coles import score_matrix as sm
        # Usamos promedios de la liga como aproximación (1.4 / 1.1)
        lam_h, lam_a = 1.4, 1.1
        matrix = sm(lam_h, lam_a, 0.0)
        p_home, p_draw, p_away = outcome_probs(matrix)
        p_btts = btts_prob(matrix)
        goals_over_map = {l: total_goals_over(matrix, l) for l in GOAL_LINES}
        n = matrix.shape[0]
        flat = [(x, y, matrix[x, y]) for x in range(n) for y in range(n)]
        flat.sort(key=lambda t: -t[2])
        top_scores = [(x, y, round(p, 4)) for x, y, p in flat[:5]]
        model_used = "fallback"
    
    # --- Métricas no-goles (Poisson sobre promedios) ---
    lambda_corners = (
        home.corners_for_home + away.corners_against_away +
        away.corners_for_away + home.corners_against_home
    ) / 2
    lambda_cards = home.cards_for_home + away.cards_for_away
    lambda_fouls = home.fouls_home + away.fouls_away
    lambda_shots = home.shots_home + away.shots_away
    
    # Posesión (los locales suelen tener ~+3% por ventaja de jugar en casa)
    raw_h = home.possession_avg + 3
    raw_a = away.possession_avg
    total = raw_h + raw_a
    poss_h = (raw_h / total) * 100
    poss_a = 100 - poss_h
    
    confidence = _confidence_from(min(home.matches, away.matches), model_used == "dixon-coles")
    
    return MatchPrediction(
        fixture_id=fixture.id,
        home_team=fixture.home_team.name,
        away_team=fixture.away_team.name,
        model=model_used,
        expected_goals_home=round(lam_h, 2),
        expected_goals_away=round(lam_a, 2),
        expected_corners=round(lambda_corners, 1),
        expected_cards=round(lambda_cards, 1),
        expected_fouls=round(lambda_fouls, 1),
        expected_shots=round(lambda_shots, 1),
        prob_home_win=round(p_home, 3),
        prob_draw=round(p_draw, 3),
        prob_away_win=round(p_away, 3),
        prob_btts_yes=round(p_btts, 3),
        prob_btts_no=round(1 - p_btts, 3),
        goals_over={l: round(p, 3) for l, p in goals_over_map.items()},
        corners_over={l: round(poisson_over(lambda_corners, l), 3) for l in CORNER_LINES},
        cards_over={l: round(poisson_over(lambda_cards, l), 3) for l in CARD_LINES},
        fouls_over={l: round(poisson_over(lambda_fouls, l), 3) for l in FOUL_LINES},
        shots_over={l: round(poisson_over(lambda_shots, l), 3) for l in SHOT_LINES},
        possession_home=round(poss_h, 1),
        possession_away=round(poss_a, 1),
        most_likely_scores=top_scores,
        confidence=confidence,
        matches_analyzed=min(home.matches, away.matches),
    )


def save_predictions(db: Session, pred: MatchPrediction):
    """Persiste predicciones en la tabla `predictions` para cachear."""
    db.query(Prediction).filter(Prediction.fixture_id == pred.fixture_id).delete()
    rows = []
    
    def add(metric: str, market: str, prob: float, expected: float | None = None):
        rows.append(Prediction(
            fixture_id=pred.fixture_id, metric=metric, market=market,
            probability=prob, expected_value=expected,
            confidence=pred.confidence, calculated_at=datetime.utcnow(),
        ))
    
    add("result", "home", pred.prob_home_win)
    add("result", "draw", pred.prob_draw)
    add("result", "away", pred.prob_away_win)
    add("btts", "yes", pred.prob_btts_yes)
    add("btts", "no", pred.prob_btts_no)
    
    total_eg = pred.expected_goals_home + pred.expected_goals_away
    for line, prob in pred.goals_over.items():
        add("goals", f"over_{line}", prob, total_eg)
    for line, prob in pred.corners_over.items():
        add("corners", f"over_{line}", prob, pred.expected_corners)
    for line, prob in pred.cards_over.items():
        add("cards", f"over_{line}", prob, pred.expected_cards)
    for line, prob in pred.fouls_over.items():
        add("fouls", f"over_{line}", prob, pred.expected_fouls)
    for line, prob in pred.shots_over.items():
        add("shots", f"over_{line}", prob, pred.expected_shots)
    
    db.add_all(rows)
    db.commit()
