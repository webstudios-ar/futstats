"""
Backtest del modelo. Camina temporalmente por los partidos:
  - Para cada fecha t, ajusta el modelo SOLO con partidos < t (sin leakage)
  - Predice los partidos del día t
  - Compara con resultados reales
  - Acumula métricas

Métricas reportadas (todas estándar en literatura de pronóstico deportivo):
  - Accuracy: % de aciertos en el resultado más probable (1X2)
  - Log Loss: -E[log p(actual)], la métrica más usada en clasificación probabilística
  - RPS (Ranked Probability Score): específica para clasificación ordinal,
    PENALIZA equivocarse "lejos" más que equivocarse "cerca". Estándar en
    forecasting deportivo (ver Constantinou & Fenton, 2012).
  - Brier Score: error cuadrático medio sobre las probabilidades.

Baselines incluidos para comparación:
  - random: 1/3 a cada outcome
  - home_bias: 45% home, 28% draw, 27% away (frecuencias históricas globales)
  - league_avg: frecuencias observadas en los datos de entrenamiento

Calibración:
  - Bins de probabilidad predicha
  - Para cada bin, frecuencia observada del evento
  - Un modelo perfectamente calibrado tiene observed ≈ predicted en cada bin
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import math
import numpy as np
from collections import defaultdict

from sqlalchemy.orm import Session
from app.models import Fixture
from app.services.dixon_coles import (
    fit_dixon_coles, predict_score, outcome_probs, btts_prob,
    total_goals_over,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------

def log_loss(predicted: list[float], actual_idx: int) -> float:
    """
    -log P(actual). predicted es la lista de probs (orden: home, draw, away).
    actual_idx es 0 (home gana), 1 (empate), 2 (away gana).
    """
    p = max(1e-12, min(1 - 1e-12, predicted[actual_idx]))
    return -math.log(p)


def rps(predicted: list[float], actual_idx: int) -> float:
    """
    Ranked Probability Score (Epstein 1969, Constantinou & Fenton 2012).
    Para 3 outcomes: RPS = 1/2 * sum_k (cum_pred_k - cum_actual_k)^2
    Más bajo = mejor. Rango [0, 1].
    """
    n = len(predicted)
    cum_p = np.cumsum(predicted)
    cum_a = np.cumsum([1 if i == actual_idx else 0 for i in range(n)])
    return float(0.5 * np.sum((cum_p[:-1] - cum_a[:-1]) ** 2))


def brier(predicted: list[float], actual_idx: int) -> float:
    """Brier score multiclase: sum_k (p_k - actual_k)^2."""
    n = len(predicted)
    actual = [1 if i == actual_idx else 0 for i in range(n)]
    return float(sum((predicted[i] - actual[i]) ** 2 for i in range(n)))


def binary_log_loss(p: float, actual: int) -> float:
    p = max(1e-12, min(1 - 1e-12, p))
    return -(actual * math.log(p) + (1 - actual) * math.log(1 - p))


def binary_brier(p: float, actual: int) -> float:
    return (p - actual) ** 2


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def baseline_random() -> list[float]:
    return [1/3, 1/3, 1/3]


def baseline_home_bias() -> list[float]:
    # Frecuencias globales típicas en fútbol europeo
    return [0.45, 0.28, 0.27]


def baseline_league_avg(matches: list[dict]) -> list[float]:
    """Frecuencia observada en el set de entrenamiento."""
    if not matches:
        return [1/3, 1/3, 1/3]
    h = sum(1 for m in matches if m["home_goals"] > m["away_goals"])
    d = sum(1 for m in matches if m["home_goals"] == m["away_goals"])
    a = sum(1 for m in matches if m["home_goals"] < m["away_goals"])
    total = h + d + a
    return [h/total, d/total, a/total]


# ---------------------------------------------------------------------------
# Resultado del backtest
# ---------------------------------------------------------------------------

@dataclass
class ModelResults:
    name: str
    n_predictions: int = 0
    correct: int = 0
    log_loss_sum: float = 0.0
    rps_sum: float = 0.0
    brier_sum: float = 0.0
    
    # Para mercados binarios
    btts_n: int = 0
    btts_correct: int = 0
    btts_log_loss: float = 0.0
    btts_brier: float = 0.0
    
    over25_n: int = 0
    over25_correct: int = 0
    over25_log_loss: float = 0.0
    over25_brier: float = 0.0
    
    # Para calibración (bins de prob predicha)
    calibration_bins: dict = field(default_factory=lambda: defaultdict(lambda: [0, 0]))
    # bin_key -> [n_total, n_event_happened]
    
    def add_outcome(self, predicted: list[float], actual_idx: int):
        self.n_predictions += 1
        pred_idx = int(np.argmax(predicted))
        if pred_idx == actual_idx:
            self.correct += 1
        self.log_loss_sum += log_loss(predicted, actual_idx)
        self.rps_sum += rps(predicted, actual_idx)
        self.brier_sum += brier(predicted, actual_idx)
    
    def add_btts(self, p_yes: float, actual: int):
        self.btts_n += 1
        if (p_yes >= 0.5) == bool(actual):
            self.btts_correct += 1
        self.btts_log_loss += binary_log_loss(p_yes, actual)
        self.btts_brier += binary_brier(p_yes, actual)
        # Calibración
        bin_key = round(p_yes * 10) / 10  # bins de 0.1
        self.calibration_bins[f"btts_{bin_key}"][0] += 1
        if actual:
            self.calibration_bins[f"btts_{bin_key}"][1] += 1
    
    def add_over25(self, p_over: float, actual: int):
        self.over25_n += 1
        if (p_over >= 0.5) == bool(actual):
            self.over25_correct += 1
        self.over25_log_loss += binary_log_loss(p_over, actual)
        self.over25_brier += binary_brier(p_over, actual)
        bin_key = round(p_over * 10) / 10
        self.calibration_bins[f"over25_{bin_key}"][0] += 1
        if actual:
            self.calibration_bins[f"over25_{bin_key}"][1] += 1
    
    def summary(self) -> dict:
        if self.n_predictions == 0:
            return {"name": self.name, "n": 0}
        
        # Calibración procesada
        cal_btts = {}
        cal_over25 = {}
        for key, (total, evt) in self.calibration_bins.items():
            market, bin_str = key.split("_", 1)
            bin_val = float(bin_str)
            if total >= 5:  # solo bins con muestra mínima
                obs_freq = evt / total
                if market == "btts":
                    cal_btts[bin_val] = {"predicted": bin_val, "observed": round(obs_freq, 3), "n": total}
                else:
                    cal_over25[bin_val] = {"predicted": bin_val, "observed": round(obs_freq, 3), "n": total}
        
        return {
            "name": self.name,
            "n_predictions": self.n_predictions,
            "accuracy_1x2": round(self.correct / self.n_predictions, 4),
            "log_loss_1x2": round(self.log_loss_sum / self.n_predictions, 4),
            "rps_1x2": round(self.rps_sum / self.n_predictions, 4),
            "brier_1x2": round(self.brier_sum / self.n_predictions, 4),
            "btts": {
                "n": self.btts_n,
                "accuracy": round(self.btts_correct / self.btts_n, 4) if self.btts_n else None,
                "log_loss": round(self.btts_log_loss / self.btts_n, 4) if self.btts_n else None,
                "brier": round(self.btts_brier / self.btts_n, 4) if self.btts_n else None,
                "calibration": dict(sorted(cal_btts.items())),
            },
            "over_2_5": {
                "n": self.over25_n,
                "accuracy": round(self.over25_correct / self.over25_n, 4) if self.over25_n else None,
                "log_loss": round(self.over25_log_loss / self.over25_n, 4) if self.over25_n else None,
                "brier": round(self.over25_brier / self.over25_n, 4) if self.over25_n else None,
                "calibration": dict(sorted(cal_over25.items())),
            },
        }


# ---------------------------------------------------------------------------
# Runner del backtest
# ---------------------------------------------------------------------------

def actual_outcome_idx(home_g: int, away_g: int) -> int:
    if home_g > away_g: return 0
    if home_g == away_g: return 1
    return 2


def run_backtest(
    db: Session,
    start_date: datetime,
    end_date: datetime,
    refit_every_days: int = 14,
    min_training_matches: int = 100,
    xi: float = 0.0019,
    leagues: list[int] | None = None,
) -> dict:
    """
    Walk-forward backtest. Re-ajusta el modelo cada `refit_every_days` para
    evitar leakage temporal y simular uso real (no usamos el futuro para
    predecir el pasado).
    
    Compara Dixon-Coles contra 3 baselines.
    
    Devuelve un dict con resultados de cada modelo.
    """
    q = db.query(Fixture).filter(Fixture.status == "FT")
    q = q.filter(Fixture.date >= start_date).filter(Fixture.date <= end_date)
    if leagues:
        q = q.filter(Fixture.league_id.in_(leagues))
    test_fixtures = q.order_by(Fixture.date).all()
    
    if not test_fixtures:
        return {"error": "No hay fixtures en el período"}
    
    logger.info(f"Backtest: {len(test_fixtures)} partidos entre {start_date.date()} y {end_date.date()}")
    
    results = {
        "dixon_coles": ModelResults("Dixon-Coles"),
        "random": ModelResults("Random (1/3)"),
        "home_bias": ModelResults("Home bias (45/28/27)"),
        "league_avg": ModelResults("League average freq"),
    }
    
    current_params = None
    last_fit_date = None
    
    for fixture in test_fixtures:
        # ¿Toca re-ajustar?
        needs_refit = (
            current_params is None
            or last_fit_date is None
            or (fixture.date - last_fit_date).days >= refit_every_days
        )
        
        if needs_refit:
            train_q = db.query(Fixture).filter(Fixture.status == "FT")
            train_q = train_q.filter(Fixture.date < fixture.date)
            if leagues:
                train_q = train_q.filter(Fixture.league_id.in_(leagues))
            training = train_q.all()
            
            if len(training) < min_training_matches:
                continue  # no suficiente histórico todavía
            
            training_data = [
                {"home_id": f.home_team_id, "away_id": f.away_team_id,
                 "home_goals": f.home_goals, "away_goals": f.away_goals,
                 "date": f.date}
                for f in training
            ]
            
            try:
                current_params = fit_dixon_coles(training_data, xi=xi, reference_date=fixture.date)
                last_fit_date = fixture.date
                logger.info(f"Re-ajuste @ {fixture.date.date()}: {len(training_data)} partidos")
                
                # Actualizar baseline league_avg con datos actuales
                league_avg_probs = baseline_league_avg(training_data)
            except Exception as e:
                logger.error(f"Fallo el ajuste: {e}")
                continue
        
        # Predecir
        if not current_params.has_team(fixture.home_team_id) or \
           not current_params.has_team(fixture.away_team_id):
            continue
        
        result = predict_score(current_params, fixture.home_team_id, fixture.away_team_id)
        if not result:
            continue
        lam_h, lam_a, matrix = result
        
        # Probabilidades del modelo DC
        p_h, p_d, p_a = outcome_probs(matrix)
        dc_probs = [p_h, p_d, p_a]
        p_btts_yes = btts_prob(matrix)
        p_over25 = total_goals_over(matrix, 2.5)
        
        # Outcome real
        actual = actual_outcome_idx(fixture.home_goals, fixture.away_goals)
        actual_btts = int(fixture.home_goals > 0 and fixture.away_goals > 0)
        actual_over25 = int(fixture.home_goals + fixture.away_goals > 2)
        
        # Acumular en cada modelo
        results["dixon_coles"].add_outcome(dc_probs, actual)
        results["dixon_coles"].add_btts(p_btts_yes, actual_btts)
        results["dixon_coles"].add_over25(p_over25, actual_over25)
        
        results["random"].add_outcome(baseline_random(), actual)
        results["home_bias"].add_outcome(baseline_home_bias(), actual)
        results["league_avg"].add_outcome(league_avg_probs, actual)
    
    return {
        "config": {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "refit_every_days": refit_every_days,
            "min_training_matches": min_training_matches,
            "xi": xi,
            "leagues": leagues,
            "n_test_fixtures": len(test_fixtures),
        },
        "models": {k: r.summary() for k, r in results.items()},
    }
