"""
Modelo Dixon-Coles (1997) — el estándar académico para pronósticos de fútbol.

Referencia:
  Dixon, M. J., & Coles, S. G. (1997). Modelling Association Football Scores
  and Inefficiencies in the Football Betting Market. Journal of the Royal
  Statistical Society: Series C, 46(2), 265-280.

Mejora sobre Poisson independiente:
  - Corrige la subestimación de resultados bajos (0-0, 1-0, 0-1, 1-1)
    introduciendo un factor de correlación τ(x,y) para esos scores.
  - Estima parámetros de ataque/defensa por equipo + factor de localía,
    en lugar de promediar ciegamente.
  - Pondera partidos por recencia mediante decay exponencial ξ
    (xi típicamente entre 0.001 y 0.005 por día).

Parametrización:
  λ_home = exp(attack_home + defense_away + home_advantage)
  λ_away = exp(attack_away + defense_home)

Para score (x, y):
  P(X=x, Y=y) = τ(x,y) · Poisson(x|λ_home) · Poisson(y|λ_away)

  donde:
    τ(0,0) = 1 - λ_h·λ_a·ρ
    τ(0,1) = 1 + λ_h·ρ
    τ(1,0) = 1 + λ_a·ρ
    τ(1,1) = 1 - ρ
    τ(x,y) = 1                 para el resto

Constraint: sum(attack) = 0  (normalización para identificabilidad)
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson
from dataclasses import dataclass, field
from datetime import datetime
import logging
import math

logger = logging.getLogger(__name__)


# Líneas estándar de mercados para over/under
GOAL_LINES = [0.5, 1.5, 2.5, 3.5, 4.5]
CORNER_LINES = [7.5, 8.5, 9.5, 10.5, 11.5, 12.5]
CARD_LINES = [2.5, 3.5, 4.5, 5.5, 6.5]
FOUL_LINES = [18.5, 21.5, 24.5, 27.5]
SHOT_LINES = [8.5, 10.5, 12.5, 14.5]

MAX_GOALS = 8  # límite para enumerar matriz de scores (P(X≥8) ≈ 0)


# ---------------------------------------------------------------------------
# Helpers de probabilidad
# ---------------------------------------------------------------------------

def _tau(x: int, y: int, lam_h: float, lam_a: float, rho: float) -> float:
    """Factor de corrección Dixon-Coles para scores bajos."""
    if x == 0 and y == 0:
        return 1 - lam_h * lam_a * rho
    if x == 0 and y == 1:
        return 1 + lam_h * rho
    if x == 1 and y == 0:
        return 1 + lam_a * rho
    if x == 1 and y == 1:
        return 1 - rho
    return 1.0


def score_matrix(lam_h: float, lam_a: float, rho: float, max_goals: int = MAX_GOALS) -> np.ndarray:
    """
    Matriz P[x,y] = probabilidad de score x-y bajo Dixon-Coles.
    Se normaliza para asegurar que sume 1 (los ajustes τ pueden romper esto).
    """
    h_probs = poisson.pmf(np.arange(max_goals + 1), lam_h)
    a_probs = poisson.pmf(np.arange(max_goals + 1), lam_a)
    matrix = np.outer(h_probs, a_probs)
    
    # Aplicar correcciones tau a los 4 scores bajos
    for x in range(2):
        for y in range(2):
            matrix[x, y] *= _tau(x, y, lam_h, lam_a, rho)
    
    # Re-normalizar
    total = matrix.sum()
    if total > 0:
        matrix /= total
    return matrix


def outcome_probs(matrix: np.ndarray) -> tuple[float, float, float]:
    """Devuelve (P_home, P_draw, P_away) sumando triángulos de la matriz."""
    n = matrix.shape[0]
    p_home = sum(matrix[x, y] for x in range(n) for y in range(n) if x > y)
    p_draw = sum(matrix[x, x] for x in range(n))
    p_away = sum(matrix[x, y] for x in range(n) for y in range(n) if x < y)
    return float(p_home), float(p_draw), float(p_away)


def btts_prob(matrix: np.ndarray) -> float:
    """P(ambos marcan) = sum sobre x≥1, y≥1."""
    return float(matrix[1:, 1:].sum())


def total_goals_over(matrix: np.ndarray, line: float) -> float:
    """P(total goles > line)."""
    n = matrix.shape[0]
    threshold = math.floor(line)  # over 2.5 -> total >= 3 -> x+y > 2
    p = sum(matrix[x, y] for x in range(n) for y in range(n) if x + y > threshold)
    return float(p)


def poisson_over(lambda_: float, line: float) -> float:
    """P(X > line) con Poisson simple. Para córners, tarjetas, etc."""
    k = math.floor(line) + 1
    return float(1 - poisson.cdf(k - 1, lambda_))


# ---------------------------------------------------------------------------
# Estimación de parámetros
# ---------------------------------------------------------------------------

@dataclass
class DixonColesParams:
    """Parámetros estimados del modelo."""
    attack: dict[int, float]              # team_id -> coef ataque
    defense: dict[int, float]             # team_id -> coef defensa (más negativo = mejor)
    home_advantage: float                  # log-factor de localía
    rho: float                             # correlación goles bajos
    fitted_at: datetime = field(default_factory=datetime.utcnow)
    matches_used: int = 0
    teams: list[int] = field(default_factory=list)
    log_likelihood: float = 0.0

    def has_team(self, team_id: int) -> bool:
        return team_id in self.attack and team_id in self.defense


def _neg_log_likelihood(
    params: np.ndarray,
    n_teams: int,
    team_idx: dict[int, int],
    matches: list[tuple],  # (home_id, away_id, home_g, away_g, weight)
) -> float:
    """
    Función a minimizar. params es un vector aplanado:
      [attack_0, ..., attack_{n-1}, defense_0, ..., defense_{n-1}, home_adv, rho]
    
    Constraint: sum(attack) = 0 (lo aplicamos forzando attack_0 = -sum(resto))
    """
    # Desempaquetar
    attack = np.empty(n_teams)
    attack[1:] = params[:n_teams - 1]
    attack[0] = -attack[1:].sum()
    
    defense = params[n_teams - 1:2 * n_teams - 1]
    home_adv = params[2 * n_teams - 1]
    rho = params[2 * n_teams]
    
    # Limitar rho a un rango seguro (la teoría exige -1 < rho < 1, pero
    # valores extremos rompen positividad de tau)
    rho = max(-0.3, min(0.3, rho))
    
    ll = 0.0
    for home_id, away_id, hg, ag, weight in matches:
        h = team_idx[home_id]
        a = team_idx[away_id]
        
        lam_h = math.exp(attack[h] + defense[a] + home_adv)
        lam_a = math.exp(attack[a] + defense[h])
        
        # Limitar lambdas para evitar overflow
        lam_h = min(lam_h, 10)
        lam_a = min(lam_a, 10)
        
        tau = _tau(hg, ag, lam_h, lam_a, rho)
        if tau <= 0:
            return 1e10  # penalizar territorio inválido
        
        # log P(X=hg) + log P(Y=ag) + log τ
        log_p = (
            math.log(tau)
            - lam_h + hg * math.log(lam_h) - math.lgamma(hg + 1)
            - lam_a + ag * math.log(lam_a) - math.lgamma(ag + 1)
        )
        ll += weight * log_p
    
    return -ll


def fit_dixon_coles(
    matches: list[dict],
    xi: float = 0.0019,
    reference_date: datetime | None = None,
) -> DixonColesParams:
    """
    Ajusta el modelo a una lista de partidos terminados.
    
    Cada match es: {
        'home_id': int, 'away_id': int,
        'home_goals': int, 'away_goals': int,
        'date': datetime
    }
    
    xi: decay rate. Valor típico de la literatura: 0.0019/día
        (Dixon-Coles usaron 0.0065 por semana ≈ 0.00093/día,
         papers posteriores prefieren xi entre 0.002-0.005)
    """
    if not matches:
        raise ValueError("No hay partidos para ajustar el modelo")
    
    if reference_date is None:
        reference_date = max(m["date"] for m in matches)
    
    # Recolectar equipos únicos
    teams = sorted({m["home_id"] for m in matches} | {m["away_id"] for m in matches})
    n = len(teams)
    team_idx = {tid: i for i, tid in enumerate(teams)}
    
    # Preparar matches con peso por recencia: w = exp(-xi * days_ago)
    prepared = []
    for m in matches:
        days_ago = max(0, (reference_date - m["date"]).days)
        weight = math.exp(-xi * days_ago)
        prepared.append((m["home_id"], m["away_id"], m["home_goals"], m["away_goals"], weight))
    
    # Estado inicial: todo en 0, home_adv pequeño positivo, rho en 0
    # attack tiene n-1 parámetros (el primero queda determinado por la suma)
    x0 = np.concatenate([
        np.zeros(n - 1),       # attack 1..n-1
        np.zeros(n),           # defense
        np.array([0.25]),      # home advantage típico
        np.array([-0.05]),     # rho (típicamente negativo para fútbol)
    ])
    
    logger.info(f"Ajustando Dixon-Coles: {len(matches)} partidos, {n} equipos, xi={xi}")
    
    result = minimize(
        _neg_log_likelihood,
        x0,
        args=(n, team_idx, prepared),
        method="L-BFGS-B",
        options={"maxiter": 200, "disp": False},
    )
    
    # Desempaquetar resultado
    attack_vals = np.empty(n)
    attack_vals[1:] = result.x[:n - 1]
    attack_vals[0] = -attack_vals[1:].sum()
    defense_vals = result.x[n - 1:2 * n - 1]
    home_adv = float(result.x[2 * n - 1])
    rho = float(result.x[2 * n])
    rho = max(-0.3, min(0.3, rho))
    
    return DixonColesParams(
        attack={tid: float(attack_vals[i]) for tid, i in team_idx.items()},
        defense={tid: float(defense_vals[i]) for tid, i in team_idx.items()},
        home_advantage=home_adv,
        rho=rho,
        matches_used=len(matches),
        teams=teams,
        log_likelihood=-float(result.fun),
    )


def predict_score(
    params: DixonColesParams,
    home_id: int,
    away_id: int,
) -> tuple[float, float, np.ndarray] | None:
    """
    Devuelve (lambda_home, lambda_away, matriz_scores) usando los params.
    Si algún equipo no está en el modelo (recién promovido, etc.), None.
    """
    if not params.has_team(home_id) or not params.has_team(away_id):
        return None
    
    lam_h = math.exp(params.attack[home_id] + params.defense[away_id] + params.home_advantage)
    lam_a = math.exp(params.attack[away_id] + params.defense[home_id])
    
    matrix = score_matrix(lam_h, lam_a, params.rho)
    return lam_h, lam_a, matrix
