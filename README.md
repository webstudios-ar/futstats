# FutStats ⚽ — Pronosticador estadístico de fútbol

Pronóstico de partidos de fútbol basado en el modelo **Dixon-Coles (1997)**, el estándar académico de referencia para predicción de scores. Valida sus predicciones con backtest walk-forward sobre datos históricos.

## Qué hace

- 🔮 **Predice** goles, córners, tarjetas, faltas, tiros, posesión y BTTS para próximos partidos
- 🔍 **Buscador inteligente**: escribís `boca`, `real madrid vs barcelona` o `river plate` y te muestra el partido + predicciones
- 📊 **Backtest académico** con métricas estándar (Accuracy, Log Loss, RPS, Brier) contra 3 baselines
- 📈 **Calibración** de probabilidades binned (probs predichas vs frecuencias observadas)
- 🎯 Modelo Dixon-Coles entrenado por máxima verosimilitud + decay exponencial por recencia

## Stack

- **Backend:** FastAPI + SQLAlchemy + PostgreSQL
- **Frontend:** Jinja2 + CSS custom (dark theme académico, mobile-first)
- **Datos:** API-Football (live) + datasets CSV de football-data.co.uk (histórico)
- **Modelo:** Dixon-Coles con scipy.optimize (L-BFGS-B)
- **Worker:** APScheduler
- **Deploy:** Railway

## Estructura

```
futstats/
├── app/
│   ├── main.py                    # FastAPI + scheduler + warm-up
│   ├── config.py                  # Settings
│   ├── database.py                # SQLAlchemy session
│   ├── models/schema.py           # Tablas DB
│   ├── services/
│   │   ├── api_football.py        # Cliente API live
│   │   ├── data_sync.py           # Worker de sync
│   │   ├── dixon_coles.py         # Modelo D-C (core académico)
│   │   ├── predictor.py           # Predictor de partido (D-C + Poisson)
│   │   └── backtest.py            # Backtest + métricas + calibración
│   ├── routers/
│   │   ├── pages.py               # HTML
│   │   └── api.py                 # JSON + búsqueda
│   ├── templates/                 # Jinja2 (home, match, backtest, about)
│   └── static/                    # CSS + search.js
└── scripts/
    ├── seed_initial.py            # Bajar datos desde API-Football
    ├── load_csv.py                # Cargar CSV histórico
    └── run_backtest.py            # Ejecutar validación
```

## Setup local

```bash
# 1. Clonar
cd futstats

# 2. Entorno
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Variables
cp .env.example .env
# Editá .env (DATABASE_URL y opcionalmente API_FOOTBALL_KEY)

# 4. Postgres local (Docker)
docker run -d --name fs-pg -p 5432:5432 \
  -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=futstats \
  postgres:16

# 5a. Cargar datos históricos vía CSV (RECOMENDADO para proyecto académico)
# Descargá CSVs de https://www.football-data.co.uk/data.php
# (Premier League 2022-23 = "Season 2022/2023" -> "England" -> E0.csv)
mkdir data && mv ~/Downloads/E0.csv data/
python -m scripts.load_csv data/E0.csv 39 2022 "Premier League"
# Cargá varios años + varias ligas para tener buen training set

# 5b. (Opcional) Bajar datos live desde API-Football
python -m scripts.seed_initial

# 6. Levantar el server
uvicorn app.main:app --reload
# http://localhost:8000
```

## Cargar datasets históricos (Kaggle / football-data.co.uk)

Para entrenar el modelo **sin gastar requests de API-Football**, descargá CSVs gratis:

**Football-Data.co.uk** (formato estándar, usado en papers académicos):
- https://www.football-data.co.uk/data.php
- Disponible: Premier League, La Liga, Serie A, Bundesliga, Ligue 1, Championship, etc.
- Cada CSV = una liga, una temporada
- Stats completas: goles, córners, tarjetas, faltas, tiros, árbitro

Ejemplos:
```bash
# Premier League (league_id=39) últimas 5 temporadas
python -m scripts.load_csv data/E0_2024.csv 39 2024 "Premier League"
python -m scripts.load_csv data/E0_2023.csv 39 2023 "Premier League"
python -m scripts.load_csv data/E0_2022.csv 39 2022 "Premier League"
python -m scripts.load_csv data/E0_2021.csv 39 2021 "Premier League"
python -m scripts.load_csv data/E0_2020.csv 39 2020 "Premier League"

# La Liga (league_id=140)
python -m scripts.load_csv data/SP1_2024.csv 140 2024 "La Liga"
```

Con 5 temporadas de Premier ya tenés ~1900 partidos: training set robusto para Dixon-Coles.

## Correr el backtest (clave para defender ante el profesor)

```bash
python -m scripts.run_backtest \
  --start 2023-08-01 \
  --end 2024-05-31 \
  --refit-days 14 \
  --min-training 200
```

Output:
- Tabla comparativa Dixon-Coles vs Random/HomeBias/LeagueAvg
- Métricas: Accuracy, Log Loss, RPS, Brier
- Calibración binned para BTTS y Over 2.5
- JSON completo en `results/backtest_<timestamp>.json`
- Visualización en `http://localhost:8000/backtest`

## Deploy en Railway

1. Pushear repo a GitHub
2. Railway → New Project → Deploy from GitHub
3. Add Plugin → PostgreSQL (auto-linkea `DATABASE_URL`)
4. Variables → agregar `API_FOOTBALL_KEY` (opcional, solo si querés sync live)
5. Deploy automático
6. Cargar datos: `railway run python -m scripts.load_csv data/E0.csv 39 2024 "Premier"`

## Endpoints

### HTML
- `GET /` — Lista de próximos partidos + buscador
- `GET /match/{id}` — Predicción completa de un partido
- `GET /backtest` — Resultados de validación
- `GET /about` — Metodología + referencias académicas

### Búsqueda
- `GET /api/search?q=boca+vs+river` — Búsqueda combinada
- `GET /api/search/teams?q=mad` — Autocomplete de equipos

### Predicciones
- `GET /api/match/{id}/prediction` — Predicción JSON
- `GET /api/fixtures/upcoming` — Próximos partidos

### Admin
- `POST /api/admin/seed-leagues` — Bajar ligas/equipos de API-Football
- `POST /api/admin/sync` — Sync manual de partidos + stats
- `POST /api/admin/fit-model` — Re-ajustar Dixon-Coles
- `GET /api/admin/model-status` — Estado del modelo en cache

## Métricas predichas

| Métrica | Líneas over/under | Modelo |
|---|---|---|
| Resultado 1X2 | — | Dixon-Coles |
| BTTS | sí/no | Dixon-Coles |
| Goles | 0.5, 1.5, 2.5, 3.5, 4.5 | Dixon-Coles |
| Top 5 marcadores | — | Dixon-Coles |
| Córners | 7.5, 8.5, 9.5, 10.5, 11.5, 12.5 | Poisson ponderado |
| Tarjetas | 2.5, 3.5, 4.5, 5.5, 6.5 | Poisson ponderado |
| Faltas | 18.5, 21.5, 24.5, 27.5 | Poisson ponderado |
| Tiros | 8.5, 10.5, 12.5, 14.5 | Poisson ponderado |
| Posesión | — | Promedio ajustado por localía |

## Referencias académicas

- **Dixon, M. J., & Coles, S. G. (1997).** Modelling Association Football Scores and Inefficiencies in the Football Betting Market. *JRSS Series C, 46(2)*, 265-280.
- **Epstein, E. S. (1969).** A scoring system for probability forecasts of ranked categories. *Journal of Applied Meteorology, 8(6)*, 985-987.
- **Constantinou, A. C., & Fenton, N. E. (2012).** Solving the problem of inadequate scoring rules for assessing probabilistic football forecasting models. *JQAS, 8(1)*.
- **Boshnakov, G., Kharrat, T., & McHale, I. G. (2017).** A bivariate Weibull count model for forecasting association football scores. *International Journal of Forecasting, 33(2)*, 458-466.

## Disclaimer

Herramienta de análisis estadístico con propósito académico. No es consejo de apuestas.
