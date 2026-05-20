"""
Corre el backtest sobre los datos cargados y genera reporte.

Uso:
  python -m scripts.run_backtest --start 2023-08-01 --end 2024-05-31

Genera:
  - results/backtest_<timestamp>.json con todas las métricas
  - Imprime tabla resumen en stdout
"""
import sys
import os
import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import SessionLocal
from app.services.backtest import run_backtest

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


def print_summary(results: dict):
    """Imprime una tabla bonita en stdout."""
    if "error" in results:
        print(f"ERROR: {results['error']}")
        return
    
    cfg = results["config"]
    print()
    print("=" * 78)
    print(f"BACKTEST RESULTS")
    print("=" * 78)
    print(f"Period:           {cfg['start_date'][:10]} → {cfg['end_date'][:10]}")
    print(f"Test fixtures:    {cfg['n_test_fixtures']}")
    print(f"Refit every:      {cfg['refit_every_days']} days")
    print(f"Min training:     {cfg['min_training_matches']} matches")
    print(f"xi (decay):       {cfg['xi']}")
    print()
    
    print("-" * 78)
    print(f"{'Model':<28} {'N':>6} {'Acc':>8} {'LogLoss':>10} {'RPS':>8} {'Brier':>8}")
    print("-" * 78)
    for key, m in results["models"].items():
        if m.get("n_predictions", 0) == 0:
            continue
        print(f"{m['name']:<28} {m['n_predictions']:>6} "
              f"{m['accuracy_1x2']:>8.4f} {m['log_loss_1x2']:>10.4f} "
              f"{m['rps_1x2']:>8.4f} {m['brier_1x2']:>8.4f}")
    print("-" * 78)
    
    dc = results["models"]["dixon_coles"]
    if dc.get("btts") and dc["btts"].get("n", 0) > 0:
        print()
        print("Mercados binarios (solo Dixon-Coles):")
        print(f"  BTTS:      N={dc['btts']['n']}  Acc={dc['btts']['accuracy']:.4f}  "
              f"LogLoss={dc['btts']['log_loss']:.4f}  Brier={dc['btts']['brier']:.4f}")
        print(f"  Over 2.5:  N={dc['over_2_5']['n']}  Acc={dc['over_2_5']['accuracy']:.4f}  "
              f"LogLoss={dc['over_2_5']['log_loss']:.4f}  Brier={dc['over_2_5']['brier']:.4f}")
    
    # Calibración resumida
    if dc.get("btts") and dc["btts"].get("calibration"):
        print()
        print("Calibración BTTS (bin → frecuencia observada):")
        for bin_val, info in sorted(dc["btts"]["calibration"].items()):
            pred = info["predicted"]
            obs = info["observed"]
            diff = obs - pred
            sign = "+" if diff >= 0 else ""
            print(f"  Predicho {pred:.1f}  →  Observado {obs:.3f} ({sign}{diff:+.3f})  [n={info['n']}]")
    
    print()


def main():
    parser = argparse.ArgumentParser(description="Backtest del modelo")
    parser.add_argument("--start", required=True, help="Fecha inicio (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="Fecha fin (YYYY-MM-DD)")
    parser.add_argument("--refit-days", type=int, default=14, help="Re-ajustar cada N días")
    parser.add_argument("--min-training", type=int, default=100, help="Mín partidos para entrenar")
    parser.add_argument("--xi", type=float, default=0.0019, help="Decay rate")
    parser.add_argument("--league", type=int, action="append", help="Filtrar por liga (repetible)")
    parser.add_argument("--output", default=None, help="Path del JSON de salida")
    args = parser.parse_args()
    
    start = datetime.fromisoformat(args.start)
    end = datetime.fromisoformat(args.end)
    
    db = SessionLocal()
    try:
        results = run_backtest(
            db,
            start_date=start, end_date=end,
            refit_every_days=args.refit_days,
            min_training_matches=args.min_training,
            xi=args.xi,
            leagues=args.league,
        )
    finally:
        db.close()
    
    print_summary(results)
    
    # Guardar JSON
    if args.output:
        out_path = Path(args.output)
    else:
        Path("results").mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = Path(f"results/backtest_{ts}.json")
    
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"📄 Reporte guardado: {out_path}")


if __name__ == "__main__":
    main()
