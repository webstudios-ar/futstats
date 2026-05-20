"""
Seed inicial: baja ligas, equipos y los partidos de las últimas 2 semanas
y próximas 2 semanas. Correr UNA SOLA VEZ al setup.

Uso:
    python -m scripts.seed_initial
"""
import asyncio
import logging
import sys
import os

# Permitir correr desde la raíz del proyecto
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import SessionLocal, engine, Base
from app.services.data_sync import (
    sync_leagues_and_teams,
    sync_recent_fixtures,
    sync_match_stats_for_finished,
    recalculate_predictions,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


async def main():
    log.info("1/5 Creando tablas...")
    Base.metadata.create_all(bind=engine)
    
    db = SessionLocal()
    try:
        log.info("2/5 Sincronizando ligas y equipos...")
        await sync_leagues_and_teams(db)
        
        log.info("3/5 Sincronizando fixtures recientes y próximos...")
        await sync_recent_fixtures(db, days_back=14, days_forward=14)
        
        log.info("4/5 Bajando stats de partidos terminados...")
        await sync_match_stats_for_finished(db, max_fixtures=30)
        
        log.info("5/5 Calculando predicciones iniciales...")
        recalculate_predictions(db)
        
        log.info("✅ Seed completado")
    finally:
        db.close()


if __name__ == "__main__":
    asyncio.run(main())
