"""FastAPI app principal. Routers + scheduler + warm-up del modelo."""
import logging
import threading
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from apscheduler.schedulers.background import BackgroundScheduler

from app.config import get_settings
from app.database import engine, Base, SessionLocal
from app.routers import pages, api
from app.services.data_sync import run_sync_in_background
from app.services.predictor import fit_and_cache

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)
settings = get_settings()

scheduler = BackgroundScheduler(timezone="UTC")


def _try_fit_model():
    """Intenta ajustar Dixon-Coles al arranque y periódicamente."""
    db = SessionLocal()
    try:
        fit_and_cache(db, min_matches=50)
    except ValueError as e:
        logger.warning(f"No se pudo ajustar el modelo todavía: {e}")
    except Exception as e:
        logger.exception(f"Error ajustando modelo: {e}")
    finally:
        db.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Creando tablas si no existen...")
    Base.metadata.create_all(bind=engine)
    
    # Ajustar el modelo en segundo plano para NO bloquear el arranque.
    # Con miles de partidos el fit tarda y haría fallar el healthcheck de Railway.
    logger.info("Lanzando warm-up del modelo Dixon-Coles en background...")
    warmup_thread = threading.Thread(target=_try_fit_model, daemon=True)
    warmup_thread.start()
    
    # Scheduler de sync periódico
    scheduler.add_job(
        run_sync_in_background,
        "interval",
        hours=settings.sync_interval_hours,
        id="sync_data",
        max_instances=1,
        coalesce=True,
    )
    # Re-fitear el modelo cada 24h
    scheduler.add_job(
        _try_fit_model,
        "interval",
        hours=24,
        id="refit_model",
        max_instances=1,
        coalesce=True,
    )
    scheduler.start()
    logger.info(f"Scheduler iniciado (sync cada {settings.sync_interval_hours}h, refit cada 24h)")
    
    yield
    
    scheduler.shutdown()
    logger.info("App detenida")


app = FastAPI(
    title=settings.app_name,
    description="Pronosticador estadístico de fútbol con modelo Dixon-Coles.",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory="app/static"), name="static")
app.include_router(pages.router)
app.include_router(api.router)
