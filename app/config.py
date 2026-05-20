"""
Configuración central. Lee de variables de entorno (.env en local, Railway vars en prod).
"""
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # API-Football (https://www.api-football.com/)
    api_football_key: str = ""
    api_football_host: str = "v3.football.api-sports.io"

    # Base de datos (Railway inyecta DATABASE_URL automáticamente)
    database_url: str = "postgresql://localhost/futstats"

    # App
    app_name: str = "FutStats"
    season: int = 2025  # Temporada actual a sincronizar
    
    # Ligas iniciales (IDs de API-Football)
    # 39=Premier, 140=La Liga, 135=Serie A, 78=Bundes, 61=Ligue 1
    # 2=Champions, 128=Liga Argentina, 71=Brasileirao
    league_ids: list[int] = [39, 140, 135, 78, 61, 2, 128, 71]

    # Worker
    sync_interval_hours: int = 6

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    return Settings()
