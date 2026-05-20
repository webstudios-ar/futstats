"""
Cliente de API-Football (RapidAPI / v3.football.api-sports.io).

Rate limits del plan FREE:
  - 100 requests/día
  - 10 requests/minuto

Con esos límites, sincronizar muchas ligas en un día es complicado.
Estrategia: priorizar fixtures próximos y stats de equipos activos.

Docs: https://www.api-football.com/documentation-v3
"""
import httpx
import asyncio
import logging
from typing import Any
from datetime import datetime
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

BASE_URL = f"https://{settings.api_football_host}"


class APIFootballError(Exception):
    pass


class RateLimitError(APIFootballError):
    pass


class APIFootballClient:
    """
    Cliente async con throttling básico (gap mínimo entre requests)
    y retry exponencial ante errores transitorios.
    """
    
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or settings.api_football_key
        if not self.api_key:
            logger.warning("API_FOOTBALL_KEY no configurada. Las llamadas van a fallar.")
        
        self.headers = {
            "x-rapidapi-key": self.api_key,
            "x-rapidapi-host": settings.api_football_host,
        }
        self._last_request_at: float = 0
        self._min_gap_seconds = 6.5  # ~9 req/min, margen sobre el limite de 10
        self._lock = asyncio.Lock()
    
    async def _throttle(self):
        """Asegura un gap mínimo entre requests para no pegarle al rate limit."""
        async with self._lock:
            now = asyncio.get_event_loop().time()
            elapsed = now - self._last_request_at
            if elapsed < self._min_gap_seconds:
                await asyncio.sleep(self._min_gap_seconds - elapsed)
            self._last_request_at = asyncio.get_event_loop().time()
    
    async def _request(self, endpoint: str, params: dict | None = None) -> dict[str, Any]:
        await self._throttle()
        url = f"{BASE_URL}/{endpoint}"
        
        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    resp = await client.get(url, headers=self.headers, params=params or {})
                
                if resp.status_code == 429:
                    wait = 60 * (attempt + 1)
                    logger.warning(f"Rate limit hit. Esperando {wait}s...")
                    await asyncio.sleep(wait)
                    continue
                
                resp.raise_for_status()
                data = resp.json()
                
                # API-Football devuelve errores dentro del body
                if data.get("errors") and isinstance(data["errors"], dict) and data["errors"]:
                    err_msg = str(data["errors"])
                    if "rateLimit" in err_msg.lower() or "limit" in err_msg.lower():
                        raise RateLimitError(err_msg)
                    raise APIFootballError(err_msg)
                
                return data
                
            except httpx.HTTPStatusError as e:
                if attempt == 2:
                    raise APIFootballError(f"HTTP {e.response.status_code}: {e.response.text}")
                await asyncio.sleep(2 ** attempt)
            except httpx.RequestError as e:
                if attempt == 2:
                    raise APIFootballError(f"Request error: {e}")
                await asyncio.sleep(2 ** attempt)
        
        raise APIFootballError("Exhausted retries")
    
    # ---------- Endpoints ----------
    
    async def get_league(self, league_id: int) -> dict | None:
        """Info de una liga."""
        data = await self._request("leagues", {"id": league_id})
        results = data.get("response", [])
        return results[0] if results else None
    
    async def get_teams(self, league_id: int, season: int) -> list[dict]:
        """Equipos de una liga en una temporada."""
        data = await self._request("teams", {"league": league_id, "season": season})
        return data.get("response", [])
    
    async def get_fixtures(
        self,
        league_id: int | None = None,
        season: int | None = None,
        team_id: int | None = None,
        from_date: str | None = None,
        to_date: str | None = None,
        status: str | None = None,
        last: int | None = None,
        next: int | None = None,
    ) -> list[dict]:
        """
        Trae partidos con filtros flexibles.
          - from_date/to_date: YYYY-MM-DD
          - status: NS (no iniciado), FT (terminado), etc
          - last/next: últimos N o próximos N
        """
        params: dict[str, Any] = {}
        if league_id: params["league"] = league_id
        if season: params["season"] = season
        if team_id: params["team"] = team_id
        if from_date: params["from"] = from_date
        if to_date: params["to"] = to_date
        if status: params["status"] = status
        if last: params["last"] = last
        if next: params["next"] = next
        
        data = await self._request("fixtures", params)
        return data.get("response", [])
    
    async def get_fixture_stats(self, fixture_id: int) -> list[dict]:
        """Estadísticas detalladas de un partido terminado (1 entrada por equipo)."""
        data = await self._request("fixtures/statistics", {"fixture": fixture_id})
        return data.get("response", [])
    
    async def get_head_to_head(self, team1_id: int, team2_id: int, last: int = 10) -> list[dict]:
        """Historial de partidos entre dos equipos."""
        data = await self._request(
            "fixtures/headtohead",
            {"h2h": f"{team1_id}-{team2_id}", "last": last}
        )
        return data.get("response", [])
    
    async def get_status(self) -> dict:
        """Status de la cuenta: requests usados, plan, etc. Útil para debug."""
        data = await self._request("status")
        return data.get("response", {})


# Singleton
_client: APIFootballClient | None = None


def get_client() -> APIFootballClient:
    global _client
    if _client is None:
        _client = APIFootballClient()
    return _client
