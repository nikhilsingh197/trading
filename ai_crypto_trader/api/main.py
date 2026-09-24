"""FastAPI application.

Provides REST and WebSocket endpoints for the dashboard.
All sensitive operations require authentication.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ai_crypto_trader.config.settings import get_settings
from ai_crypto_trader.core.logging import configure_logging, get_logger

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Application startup and shutdown."""
    settings = get_settings()
    configure_logging(settings.app_log_level)
    settings.ensure_directories()
    log.info("api_starting", env=settings.app_env, mode=settings.trading_mode)
    yield
    log.info("api_shutting_down")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="AI Crypto Trader",
        description="Production-grade AI cryptocurrency trading platform",
        version="0.1.0",
        lifespan=lifespan,
        # Disable detailed error messages in production
        docs_url="/docs" if settings.app_env != "live" else None,
        redoc_url="/redoc" if settings.app_env != "live" else None,
    )

    # CORS (restrict in production)
    origins = ["http://localhost:3000", "http://127.0.0.1:3000"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"],
    )

    # Register routes
    from ai_crypto_trader.api.routes import (
        models,
        portfolio,
        positions,
        strategies,
        system,
        trades,
        websocket,
    )
    app.include_router(system.router, prefix="/api/v1", tags=["system"])
    app.include_router(portfolio.router, prefix="/api/v1", tags=["portfolio"])
    app.include_router(positions.router, prefix="/api/v1", tags=["positions"])
    app.include_router(trades.router, prefix="/api/v1", tags=["trades"])
    app.include_router(strategies.router, prefix="/api/v1", tags=["strategies"])
    app.include_router(models.router, prefix="/api/v1", tags=["models"])
    app.include_router(websocket.router, prefix="/api/v1", tags=["websocket"])

    @app.get("/health")
    async def health_check():
        """Health check endpoint for Docker/load balancer."""
        return {"status": "ok", "version": "0.1.0"}

    @app.get("/dashboard")
    @app.get("/")
    async def dashboard_view():
        """Serve the interactive HTML mission control dashboard."""
        from pathlib import Path
        from fastapi.responses import HTMLResponse
        for p in [Path("dashboard.html"), Path("C:/Users/singh/.gemini/antigravity/brain/00611cef-b266-4ba6-91ef-c84fca3d4fbc/dashboard.html")]:
            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    return HTMLResponse(content=f.read())
        return HTMLResponse(content="<h1>AI Crypto Trader Dashboard</h1><p>Initializing...</p>")

    return app


app = create_app()
