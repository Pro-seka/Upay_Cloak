"""Central FastAPI entrypoint for UpayShield.
Combines Core API (Part 1), Intelligence Services (Part 2), Frontend Routes,
and static frontend hosting into a single unified application.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app.api import routes_core, routes_frontend, routes_intel
from backend.app.config import ROOT, get_settings
from backend.app.contracts.interfaces import Container
from backend.app.deps import get_container, set_container
from backend.app.intelligence import build_intelligence
from backend.app.store.case_store import CaseStore

logger = logging.getLogger("upayshield")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup & shutdown sequence for UpayShield."""
    settings = get_settings()
    logger.info("Initializing UpayShield Backend...")

    # 1. Initialize store & data
    store = CaseStore(txns_path=settings.data_path, reports_dir=settings.reports_dir)

    # 2. Build live intelligence services (NetworkX graph, agent analytics, assistant)
    intel = build_intelligence(settings, live=True)

    # 3. Populate graph & agent models with transaction history
    df = store.history_frame()
    if not df.empty:
        logger.info(f"Building graph and agent models from {len(df)} transactions...")
        intel.graph.build(df)
        intel.agents.build(df)

    # 4. Wire dependency container
    container = Container(cases=store, intel=intel)
    set_container(container)
    logger.info("UpayShield Backend initialisation complete!")

    yield

    # Shutdown
    set_container(None)


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="UpayShield Trust & Risk Intelligence API",
        description="Case-centric fraud detection, graph analytics, and AI compliance assistant for mobile money.",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Enable CORS for static frontend / external calls
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API Routers
    app.include_router(routes_core.router)        # /api/v1/score, /api/v1/cases, /api/v1/config...
    app.include_router(routes_intel.router)       # /api/v1/graph, /api/v1/agents, /api/v1/cases/{id}/ask...
    app.include_router(routes_frontend.router)    # /api/overview, /api/transactions, /api/cases...

    # Health check endpoint
    @app.get("/health", tags=["system"])
    def health_check() -> dict[str, Any]:
        cnt = get_container()
        return {
            "status": "ok",
            "model_loaded": settings.model_path.exists(),
            "intelligence": "live" if cnt.intel.live else "stubs",
            "cases_count": len(cnt.cases.list_cases()) if hasattr(cnt.cases, "list_cases") else 0,
        }

    # Direct aliases for frontend root-relative paths if called without /api prefix
    @app.get("/overview", include_in_schema=False)
    def root_overview():
        return routes_frontend.get_frontend_overview(get_container())

    @app.get("/transactions", include_in_schema=False)
    def root_transactions():
        return routes_frontend.get_frontend_transactions(get_container())

    @app.get("/cases", include_in_schema=False)
    def root_cases(alert_type: str | None = None):
        return routes_frontend.get_frontend_cases(alert_type, get_container())

    @app.get("/cases/{case_id}", include_in_schema=False)
    def root_case_by_id(case_id: str):
        return routes_frontend.get_frontend_case(case_id, get_container())

    @app.get("/graph/{entity_id}", include_in_schema=False)
    def root_graph_by_id(entity_id: str):
        return routes_frontend.get_frontend_graph_by_id(entity_id, 2, get_container())

    @app.get("/graph", include_in_schema=False)
    def root_graph(center: str | None = None):
        return routes_frontend.get_frontend_full_graph(center, 2, get_container())

    # Mount static frontend directory at /
    frontend_dir = ROOT / "Frontend"
    if not frontend_dir.exists():
        frontend_dir = ROOT / "frontend"
    if frontend_dir.exists():
        app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
        logger.info(f"Mounted static frontend from {frontend_dir}")

    return app


app = create_app()
