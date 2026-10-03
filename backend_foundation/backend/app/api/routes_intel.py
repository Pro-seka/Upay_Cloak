"""PART 2 OWNS THIS FILE: graph / agents / evidence / narrative / ask / report endpoints.

Part 1's `create_app()` already does `app.include_router(routes_intel.router)`.
All routes live under the `/api/v1` prefix (see docs/backend/API_CONTRACT.md).
"""
from fastapi import APIRouter

router = APIRouter(prefix="/api/v1", tags=["intelligence"])
