"""Dependency wiring. Routes use `Depends(get_container)`; tests use `set_container()` or
`app.dependency_overrides[get_container]`."""
from __future__ import annotations

from backend.app.contracts.interfaces import Container

_container: Container | None = None


def set_container(c: Container | None) -> None:
    global _container
    _container = c


def get_container() -> Container:
    global _container
    if _container is None:
        from backend.app.config import get_settings
        from backend.app.intelligence import build_intelligence
        from backend.app.store.case_store import CaseStore
        settings = get_settings()
        store = CaseStore()
        intel = build_intelligence(settings, live=True)
        df = store.history_frame()
        if not df.empty:
            intel.graph.build(df.head(5000))
            intel.agents.build(df.head(5000))
        _container = Container(cases=store, intel=intel)
    return _container
