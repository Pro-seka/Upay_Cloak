"""Dependency wiring.  Routes use `Depends(get_container)`; tests use `set_container()` or
`app.dependency_overrides[get_container]`."""
from __future__ import annotations

from backend.app.contracts.interfaces import Container

_container: Container | None = None


def set_container(c: Container | None) -> None:
    global _container
    _container = c


def get_container() -> Container:
    if _container is None:
        raise RuntimeError("Container not initialised (startup not finished)")
    return _container
