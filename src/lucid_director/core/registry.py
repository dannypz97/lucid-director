"""Maps config names (``type: stub``) to implementation classes.

Entries can be classes or ``"module:ClassName"`` strings. Strings are imported only when
that name is actually selected, so choosing ``pipeline.type: stub`` never imports torch.
"""

from __future__ import annotations

import importlib
from typing import Any, Literal, get_args

Kind = Literal["frame_source", "pipeline", "display", "llm"]
KINDS: tuple[str, ...] = get_args(Kind)


class RegistryError(LookupError):
    pass


class Registry:
    def __init__(self) -> None:
        self._entries: dict[str, dict[str, type | str]] = {kind: {} for kind in KINDS}

    def register(self, kind: Kind, name: str, target: type | str) -> None:
        entries = self._kind(kind)
        if name in entries:
            raise RegistryError(f"{kind} {name!r} is already registered")
        entries[name] = target

    def available(self, kind: Kind) -> list[str]:
        return sorted(self._kind(kind))

    def resolve(self, kind: Kind, name: str) -> type:
        entries = self._kind(kind)
        if name not in entries:
            known = ", ".join(self.available(kind)) or "(none)"
            raise RegistryError(f"unknown {kind} type {name!r}; known: {known}")
        target = entries[name]
        if isinstance(target, str):
            target = _import(target)
            entries[name] = target  # cache the class after the first import
        return target

    def create(self, kind: Kind, name: str, *args: Any, **kwargs: Any) -> Any:
        return self.resolve(kind, name)(*args, **kwargs)

    def _kind(self, kind: str) -> dict[str, type | str]:
        if kind not in self._entries:
            raise RegistryError(f"unknown kind {kind!r}; valid: {', '.join(KINDS)}")
        return self._entries[kind]


def _import(path: str) -> type:
    module_name, _, attr = path.partition(":")
    if not attr:
        raise RegistryError(f"bad import path {path!r}; expected 'module:ClassName'")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        # Usually a missing optional dependency, e.g. torch for the sdturbo pipeline.
        raise RegistryError(f"cannot import {path!r}: {exc}") from exc
    try:
        return getattr(module, attr)
    except AttributeError as exc:
        raise RegistryError(f"{module_name!r} has no attribute {attr!r}") from exc


registry = Registry()

# Built-in implementations are added here as they're written (0.5: stub source, 0.6: stub
# pipeline, 0.7: null display).
