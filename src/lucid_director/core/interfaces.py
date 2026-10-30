"""The four component contracts. Core code depends only on these.

Construction convention: every implementation is built as ``cls(section)``, where
``section`` is its config section (e.g. ``FrameSourceConfig``). It reads the shared
fields and validates its own ``options``.

All components hold resources (windows, GPU memory, game processes, HTTP sessions), so
each has ``close()`` and works as a context manager.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from types import TracebackType
from typing import Self

import numpy as np

from lucid_director.core.config import StyleSettings
from lucid_director.core.types import Frame, LLMResponse, Message, PipelineResult, ToolSpec


class Component(ABC):
    def close(self) -> None:
        """Release resources. Safe to call more than once."""

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


class FrameSource(Component):
    """Produces frames: synthetic, a game, or a screen capture."""

    @abstractmethod
    def next_frame(self) -> Frame | None:
        """Return the next frame, or ``None`` when the episode has ended."""

    def provides(self) -> frozenset[str]:
        """Optional extras this source fills in: any of ``"depth"``, ``"labels"``, ``"scene"``.

        The agent's tool list is built from this, e.g. depth conditioning is only offered
        when ``"depth"`` is present.
        """
        return frozenset()

    def scene_summary(self) -> str | None:
        """Short text about the current scene (e.g. ``"2 imps visible, health low"``).

        Called by the agent every few seconds, not per frame. ``None`` if unsupported.
        """
        return None


class Pipeline(Component):
    """Turns a source frame into a styled image."""

    @abstractmethod
    def process(self, frame: Frame, settings: StyleSettings) -> PipelineResult:
        """Style one frame.

        ``settings`` is passed on every call instead of being stored on the pipeline: the
        render loop takes one snapshot of the current settings per frame, so a change by
        the agent can only take effect between frames, never halfway through one.
        Implementations may cache work keyed on settings (e.g. the encoded prompt).
        """


class Display(Component):
    """Shows (or writes, or discards) styled images."""

    @abstractmethod
    def show(self, image: np.ndarray, overlay: dict[str, str] | None = None) -> bool:
        """Present one image, optionally with overlay text (e.g. ``{"fps": "14.2"}``).

        Return ``False`` if the user asked to quit (e.g. closed the window), else ``True``.
        """


class LLMClient(Component):
    """A chat model that can call tools."""

    @abstractmethod
    def complete(self, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> LLMResponse:
        """Send the conversation and available tools; return the model's reply.

        Blocking. The agent calls this from its own thread, so the render loop never
        waits on it. Implementations translate the neutral message types to their API.
        """
