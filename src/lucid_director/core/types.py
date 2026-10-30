"""Data passed between components: frames, pipeline results, and LLM messages.

Image convention everywhere: RGB, ``uint8``, shape ``(height, width, 3)``. OpenCV uses
BGR, so conversion happens only inside OpenCV-based displays/sources.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np


def _check_image(name: str, image: np.ndarray) -> None:
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f"{name} must be uint8 with shape (H, W, 3); got {image.dtype} {image.shape}"
        )


@dataclass(frozen=True, slots=True)
class Frame:
    """One frame from a FrameSource.

    ``depth`` and ``labels`` travel with the image so they always describe the same
    instant; sources that lack them leave them as ``None``.
    """

    image: np.ndarray
    index: int  # 0-based position in the episode
    timestamp: float  # seconds, from the run's clock
    depth: np.ndarray | None = None  # (H, W), larger = farther; units are source-specific
    labels: np.ndarray | None = None  # (H, W) integer object ids, 0 = background

    def __post_init__(self) -> None:
        _check_image("Frame.image", self.image)
        for name, buffer in (("depth", self.depth), ("labels", self.labels)):
            if buffer is not None and buffer.shape != self.image.shape[:2]:
                raise ValueError(
                    f"Frame.{name} shape {buffer.shape} != image shape {self.image.shape[:2]}"
                )

    @property
    def size(self) -> tuple[int, int]:
        """(width, height), like config resolutions. Note ``image.shape`` is (height, width, 3)."""
        height, width = self.image.shape[:2]
        return width, height


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """Output of one Pipeline.process call.

    ``stage_ms`` holds per-stage wall time (e.g. ``{"unet": 41.2, "vae": 18.0}``).
    ``metrics`` holds anything the pipeline itself can report; the stub uses it for its
    simulated quality numbers, real pipelines usually leave it empty and let the metrics
    module measure output frames instead.
    """

    image: np.ndarray
    stage_ms: dict[str, float] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _check_image("PipelineResult.image", self.image)


# --- LLM messages (provider-neutral; each LLMClient translates to its API's format) ---


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]  # JSON Schema for the arguments


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str  # provider-assigned; the tool result must echo it back
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class Message:
    role: Literal["user", "assistant", "tool"]
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()  # set on assistant messages
    tool_call_id: str | None = None  # set on tool messages


@dataclass(frozen=True, slots=True)
class LLMResponse:
    message: Message  # role == "assistant"
    stop_reason: str = ""  # e.g. "tool_use", "end_turn"; provider-specific
    usage: dict[str, int] = field(default_factory=dict)  # token counts, for cost tracking
