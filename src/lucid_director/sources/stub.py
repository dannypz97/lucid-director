"""Synthetic frame source: shapes drifting over a gradient, with optional fake depth.

Everything random (colours, sizes, starting positions, velocities) is drawn once from
the config seed, so the same seed always produces the same frames.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from lucid_director.core.config import FrameSourceConfig
from lucid_director.core.interfaces import FrameSource
from lucid_director.core.timing import Clock, RealClock
from lucid_director.core.types import Frame

BACKGROUND_DEPTH = 1.0  # depth of the gradient; shapes are always nearer than this


class StubSourceOptions(BaseModel):
    """``frame_source.options`` for ``type: stub``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    frames: int = Field(default=300, ge=1)  # episode length; next_frame() returns None after
    shapes: int = Field(default=3, ge=1, le=16)
    max_speed: float = Field(default=3.0, gt=0)  # pixels per frame, per axis
    depth: bool = False  # also produce a depth map


@dataclass(slots=True)
class _Shape:
    kind: str  # "circle" or "square"
    x: float  # centre, pixels
    y: float
    vx: float  # velocity, pixels per frame
    vy: float
    radius: float  # half-width for squares
    color: np.ndarray  # (3,) uint8
    depth: float  # in (0, BACKGROUND_DEPTH)


class StubSource(FrameSource):
    def __init__(self, section: FrameSourceConfig, clock: Clock | None = None) -> None:
        self.options = StubSourceOptions.model_validate(section.options)
        self.width, self.height = section.resolution
        self._clock = clock if clock is not None else RealClock()
        self._index = 0

        rng = np.random.default_rng(section.seed)
        self._background = _gradient(self.width, self.height, rng)
        self._shapes = [self._random_shape(rng) for _ in range(self.options.shapes)]
        # Draw farthest first so nearer shapes end up on top.
        self._shapes.sort(key=lambda s: s.depth, reverse=True)

        # Pixel coordinate grids, reused for every frame's shape masks.
        # _xs holds column numbers, and _ys holds row numbers.
        self._ys, self._xs = np.mgrid[0 : self.height, 0 : self.width].astype(np.float32)

    def next_frame(self) -> Frame | None:
        if self._index >= self.options.frames:
            return None

        image = self._background.copy()
        depth = np.full((self.height, self.width), BACKGROUND_DEPTH, dtype=np.float32)
        for shape in self._shapes:
            mask = self._mask(shape)
            # Boolean indexing: only pixels where the mask is True are overwritten.
            image[mask] = shape.color
            depth[mask] = shape.depth

        frame = Frame(
            image=image,
            index=self._index,
            timestamp=self._clock.now(),
            depth=depth if self.options.depth else None,
        )
        self._index += 1
        self._move_shapes()
        return frame

    def provides(self) -> frozenset[str]:
        return frozenset({"scene", "depth"} if self.options.depth else {"scene"})

    def scene_summary(self) -> str | None:
        kinds = ", ".join(f"{_color_name(s.color)} {s.kind}" for s in self._shapes)
        return f"synthetic scene: {len(self._shapes)} shapes ({kinds}) drifting over a gradient"

    def _random_shape(self, rng: np.random.Generator) -> _Shape:
        short_side = min(self.width, self.height)
        radius = float(rng.uniform(0.06, 0.15) * short_side)
        speed = self.options.max_speed
        return _Shape(
            kind=str(rng.choice(["circle", "square"])),
            x=float(rng.uniform(radius, self.width - radius)),
            y=float(rng.uniform(radius, self.height - radius)),
            vx=float(rng.uniform(-speed, speed)),
            vy=float(rng.uniform(-speed, speed)),
            radius=radius,
            color=rng.integers(0, 256, size=3, dtype=np.uint8),
            depth=float(rng.uniform(0.2, 0.8)),
        )

    def _mask(self, shape: _Shape) -> np.ndarray:
        """(H, W) bool array, True for pixels inside ``shape``."""
        # Each pixel's offset from the shape's centre, computed for all pixels at once.
        dx = self._xs - shape.x
        dy = self._ys - shape.y
        if shape.kind == "circle":
            # Inside if distance² <= radius² (avoids a square root).
            return dx * dx + dy * dy <= shape.radius * shape.radius
        # Square: within radius of the centre on both axes.
        return (np.abs(dx) <= shape.radius) & (np.abs(dy) <= shape.radius)

    def _move_shapes(self) -> None:
        """Advance each shape one frame, bouncing off the frame edges."""
        for s in self._shapes:
            s.x += s.vx
            s.y += s.vy
            if not s.radius <= s.x <= self.width - s.radius:
                s.vx = -s.vx
                s.x = min(max(s.x, s.radius), self.width - s.radius)
            if not s.radius <= s.y <= self.height - s.radius:
                s.vy = -s.vy
                s.y = min(max(s.y, s.radius), self.height - s.radius)


def _gradient(width: int, height: int, rng: np.random.Generator) -> np.ndarray:
    """A diagonal blend between two random colours, as an (H, W, 3) uint8 image."""
    start = rng.integers(0, 256, size=3).astype(np.float32)
    end = rng.integers(0, 256, size=3).astype(np.float32)
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    t = (xs / max(width - 1, 1) + ys / max(height - 1, 1)) / 2  # 0 at top-left, 1 at bottom-right
    image = start + t[..., None] * (end - start)
    return image.round().astype(np.uint8)


_NAMED_COLORS = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "red": (220, 40, 40),
    "green": (40, 180, 60),
    "blue": (40, 70, 220),
    "yellow": (230, 220, 50),
    "cyan": (50, 210, 220),
    "magenta": (210, 50, 200),
    "orange": (240, 140, 30),
    "grey": (128, 128, 128),
}


def _color_name(color: np.ndarray) -> str:
    """Nearest of a few named colours, for the scene summary."""
    rgb = color.astype(np.int32)
    return min(_NAMED_COLORS, key=lambda name: int(((rgb - _NAMED_COLORS[name]) ** 2).sum()))
