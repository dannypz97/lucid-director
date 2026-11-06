"""Fake pipeline: returns the input lightly tinted, with simulated cost and quality.

The simulated numbers follow the same trade-offs as real img2img diffusion:

* latency grows with pixel count, and the UNet part also with steps;
* style rises with strength (diminishing returns), plus a small, shrinking bonus per step;
* flicker rises with strength squared, three times faster under the ``per_frame`` seed policy.

Optional noise is drawn from ``options.seed``, so runs are reproducible.
"""

from __future__ import annotations

import math
import time

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from lucid_director.core.config import PipelineConfig, StyleSettings
from lucid_director.core.interfaces import Pipeline
from lucid_director.core.timing import Clock, FakeClock, RealClock
from lucid_director.core.types import Frame, PipelineResult

TINT_COLOR = np.array([255, 0, 160], dtype=np.float32)  # magenta, so styled frames are recognisable
PER_FRAME_FLICKER_FACTOR = 3.0  # new noise every frame makes outputs far less consistent
STYLE_CURVE = 3.0  # how quickly style saturates as strength rises


class StubPipelineOptions(BaseModel):
    """``pipeline.options`` for ``type: stub``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    overhead_ms: float = Field(default=4.0, ge=0)  # fixed cost per frame
    vae_ms_per_mp: float = Field(default=110.0, ge=0)  # VAE encode + decode, per megapixel
    unet_ms_per_mp: float = Field(default=120.0, ge=0)  # one UNet step, per megapixel
    flicker_scale: float = Field(default=0.25, ge=0, le=1)  # flicker at strength 1, fixed seed
    noise: float = Field(default=0.02, ge=0, le=0.5)  # std of the random jitter; 0 = exact
    tint: float = Field(default=0.15, ge=0, le=1)  # tint amount at strength 1
    seed: int | None = 0  # for the jitter; None = nondeterministic


class StubPipeline(Pipeline):
    def __init__(self, section: PipelineConfig, clock: Clock | None = None) -> None:
        self.options = StubPipelineOptions.model_validate(section.options)
        self._clock = clock if clock is not None else RealClock()
        self._rng = np.random.default_rng(self.options.seed)

    def process(self, frame: Frame, settings: StyleSettings) -> PipelineResult:
        stage_ms = self._stage_ms(frame, settings)
        latency_s = sum(stage_ms.values()) / 1000.0
        # Simulated time passes on a fake clock; with a real clock, actually wait.
        if isinstance(self._clock, FakeClock):
            self._clock.advance(latency_s)
        else:
            time.sleep(latency_s)

        metrics = {
            "style": self._jitter_score(_style(settings)),
            "flicker": self._jitter_score(_flicker(settings, self.options.flicker_scale)),
        }
        return PipelineResult(
            image=_tint(frame.image, self.options.tint * settings.strength),
            stage_ms=stage_ms,
            metrics=metrics,
        )

    def _stage_ms(self, frame: Frame, settings: StyleSettings) -> dict[str, float]:
        width, height = frame.size
        megapixels = width * height / 1e6
        o = self.options
        return {
            "overhead": self._jitter_ms(o.overhead_ms),
            "vae": self._jitter_ms(o.vae_ms_per_mp * megapixels),
            "unet": self._jitter_ms(o.unet_ms_per_mp * megapixels * settings.steps),
        }

    def _jitter_ms(self, ms: float) -> float:
        """``ms`` scaled by a random factor near 1, never negative."""
        if self.options.noise == 0:
            return ms
        return max(0.0, ms * (1.0 + self._rng.normal(0.0, self.options.noise)))

    def _jitter_score(self, score: float) -> float:
        """``score`` plus a small random offset, kept within 0..1."""
        if self.options.noise == 0:
            return score
        return min(1.0, max(0.0, score + self._rng.normal(0.0, self.options.noise)))


def _style(settings: StyleSettings) -> float:
    """0..1. Saturating in strength, times a step factor from 0.85 (1 step) toward 1."""
    strength_term = (1 - math.exp(-STYLE_CURVE * settings.strength)) / (1 - math.exp(-STYLE_CURVE))
    step_factor = 0.85 + 0.15 * (1 - 1 / settings.steps)
    return strength_term * step_factor


def _flicker(settings: StyleSettings, scale: float) -> float:
    """0..1. Grows with strength squared; much larger under the per_frame seed policy."""
    flicker = scale * settings.strength**2
    if settings.seed_policy == "per_frame":
        flicker *= PER_FRAME_FLICKER_FACTOR
    return min(1.0, flicker)


def _tint(image: np.ndarray, amount: float) -> np.ndarray:
    """Blend ``image`` toward ``TINT_COLOR`` by ``amount`` (0 = unchanged)."""
    if amount == 0:
        return image.copy()
    blended = image.astype(np.float32) + amount * (TINT_COLOR - image)
    return blended.round().astype(np.uint8)
