"""Typed run configuration, loaded from YAML.

Design notes:

* Every model forbids unknown keys, so a typo such as ``stregth`` fails at startup
  instead of being silently ignored.
* Each swappable section has a ``type`` (which implementation to use) and an
  ``options`` dict passed through to that implementation. Core config validates only
  what every implementation shares; implementation-specific keys (e.g. a ViZDoom
  ``scenario``) are validated by the implementation itself.
* ``type`` is a plain string, not a fixed list: the registry (sub-task 0.3) decides
  which names exist, so adding an implementation never requires editing this file.
* Bounds here are generic sanity limits. Hardware profiles and calibration narrow them
  later (e.g. max steps on a 6 GB GPU).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


class ConfigError(ValueError):
    """Raised when a config file cannot be read or fails validation."""


class _Strict(BaseModel):
    # frozen: config is read-only once loaded. Settings the agent changes at runtime
    # live in a separate thread-safe object (Phase 1), never in the config.
    model_config = ConfigDict(extra="forbid", frozen=True)


class HardwareConfig(_Strict):
    profile: Literal["auto", "low", "mid", "high"] = "auto"


class FrameSourceConfig(_Strict):
    type: str = "stub"
    resolution: tuple[int, int] = (320, 200)  # (width, height) in pixels
    seed: int | None = 0  # None = nondeterministic
    options: dict[str, Any] = Field(default_factory=dict)

    @field_validator("resolution")
    @classmethod
    def _check_resolution(cls, value: tuple[int, int]) -> tuple[int, int]:
        width, height = value
        if not (16 <= width <= 4096 and 16 <= height <= 4096):
            raise ValueError(f"resolution {value} out of range; each side must be 16..4096")
        return value


class StyleSettings(_Strict):
    """The knobs the director turns. Used here as the starting point of a run."""

    prompt: str = Field(default="", max_length=200)
    strength: float = Field(default=0.5, ge=0.0, le=1.0)  # img2img denoise strength
    steps: int = Field(default=2, ge=1, le=50)
    seed_policy: Literal["fixed", "per_frame"] = "fixed"

    def with_changes(self, **changes: Any) -> StyleSettings:
        """Return a validated copy. (``model_copy(update=...)`` skips validation.)"""
        return StyleSettings.model_validate({**self.model_dump(), **changes})


class PipelineConfig(_Strict):
    type: str = "stub"
    model: str | None = None  # e.g. "stabilityai/sd-turbo"; unused by stubs
    resolution: tuple[int, int] = (384, 240)  # (width, height) the model works at
    dtype: Literal["fp16", "bf16", "fp32"] = "fp16"
    conditioning: Literal["none", "depth", "edges"] = "none"
    initial: StyleSettings = Field(default_factory=StyleSettings)
    options: dict[str, Any] = Field(default_factory=dict)

    @field_validator("resolution")
    @classmethod
    def _check_resolution(cls, value: tuple[int, int]) -> tuple[int, int]:
        width, height = value
        if not (64 <= width <= 2048 and 64 <= height <= 2048):
            raise ValueError(f"resolution {value} out of range; each side must be 64..2048")
        # Stable Diffusion's VAE downsamples by 8, so other sizes break or get cropped.
        if width % 8 or height % 8:
            raise ValueError(f"resolution {value} must be a multiple of 8 on each side")
        return value


class DisplayConfig(_Strict):
    type: str = "none"
    options: dict[str, Any] = Field(default_factory=dict)


class LLMConfig(_Strict):
    provider: str = "scripted"
    model: str | None = None
    call_interval_s: float = Field(default=5.0, gt=0)
    options: dict[str, Any] = Field(default_factory=dict)


class Budgets(_Strict):
    min_fps: float = Field(default=15.0, gt=0)
    max_p95_latency_ms: float = Field(default=120.0, gt=0)
    max_flicker: float = Field(default=0.15, ge=0.0, le=1.0)


class AgentConfig(_Strict):
    goal: str = ""
    budgets: Budgets = Field(default_factory=Budgets)
    max_steps_per_episode: int = Field(default=30, ge=1)


class LoggingConfig(_Strict):
    dir: Path = Path("runs")


class Config(_Strict):
    """Root config. Every section has defaults, so an empty file is a valid stub run."""

    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    frame_source: FrameSourceConfig = Field(default_factory=FrameSourceConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    display: DisplayConfig = Field(default_factory=DisplayConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def load_config(path: str | Path) -> Config:
    """Read a YAML file and validate it into a :class:`Config`.

    Raises :class:`ConfigError` (with the file name in the message) on any problem.
    """
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"{path}: cannot read config ({exc})") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML ({exc})") from exc

    if raw is None:  # empty file
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top level must be a mapping, got {type(raw).__name__}")

    try:
        return Config.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"{path}: {exc}") from exc
