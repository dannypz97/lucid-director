"""Clocks and rolling timing statistics.

Components that need the time take a ``Clock`` instead of calling ``time`` directly, so
tests can pass a ``FakeClock`` and control time exactly.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


class Clock(Protocol):
    def now(self) -> float:
        """Current time in seconds. Only differences between readings are meaningful."""
        ...


class RealClock:
    """Wall-clock time from ``time.perf_counter`` (monotonic, high resolution)."""

    def now(self) -> float:
        return time.perf_counter()


class FakeClock:
    """A clock that only moves when ``advance`` is called."""

    def __init__(self, start: float = 0.0) -> None:
        self._now = start

    def now(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError(f"FakeClock cannot go backwards (advance({seconds}))")
        self._now += seconds


@dataclass(frozen=True, slots=True)
class TimingSummary:
    """Snapshot of ``TimingStats`` over its current window. All zeros when empty."""

    frames: int  # frames in the window
    window_s: float  # time from the oldest to the newest frame completion in the window
    fps: float
    p50_ms: float  # median end-to-end latency
    p95_ms: float  # 95% of frames were at or under this latency
    stage_ms: dict[str, float] = field(default_factory=dict)  # mean per stage


@dataclass(frozen=True, slots=True)
class _Record:
    start: float
    end: float
    stage_ms: dict[str, float]


class TimingStats:
    """Rolling-window frame timing: fps, latency percentiles, and mean per-stage ms.

    Only the most recent ``window`` frames count, so the numbers follow the current
    behaviour instead of averaging over the whole run.
    """

    def __init__(self, window: int = 120) -> None:
        if window < 2:
            raise ValueError(f"window must be at least 2 frames to measure fps; got {window}")
        self._records: deque[_Record] = deque(maxlen=window)

    def record(self, start: float, end: float, stage_ms: dict[str, float] | None = None) -> None:
        """Add one frame.

        ``start`` is when the source frame was taken and ``end`` when its styled image was
        shown, both from the same clock, so ``end - start`` is the end-to-end latency.
        """
        if end < start:
            raise ValueError(f"end ({end}) is before start ({start})")
        self._records.append(_Record(start, end, dict(stage_ms or {})))

    def summary(self) -> TimingSummary:
        records = self._records
        if not records:
            return TimingSummary(frames=0, window_s=0.0, fps=0.0, p50_ms=0.0, p95_ms=0.0)

        # fps counts the gaps between frame completions, not the frames themselves.
        window_s = records[-1].end - records[0].end
        fps = (len(records) - 1) / window_s if window_s > 0 else 0.0

        latencies_ms = [(r.end - r.start) * 1000.0 for r in records]
        # "inverted_cdf" returns an actual recorded latency rather than interpolating.
        p50, p95 = np.percentile(latencies_ms, [50, 95], method="inverted_cdf")

        stage_values: dict[str, list[float]] = {}
        for r in records:
            for name, ms in r.stage_ms.items():
                stage_values.setdefault(name, []).append(ms)
        stage_means = {name: sum(v) / len(v) for name, v in stage_values.items()}

        return TimingSummary(
            frames=len(records),
            window_s=window_s,
            fps=fps,
            p50_ms=float(p50),
            p95_ms=float(p95),
            stage_ms=stage_means,
        )
