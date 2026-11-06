import numpy as np
import pytest
from pydantic import ValidationError

from lucid_director.core.config import PipelineConfig, StyleSettings
from lucid_director.core.registry import registry
from lucid_director.core.timing import FakeClock
from lucid_director.core.types import Frame, PipelineResult
from lucid_director.pipelines.stub import StubPipeline

BASE = StyleSettings(prompt="neon", strength=0.5, steps=2, seed_policy="fixed")


def make_pipeline(clock: FakeClock | None = None, **options) -> StubPipeline:
    options.setdefault("noise", 0.0)
    return StubPipeline(PipelineConfig(options=options), clock=clock or FakeClock())


def make_frame(width: int = 384, height: int = 240, value: int = 100) -> Frame:
    image = np.full((height, width, 3), value, dtype=np.uint8)
    return Frame(image=image, index=0, timestamp=0.0)


def run(pipeline: StubPipeline, **changes) -> PipelineResult:
    return pipeline.process(make_frame(), BASE.with_changes(**changes))


def latency_ms(result: PipelineResult) -> float:
    return sum(result.stage_ms.values())


def test_more_steps_means_lower_fps() -> None:
    pipeline = make_pipeline()
    latencies = [latency_ms(run(pipeline, steps=n)) for n in (1, 2, 3, 4)]
    assert latencies == sorted(latencies)
    assert latencies[0] < latencies[-1]


def test_latency_grows_with_resolution() -> None:
    pipeline = make_pipeline()
    small = latency_ms(pipeline.process(make_frame(192, 120), BASE))
    large = latency_ms(pipeline.process(make_frame(384, 240), BASE))
    assert large > 3 * small  # 4x the pixels, plus a fixed overhead


def test_more_strength_means_more_style_with_diminishing_returns() -> None:
    pipeline = make_pipeline()
    style = [run(pipeline, strength=s).metrics["style"] for s in (0.2, 0.4, 0.6, 0.8)]
    assert style == sorted(style)
    gains = np.diff(style)
    assert all(later < earlier for earlier, later in zip(gains, gains[1:]))


def test_more_strength_means_more_flicker() -> None:
    pipeline = make_pipeline()
    flicker = [run(pipeline, strength=s).metrics["flicker"] for s in (0.2, 0.4, 0.6, 0.8)]
    assert flicker == sorted(flicker)
    assert flicker[0] < flicker[-1]


def test_fixed_seed_means_less_flicker() -> None:
    pipeline = make_pipeline()
    fixed = run(pipeline, seed_policy="fixed").metrics["flicker"]
    per_frame = run(pipeline, seed_policy="per_frame").metrics["flicker"]
    assert per_frame > 2 * fixed


def test_best_setting_under_budgets_is_not_a_corner() -> None:
    # Demo budgets from configs/stub.yaml: fps >= 20, flicker <= 0.15. Maximise style.
    pipeline = make_pipeline()
    feasible = []
    for steps in range(1, 9):
        for strength in np.linspace(0.0, 1.0, 21):
            for policy in ("fixed", "per_frame"):
                result = run(pipeline, steps=steps, strength=float(strength), seed_policy=policy)
                if 1000 / latency_ms(result) >= 20 and result.metrics["flicker"] <= 0.15:
                    feasible.append((result.metrics["style"], steps, float(strength), policy))

    _, steps, strength, policy = max(feasible)
    assert 1 < steps < 8
    assert 0.0 < strength < 1.0
    assert policy == "fixed"


def test_advances_fake_clock_by_simulated_latency(fake_clock: FakeClock) -> None:
    pipeline = make_pipeline(clock=fake_clock)
    result = run(pipeline)
    assert fake_clock.now() == pytest.approx(latency_ms(result) / 1000)


def test_output_is_tinted_more_at_higher_strength() -> None:
    pipeline = make_pipeline()
    frame = make_frame(value=100)

    untouched = pipeline.process(frame, BASE.with_changes(strength=0.0)).image
    light = pipeline.process(frame, BASE.with_changes(strength=0.3)).image
    heavy = pipeline.process(frame, BASE.with_changes(strength=0.9)).image

    assert np.array_equal(untouched, frame.image)
    assert untouched is not frame.image
    assert heavy.shape == frame.image.shape and heavy.dtype == np.uint8
    original = frame.image.astype(int)
    assert np.abs(light.astype(int) - original).sum() < np.abs(heavy.astype(int) - original).sum()


def test_noise_is_reproducible_for_a_seed() -> None:
    a = make_pipeline(noise=0.05, seed=3)
    b = make_pipeline(noise=0.05, seed=3)
    for _ in range(5):
        ra, rb = run(a), run(b)
        assert ra.stage_ms == rb.stage_ms
        assert ra.metrics == rb.metrics
    assert ra.metrics != run(make_pipeline(noise=0.0)).metrics


def test_noisy_scores_stay_in_range() -> None:
    pipeline = make_pipeline(noise=0.5)
    for strength in (0.0, 1.0):
        for _ in range(50):
            result = run(pipeline, strength=strength, seed_policy="per_frame")
            assert all(0.0 <= v <= 1.0 for v in result.metrics.values())
            assert all(ms >= 0.0 for ms in result.stage_ms.values())


def test_unknown_option_rejected() -> None:
    with pytest.raises(ValidationError, match="nois"):
        make_pipeline(nois=0.1)


def test_registered_as_stub() -> None:
    assert registry.resolve("pipeline", "stub") is StubPipeline

    pipeline = registry.create("pipeline", "stub", PipelineConfig(), FakeClock())
    assert isinstance(pipeline, StubPipeline)
