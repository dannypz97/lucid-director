import numpy as np
import pytest
from pydantic import ValidationError

from lucid_director.core.config import FrameSourceConfig
from lucid_director.core.registry import registry
from lucid_director.core.timing import FakeClock
from lucid_director.core.types import Frame
from lucid_director.sources.stub import BACKGROUND_DEPTH, StubSource


def make_source(clock: FakeClock | None = None, seed: int | None = 0, **options) -> StubSource:
    section = FrameSourceConfig(resolution=(64, 48), seed=seed, options=options)
    return StubSource(section, clock=clock)


def all_frames(source: StubSource) -> list[Frame]:
    frames = []
    while (frame := source.next_frame()) is not None:
        frames.append(frame)
    return frames


def test_yields_configured_count_and_resolution() -> None:
    frames = all_frames(make_source(frames=5))

    assert [f.index for f in frames] == [0, 1, 2, 3, 4]
    for frame in frames:
        assert frame.image.shape == (48, 64, 3)
        assert frame.image.dtype == np.uint8
        assert frame.size == (64, 48)


def test_keeps_returning_none_after_the_end() -> None:
    source = make_source(frames=1)
    assert source.next_frame() is not None
    assert source.next_frame() is None
    assert source.next_frame() is None


def test_same_seed_gives_identical_frames() -> None:
    first = all_frames(make_source(seed=7, frames=10, depth=True))
    second = all_frames(make_source(seed=7, frames=10, depth=True))

    for a, b in zip(first, second, strict=True):
        assert np.array_equal(a.image, b.image)
        assert np.array_equal(a.depth, b.depth)


def test_different_seeds_give_different_frames() -> None:
    a = make_source(seed=1).next_frame()
    b = make_source(seed=2).next_frame()
    assert not np.array_equal(a.image, b.image)


def test_shapes_move_between_frames() -> None:
    frames = all_frames(make_source(frames=3))
    assert not np.array_equal(frames[0].image, frames[1].image)
    assert not np.array_equal(frames[1].image, frames[2].image)


def test_shapes_stay_in_frame_over_a_long_run() -> None:
    source = make_source(frames=500, max_speed=10.0)
    all_frames(source)
    for shape in source._shapes:
        assert shape.radius <= shape.x <= source.width - shape.radius
        assert shape.radius <= shape.y <= source.height - shape.radius


def test_timestamps_come_from_the_injected_clock(fake_clock: FakeClock) -> None:
    source = make_source(clock=fake_clock, frames=3)
    timestamps = []
    for _ in range(3):
        timestamps.append(source.next_frame().timestamp)
        fake_clock.advance(0.05)

    assert timestamps == pytest.approx([0.0, 0.05, 0.10])


def test_depth_is_off_by_default() -> None:
    source = make_source()
    assert source.next_frame().depth is None
    assert source.provides() == frozenset({"scene"})


def test_depth_marks_shapes_nearer_than_background() -> None:
    source = make_source(depth=True)
    depth = source.next_frame().depth

    assert source.provides() == frozenset({"scene", "depth"})
    assert depth.shape == (48, 64)
    assert depth.max() == BACKGROUND_DEPTH  # some background is visible
    assert depth.min() < BACKGROUND_DEPTH  # and at least one shape


def test_scene_summary_describes_the_shapes() -> None:
    summary = make_source(shapes=2).scene_summary()
    assert summary is not None
    assert "2 shapes" in summary


def test_unknown_option_rejected() -> None:
    with pytest.raises(ValidationError, match="speeed"):
        make_source(speeed=2.0)


def test_registered_as_stub() -> None:
    assert registry.resolve("frame_source", "stub") is StubSource

    source = registry.create("frame_source", "stub", FrameSourceConfig(options={"frames": 2}))
    assert isinstance(source, StubSource)
    assert source.next_frame().size == (320, 200)  # FrameSourceConfig's default resolution
