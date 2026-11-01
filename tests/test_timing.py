import pytest

from lucid_director.core.timing import FakeClock, RealClock, TimingStats


def test_fake_clock_moves_only_when_advanced(fake_clock: FakeClock) -> None:
    assert fake_clock.now() == 0.0
    fake_clock.advance(0.25)
    fake_clock.advance(0.5)
    assert fake_clock.now() == pytest.approx(0.75)


def test_fake_clock_rejects_going_backwards(fake_clock: FakeClock) -> None:
    with pytest.raises(ValueError, match="backwards"):
        fake_clock.advance(-1.0)


def test_real_clock_is_monotonic() -> None:
    clock = RealClock()
    first = clock.now()
    assert clock.now() >= first


def test_fps_and_latency_percentiles() -> None:
    stats = TimingStats(window=100)
    # Frames finish every 50 ms (20 fps); frame i has latency (i + 1) ms, i.e. 1..100 ms.
    # The 1.0 offset only keeps every start positive for readability; stats depend on
    # differences between timestamps, so any offset (including none) gives the same result.
    for i in range(100):
        end = 1.0 + i * 0.050
        stats.record(start=end - (i + 1) / 1000, end=end)

    summary = stats.summary()

    assert summary.frames == 100
    assert summary.window_s == pytest.approx(4.95)  # 99 gaps of 50 ms
    assert summary.fps == pytest.approx(20.0)
    assert summary.p50_ms == pytest.approx(50.0)
    assert summary.p95_ms == pytest.approx(95.0)


def test_window_keeps_only_recent_frames(fake_clock: FakeClock) -> None:
    stats = TimingStats(window=10)
    # 10 slow frames (200 ms latency), then 10 fast ones (10 ms) that push them out.
    for latency in [0.200] * 10 + [0.010] * 10:
        start = fake_clock.now()
        fake_clock.advance(latency)
        stats.record(start, fake_clock.now())

    summary = stats.summary()

    assert summary.frames == 10
    assert summary.window_s == pytest.approx(0.090)
    assert summary.p95_ms == pytest.approx(10.0)
    assert summary.fps == pytest.approx(100.0)


def test_stage_ms_is_mean_per_stage() -> None:
    stats = TimingStats()
    stats.record(0.0, 0.1, {"unet": 40.0, "vae": 10.0})
    stats.record(0.1, 0.2, {"unet": 60.0})

    assert stats.summary().stage_ms == pytest.approx({"unet": 50.0, "vae": 10.0})


def test_empty_and_single_frame_report_zero_fps() -> None:
    stats = TimingStats()
    assert stats.summary().fps == 0.0
    assert stats.summary().frames == 0
    assert stats.summary().window_s == 0.0

    stats.record(0.0, 0.05)
    summary = stats.summary()
    assert summary.fps == 0.0
    assert summary.window_s == 0.0
    assert summary.p95_ms == pytest.approx(50.0)


def test_record_rejects_end_before_start() -> None:
    with pytest.raises(ValueError, match="before start"):
        TimingStats().record(start=1.0, end=0.5)
