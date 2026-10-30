import dataclasses

import numpy as np
import pytest

from lucid_director.core.config import StyleSettings
from lucid_director.core.types import Frame, PipelineResult


def image(width: int = 32, height: int = 20) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def test_frame_size_is_width_height() -> None:
    assert Frame(image(32, 20), index=0, timestamp=0.0).size == (32, 20)


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((20, 32, 3), dtype=np.float32),  # wrong dtype
        np.zeros((20, 32), dtype=np.uint8),  # no channel axis
        np.zeros((20, 32, 4), dtype=np.uint8),  # RGBA
    ],
)
def test_frame_rejects_bad_images(bad: np.ndarray) -> None:
    with pytest.raises(ValueError, match="uint8"):
        Frame(bad, index=0, timestamp=0.0)


def test_frame_rejects_mismatched_depth() -> None:
    with pytest.raises(ValueError, match="depth"):
        Frame(image(32, 20), index=0, timestamp=0.0, depth=np.zeros((10, 10)))


def test_frame_is_immutable() -> None:
    frame = Frame(image(), index=0, timestamp=0.0)

    with pytest.raises(dataclasses.FrozenInstanceError):
        frame.index = 1  # type: ignore[misc]


def test_pipeline_result_validates_image() -> None:
    with pytest.raises(ValueError):
        PipelineResult(np.zeros((4, 4), dtype=np.uint8))


def test_settings_with_changes_returns_new_copy() -> None:
    before = StyleSettings(strength=0.5)
    after = before.with_changes(strength=0.7)

    assert (before.strength, after.strength) == (0.5, 0.7)


def test_settings_with_changes_validates() -> None:
    with pytest.raises(ValueError, match="strength"):
        StyleSettings().with_changes(strength=1.5)
