import pytest

from lucid_director.core.timing import FakeClock


@pytest.fixture
def fake_clock() -> FakeClock:
    return FakeClock()
