from pathlib import Path

import pytest
from numpy.typing import ArrayLike  # type: ignore

import tiny_eva


@pytest.fixture
def traffic_frame_path() -> ArrayLike:
    relative = Path(tiny_eva.__file__).parent.parent / "data" / "traffic.jpeg"
    yield relative.resolve()
