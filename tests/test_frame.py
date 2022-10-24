from pathlib import Path

import numpy as np
import pytest
from numpy.types import ArrayLike  # type: ignore

import tiny_eva
from tiny_eva.decoder import Frame


@pytest.fixture
def traffic_frame_path() -> ArrayLike:
    relative = Path(tiny_eva.__file__).parent.parent / "data" / "traffic.jpeg"
    yield relative.resolve()


def test_initialize_from_array():
    np_frame = np.ones((3, 200, 300))
    frame = Frame.from_numpy(np_frame)
    assert np.array_equal(frame.to_numpy(), np_frame)


def test_intialize_from_source(traffic_frame_path):
    frame = Frame.from_source(traffic_frame_path)
    assert frame.to_numpy().shape == (3, 853, 1280)
