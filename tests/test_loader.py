import numpy as np  # type: ignore
from numpy.typing import ArrayLike  # type: ignore
import pytest

from tiny_eva.loader import UDF


@pytest.fixture
def all_zeros():
    def all_zeros(frame: ArrayLike) -> bool:
        return not np.any(frame)

    yield UDF.from_callable(all_zeros)


def test_all_zeros_true(all_zeros):
    sample_frame = np.zeros((3, 2, 3))
    assert all_zeros(sample_frame)


def test_all_zeros_false(all_zeros):
    sample_frame = np.ones((3, 2, 3))
    assert not all_zeros(sample_frame)
