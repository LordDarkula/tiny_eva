import numpy as np  # type: ignore
import pytest

from tiny_eva.query import Query, Condition
from tiny_eva.udf_loader import UDF
from tiny_eva.frame import AbstractFrame
from tiny_eva.video import Video
from tiny_eva.frame import Frame
from tiny_eva.result import SingularResult


@pytest.fixture
def white_frame():
    return Frame.from_numpy(np.zeros((1, 100, 300)))


@pytest.fixture
def black_frame():
    return Frame.from_numpy(np.ones((1, 100, 300)))


@pytest.fixture
def num_black_pixels():
    """
    UDF to calculate the number of pixels that are 1 in a frame.
    """

    def num_one_pixels_callable(frame: AbstractFrame):
        return SingularResult(np.count_nonzero(frame.to_numpy() == 1))

    return UDF.from_callable(num_one_pixels_callable)


@pytest.fixture
def num_pixels_is_two(num_black_pixels):
    def two_occurrences_of_one_pixel(result):
        return int(result) == 2

    return Query().filter(
        Condition(udf=num_black_pixels, result=two_occurrences_of_one_pixel)
    )


def test_num_black_pixels_in_white_frame_is_zero(white_frame, num_black_pixels):
    assert int(num_black_pixels(white_frame)) == 0


def test_num_black_pixels_in_black_frame_is_every_pixel(black_frame, num_black_pixels):
    assert int(num_black_pixels(black_frame)) == 30_000
