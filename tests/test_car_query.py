import numpy as np  # type: ignore
import pytest

from tiny_eva.query import Query, Condition
from tiny_eva.udf_loader import UDF
from tiny_eva.frame import AbstractFrame


@pytest.fixture
def num_one_pixels():
    """
    UDF to calculate the number of pixels that are 1 in a frame.
    """

    def num_one_pixels_callable(frame: AbstractFrame):
        return np.count_nonzero(frame.to_numpy() == 1)

    return UDF.from_callable(num_one_pixels_callable)


@pytest.fixture
def num_pixels_is_two(num_one_pixels):
    def two_occurrences_of_one_pixel(result):
        return int(result.result) == 2

    return Query().filter(
        Condition(udf=num_one_pixels, result=two_occurrences_of_one_pixel)
    )
