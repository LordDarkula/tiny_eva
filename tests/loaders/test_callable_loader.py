import numpy as np  # type: ignore
import pytest

from tiny_eva.frame import Frame


@pytest.fixture
def zeros_frame():
    return Frame.from_numpy(np.zeros((3, 2, 3)))


def test_all_zeros_len_is_one(zeros_frame, all_zeros_udf):
    assert len(all_zeros_udf(zeros_frame)) == 1


def test_all_zeros_true_on_zeros_frame(zeros_frame, all_zeros_udf):
    assert all_zeros_udf(zeros_frame)


def test_all_zeros_false_in_on_ones_frame(all_zeros_udf):
    sample_frame = Frame.from_numpy(np.ones((3, 2, 3)))
    assert not all_zeros_udf(sample_frame)
