from dataclasses import FrozenInstanceError

import pytest


from tiny_eva.result import MultiResult, Result
from tiny_eva.bbox import Bbox


@pytest.fixture
def sample_multi_result():
    bboxes = {
        Bbox.from_coords(c1=(0, 0), c2=(20, 30), label="car"),
        Bbox.from_coords(c1=(32, 41), c2=(58, 55), label="tree"),
        Bbox.from_coords(c1=(3, 41), c2=(44, 45), label="car"),
    }
    return MultiResult(bboxes)


def test_intialize_abstract_result():
    with pytest.raises(TypeError):
        Result()


def test_sample_result_is_immutable(sample_multi_result):
    with pytest.raises(FrozenInstanceError):
        sample_multi_result.bboxes = []


def test_sample_result_len_is_three(sample_multi_result):
    assert len(sample_multi_result) == 3


def test_sample_single_result_len_is_one(sample_single_result):
    assert len(sample_single_result) == 1


def test_sample_result_truthiness(sample_multi_result):
    assert sample_multi_result
