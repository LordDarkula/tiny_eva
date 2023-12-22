from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd  # type: ignore
import pytest

from tiny_eva.bbox import Bbox
from tiny_eva.result import MultiResult, PandasResult, Result


@pytest.fixture
def sample_multi_result():
    bboxes = {
        Bbox.from_coords(c1=(0, 0), c2=(20, 30), label="car"),
        Bbox.from_coords(c1=(32, 41), c2=(58, 55), label="tree"),
        Bbox.from_coords(c1=(3, 41), c2=(44, 45), label="car"),
    }
    return MultiResult(frame=np.zeros([3, 3, 3]), bboxes=bboxes)


@pytest.fixture
def sample_result_df():
    return pd.DataFrame.from_dict(
        {
            "xmin": [0, 30, 20],
            "ymin": [0, 0, 20],
            "xmax": [50, 40, 30],
            "ymax": [10, 40, 30],
            "confidence": [0.9, 0.5, 0.7],
            "class": [0, 1, 2],
            "name": ["car", "bus", "person"],
        }
    )


@pytest.fixture
def sample_pandas_result(sample_result_df):
    return PandasResult(frame=np.zeros([3, 3, 3]), df=sample_result_df)


@pytest.fixture
def sample_pandas_result_first_bbox():
    return Bbox.from_coords(c1=(0, 0), c2=(50, 10), label="car")


def test_intialize_abstract_result():
    with pytest.raises(TypeError):
        Result()


def test_sample_result_is_immutable(sample_multi_result):
    with pytest.raises(FrozenInstanceError):
        sample_multi_result.bboxes = []


def test_sample_result_len_is_three(sample_multi_result):
    assert len(sample_multi_result) == 3


def test_sample_result_truthiness(sample_multi_result):
    assert sample_multi_result


def test_pandas_result_bool(sample_pandas_result):
    assert not bool(sample_pandas_result)


def test_pandas_result_len(sample_pandas_result):
    assert len(sample_pandas_result) == 3


def test_pandas_result_index(sample_pandas_result, sample_pandas_result_first_bbox):
    assert sample_pandas_result[0] == sample_pandas_result_first_bbox
