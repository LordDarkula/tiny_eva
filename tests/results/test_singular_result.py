from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from tiny_eva.result import SingularResult


@pytest.fixture
def sample_single_result():
    return SingularResult(frame=np.zeros([3, 3, 3]), result=True)


def test_singular_result_is_immutable(sample_single_result):
    with pytest.raises(FrozenInstanceError):
        sample_single_result.result = False


def test_sample_single_result_len_is_one(sample_single_result):
    assert len(sample_single_result) == 1
