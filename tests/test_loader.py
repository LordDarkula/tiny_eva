import numpy as np
from numpy.typing import ArrayLike
import pytest

from tiny_eva.loader import UDF

@pytest.fixture
def all_zeros_callable():
    def all_zeros(frame: ArrayLike) -> bool:
        return not np.any(frame)
    
    yield all_zeros

def test_load_bool_udf_from_callable(all_zeros_callable):
    is_empty_udf = UDF.from_callable(all_zeros_callable)
    sample_frame = np.zeros((3, 2, 3))
    assert is_empty_udf(sample_frame)

