from dataclasses import FrozenInstanceError

import pytest

from tiny_eva.result import SingularResult


@pytest.fixture
def sample_single_result():
    return SingularResult(True)


def test_singular_result_is_immutable(sample_single_result):
    with pytest.raises(FrozenInstanceError):
        sample_single_result.result = False
