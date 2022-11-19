import pytest

from tiny_eva.query import Query


@pytest.fixture
def times_two_query():
    return Query().map(udf=lambda x: 2 * x)


def test_map_len(times_two_query):
    assert len(times_two_query) == 1
