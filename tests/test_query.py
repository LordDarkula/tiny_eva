import pytest

from tiny_eva.query import Query


@pytest.fixture
def times_two_query():
    return Query().map(udf=lambda x: 2 * x)


@pytest.fixture
def square_greater_than_thirty_query():
    return Query().map(udf=lambda x: x**2).filter(lambda x: x > 30)


def test_times_two_query_len(times_two_query):
    assert len(times_two_query) == 1


def test_square_greater_than_thirty_len(square_greater_than_thirty_query):
    assert len(square_greater_than_thirty_query) == 2


def test_call_times_two_query(times_two_query):
    assert list(times_two_query([1, 2])) == [2, 4]
