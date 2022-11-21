import pytest

from tiny_eva.query import Query, Condition


@pytest.fixture
def times_two_query():
    return Query().map(udf=lambda x: 2 * x)


@pytest.fixture
def square_greater_than_thirty_query():
    return Query().map(udf=lambda x: x**2).filter(lambda x: x > 30)


@pytest.fixture
def filter_values_if_square_greater_than_thirty_query():
    return Query().filter(Condition(udf=lambda x: x**2, result=lambda x: x > 30))


def test_times_two_query_len(times_two_query):
    assert len(times_two_query) == 1


def test_square_greater_than_thirty_len(square_greater_than_thirty_query):
    assert len(square_greater_than_thirty_query) == 2


def test_call_times_two_query(times_two_query):
    assert list(times_two_query([1, 2])) == [2, 4]


def test_call_square_greater_than_thirty(square_greater_than_thirty_query):
    assert list(square_greater_than_thirty_query([2, 4, 3, 6, 7, 5])) == [36, 49]


def test_call_filter_values_if_square_greater_than_thirty_query(
    filter_values_if_square_greater_than_thirty_query,
):
    assert list(
        filter_values_if_square_greater_than_thirty_query([2, 4, 3, 6, 7, 5])
    ) == [6, 7]
