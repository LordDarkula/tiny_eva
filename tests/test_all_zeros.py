import pytest

from tiny_eva.executor import NaiveExecutor
from tiny_eva.query import Query


@pytest.mark.skip
def test_all_query(blank_frame_video, all_zeros_udf):
    query = Query().map(all_zeros_udf).filter(udf="all_zeros_udf", result=True).all()
    executor = NaiveExecutor()
    result_video = executor.execute(blank_frame_video, query)
    assert len(result_video) == 15


@pytest.mark.skip
def test_limit_query(blank_frame_video, all_zeros_udf):
    query = Query().map(all_zeros_udf).filter(udf="all_zeros_udf", result=True).limit(2)
    executor = NaiveExecutor()
    result_video = executor.execute(blank_frame_video, query)
    assert len(result_video) == 2
