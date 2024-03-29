import pytest

from tiny_eva.executor import NaiveExecutor
from tiny_eva.loaders.udf_loader import UDF
from tiny_eva.query import Query
from tiny_eva.video import Video


def test_all_query(blank_frame_video: Video, all_zeros_udf: UDF):
    query = Query().map(all_zeros_udf).filter(condition=all_zeros_udf)
    result_video = query(blank_frame_video)
    assert len(list(result_video)) == 15


@pytest.mark.skip
def test_limit_query(blank_frame_video: Video, all_zeros_udf: UDF):
    query = Query().map(all_zeros_udf).filter(udf="all_zeros_udf", result=True).limit(2)
    executor = NaiveExecutor()
    result_video = executor.execute(blank_frame_video, query)
    assert len(result_video) == 2
