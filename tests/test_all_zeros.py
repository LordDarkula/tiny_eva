import pytest


@pytest.mark.skip
def test_simple_query(blank_frame_video, all_zeros_udf):
    query = Query().filter(all_zeros_udf).all()
    executor = NaiveExecutor()
    result_video = executor.execute(blank_frame_video, query)
    assert len(result_video) == 40
