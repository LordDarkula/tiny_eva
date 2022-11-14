import pytest
import numpy as np  # type: ignore

from tiny_eva.frame import Frame
from tiny_eva.video import Video


@pytest.mark.skip
def test_simple_query(blank_frame_video):
    all_zeros = all_zeros_udf()
    query = Query().filter(all_zeros).all()
    executor = NaiveExecutor()
    executor.execute(frames, query)
